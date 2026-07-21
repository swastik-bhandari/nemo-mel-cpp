// indic_mel.cpp
// ============================================================================
// Standalone C++ port of indic_frontend.py (the torch-free numpy front-end for
// ai4bharat Indic-Conformer). WAV -> mel spectrogram [80, T], no Python.
//
// Reproduces, step for step, indic_frontend.py:
//   1. load wav        : minimal 16-bit PCM RIFF parser (== stdlib `wave`)
//   2. stereo downmix  : mean across channels, float32   (server.py path)
//   3. resample        : EXACT port of torchaudio sinc_interp_hann kernel
//                        (lowpass_filter_width=6, rolloff=0.99). NOT libsoxr —
//                        soxr flips tokens; this matches torchaudio to ~1e-8.
//   4. preemphasis     : y[0]=x[0]; y[i]=x[i]-0.97*x[i-1]
//   5. STFT            : reflect-pad n_fft/2 (=256), hann(400,periodic=False)
//                        zero-padded centered to n_fft=512, rFFT via PocketFFT
//                        in double (matches numpy np.fft.rfft precision)
//   6. power           : re^2 + im^2
//   7. mel projection  : slaney filterbank [80,257] (librosa htk=False,
//                        norm="slaney"), computed here — no constants file
//   8. log             : log(mel + 2^-24)
//   9. normalize       : per-feature over ALL frames, (N-1) denom, +1e-5
//
// The mel filterbank and Hann window are computed in-code (double, then
// rounded to float32 like numpy), so the binary needs ONLY the wav file.
//
// Build:
//   g++ -O2 -std=c++17 indic_mel.cpp -o indic_mel
//
// Run:
//   ./indic_mel input.wav output_mel.bin
//
// Output binary layout (little-endian):
//   int32 n_mels (=80), int32 T
//   float32[n_mels * T]  row-major [n_mels, T]     (this is audio_signal[0])
// ============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "pocketfft_hdronly.h"

// ---- constants (must match indic_frontend.py) ------------------------------
static constexpr int    SR       = 16000;
static constexpr int    N_FFT    = 512;
static constexpr int    WIN      = 400;      // 25 ms
static constexpr int    HOP      = 160;      // 10 ms
static constexpr int    N_MELS   = 80;
static constexpr double FMIN     = 0.0;
static constexpr double FMAX     = SR / 2.0; // 8000
static constexpr float  PREEMPH  = 0.97f;
static constexpr float  NORM_EPS = 1e-5f;

// ============================================================================
// 1+2. minimal WAV reader: 16-bit PCM, any channels/rate -> mono float32
// ============================================================================
static std::vector<float> load_wav_mono(const std::string& path, int& sr_out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(sz);
    if (std::fread(b.data(), 1, sz, f) != static_cast<size_t>(sz)) {
        std::fclose(f);
        throw std::runtime_error("short read on wav file");
    }
    std::fclose(f);

    auto u16 = [&](size_t o) { return static_cast<uint16_t>(b[o] | (b[o + 1] << 8)); };
    auto u32 = [&](size_t o) {
        return static_cast<uint32_t>(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24));
    };
    if (sz < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF/WAVE file");

    int channels = 0, bits = 0;
    sr_out = 0;
    size_t data_off = 0, data_len = 0;

    // walk chunks: [id(4)][size(4)][payload(size)] (payload padded to even)
    size_t p = 12;
    while (p + 8 <= static_cast<size_t>(sz)) {
        const char* id = reinterpret_cast<const char*>(b.data() + p);
        uint32_t csz = u32(p + 4);
        size_t body = p + 8;
        if (std::memcmp(id, "fmt ", 4) == 0 && body + 16 <= static_cast<size_t>(sz)) {
            uint16_t fmt = u16(body);
            channels = u16(body + 2);
            sr_out = static_cast<int>(u32(body + 4));
            bits = u16(body + 14);
            if (fmt != 1) throw std::runtime_error("only PCM (fmt=1) WAV supported");
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_off = body;
            data_len = csz;
            if (data_off + data_len > static_cast<size_t>(sz)) data_len = sz - data_off; // be lenient
        }
        p = body + csz + (csz & 1); // chunks are word-aligned
    }
    if (bits != 16) throw std::runtime_error("expected 16-bit PCM WAV");
    if (channels < 1 || sr_out <= 0 || data_off == 0)
        throw std::runtime_error("malformed WAV (no fmt/data)");

    const size_t n_frames = data_len / (2 * channels);
    const int16_t* pcm = reinterpret_cast<const int16_t*>(b.data() + data_off);
    std::vector<float> mono(n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c)
            acc += static_cast<float>(pcm[i * channels + c]) / 32768.0f;
        mono[i] = acc / static_cast<float>(channels);
    }
    return mono;
}

// ============================================================================
// 3. resample — exact port of torchaudio.functional.resample
//    (sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99)
// ============================================================================
static std::vector<float> resample(const std::vector<float>& wav, int orig_freq, int new_freq) {
    if (orig_freq == new_freq) return wav;
    const int lpw = 6;
    const double rolloff = 0.99;
    int g = std::gcd(orig_freq, new_freq);
    int of = orig_freq / g, nf = new_freq / g;
    double base_freq = std::min(of, nf) * rolloff;
    int width = static_cast<int>(std::ceil(lpw * of / base_freq));
    int K = 2 * width + of;

    // kernels[j*K + k], j in [0,nf), k in [0,K)   (computed in double -> float32)
    std::vector<float> kernels(static_cast<size_t>(nf) * K);
    const double scale = base_freq / of;
    for (int j = 0; j < nf; ++j) {
        for (int k = 0; k < K; ++k) {
            double idx = static_cast<double>(-width + k) / of;
            double t = (static_cast<double>(-j) / nf + idx) * base_freq;
            if (t < -lpw) t = -lpw;
            if (t > lpw) t = lpw;
            double window = std::pow(std::cos(t * M_PI / lpw / 2.0), 2.0);
            double tp = t * M_PI;
            double sinc = (tp == 0.0) ? 1.0 : std::sin(tp) / tp;
            kernels[static_cast<size_t>(j) * K + k] = static_cast<float>(sinc * window * scale);
        }
    }

    const long L = static_cast<long>(wav.size());
    // padded = pad(wav, (width, width+of))
    std::vector<float> padded(static_cast<size_t>(L) + 2 * width + of, 0.0f);
    std::memcpy(padded.data() + width, wav.data(), static_cast<size_t>(L) * sizeof(float));

    long n_out = (static_cast<long>(padded.size()) - K) / of + 1;
    long target_length = static_cast<long>(std::ceil(static_cast<double>(new_freq) * L / orig_freq));

    // resampled[i*nf + j] = sum_k padded[i*of + k] * kernels[j][k]   (float32 accum, like numpy sgemm)
    std::vector<float> out(static_cast<size_t>(n_out) * nf);
    for (long i = 0; i < n_out; ++i) {
        const float* frame = padded.data() + i * of;
        for (int j = 0; j < nf; ++j) {
            const float* ker = kernels.data() + static_cast<size_t>(j) * K;
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += frame[k] * ker[k];
            out[static_cast<size_t>(i) * nf + j] = acc;
        }
    }
    if (static_cast<long>(out.size()) > target_length) out.resize(target_length);
    return out;
}

// ============================================================================
// slaney mel filterbank (librosa htk=False, norm="slaney"), computed in double
// ============================================================================
static double hz_to_mel(double f) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0, min_log_mel = 1000.0 / f_sp, logstep = std::log(6.4) / 27.0;
    return (f >= min_log_hz) ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
}
static double mel_to_hz(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0, min_log_mel = 1000.0 / f_sp, logstep = std::log(6.4) / 27.0;
    return (m >= min_log_mel) ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
}
// returns fb[N_MELS * n_freq] row-major, float32-rounded like numpy
static std::vector<float> mel_filterbank(int n_freq) {
    std::vector<double> fftfreqs(n_freq);
    for (int i = 0; i < n_freq; ++i) fftfreqs[i] = (SR / 2.0) * i / (n_freq - 1);
    std::vector<double> freq_pts(N_MELS + 2);
    double m_lo = hz_to_mel(FMIN), m_hi = hz_to_mel(FMAX);
    for (int i = 0; i < N_MELS + 2; ++i)
        freq_pts[i] = mel_to_hz(m_lo + (m_hi - m_lo) * i / (N_MELS + 1));
    std::vector<float> fb(static_cast<size_t>(N_MELS) * n_freq, 0.0f);
    for (int i = 0; i < N_MELS; ++i) {
        double fdiff0 = freq_pts[i + 1] - freq_pts[i];
        double fdiff1 = freq_pts[i + 2] - freq_pts[i + 1];
        double enorm = 2.0 / (freq_pts[i + 2] - freq_pts[i]); // slaney
        for (int k = 0; k < n_freq; ++k) {
            double lower = (fftfreqs[k] - freq_pts[i]) / fdiff0;
            double upper = (freq_pts[i + 2] - fftfreqs[k]) / fdiff1;
            double v = std::max(0.0, std::min(lower, upper));
            fb[static_cast<size_t>(i) * n_freq + k] = static_cast<float>(v * enorm);
        }
    }
    return fb;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input.wav output_mel.bin\n", argv[0]);
        return 1;
    }
    const std::string wav_path = argv[1], out_path = argv[2];
    const int n_freq = N_FFT / 2 + 1; // 257

    try {
        // ---- 1+2 load + downmix ----
        int sr_file = 0;
        std::vector<float> x = load_wav_mono(wav_path, sr_file);
        // ---- 3 resample to 16k (sinc) ----
        x = resample(x, sr_file, SR);
        const long L = static_cast<long>(x.size());
        if (L < N_FFT) throw std::runtime_error("audio too short after resample");

        // ---- 4 preemphasis: y[0]=x[0], y[i]=x[i]-0.97*x[i-1] (right-to-left in place) ----
        for (long i = L - 1; i >= 1; --i) x[i] = x[i] - PREEMPH * x[i - 1];

        // ---- 5 reflect pad n_fft/2 = 256 each side ----
        const int pad = N_FFT / 2; // 256
        std::vector<float> xp(static_cast<size_t>(L) + 2 * pad);
        for (int j = 0; j < pad; ++j) xp[j] = x[pad - j];               // reflect (no edge repeat)
        std::memcpy(xp.data() + pad, x.data(), static_cast<size_t>(L) * sizeof(float));
        for (int k = 0; k < pad; ++k) xp[pad + L + k] = x[L - 2 - k];

        // Hann window (periodic=False) length WIN, zero-padded centered to N_FFT
        std::vector<float> win_full(N_FFT, 0.0f);
        const int wpad = (N_FFT - WIN) / 2; // 56
        for (int i = 0; i < WIN; ++i)
            win_full[wpad + i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (WIN - 1)));

        const long T = (static_cast<long>(xp.size()) - N_FFT) / HOP + 1;
        const std::vector<float> fb = mel_filterbank(n_freq);

        // ---- 5/6 STFT + power via PocketFFT (double, matching numpy rfft) ----
        using pocketfft::shape_t;
        using pocketfft::stride_t;
        shape_t shape_in{static_cast<size_t>(N_FFT)};
        stride_t stride_in{sizeof(double)};
        stride_t stride_out{sizeof(std::complex<double>)};
        shape_t axes{0};

        std::vector<double> frame(N_FFT);
        std::vector<std::complex<double>> spec(n_freq);
        // mel[T * N_MELS] row-major [T, N_MELS]
        std::vector<float> mel(static_cast<size_t>(T) * N_MELS);
        const float guard = std::pow(2.0f, -24.0f);

        for (long t = 0; t < T; ++t) {
            const float* src = xp.data() + t * HOP;
            for (int i = 0; i < N_FFT; ++i)
                frame[i] = static_cast<double>(src[i] * win_full[i]); // multiply in float32, then widen
            pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                           frame.data(), spec.data(), 1.0);
            // power = re^2 + im^2 (double), cast to float32 like indic_frontend
            float power[257];
            for (int k = 0; k < n_freq; ++k) {
                double re = spec[k].real(), im = spec[k].imag();
                power[k] = static_cast<float>(re * re + im * im);
            }
            // mel[t] = power @ fb.T   (float32 accum), then log(mel + 2^-24)
            float* mrow = &mel[static_cast<size_t>(t) * N_MELS];
            for (int m = 0; m < N_MELS; ++m) {
                const float* fbrow = &fb[static_cast<size_t>(m) * n_freq];
                float acc = 0.0f;
                for (int k = 0; k < n_freq; ++k) acc += power[k] * fbrow[k];
                mrow[m] = std::log(acc + guard);
            }
        }

        // ---- 9 per-feature normalization over ALL T frames, ddof=1, +1e-5 ----
        // (indic_frontend normalizes across the full time axis; no seq_len mask)
        for (int m = 0; m < N_MELS; ++m) {
            float sum = 0.0f;
            for (long t = 0; t < T; ++t) sum += mel[static_cast<size_t>(t) * N_MELS + m];
            float mean = sum / static_cast<float>(T);
            float ss = 0.0f;
            for (long t = 0; t < T; ++t) {
                float d = mel[static_cast<size_t>(t) * N_MELS + m] - mean;
                ss += d * d;
            }
            float sd = std::sqrt(ss / (static_cast<float>(T) - 1.0f)) + NORM_EPS;
            for (long t = 0; t < T; ++t) {
                float& v = mel[static_cast<size_t>(t) * N_MELS + m];
                v = (v - mean) / sd;
            }
        }

        // ---- write as [n_mels, T] row-major (transpose from [T, n_mels]) ----
        std::vector<float> out(static_cast<size_t>(N_MELS) * T);
        for (long t = 0; t < T; ++t)
            for (int m = 0; m < N_MELS; ++m)
                out[static_cast<size_t>(m) * T + t] = mel[static_cast<size_t>(t) * N_MELS + m];

        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output: " + out_path);
        const int32_t hdr[2] = {N_MELS, static_cast<int32_t>(T)};
        std::fwrite(hdr, sizeof(int32_t), 2, f);
        std::fwrite(out.data(), sizeof(float), out.size(), f);
        std::fclose(f);

        std::printf("wrote %s: mel [%d, %ld]  (input %s -> %ld samples @ %d Hz)\n",
                    out_path.c_str(), N_MELS, T, wav_path.c_str(), L, SR);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
