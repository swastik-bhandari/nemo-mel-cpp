// nemo_mel.cpp
// ============================================================================
// C++ reimplementation of NeMo's AudioToMelSpectrogramPreprocessor default
// pipeline (FilterbankFeatures, eval mode), targeting float32-rounding-level
// agreement with the Python/PyTorch reference.
//
// Pipeline replicated (defaults: sr=16000, win=320, hop=160, n_fft=512,
// n_mels=64, preemph=0.97, log add 2^-24, per_feature norm, pad_to=16,
// dither skipped = eval mode):
//
//   1. load wav        : libsndfile float32 read      (== python-soundfile)
//   2. stereo downmix  : mean across channels          (channel_selector='average')
//   3. resample        : libsoxr float32 SOXR_HQ       (== librosa soxr_hq default)
//   4. preemphasis     : y[0]=x[0]; y[i]=x[i]-0.97*x[i-1]
//   5. STFT torch-style: center=True, pad_mode="constant" (zero pad n_fft/2
//                        each side), hann(320,periodic=False) zero-padded
//                        centered to n_fft, rFFT via PocketFFT (the same FFT
//                        PyTorch bundles for CPU)
//   6. magnitude       : sqrt(re^2+im^2) then square    (same op order as
//                        torch: view_as_real -> sqrt(sum(pow(2))) -> pow(2))
//   7. mel projection  : fb[64,257] x power[257,T]      (fb dumped from
//                        librosa via dump_constants.py -- bit-identical
//                        weights)
//   8. log             : log(x + 2^-24)
//   9. normalize       : per_feature, (N-1) denominator, +1e-5, computed
//                        in float32
//  10. mask + pad_to   : frames >= seq_len zeroed; time dim padded with 0
//                        to a multiple of 16.  seq_len = floor(L/hop).
//
// All arithmetic is float32 throughout, matching the reference.
//
// Build:
//   g++ -O2 -std=c++17 nemo_mel.cpp -o nemo_mel -lsndfile -lsoxr
//
// Run:
//   ./nemo_mel input.wav nemo_mel_constants.bin output_mel.bin
//
// Output binary layout (little-endian):
//   int32 n_mels, int32 total_frames (incl. padding), int32 seq_len,
//   float32[n_mels * total_frames]  row-major [n_mels, total_frames]
// ============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sndfile.h>
#include <soxr.h>

#include "pocketfft_hdronly.h"

struct Constants {
    int32_t sr, n_fft, win, hop, n_mels, pad_to;
    std::vector<float> window;  // [win]
    std::vector<float> fb;      // [n_mels, n_fft/2+1] row-major
};

static Constants load_constants(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open constants file: " + path);
    Constants c{};
    int32_t hdr[6];
    if (std::fread(hdr, sizeof(int32_t), 6, f) != 6) throw std::runtime_error("bad constants header");
    c.sr = hdr[0]; c.n_fft = hdr[1]; c.win = hdr[2]; c.hop = hdr[3]; c.n_mels = hdr[4]; c.pad_to = hdr[5];
    const int n_freq = c.n_fft / 2 + 1;
    c.window.resize(c.win);
    c.fb.resize(static_cast<size_t>(c.n_mels) * n_freq);
    if (std::fread(c.window.data(), sizeof(float), c.win, f) != static_cast<size_t>(c.win))
        throw std::runtime_error("bad window data");
    if (std::fread(c.fb.data(), sizeof(float), c.fb.size(), f) != c.fb.size())
        throw std::runtime_error("bad filterbank data");
    std::fclose(f);
    return c;
}

// --- 1+2: load wav as float32 (libsndfile == python-soundfile), downmix ----
static std::vector<float> load_wav_mono(const std::string& path, int& sr_out) {
    SF_INFO info{};
    SNDFILE* snd = sf_open(path.c_str(), SFM_READ, &info);
    if (!snd) throw std::runtime_error("cannot open wav: " + path);
    sr_out = info.samplerate;
    std::vector<float> interleaved(static_cast<size_t>(info.frames) * info.channels);
    sf_count_t got = sf_readf_float(snd, interleaved.data(), info.frames);
    sf_close(snd);
    if (got != info.frames) throw std::runtime_error("short read on wav");

    if (info.channels == 1) return interleaved;

    // channel_selector='average'  (np.mean over channel axis, float32)
    std::vector<float> mono(info.frames);
    for (sf_count_t i = 0; i < info.frames; ++i) {
        float acc = 0.0f;
        for (int ch = 0; ch < info.channels; ++ch)
            acc += interleaved[static_cast<size_t>(i) * info.channels + ch];
        mono[i] = acc / static_cast<float>(info.channels);
    }
    return mono;
}

// --- 3: resample via libsoxr, HQ, float32 (librosa's soxr_hq backend) ------
static std::vector<float> resample_soxr(const std::vector<float>& in, int sr_in, int sr_out) {
    if (sr_in == sr_out) return in;
    const double ratio = static_cast<double>(sr_out) / sr_in;
    // librosa: int(np.ceil(n * ratio)) output samples
    const size_t olen = static_cast<size_t>(std::ceil(in.size() * ratio));
    std::vector<float> out(olen);
    soxr_quality_spec_t q = soxr_quality_spec(SOXR_HQ, 0);
    soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    size_t idone = 0, odone = 0;
    soxr_error_t err = soxr_oneshot(sr_in, sr_out, 1,
                                    in.data(), in.size(), &idone,
                                    out.data(), olen, &odone,
                                    &io, &q, nullptr);
    if (err) throw std::runtime_error(std::string("soxr: ") + err);
    out.resize(odone);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s input.wav constants.bin output_mel.bin\n", argv[0]);
        return 1;
    }
    const std::string wav_path = argv[1], const_path = argv[2], out_path = argv[3];

    const Constants C = load_constants(const_path);
    const int n_freq = C.n_fft / 2 + 1;

    // ---- load + downmix + resample ----
    int sr_file = 0;
    std::vector<float> x = load_wav_mono(wav_path, sr_file);
    x = resample_soxr(x, sr_file, C.sr);
    const int64_t L = static_cast<int64_t>(x.size());

    // seq_len exactly as FilterbankFeatures.get_seq_len:
    //   floor((L + 2*(n_fft//2) - n_fft) / hop)  ==  floor(L / hop)
    const int64_t seq_len = (L + 2 * (C.n_fft / 2) - C.n_fft) / C.hop;

    // ---- 4: preemphasis (in-place, right-to-left preserves x[i-1]) ----
    for (int64_t i = L - 1; i >= 1; --i) x[i] = x[i] - 0.97f * x[i - 1];
    // x[0] unchanged; time-mask is a no-op for a single full-length signal.

    // ---- 5: torch-style STFT ----
    // center=True: zero-pad n_fft/2 on both sides (pad_mode="constant")
    const int pad = C.n_fft / 2;
    std::vector<float> xp(static_cast<size_t>(L) + 2 * pad, 0.0f);
    std::memcpy(xp.data() + pad, x.data(), static_cast<size_t>(L) * sizeof(float));

    // window of length win < n_fft is zero-padded *centered* to n_fft
    std::vector<float> win_full(C.n_fft, 0.0f);
    const int wpad_left = (C.n_fft - C.win) / 2;
    std::memcpy(win_full.data() + wpad_left, C.window.data(), C.win * sizeof(float));

    const int64_t T = (static_cast<int64_t>(xp.size()) - C.n_fft) / C.hop + 1;

    // PocketFFT real-to-complex, one frame at a time.
    // (PyTorch's own CPU FFT is PocketFFT, so algorithmic parity is high.)
    using pocketfft::shape_t;
    using pocketfft::stride_t;
    shape_t shape_in{static_cast<size_t>(C.n_fft)};
    stride_t stride_in{sizeof(float)};
    stride_t stride_out{sizeof(std::complex<float>)};
    shape_t axes{0};

    std::vector<float> frame(C.n_fft);
    std::vector<std::complex<float>> spec(n_freq);
    // power spectrum, layout [n_freq, T] to make the mel matmul cache-friendly
    std::vector<float> power(static_cast<size_t>(n_freq) * T);

    for (int64_t t = 0; t < T; ++t) {
        const float* src = xp.data() + t * C.hop;
        for (int i = 0; i < C.n_fft; ++i) frame[i] = src[i] * win_full[i];
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       frame.data(), spec.data(), 1.0f);
        // ---- 6: same op order as torch: sqrt(re^2+im^2), then square ----
        for (int k = 0; k < n_freq; ++k) {
            const float re = spec[k].real(), im = spec[k].imag();
            const float mag = std::sqrt(re * re + im * im);
            power[static_cast<size_t>(k) * T + t] = mag * mag;
        }
    }

    // ---- 7: mel projection  fb[n_mels, n_freq] x power[n_freq, T] ----
    // ---- 8: log(x + 2^-24) ----
    const float guard = std::pow(2.0f, -24.0f);
    std::vector<float> mel(static_cast<size_t>(C.n_mels) * T, 0.0f);
    for (int m = 0; m < C.n_mels; ++m) {
        const float* fbrow = &C.fb[static_cast<size_t>(m) * n_freq];
        float* melrow = &mel[static_cast<size_t>(m) * T];
        for (int k = 0; k < n_freq; ++k) {
            const float w = fbrow[k];
            if (w == 0.0f) continue;  // filterbank is sparse (triangles)
            const float* prow = &power[static_cast<size_t>(k) * T];
            for (int64_t t = 0; t < T; ++t) melrow[t] += w * prow[t];
        }
        for (int64_t t = 0; t < T; ++t) melrow[t] = std::log(melrow[t] + guard);
    }

    // ---- 9: per_feature normalization over VALID frames (< seq_len) ----
    // mean = sum/N ; std = sqrt( sum((x-mean)^2) / (N-1) ) ; std += 1e-5
    const float CONSTANT = 1e-5f;
    for (int m = 0; m < C.n_mels; ++m) {
        float* row = &mel[static_cast<size_t>(m) * T];
        float sum = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) sum += row[t];
        const float mean = sum / static_cast<float>(seq_len);
        float ss = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) {
            const float d = row[t] - mean;
            ss += d * d;
        }
        float sd = std::sqrt(ss / (static_cast<float>(seq_len) - 1.0f));
        if (std::isnan(sd)) sd = 0.0f;  // edge case: seq_len == 1
        sd += CONSTANT;
        for (int64_t t = 0; t < T; ++t) row[t] = (row[t] - mean) / sd;
    }

    // ---- 10: zero frames beyond seq_len, pad time dim to multiple of pad_to
    int64_t total = T;
    if (C.pad_to > 0) {
        const int64_t rem = T % C.pad_to;
        if (rem != 0) total = T + (C.pad_to - rem);
    }
    std::vector<float> out(static_cast<size_t>(C.n_mels) * total, 0.0f);
    for (int m = 0; m < C.n_mels; ++m)
        for (int64_t t = 0; t < seq_len && t < T; ++t)
            out[static_cast<size_t>(m) * total + t] = mel[static_cast<size_t>(m) * T + t];

    // ---- write output ----
    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    const int32_t hdr[3] = {C.n_mels, static_cast<int32_t>(total), static_cast<int32_t>(seq_len)};
    std::fwrite(hdr, sizeof(int32_t), 3, f);
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);

    std::printf("wrote %s: mel [%d, %lld], valid frames %lld  (input: %lld samples @ %d Hz)\n",
                out_path.c_str(), C.n_mels, static_cast<long long>(total),
                static_cast<long long>(seq_len), static_cast<long long>(L), C.sr);
    return 0;
}
