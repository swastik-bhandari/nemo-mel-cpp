// indic_asr.cpp
// ============================================================================
// Fully-native Indic ASR: WAV -> text, in ONE C++ binary. No Python, no torch,
// no NeMo. onnxruntime (C++ API) runs the model; everything else is in here.
//
//   wav --(mel, this file)--> [80,T] --(encoder.onnx)--> --(ctc_decoder.onnx)-->
//       --(language mask + greedy CTC + SentencePiece detokenize)--> text
//
// The mel front-end is the verified port from indic_mel.cpp (own WAV parser,
// torchaudio sinc resampler, PocketFFT, slaney mel). Decode tables (per-language
// column indices + vocab) are loaded from <exe_dir>/lang/<code>.tbl, produced by
// dump_lang_tables.py — so no JSON/SentencePiece library is needed here.
//
// Model files are expected at <exe_dir>/assets/encoder.onnx + ctc_decoder.onnx
// (with their external weight blobs in the same dir).
//
// Build:
//   ORT=/path/to/onnxruntime-linux-x64-1.20.1
//   g++ -O2 -std=c++17 indic_asr.cpp -o indic_asr \
//       -I$ORT/include -L$ORT/lib -lonnxruntime -Wl,-rpath,'$ORIGIN/lib'
//
// Run:
//   ./indic_asr input.wav [lang=ne]
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
#include <unistd.h>
#include <limits.h>

#include "pocketfft_hdronly.h"
#include "onnxruntime_cxx_api.h"

static constexpr int    SR = 16000, N_FFT = 512, WIN = 400, HOP = 160, N_MELS = 80;
static constexpr double FMIN = 0.0, FMAX = SR / 2.0;
static constexpr float  PREEMPH = 0.97f, NORM_EPS = 1e-5f;

// ---- directory of this executable (so assets/ and lang/ are found relatively) ----
static std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    std::string p(buf);
    size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

// ============================================================================
// WAV -> mono float32 (16-bit PCM RIFF parser)
// ============================================================================
static std::vector<float> load_wav_mono(const std::string& path, int& sr_out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(sz);
    if (std::fread(b.data(), 1, sz, f) != (size_t)sz) { std::fclose(f); throw std::runtime_error("short read"); }
    std::fclose(f);
    auto u16 = [&](size_t o){ return (uint16_t)(b[o] | (b[o+1] << 8)); };
    auto u32 = [&](size_t o){ return (uint32_t)(b[o] | (b[o+1]<<8) | (b[o+2]<<16) | (b[o+3]<<24)); };
    if (sz < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data()+8, "WAVE", 4))
        throw std::runtime_error("not a RIFF/WAVE file");
    int channels = 0, bits = 0; sr_out = 0; size_t data_off = 0, data_len = 0;
    for (size_t p = 12; p + 8 <= (size_t)sz; ) {
        const char* id = (const char*)(b.data() + p); uint32_t csz = u32(p+4); size_t body = p + 8;
        if (!std::memcmp(id,"fmt ",4) && body+16 <= (size_t)sz) {
            if (u16(body) != 1) throw std::runtime_error("only PCM WAV supported");
            channels = u16(body+2); sr_out = (int)u32(body+4); bits = u16(body+14);
        } else if (!std::memcmp(id,"data",4)) {
            data_off = body; data_len = csz;
            if (data_off + data_len > (size_t)sz) data_len = sz - data_off;
        }
        p = body + csz + (csz & 1);
    }
    if (bits != 16) throw std::runtime_error("expected 16-bit PCM WAV");
    if (channels < 1 || sr_out <= 0 || !data_off) throw std::runtime_error("malformed WAV");
    size_t nfr = data_len / (2 * channels);
    const int16_t* pcm = (const int16_t*)(b.data() + data_off);
    std::vector<float> mono(nfr);
    for (size_t i = 0; i < nfr; ++i) {
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c) acc += (float)pcm[i*channels + c] / 32768.0f;
        mono[i] = acc / channels;
    }
    return mono;
}

// ---- torchaudio sinc_interp_hann resample (exact port) ----
static std::vector<float> resample(const std::vector<float>& wav, int orig, int nw) {
    if (orig == nw) return wav;
    const int lpw = 6; const double rolloff = 0.99;
    int g = std::gcd(orig, nw), of = orig/g, nf = nw/g;
    double base = std::min(of, nf) * rolloff;
    int width = (int)std::ceil(lpw * of / base), K = 2*width + of;
    std::vector<float> ker((size_t)nf * K); const double scale = base / of;
    for (int j = 0; j < nf; ++j) for (int k = 0; k < K; ++k) {
        double idx = (double)(-width + k) / of, t = ((double)(-j)/nf + idx) * base;
        t = std::max((double)-lpw, std::min((double)lpw, t));
        double win = std::pow(std::cos(t*M_PI/lpw/2.0), 2.0), tp = t*M_PI;
        double sinc = (tp == 0.0) ? 1.0 : std::sin(tp)/tp;
        ker[(size_t)j*K + k] = (float)(sinc * win * scale);
    }
    long L = (long)wav.size();
    std::vector<float> pad((size_t)L + 2*width + of, 0.0f);
    std::memcpy(pad.data()+width, wav.data(), (size_t)L*sizeof(float));
    long n_out = ((long)pad.size() - K)/of + 1;
    long target = (long)std::ceil((double)nw * L / orig);
    std::vector<float> out((size_t)n_out * nf);
    for (long i = 0; i < n_out; ++i) { const float* fr = pad.data() + i*of;
        for (int j = 0; j < nf; ++j) { const float* kp = ker.data() + (size_t)j*K;
            float acc = 0.0f; for (int k = 0; k < K; ++k) acc += fr[k]*kp[k];
            out[(size_t)i*nf + j] = acc; } }
    if ((long)out.size() > target) out.resize(target);
    return out;
}

static double hz_to_mel(double f){ const double s=200.0/3.0, mlh=1000.0, mlm=1000.0/s, ls=std::log(6.4)/27.0;
    return f>=mlh ? mlm + std::log(f/mlh)/ls : f/s; }
static double mel_to_hz(double m){ const double s=200.0/3.0, mlh=1000.0, mlm=1000.0/s, ls=std::log(6.4)/27.0;
    return m>=mlm ? mlh*std::exp(ls*(m-mlm)) : s*m; }
static std::vector<float> mel_filterbank(int nfreq){
    std::vector<double> ff(nfreq); for (int i=0;i<nfreq;++i) ff[i]=(SR/2.0)*i/(nfreq-1);
    std::vector<double> fp(N_MELS+2); double lo=hz_to_mel(FMIN), hi=hz_to_mel(FMAX);
    for (int i=0;i<N_MELS+2;++i) fp[i]=mel_to_hz(lo+(hi-lo)*i/(N_MELS+1));
    std::vector<float> fb((size_t)N_MELS*nfreq, 0.0f);
    for (int i=0;i<N_MELS;++i){ double d0=fp[i+1]-fp[i], d1=fp[i+2]-fp[i+1], en=2.0/(fp[i+2]-fp[i]);
        for (int k=0;k<nfreq;++k){ double lo2=(ff[k]-fp[i])/d0, up=(fp[i+2]-ff[k])/d1;
            fb[(size_t)i*nfreq+k]=(float)(std::max(0.0,std::min(lo2,up))*en); } }
    return fb;
}

// wav -> mel [80*T] row-major, returns T
static std::vector<float> compute_mel(const std::string& wav, long& T_out) {
    int srf = 0;
    std::vector<float> x = load_wav_mono(wav, srf);
    x = resample(x, srf, SR);
    long L = (long)x.size();
    if (L < N_FFT) throw std::runtime_error("audio too short after resample");
    for (long i = L-1; i >= 1; --i) x[i] = x[i] - PREEMPH * x[i-1];
    const int pad = N_FFT/2, nfreq = N_FFT/2 + 1;
    std::vector<float> xp((size_t)L + 2*pad);
    for (int j=0;j<pad;++j) xp[j]=x[pad-j];
    std::memcpy(xp.data()+pad, x.data(), (size_t)L*sizeof(float));
    for (int k=0;k<pad;++k) xp[pad+L+k]=x[L-2-k];
    std::vector<float> wf(N_FFT, 0.0f); int wpad=(N_FFT-WIN)/2;
    for (int i=0;i<WIN;++i) wf[wpad+i]=(float)(0.5-0.5*std::cos(2.0*M_PI*i/(WIN-1)));
    long T = ((long)xp.size() - N_FFT)/HOP + 1; T_out = T;
    std::vector<float> fb = mel_filterbank(nfreq);
    using pocketfft::shape_t; using pocketfft::stride_t;
    shape_t sh{(size_t)N_FFT}; stride_t si{sizeof(double)}, so{sizeof(std::complex<double>)}; shape_t ax{0};
    std::vector<double> frame(N_FFT); std::vector<std::complex<double>> spec(nfreq);
    std::vector<float> mel((size_t)T * N_MELS); const float guard = std::pow(2.0f,-24.0f);
    for (long t=0;t<T;++t){ const float* src=xp.data()+t*HOP;
        for (int i=0;i<N_FFT;++i) frame[i]=(double)(src[i]*wf[i]);
        pocketfft::r2c(sh, si, so, ax, pocketfft::FORWARD, frame.data(), spec.data(), 1.0);
        float power[257]; for (int k=0;k<nfreq;++k){ double re=spec[k].real(), im=spec[k].imag(); power[k]=(float)(re*re+im*im); }
        float* mr=&mel[(size_t)t*N_MELS];
        for (int m=0;m<N_MELS;++m){ const float* fr=&fb[(size_t)m*nfreq]; float acc=0.0f;
            for (int k=0;k<nfreq;++k) acc+=power[k]*fr[k]; mr[m]=std::log(acc+guard); } }
    for (int m=0;m<N_MELS;++m){ float s=0.0f; for (long t=0;t<T;++t) s+=mel[(size_t)t*N_MELS+m];
        float mean=s/T, ss=0.0f; for (long t=0;t<T;++t){ float d=mel[(size_t)t*N_MELS+m]-mean; ss+=d*d; }
        float sd=std::sqrt(ss/((float)T-1.0f))+NORM_EPS;
        for (long t=0;t<T;++t){ float& v=mel[(size_t)t*N_MELS+m]; v=(v-mean)/sd; } }
    // transpose to [80, T]
    std::vector<float> out((size_t)N_MELS*T);
    for (long t=0;t<T;++t) for (int m=0;m<N_MELS;++m) out[(size_t)m*T+t]=mel[(size_t)t*N_MELS+m];
    return out;
}

// ---- per-language decode table ----
struct LangTable { std::vector<int32_t> cols; int32_t blank; std::vector<std::string> vocab; };
static LangTable load_table(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open language table: " + path);
    LangTable t; int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) throw std::runtime_error("bad table header");
    t.cols.resize(n);
    if (std::fread(t.cols.data(), 4, n, f) != (size_t)n) throw std::runtime_error("bad table cols");
    if (std::fread(&t.blank, 4, 1, f) != 1) throw std::runtime_error("bad table blank");
    t.vocab.resize(n);
    for (int i = 0; i < n; ++i) {
        int32_t len = 0; if (std::fread(&len, 4, 1, f) != 1) throw std::runtime_error("bad token len");
        std::string s(len, '\0'); if (len && std::fread(&s[0], 1, len, f) != (size_t)len) throw std::runtime_error("bad token");
        t.vocab[i] = std::move(s);
    }
    std::fclose(f);
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s input.wav [lang=ne]\n", argv[0]); return 1; }
    const std::string wav = argv[1];
    const std::string lang = argc > 2 ? argv[2] : "ne";
    const std::string dir = exe_dir();

    try {
        // 1. mel (this file)
        long T = 0;
        std::vector<float> mel = compute_mel(wav, T);              // [80*T]
        int64_t len_val = T;

        // 2. onnxruntime: encoder -> ctc
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "indic_asr");
        Ort::SessionOptions opt; opt.SetIntraOpNumThreads(0);
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session enc(env, (dir + "/assets/encoder.onnx").c_str(), opt);
        Ort::Session ctc(env, (dir + "/assets/ctc_decoder.onnx").c_str(), opt);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        int64_t feat_shape[3] = {1, N_MELS, T};
        int64_t len_shape[1] = {1};
        Ort::Value feat_t = Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(), feat_shape, 3);
        Ort::Value len_t  = Ort::Value::CreateTensor<int64_t>(mem, &len_val, 1, len_shape, 1);

        const char* enc_in[]  = {"audio_signal", "length"};
        const char* enc_out[] = {"outputs", "encoded_lengths"};
        Ort::Value enc_inputs[] = {std::move(feat_t), std::move(len_t)};
        auto eo = enc.Run(Ort::RunOptions{nullptr}, enc_in, enc_inputs, 2, enc_out, 2);

        // enc "outputs" -> ctc "encoder_output"
        float* enc_data = eo[0].GetTensorMutableData<float>();
        auto enc_shape = eo[0].GetTensorTypeAndShapeInfo().GetShape();  // [1,1024,T']
        size_t enc_size = 1; for (auto d : enc_shape) enc_size *= (size_t)d;
        Ort::Value ctc_in = Ort::Value::CreateTensor<float>(mem, enc_data, enc_size,
                                                            enc_shape.data(), enc_shape.size());
        const char* ctc_in_names[]  = {"encoder_output"};
        const char* ctc_out_names[] = {"logprobs"};
        auto co = ctc.Run(Ort::RunOptions{nullptr}, ctc_in_names, &ctc_in, 1, ctc_out_names, 1);

        const float* lp = co[0].GetTensorData<float>();
        auto lp_shape = co[0].GetTensorTypeAndShapeInfo().GetShape();   // [1, Tp, 5633]
        long Tp = lp_shape[1]; long V = lp_shape[2];

        // 3. language mask + greedy CTC + detokenize
        LangTable tb = load_table(dir + "/lang/" + lang + ".tbl");
        const int NC = (int)tb.cols.size();
        std::string text; int prev = -1;
        for (long t = 0; t < Tp; ++t) {
            const float* row = lp + (size_t)t * V;
            int best = 0; float bv = row[tb.cols[0]];
            for (int j = 1; j < NC; ++j) { float v = row[tb.cols[j]]; if (v > bv) { bv = v; best = j; } }
            if (best != prev && best != tb.blank) text += tb.vocab[best];  // collapse repeats, drop blank
            prev = best;
        }
        // SentencePiece: replace U+2581 (E2 96 81) word-boundary marker with a space
        std::string out; out.reserve(text.size());
        for (size_t i = 0; i < text.size(); ) {
            if (i + 2 < text.size() && (uint8_t)text[i]==0xE2 && (uint8_t)text[i+1]==0x96 && (uint8_t)text[i+2]==0x81)
                { out += ' '; i += 3; } else { out += text[i]; ++i; }
        }
        size_t a = out.find_first_not_of(' '), b = out.find_last_not_of(' ');
        std::printf("%s\n", (a == std::string::npos) ? "" : out.substr(a, b - a + 1).c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
