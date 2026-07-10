// asr.cpp
// ============================================================================
// Self-contained C++ speech-to-text: WAV -> text.
// No Python, no NeMo, no PyTorch at runtime.
//
// Pipeline:
//   1. WAV -> mel spectrogram   (the exact NeMo mel front-end from nemo_mel.cpp,
//                                inlined here: libsndfile + libsoxr + PocketFFT)
//   2. mel -> char logits       (quartznet_full.onnx via ONNX Runtime C++ API)
//   3. logits -> text           (greedy CTC: argmax, collapse repeats, drop blank)
//
// Build:
//   g++ -O2 -std=c++17 asr.cpp -o asr \
//       -I<onnxruntime>/include -L<onnxruntime>/lib \
//       -lsndfile -lsoxr -lonnxruntime \
//       -Wl,-rpath,<onnxruntime>/lib
//
//   where <onnxruntime> is the extracted onnxruntime-linux-x64-<ver> folder.
//
// Run:
//   ./asr example.wav quartznet_full.onnx quartznet_labels.txt nemo_mel_constants.bin
//
// ============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sndfile.h>
#include <soxr.h>

#include "pocketfft_hdronly.h"
#include "onnxruntime_cxx_api.h"

// ============================================================================
// PART 1 — MEL FRONT-END  (inlined from nemo_mel.cpp, unchanged logic)
// ============================================================================

struct Constants {
    int32_t sr, n_fft, win, hop, n_mels, pad_to;
    std::vector<float> window;   // [win]
    std::vector<float> fb;       // [n_mels, n_fft/2+1] row-major
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
    // channel_selector='average'
    std::vector<float> mono(info.frames);
    for (sf_count_t i = 0; i < info.frames; ++i) {
        float acc = 0.0f;
        for (int ch = 0; ch < info.channels; ++ch)
            acc += interleaved[static_cast<size_t>(i) * info.channels + ch];
        mono[i] = acc / static_cast<float>(info.channels);
    }
    return mono;
}

static std::vector<float> resample_soxr(const std::vector<float>& in, int sr_in, int sr_out) {
    if (sr_in == sr_out) return in;
    const double ratio = static_cast<double>(sr_out) / sr_in;
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

// Produces mel [n_mels, total_frames] (row-major) and sets seq_len (valid frames).
static std::vector<float> wav_to_mel(const std::string& wav_path, const Constants& C,
                                     int64_t& n_mels_out, int64_t& total_out, int64_t& seq_len_out) {
    const int n_freq = C.n_fft / 2 + 1;

    int sr_file = 0;
    std::vector<float> x = load_wav_mono(wav_path, sr_file);
    x = resample_soxr(x, sr_file, C.sr);
    const int64_t L = static_cast<int64_t>(x.size());
    const int64_t seq_len = (L + 2 * (C.n_fft / 2) - C.n_fft) / C.hop;

    // preemphasis (right-to-left in place)
    for (int64_t i = L - 1; i >= 1; --i) x[i] = x[i] - 0.97f * x[i - 1];

    // center pad n_fft/2 each side
    const int pad = C.n_fft / 2;
    std::vector<float> xp(static_cast<size_t>(L) + 2 * pad, 0.0f);
    std::memcpy(xp.data() + pad, x.data(), static_cast<size_t>(L) * sizeof(float));

    // window zero-padded centered to n_fft
    std::vector<float> win_full(C.n_fft, 0.0f);
    const int wpad_left = (C.n_fft - C.win) / 2;
    std::memcpy(win_full.data() + wpad_left, C.window.data(), C.win * sizeof(float));

    const int64_t T = (static_cast<int64_t>(xp.size()) - C.n_fft) / C.hop + 1;

    using pocketfft::shape_t;
    using pocketfft::stride_t;
    shape_t shape_in{static_cast<size_t>(C.n_fft)};
    stride_t stride_in{sizeof(float)};
    stride_t stride_out{sizeof(std::complex<float>)};
    shape_t axes{0};

    std::vector<float> frame(C.n_fft);
    std::vector<std::complex<float>> spec(n_freq);
    std::vector<float> power(static_cast<size_t>(n_freq) * T);

    for (int64_t t = 0; t < T; ++t) {
        const float* src = xp.data() + t * C.hop;
        for (int i = 0; i < C.n_fft; ++i) frame[i] = src[i] * win_full[i];
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       frame.data(), spec.data(), 1.0f);
        for (int k = 0; k < n_freq; ++k) {
            const float re = spec[k].real(), im = spec[k].imag();
            const float mag = std::sqrt(re * re + im * im);
            power[static_cast<size_t>(k) * T + t] = mag * mag;
        }
    }

    const float guard = std::pow(2.0f, -24.0f);
    std::vector<float> mel(static_cast<size_t>(C.n_mels) * T, 0.0f);
    for (int m = 0; m < C.n_mels; ++m) {
        const float* fbrow = &C.fb[static_cast<size_t>(m) * n_freq];
        float* melrow = &mel[static_cast<size_t>(m) * T];
        for (int k = 0; k < n_freq; ++k) {
            const float w = fbrow[k];
            if (w == 0.0f) continue;
            const float* prow = &power[static_cast<size_t>(k) * T];
            for (int64_t t = 0; t < T; ++t) melrow[t] += w * prow[t];
        }
        for (int64_t t = 0; t < T; ++t) melrow[t] = std::log(melrow[t] + guard);
    }

    const float CONSTANT = 1e-5f;
    for (int m = 0; m < C.n_mels; ++m) {
        float* row = &mel[static_cast<size_t>(m) * T];
        float sum = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) sum += row[t];
        const float mean = sum / static_cast<float>(seq_len);
        float ss = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) { const float d = row[t] - mean; ss += d * d; }
        float sd = std::sqrt(ss / (static_cast<float>(seq_len) - 1.0f));
        if (std::isnan(sd)) sd = 0.0f;
        sd += CONSTANT;
        for (int64_t t = 0; t < T; ++t) row[t] = (row[t] - mean) / sd;
    }

    int64_t total = T;
    if (C.pad_to > 0) {
        const int64_t rem = T % C.pad_to;
        if (rem != 0) total = T + (C.pad_to - rem);
    }
    std::vector<float> out(static_cast<size_t>(C.n_mels) * total, 0.0f);
    for (int m = 0; m < C.n_mels; ++m)
        for (int64_t t = 0; t < seq_len && t < T; ++t)
            out[static_cast<size_t>(m) * total + t] = mel[static_cast<size_t>(m) * T + t];

    n_mels_out = C.n_mels;
    total_out = total;
    seq_len_out = seq_len;
    return out;  // [n_mels, total]
}

// ============================================================================
// PART 2 — LABELS + CTC DECODE
// ============================================================================

static std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open labels file: " + path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF safety
        labels.push_back(line);
    }
    return labels;  // blank id = labels.size()
}

// logits: [T, V] row-major (V = vocab+1, blank = V-1). Greedy CTC.
static std::string ctc_greedy_decode(const float* logits, int64_t T, int64_t V,
                                     const std::vector<std::string>& labels) {
    const int64_t blank_id = static_cast<int64_t>(labels.size());  // == V-1
    std::string text;
    int64_t prev = -1;
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits + t * V;
        int64_t best = 0;
        float bestv = row[0];
        for (int64_t v = 1; v < V; ++v)
            if (row[v] > bestv) { bestv = row[v]; best = v; }
        if (best != prev && best != blank_id)
            text += labels[best];
        prev = best;
    }
    return text;
}

// ============================================================================
// PART 3 — ONNX INFERENCE + MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s input.wav quartznet_full.onnx quartznet_labels.txt nemo_mel_constants.bin\n",
            argv[0]);
        return 1;
    }
    const std::string wav_path = argv[1];
    const std::string onnx_path = argv[2];
    const std::string labels_path = argv[3];
    const std::string const_path = argv[4];

    try {
        // ---- 1. WAV -> mel ----
        Constants C = load_constants(const_path);
        int64_t n_mels = 0, total = 0, seq_len = 0;
        std::vector<float> mel = wav_to_mel(wav_path, C, n_mels, total, seq_len);
        std::printf("mel: [%lld, %lld], valid frames %lld\n",
                    (long long)n_mels, (long long)total, (long long)seq_len);

        // ---- 2. mel -> logits via ONNX Runtime ----
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "asr");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, onnx_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;

        // Resolve input/output names from the model (robust to naming).
        std::vector<Ort::AllocatedStringPtr> in_name_ptrs, out_name_ptrs;
        std::vector<const char*> in_names, out_names;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            in_name_ptrs.push_back(session.GetInputNameAllocated(i, alloc));
            in_names.push_back(in_name_ptrs.back().get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            out_name_ptrs.push_back(session.GetOutputNameAllocated(i, alloc));
            out_names.push_back(out_name_ptrs.back().get());
        }

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input "mel": shape [1, n_mels, total]
        std::array<int64_t, 3> mel_shape{1, n_mels, total};
        Ort::Value mel_tensor = Ort::Value::CreateTensor<float>(
            mem, mel.data(), mel.size(), mel_shape.data(), mel_shape.size());

        // Input "length": shape [1], the number of VALID frames (int64)
        std::array<int64_t, 1> len_val{seq_len};
        std::array<int64_t, 1> len_shape{1};
        Ort::Value len_tensor = Ort::Value::CreateTensor<int64_t>(
            mem, len_val.data(), len_val.size(), len_shape.data(), len_shape.size());

        // Order inputs to match the model's declared input names (mel, length).
        std::vector<Ort::Value> inputs;
        for (auto* name : in_names) {
            if (std::string(name) == "length") inputs.push_back(std::move(len_tensor));
            else                                inputs.push_back(std::move(mel_tensor));
        }

        auto out = session.Run(Ort::RunOptions{nullptr},
                               in_names.data(), inputs.data(), inputs.size(),
                               out_names.data(), out_names.size());

        // logits shape: [1, T, V]
        auto info = out[0].GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape();  // {1, T, V}
        const int64_t T = shp[1], V = shp[2];
        const float* logits = out[0].GetTensorData<float>();

        // ---- 3. CTC decode ----
        std::vector<std::string> labels = load_labels(labels_path);
        if (V != static_cast<int64_t>(labels.size()) + 1) {
            std::fprintf(stderr,
                "warning: model vocab %lld != labels %zu + 1 blank\n",
                (long long)V, labels.size());
        }
        std::string text = ctc_greedy_decode(logits, T, V, labels);

        std::printf("logits: [%lld, %lld]\n", (long long)T, (long long)V);
        std::printf("\nTranscription: '%s'\n", text.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
