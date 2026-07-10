// mel_wasm.cpp
// ============================================================================
// WebAssembly mel-spectrogram module for the browser.
//
// This is the mel DSP core of nemo_mel.cpp, adapted for Emscripten:
//   * NO libsndfile  — the browser decodes audio (Web Audio API)
//   * NO libsoxr     — the browser resamples (OfflineAudioContext to 16 kHz)
//   * NO file I/O    — samples and constants come from JS as typed arrays
//   * PocketFFT stays (header-only, compiles to wasm cleanly)
//
// It exposes ONE function to JavaScript, mel_extract(), which takes:
//   - a pointer to float32 mono samples already at 16 kHz
//   - the sample count
//   - a pointer to the constants blob (same nemo_mel_constants.bin bytes)
//   - the constants byte length
//   - output pointers for n_mels, total_frames, seq_len
// and returns a pointer to a freshly malloc'd float32 [n_mels * total_frames]
// row-major mel. JS reads it out of the wasm heap and frees it.
//
// Build (on your machine, after `source emsdk_env.sh`):
//   emcc -O2 -std=c++17 mel_wasm.cpp -o mel.js \
//     -I. \
//     -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web \
//     -s EXPORTED_FUNCTIONS='["_mel_extract","_malloc","_free"]' \
//     -s EXPORTED_RUNTIME_METHODS='["HEAPF32","HEAP32","HEAPU8","cwrap","getValue"]' \
//     -s ALLOW_MEMORY_GROWTH=1
//
// Produces mel.js + mel.wasm.
// ============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "pocketfft_hdronly.h"

#include <emscripten/emscripten.h>

// ---- constants blob layout (same as nemo_mel_constants.bin) ----
//   int32 sr, n_fft, win, hop, n_mels, pad_to
//   float32[win]                  window
//   float32[n_mels*(n_fft/2+1)]   filterbank row-major
struct Constants {
    int32_t sr, n_fft, win, hop, n_mels, pad_to;
    const float* window;  // into blob
    const float* fb;      // into blob
};

static Constants parse_constants(const uint8_t* blob) {
    Constants c{};
    const int32_t* hdr = reinterpret_cast<const int32_t*>(blob);
    c.sr = hdr[0]; c.n_fft = hdr[1]; c.win = hdr[2];
    c.hop = hdr[3]; c.n_mels = hdr[4]; c.pad_to = hdr[5];
    const uint8_t* p = blob + 6 * sizeof(int32_t);
    c.window = reinterpret_cast<const float*>(p);
    p += static_cast<size_t>(c.win) * sizeof(float);
    c.fb = reinterpret_cast<const float*>(p);
    return c;
}

extern "C" {

// Returns malloc'd float32[n_mels*total] (row-major). Caller frees via _free.
// Writes n_mels, total_frames, seq_len through the out pointers.
EMSCRIPTEN_KEEPALIVE
float* mel_extract(const float* samples, int32_t n_samples,
                   const uint8_t* const_blob, int32_t /*const_len*/,
                   int32_t* out_n_mels, int32_t* out_total, int32_t* out_seq_len) {
    const Constants C = parse_constants(const_blob);
    const int n_freq = C.n_fft / 2 + 1;
    const int64_t L = n_samples;

    // seq_len = floor((L + 2*(n_fft/2) - n_fft) / hop)  == floor(L/hop)
    const int64_t seq_len = (L + 2 * (C.n_fft / 2) - C.n_fft) / C.hop;

    // copy samples (we modify in place for preemphasis)
    std::vector<float> x(samples, samples + L);

    // preemphasis 0.97, right-to-left
    for (int64_t i = L - 1; i >= 1; --i) x[i] = x[i] - 0.97f * x[i - 1];

    // center pad n_fft/2 each side (pad_mode="constant")
    const int pad = C.n_fft / 2;
    std::vector<float> xp(static_cast<size_t>(L) + 2 * pad, 0.0f);
    std::memcpy(xp.data() + pad, x.data(), static_cast<size_t>(L) * sizeof(float));

    // window zero-padded centered to n_fft
    std::vector<float> win_full(C.n_fft, 0.0f);
    const int wpad_left = (C.n_fft - C.win) / 2;
    std::memcpy(win_full.data() + wpad_left, C.window, C.win * sizeof(float));

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

    // malloc output the JS side will read then free
    float* out = static_cast<float*>(std::malloc(sizeof(float) * C.n_mels * total));
    std::memset(out, 0, sizeof(float) * C.n_mels * total);
    for (int m = 0; m < C.n_mels; ++m)
        for (int64_t t = 0; t < seq_len && t < T; ++t)
            out[static_cast<size_t>(m) * total + t] = mel[static_cast<size_t>(m) * T + t];

    *out_n_mels = C.n_mels;
    *out_total = static_cast<int32_t>(total);
    *out_seq_len = static_cast<int32_t>(seq_len);
    return out;
}

}  // extern "C"
