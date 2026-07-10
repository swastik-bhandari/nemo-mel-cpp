# Browser ASR Skeleton (QuartzNet, English)

Upload a `.wav` in the browser → get text. Fully client-side: the audio never
leaves the page. Proves the whole web pipeline end-to-end on the trusted
English model, so IndicConformer can later be swapped in piece by piece.

```
wav → Web Audio decode+resample(16k) → mel (WASM) → quartznet_full.onnx
      (browser, free)                   (your C++)   (onnxruntime-web)
    → greedy CTC decode (JS) → text
```

## Files

| File | Role | Have it? |
|---|---|---|
| `index.html` | the page + all glue JS | ✅ built |
| `mel_wasm.cpp` | mel DSP for wasm (from nemo_mel.cpp, no libsndfile/libsoxr) | ✅ built |
| `pocketfft_hdronly.h` | FFT, header-only | ✅ have |
| `mel.js` + `mel.wasm` | **compiled from mel_wasm.cpp** (step below) | ⬜ build |
| `quartznet_full.onnx` | acoustic model | ✅ have |
| `quartznet_labels.txt` | vocab (28 lines) | ✅ have |
| `nemo_mel_constants.bin` | mel constants | ✅ have |

## Step 1 — install Emscripten (one time)

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh        # run this in each new shell
cd ..
```

## Step 2 — compile the mel module to wasm

```bash
emcc -O2 -std=c++17 mel_wasm.cpp -o mel.js \
  -I. \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web \
  -s EXPORTED_FUNCTIONS='["_mel_extract","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap","getValue"]' \
  -s ALLOW_MEMORY_GROWTH=1
```

Produces `mel.js` and `mel.wasm`. (HEAPF32/HEAP32/HEAPU8 are accessible on the
module instance by default with these flags.)

## Step 3 — put everything in one folder and serve it

Browsers block `fetch()` of local files over `file://`, and wasm needs a real
MIME type, so serve over HTTP (any static server):

```bash
# folder should contain:
#   index.html  mel.js  mel.wasm  quartznet_full.onnx
#   quartznet_labels.txt  nemo_mel_constants.bin
python3 -m http.server 8000
```

Open <http://localhost:8000> and drop a `.wav`. You should see the steps light
up (decode → mel → model → decode text) and the transcription appear.

## Verified already (before you build)

- The **JS CTC decode + label parsing** were unit-tested in Node against a
  known token sequence → correct text ("hi"). See `verify_logic.mjs`.
- The **mel_wasm.cpp core** compiles clean with g++ (emscripten macros aside);
  its DSP is byte-identical to `nemo_mel.cpp`, already verified ~1e-5 vs NeMo.
- What can only be checked in-browser: the Emscripten build itself and the
  onnxruntime-web run. Those are standard and well-trodden.

## When it works, swapping in IndicConformer later

Only three things change (the architecture stays):
1. `nemo_mel_constants.bin` → regenerate for its config (likely 80 mels)
2. `quartznet_full.onnx` → its Conformer-CTC ONNX
3. `ctcDecode()` in index.html → subword detokenize (Devanagari), not char-join

`mel.wasm`, onnxruntime-web, and the Web Audio decode are model-agnostic and
stay as-is.

## Notes / caveats

- **onnxruntime-web version** is pinned to 1.20.1 in the `<script>` tag to match
  the runtime you have; bump both together if you change it.
- **Model size:** QuartzNet ONNX is ~19 MB, fetched once then cached. A 600M
  Conformer will be far larger (~1–2 GB) — fine functionally, but a real
  first-load cost to plan for.
- **Input names:** the page routes model inputs by name (anything containing
  "len" → length tensor, else mel), so minor naming differences are tolerated.
