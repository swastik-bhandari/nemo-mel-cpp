# Indic ASR — Speech → Text for 22 Indian Languages

A local web app that transcribes speech in **all 22 official Indian languages** using
[`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).

The model runs on **this computer as a server**; the browser is just a thin interface
that uploads a `.wav` and displays the returned text. Verified reference: `test_nepali.wav`
→ **म सचै छु**.

---

## Why server-side (and not in-browser)

The English demo (`index.html`, QuartzNet) runs entirely in the browser because QuartzNet
is only **76 MB**. Indic-Conformer is a **600M-parameter** model — its `encoder.onnx` alone
is **~2.2 GB in fp32**, which:

- exceeds `onnxruntime-web`'s ~2 GB WebAssembly memory ceiling (won't load), and
- would be a multi-GB download on every page visit.

Quantizing to int8 (~550 MB) was considered but rejected in favour of keeping full fp32
accuracy. So the model runs on the machine (no memory ceiling, no giant download) and the
browser talks to it over HTTP.

**Trade-off:** the audio is sent to the local server, so it is *not* "on-device" like the
English page. For a local/demo assignment this is fine.

---

## Full pipeline

```
                          BROWSER (interface.html)
   ┌─────────────────────────────────────────────────────────────┐
   │  pick language  +  drop a .wav                                │
   │            │                                                  │
   │            └──►  POST /transcribe   (multipart: audio, lang)  │
   └─────────────────────────────┬───────────────────────────────┘
                                 │  HTTP (localhost:8000)
                                 ▼
                          SERVER (server.py)
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. decode WAV      wave module → int16 → float32, downmix     │
   │                    to mono                                    │
   │ 2. resample        torchaudio.functional.resample → 16 kHz    │
   │                    (SINC — linear resampling changes tokens!) │
   │ 3. mel front-end   preprocessor.ts  (TorchScript)            │
   │                    → audio_signal [1, 80, T]                  │
   │ 4. encoder         encoder.onnx (onnxruntime)                 │
   │                    → outputs [1, 1024, T']                    │
   │ 5. CTC head        ctc_decoder.onnx (onnxruntime)             │
   │                    → logprobs [1, T', 5633]                   │
   │ 6. language mask   logprobs[:, :, language_masks[lang]]       │
   │                    boolean mask, 5633 → 257 columns           │
   │ 7. greedy CTC      argmax → collapse repeats → drop blank     │
   │ 8. detokenize      vocab[lang] lookup → replace '▁' with ' '  │
   └─────────────────────────────┬───────────────────────────────┘
                                 │  JSON { text, lang, duration_s }
                                 ▼
                     browser shows the transcription
```

### The mel front-end (step 3)

Parameters extracted directly from `preprocessor.ts` (not assumed):

| param | value |
|---|---|
| sample rate | 16000 |
| **n_mels** | **80** (QuartzNet was 64) |
| n_fft | 512 |
| win_length | 400 (25 ms) |
| hop_length | 160 (10 ms) |
| window | Hann (periodic=False) |
| padding | reflect, 256 each side (center) |
| spectrogram | power (magnitude²) |
| log | `ln(mel + 2⁻²⁴)` |
| normalization | per-feature: `(x − mean) / (std + 1e-5)` over time |

### The ONNX part (steps 4–6) — the core

The HuggingFace repo **already ships the exported ONNX** — we did **not** export it
ourselves. Under `trust_remote_code`, the model (`model_onnx.py`) is not a PyTorch network;
it is a thin wrapper that runs pre-exported ONNX graphs via `onnxruntime`:

- **`encoder.onnx`** — the Conformer encoder.
  - input:  `audio_signal [B, 80, T]` (mel), `length [B]`
  - output: `outputs [B, 1024, T']`, `encoded_lengths [B]`
  - Weights are stored as **external data** (hundreds of `onnx__MatMul_*` / `onnx__Conv_*`
    blobs in the repo's `assets/`), which is why the graph file is tiny but the real
    footprint is ~2.2 GB.
- **`ctc_decoder.onnx`** — the CTC projection head.
  - input:  `encoder_output [B, 1024, T']`
  - output: `logprobs [B, T', 5633]`  ← **5633 = the full multilingual vocabulary**

**Why one ONNX serves all 22 languages.** The encoder and CTC head are shared. Each
language is a **boolean mask** of length 5633 (`language_masks.json`) that selects that
language's columns out of the shared 5633-wide output. For Nepali the mask has exactly
**257 True** entries → the masked logits are `[T', 257]`, and `vocab['ne']` is the matching
list of 257 **SentencePiece** subwords (blank at index 256, `'▁'` marks a word boundary).
Switching language = switching mask + vocab; no model reload.

> RNNT branch is intentionally **not** used — it is autoregressive and does not export as
> one clean graph. This project uses the **CTC path only**.

---

## File structure (Indic-related only)

```
Project files we wrote
├── server.py                  # Flask server: loads model once, /transcribe + /languages
├── interface.html             # browser UI (upload, language picker, waveform, result)
├── README_indic.md            # this file
│
├── cp1_transcribe.py          # Checkpoint 1: PyTorch reference transcription (ground truth)
├── cp2_config.py              # Checkpoint 2: extract n_mels / DSP config + vocab + mask
├── cp3_onnx_verify.py         # Checkpoint 3: prove shipped ONNX reproduces the reference
├── cp4_mel_check.py           # investigation: WASM-mel vs preprocessor.ts (preemph/pad diff)
├── cp4_quantize.py            # abandoned int8 experiment (kept for reference; not used)
│
├── indic_preproc_config.json  # extracted preprocessor + tokenizer config (CP2 output)
├── indic_vocab_ne.json        # Nepali vocab, 257 SentencePiece tokens
├── indic_langmask_ne.json     # Nepali boolean language mask (len 5633)
├── indic_ctc_ne_columns.json  # the 257 CTC column indices selected for Nepali
│
├── test_nepali.wav            # test clip → expected "म सचै छु"
└── indic_env/                 # Python 3.12 venv (torch/torchaudio CPU, onnxruntime, flask)

Model assets (downloaded from HuggingFace into ~/.cache/huggingface, not committed)
├── config.json                # BLANK_ID=256, RNNT params
├── model_onnx.py              # trust_remote_code wrapper (runs the ONNX graphs)
├── preprocessor.ts            # TorchScript mel front-end
├── encoder.onnx  (+ external weight blobs)
├── ctc_decoder.onnx
├── vocab.json                 # per-language subword lists (all 22)
└── language_masks.json        # per-language boolean masks (all 22)
```

---

## How we built it (the process)

Development was gated behind four checkpoints; nothing downstream was trusted until the
step above it was verified against the PyTorch reference.

1. **Checkpoint 1 — ground truth** (`cp1_transcribe.py`)
   Loaded the model with `AutoModel.from_pretrained(..., trust_remote_code=True)` and ran
   the CTC path `model(wav, "ne", "ctc")`. Output **म सचै छु**, confirmed correct. This is
   the reference every later step must match.

2. **Checkpoint 2 — extract the real config** (`cp2_config.py`)
   Read `n_mels = 80` off `encoder.onnx`'s input tensor (cross-checked against the live
   preprocessor output), pulled the STFT params out of the traced `preprocessor.ts`
   (`n_fft=512, win=400, hop=160`), and discovered the tokenizer is **SentencePiece** with
   a **boolean per-language mask** (257 True for Nepali). No values assumed.

3. **Checkpoint 3 — verify the ONNX** (`cp3_onnx_verify.py`)
   Ran the **shipped** `encoder.onnx` + `ctc_decoder.onnx` through a standalone
   `onnxruntime` reimplementation of the mask + SentencePiece decode. It reproduced
   **म सचै छु exactly**, proving both the ONNX graphs and the decode logic are correct.
   *Gotcha found here:* a crude linear (`np.interp`) resample produced a **wrong** token
   (`म सन्चै छु`) — only the SINC resampler matches. Hence the server uses
   `torchaudio.functional.resample`.

4. **Checkpoint 4 — ship it**
   The 2.2 GB fp32 model can't run in-browser, so the decision was **server-side**:
   `server.py` wraps the exact CP3 pipeline behind a Flask endpoint, and `interface.html`
   uploads audio and shows text. Then generalized from Nepali-only to **all 22 languages**
   by loading every mask/vocab and adding a `lang` parameter + language picker.

---

## Running it

```bash
source indic_env/bin/activate
python server.py            # loads the model once (~10–30 s), serves on :8000
# open http://localhost:8000 , pick a language, drop a .wav
```

- **Endpoints:** `GET /` (UI), `GET /languages` (list of 22), `POST /transcribe`
  (form fields: `audio` = wav file, `lang` = language code, default `ne`).
- **Audio format:** 16-bit PCM WAV (mono or stereo, any sample rate — resampled to 16 kHz).
- **Note:** the model does **not** auto-detect language; select the one matching your audio.

## Setup notes

- venv `indic_env` (Python 3.12): CPU builds of `torch==2.11.0` and `torchaudio==2.11.0+cpu`
  (must both be CPU), plus `onnxruntime`, `flask`, `librosa`.
- `torchaudio 2.11`'s `.load()` needs `torchcodec` (not installed) → WAV is decoded with the
  stdlib `wave` module instead.
- The model repo is **gated**: accept the terms on the model page and place a token at
  `~/.cache/huggingface/token`. The model itself needs **no NeMo toolkit** — it's
  self-contained ONNX + TorchScript.
