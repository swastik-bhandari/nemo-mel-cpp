#!/usr/bin/env python
"""
Nepali ASR server — your computer runs the model, the browser is just an interface.

Pipeline (torch-free — numpy front-end verified bit-exact vs preprocessor.ts):
  uploaded wav -> mono/16k (numpy sinc) -> numpy mel -> encoder.onnx
              -> ctc_decoder.onnx -> language-mask[ne] -> greedy CTC -> SentencePiece text

No torch/torchaudio at runtime: onnxruntime runs the model, indic_frontend does the DSP.

Run:  python server.py      then open  http://localhost:8000
"""
import io, os, json, wave
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import snapshot_download

import indic_frontend as F

HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = 256
SR = 16000

# display names for the 22 supported languages (keys must match vocab.json)
LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

print("[server] loading model (one time) ...", flush=True)
A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")
ENC = ort.InferenceSession(f"{A}/encoder.onnx", providers=["CPUExecutionProvider"])
CTC = ort.InferenceSession(f"{A}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
# per-language decode tables (encoder + ctc graph are shared across all languages)
VOCABS = json.load(open(f"{A}/vocab.json"))                                       # {lang: [subwords]}
MASKS = {lg: np.array(m, dtype=bool) for lg, m in json.load(open(f"{A}/language_masks.json")).items()}
LANGS = [lg for lg in LANG_NAMES if lg in VOCABS and lg in MASKS]
print(f"[server] ready · n_mels=80 · {len(LANGS)} languages: {', '.join(LANGS)}", flush=True)


def decode_wav_bytes(raw: bytes) -> np.ndarray:
    """WAV bytes -> mono float32 @ 16 kHz (sinc), matching the verified pipeline."""
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={sw} bytes")
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    if sr != SR:
        pcm = F.resample(pcm, sr, SR)     # numpy sinc port of torchaudio (verified 1e-8)
    return pcm.astype(np.float32)


def transcribe(samples: np.ndarray, lang: str) -> str:
    mask, vocab = MASKS[lang], VOCABS[lang]
    feat, length = F.mel_spectrogram(samples)   # numpy mel (verified vs preprocessor.ts, 1e-6)
    enc_out, _ = ENC.run(["outputs", "encoded_lengths"],
                         {"audio_signal": feat, "length": length})
    logprobs = CTC.run(["logprobs"], {"encoder_output": enc_out})[0]       # [1, T, 5633]
    masked = logprobs[0][:, mask]                                          # [T, vocab]
    idx = masked.argmax(axis=-1)
    collapsed = idx[np.insert(np.diff(idx) != 0, 0, True)]                 # unique_consecutive
    return "".join(vocab[x] for x in collapsed if x != BLANK).replace("▁", " ").strip()


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "interface.html")


@app.route("/languages")
def languages():
    return jsonify(languages=[{"code": lg, "name": LANG_NAMES[lg]} for lg in LANGS],
                   default="ne")


@app.route("/transcribe", methods=["POST"])
def do_transcribe():
    if "audio" not in request.files:
        return jsonify(error="no 'audio' file in request"), 400
    lang = (request.form.get("lang") or "ne").strip()
    if lang not in MASKS:
        return jsonify(error=f"unsupported language '{lang}'"), 400
    try:
        raw = request.files["audio"].read()
        samples = decode_wav_bytes(raw)
        text = transcribe(samples, lang)
        return jsonify(text=text, lang=lang, language=LANG_NAMES.get(lang, lang),
                       duration_s=round(len(samples) / SR, 2))
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True)
