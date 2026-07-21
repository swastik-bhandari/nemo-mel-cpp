#!/usr/bin/env python
"""Gate for the C++ mel front-end (indic_mel.cpp).

Proves the standalone C++ binary matches indic_frontend.py (numpy) and that the
C++ mel still decodes म सचै छु through the real ONNX model.

Usage:
    g++ -O2 -std=c++17 indic_mel.cpp -o indic_mel
    ./indic_mel test_nepali.wav /tmp/indic_mel_cpp.bin
    python verify_indic_mel.py            # (uses /tmp/indic_mel_cpp.bin)
"""
import os, json, struct, subprocess, numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
import indic_frontend as F

REF = "म सचै छु"; LANG = "ne"; BLANK = 256
WAV = "test_nepali.wav"; BIN = "/tmp/indic_mel_cpp.bin"

# build + run the C++ binary if needed
if not os.path.exists("indic_mel"):
    subprocess.run(["g++", "-O2", "-std=c++17", "indic_mel.cpp", "-o", "indic_mel"], check=True)
subprocess.run(["./indic_mel", WAV, BIN], check=True)

# ---- numpy reference (server path) ----
import wave
with wave.open(WAV, "rb") as w:
    ch, sr = w.getnchannels(), w.getframerate()
    raw = w.readframes(w.getnframes())
pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
if ch > 1: pcm = pcm.reshape(-1, ch).mean(1)
ref_feat, _ = F.mel_spectrogram(F.resample(np.ascontiguousarray(pcm, np.float32), sr, 16000))
ref = ref_feat[0]

# ---- C++ output ----
with open(BIN, "rb") as f:
    n_mels, T = struct.unpack("<ii", f.read(8))
    cpp = np.frombuffer(f.read(), dtype=np.float32).reshape(n_mels, T)

Tm = min(ref.shape[1], cpp.shape[1])
d = np.abs(ref[:, :Tm] - cpp[:, :Tm])
print("=== mel: C++ vs numpy indic_frontend ===")
print(f"  numpy {ref.shape}   cpp {cpp.shape}   T match: {ref.shape[1]==cpp.shape[1]}")
print(f"  meanAbsDiff={d.mean():.3e}   maxAbsDiff={d.max():.3e}")

# ---- end-to-end: C++ mel through the real ONNX model ----
A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")
ENC = ort.InferenceSession(f"{A}/encoder.onnx", providers=["CPUExecutionProvider"])
CTC = ort.InferenceSession(f"{A}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
mask = np.array(json.load(open(f"{A}/language_masks.json"))[LANG], dtype=bool)
vocab = json.load(open(f"{A}/vocab.json"))[LANG]

feat = np.ascontiguousarray(cpp[None], dtype=np.float32)
eo, _ = ENC.run(["outputs", "encoded_lengths"], {"audio_signal": feat, "length": np.array([T], np.int64)})
lp = CTC.run(["logprobs"], {"encoder_output": eo})[0]
idx = lp[0][:, mask].argmax(-1)
col = idx[np.insert(np.diff(idx) != 0, 0, True)]
text = "".join(vocab[i] for i in col if i != BLANK).replace("▁", " ").strip()

print("\n=== transcription (C++ mel -> ONNX) ===")
print(f"  reference : {REF!r}")
print(f"  C++ mel   : {text!r}")
print("\nRESULT:", "PASS ✅" if (text == REF and d.max() < 1e-3) else "FAIL ❌")
