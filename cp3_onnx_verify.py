#!/usr/bin/env python
"""CHECKPOINT 3 — verify the SHIPPED encoder.onnx + ctc_decoder.onnx reproduce the
Checkpoint-1 PyTorch reference, using a STANDALONE onnxruntime + decode reimplementation
(the exact logic the browser JS must replicate). No model.forward() shortcut.
"""
import os, json, wave, numpy as np, torch, torchaudio
import onnxruntime as ort
from huggingface_hub import snapshot_download

REFERENCE = "म सचै छु"          # Checkpoint 1, user-confirmed
LANG = "ne"; BLANK = 256
A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")

# ---- load audio (stdlib wave), mono, 16k ----
with wave.open("test_nepali.wav","rb") as w:
    ch,sw,sr = w.getnchannels(),w.getsampwidth(),w.getframerate()
    raw=w.readframes(w.getnframes())
pcm=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
if ch>1: pcm=pcm.reshape(-1,ch).mean(1)
wav=torch.from_numpy(pcm).unsqueeze(0)
if sr!=16000:                                   # SAME sinc resampler as Checkpoint 1
    wav=torchaudio.functional.resample(wav, sr, 16000)

# ---- mel front-end (preprocessor.ts for now; the WASM C++ mel is CP4) ----
pp = torch.jit.load(f"{A}/preprocessor.ts", map_location="cpu")
audio_signal, length = pp(input_signal=wav, length=torch.tensor([wav.shape[-1]]))
audio_signal = audio_signal.cpu().numpy(); length = length.cpu().numpy()
print("mel:", audio_signal.shape, "length:", length.tolist())

# ---- encoder.onnx (shipped) ----
enc = ort.InferenceSession(f"{A}/encoder.onnx", providers=["CPUExecutionProvider"])
enc_out, enc_len = enc.run(["outputs","encoded_lengths"],
                           {"audio_signal":audio_signal, "length":length})
print("encoder_out:", enc_out.shape, "encoded_lengths:", enc_len.tolist())

# ---- ctc_decoder.onnx (shipped) ----
ctc = ort.InferenceSession(f"{A}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
logprobs = ctc.run(["logprobs"], {"encoder_output": enc_out})[0]   # [1, T, 5633]
print("ctc logprobs:", logprobs.shape)

# ---- STANDALONE decode (mirrors model_onnx.py _ctc_decode; what the browser JS will do) ----
mask = np.array(json.load(open(f"{A}/language_masks.json"))[LANG], dtype=bool)  # len 5633, 257 True
vocab = json.load(open(f"{A}/vocab.json"))[LANG]                                 # 257 tokens
masked = logprobs[0][:, mask]                       # [T, 257]
indices = masked.argmax(axis=-1)                    # greedy
collapsed = indices[np.insert(np.diff(indices)!=0, 0, True)]   # unique_consecutive
hyp = "".join(vocab[x] for x in collapsed if x != BLANK).replace("▁"," ").strip()

print("\n==================== CHECKPOINT 3 — ONNX (standalone) ====================")
print("  reference (CP1):", repr(REFERENCE))
print("  onnx output    :", repr(hyp))
print("  MATCH          :", hyp == REFERENCE)
print("=========================================================================")

# external-data footprint for the browser bundle (CP4 needs these)
def onnx_bytes(path):
    tot=os.path.getsize(path); d=os.path.dirname(path)
    return tot
print("\nencoder.onnx graph file:", os.path.getsize(f"{A}/encoder.onnx"), "bytes (weights are external in assets/)")
print("ctc_decoder.onnx      :", os.path.getsize(f"{A}/ctc_decoder.onnx"), "bytes")
