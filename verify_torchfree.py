#!/usr/bin/env python
"""Gate: prove the numpy front-end (indic_frontend.py) matches the torch reference
and that the fully torch-free pipeline still transcribes म सचै छु.

torch/torchaudio are used ONLY as the oracle here, never in server.py.
"""
import os, json, wave, numpy as np, torch, torchaudio
import onnxruntime as ort
from huggingface_hub import snapshot_download
import indic_frontend as F

REF = "म सचै छु"; LANG = "ne"; BLANK = 256
A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")

# ---- load audio (stdlib wave -> numpy), mono ----
with wave.open("test_nepali.wav", "rb") as w:
    ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
    raw = w.readframes(w.getnframes())
pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
if ch > 1: pcm = pcm.reshape(-1, ch).mean(1)
pcm = pcm.astype(np.float32)

# ============ 1. resample: numpy vs torchaudio ============
ref_res = torchaudio.functional.resample(torch.from_numpy(pcm).unsqueeze(0), sr, 16000).numpy()[0]
my_res = F.resample(pcm, sr, 16000)
print("=== resample ===")
print(f"  torchaudio: {ref_res.shape}   numpy: {my_res.shape}")
n = min(len(ref_res), len(my_res))
d = np.abs(ref_res[:n] - my_res[:n])
print(f"  len match: {len(ref_res)==len(my_res)}   meanAbsDiff={d.mean():.3e}  maxAbsDiff={d.max():.3e}")

# ============ 2. mel: numpy vs preprocessor.ts ============
pp = torch.jit.load(f"{A}/preprocessor.ts", map_location="cpu").eval()
# feed the SAME (torch-resampled) audio so this isolates the mel from the resampler
ref_feat, ref_len = pp(input_signal=torch.from_numpy(ref_res).unsqueeze(0),
                       length=torch.tensor([len(ref_res)]))
ref_feat = ref_feat.numpy()[0]
my_feat, my_len = F.mel_spectrogram(ref_res)
print("\n=== mel (on identical audio) ===")
print(f"  preprocessor.ts: {ref_feat.shape}   numpy: {my_feat[0].shape}")
Tm = min(ref_feat.shape[1], my_feat.shape[2])
dm = np.abs(ref_feat[:, :Tm] - my_feat[0][:, :Tm])
print(f"  meanAbsDiff={dm.mean():.3e}  maxAbsDiff={dm.max():.3e}   len ref={int(ref_len[0])} numpy={int(my_len[0])}")

# ============ 3. full torch-free pipeline -> text ============
ENC = ort.InferenceSession(f"{A}/encoder.onnx", providers=["CPUExecutionProvider"])
CTC = ort.InferenceSession(f"{A}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
mask = np.array(json.load(open(f"{A}/language_masks.json"))[LANG], dtype=bool)
vocab = json.load(open(f"{A}/vocab.json"))[LANG]

def run(samples):
    feat, length = F.mel_spectrogram(samples)
    eo, _ = ENC.run(["outputs", "encoded_lengths"], {"audio_signal": feat, "length": length})
    lp = CTC.run(["logprobs"], {"encoder_output": eo})[0]
    idx = lp[0][:, mask].argmax(-1)
    col = idx[np.insert(np.diff(idx) != 0, 0, True)]
    return "".join(vocab[i] for i in col if i != BLANK).replace("▁", " ").strip()

# end-to-end fully torch-free: numpy resample -> numpy mel -> onnx -> numpy decode
hyp_full = run(F.resample(pcm, sr, 16000))
# also on torch-resampled audio, to isolate any resampler effect
hyp_mel = run(ref_res)

print("\n=== transcription ===")
print(f"  reference                    : {REF!r}")
print(f"  numpy mel (torch resample)   : {hyp_mel!r}   MATCH={hyp_mel==REF}")
print(f"  FULLY torch-free (numpy all) : {hyp_full!r}   MATCH={hyp_full==REF}")
print("\nRESULT:", "PASS ✅" if hyp_full == REF else "FAIL ❌")
