#!/usr/bin/env python
"""CHECKPOINT 1 — load indic-conformer-600m and transcribe test_nepali.wav (CTC path).

Ground-truth reference. Nothing downstream is valid until the user confirms
the Nepali printed here is correct.
"""
import sys
import torch
import torchaudio

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
WAV = "test_nepali.wav"
LANG = "ne"

def main():
    from transformers import AutoModel
    print(f"[cp1] loading {MODEL_ID} (trust_remote_code=True) ...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    print("[cp1] model loaded OK", flush=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--load-only":
        print("[cp1] --load-only: skipping transcription (no wav yet)")
        return

    # load audio via stdlib wave (torchaudio 2.11 .load needs torchcodec, not installed)
    import wave, numpy as np
    with wave.open(WAV, "rb") as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    assert sw == 2, f"expected 16-bit PCM, got sampwidth={sw}"
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)          # downmix to mono
    wav = torch.from_numpy(pcm).unsqueeze(0)            # [1, n]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)  # pure-tensor, no torchcodec
        sr = 16000
    print(f"[cp1] audio: {WAV}  sr={sr}  samples={wav.shape[-1]}", flush=True)

    with torch.no_grad():
        text = model(wav, LANG, "ctc")
    print("\n==================== CHECKPOINT 1 — CTC TRANSCRIPTION ====================")
    print(repr(text))
    print(text)
    print("=========================================================================")

if __name__ == "__main__":
    main()
