"""
Dump the exact constants that NeMo's FilterbankFeatures uses into a binary
blob for the C++ implementation. This removes ALL reimplementation risk for:

  * the Hann window:      torch.hann_window(320, periodic=False)  (float32)
  * the mel filterbank:   librosa.filters.mel(sr=16000, n_fft=512, n_mels=64,
                          fmin=0, fmax=8000, norm='slaney')        (float32,
                          cast exactly as FilterbankFeatures does)

Binary layout (little-endian):
  int32 : sample_rate        (16000)
  int32 : n_fft              (512)
  int32 : win_length         (320)
  int32 : hop_length         (160)
  int32 : n_mels             (64)
  int32 : pad_to             (16)
  float32[win_length]        : window
  float32[n_mels * (n_fft/2+1)] : mel filterbank, row-major [n_mels, n_freq]

Usage:  python dump_constants.py [out.bin]
"""

import struct
import sys

import librosa
import numpy as np
import torch

SR, WIN, HOP, NMELS, PAD_TO = 16000, 320, 160, 64, 16
NFFT = 2 ** int(np.ceil(np.log2(WIN)))  # 512, same formula as FilterbankFeatures

out = sys.argv[1] if len(sys.argv) > 1 else "nemo_mel_constants.bin"

# Exactly as FilterbankFeatures.__init__ builds them:
window = torch.hann_window(WIN, periodic=False).numpy().astype(np.float32)
fb = torch.tensor(
    librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=NMELS, fmin=0, fmax=SR / 2, norm="slaney"),
    dtype=torch.float,
).numpy()  # [n_mels, n_fft//2+1] float32

assert window.shape == (WIN,)
assert fb.shape == (NMELS, NFFT // 2 + 1)

with open(out, "wb") as f:
    f.write(struct.pack("<6i", SR, NFFT, WIN, HOP, NMELS, PAD_TO))
    f.write(window.tobytes())
    f.write(fb.astype("<f4").tobytes())

print(f"wrote {out}: sr={SR} n_fft={NFFT} win={WIN} hop={HOP} n_mels={NMELS} pad_to={PAD_TO}")
print(f"window[0:3]={window[:3]}, fb.sum()={fb.sum():.6f}")
