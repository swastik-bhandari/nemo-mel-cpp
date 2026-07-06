# NeMo Mel Spectrogram — C++ Implementation

C++ port of NeMo's `AudioToMelSpectrogramPreprocessor` default pipeline
(`FilterbankFeatures`, eval mode). Verified against the Python standalone
(itself bit-exact vs. genuine NeMo) at **float32-rounding level**:
max abs diff ~6e-5 on normalized log-mel (~1e-5 relative), across mono,
stereo, 44.1 kHz (resample path), float32-encoded, and very short inputs.

## Files

| File | Purpose |
|---|---|
| `nemo_mel.cpp` | The implementation (single file) |
| `pocketfft_hdronly.h` | Header-only FFT — the same FFT PyTorch bundles for CPU |
| `dump_constants.py` | Dumps hann window + librosa mel filterbank to binary (run once) |
| `nemo_mel_constants.bin` | The dumped constants (pre-generated, sr=16000/n_fft=512/64 mels) |
| `compare_cpp.py` | Verification harness vs. `nemo_mel_standalone.py` |

## Dependencies

- `libsndfile` — identical to python-soundfile (it *is* the same C library)
- `libsoxr` — identical to librosa's default `soxr_hq` resampler backend

```bash
sudo apt install libsndfile1-dev libsoxr-dev
```

## Build & run

```bash
g++ -O2 -std=c++17 nemo_mel.cpp -o nemo_mel -lsndfile -lsoxr
./nemo_mel input.wav nemo_mel_constants.bin output_mel.bin
```

Output binary: `int32 n_mels, int32 total_frames, int32 seq_len`, then
`float32[n_mels * total_frames]` row-major. Read in Python:

```python
import struct, numpy as np
with open("output_mel.bin", "rb") as f:
    n_mels, total, seq_len = struct.unpack("<3i", f.read(12))
    mel = np.frombuffer(f.read(), dtype="<f4").reshape(n_mels, total)
```

## Verify against NeMo

```bash
python dump_constants.py                       # regenerate constants if needed
./nemo_mel your.wav nemo_mel_constants.bin output_mel.bin
python compare_cpp.py your.wav output_mel.bin  # needs nemo_mel_standalone.py
```

## Exactness statement (for provenance)

- **Bit-exact to PyTorch: no.** Different FFT planning and reduction orders
  make bit-identity across frameworks unattainable; the only bit-exact C++
  route is libtorch running the TorchScript-exported featurizer.
- **Float32-rounding-exact: yes.** Constants (window, mel filterbank) are
  byte-identical (dumped, not reimplemented); audio loading and resampling
  use the same underlying C libraries as the Python stack; op order mirrors
  torch (incl. sqrt-then-square magnitude). Residual ~1e-5 relative diffs
  come from float32 accumulation order only. For reference, NeMo's own
  docs note that even bf16-level errors (~1e-2) move WER by only ~0.1%.

## Config changes

The constants file hard-codes sr=16000, n_fft=512, win=320, hop=160,
n_mels=64, pad_to=16 (NeMo defaults). For other configs, edit the values at
the top of `dump_constants.py` and regenerate — the C++ reads everything
from the binary, no recompile needed. (preemph=0.97, log guard 2^-24, and
per_feature normalization are hard-coded in the .cpp; change there if your
NeMo config differs.)
