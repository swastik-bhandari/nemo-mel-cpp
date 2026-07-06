"""
Compare the C++ implementation's output DIRECTLY against genuine NeMo code —
no nemo_mel_standalone.py involved.

Reference = NVIDIA's own, UNMODIFIED features.py + segment.py from your
installed nemo package:
  Path A: normal `import nemo` (used if it works on your machine)
  Path B: if Path A fails (broken onnx/protobuf chain), NVIDIA's untouched
          .py files are loaded directly from site-packages via importlib,
          with stubs registered ONLY for import scaffolding (logging, the
          no-op AudioAugmentor). Every line of DSP that executes is NVIDIA's.

Steps it performs:
  1. run the reference NeMo pipeline on the wav (AudioSegment.from_file ->
     FilterbankFeatures with AudioToMelSpectrogramPreprocessor defaults,
     eval mode, channel_selector='average' for multichannel)
  2. load the C++ binary output
  3. report shape/seq_len match + max abs/relative differences

Usage:
    # first build and run the C++ side:
    g++ -O2 -std=c++17 nemo_mel.cpp -o nemo_mel -lsndfile -lsoxr
    ./nemo_mel example.wav nemo_mel_constants.bin output_mel.bin
    # then compare:
    python compare_cpp_vs_nemo.py example.wav output_mel.bin
"""

import importlib.util
import os
import struct
import sys
import types

import numpy as np
import soundfile as sf
import torch

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "example.wav"
CPP_BIN = sys.argv[2] if len(sys.argv) > 2 else "output_mel.bin"
SR = 16000

if not os.path.exists(AUDIO):
    sys.exit(f"audio file not found: {AUDIO}")
if not os.path.exists(CPP_BIN):
    sys.exit(
        f"C++ output not found: {CPP_BIN}\n"
        f"Build and run the C++ extractor first:\n"
        f"  g++ -O2 -std=c++17 nemo_mel.cpp -o nemo_mel -lsndfile -lsoxr\n"
        f"  ./nemo_mel {AUDIO} nemo_mel_constants.bin {CPP_BIN}"
    )

CS = 'average' if sf.info(AUDIO).channels > 1 else None
if CS:
    print(f"[setup] {AUDIO} is multichannel -> channel_selector='average' (matches C++ downmix)\n")

# ----------------------------------------------------------------------------
# 1. Genuine NeMo reference
# ----------------------------------------------------------------------------
FilterbankFeaturesRef = None
AudioSegmentRef = None

try:  # ---- Path A ----
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures as FilterbankFeaturesRef
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment as AudioSegmentRef

    print("[ref] Path A: normal `import nemo`")
except Exception as e:  # ---- Path B ----
    print(f"[ref] normal nemo import failed ({type(e).__name__}); "
          f"loading NVIDIA's unmodified files directly (Path B)")

    def find_nemo_dir():
        for p in sys.path:
            cand = os.path.join(p, "nemo")
            if os.path.isdir(os.path.join(cand, "collections", "asr", "parts", "preprocessing")):
                return cand
        raise FileNotFoundError(
            "Could not find installed nemo package. "
            "Set NEMO_DIR env var to your site-packages/nemo directory."
        )

    NEMO_DIR = os.environ.get("NEMO_DIR") or find_nemo_dir()
    PREP = os.path.join(NEMO_DIR, "collections", "asr", "parts", "preprocessing")
    print(f"[ref] using NVIDIA's files at {PREP}\n")

    def register_stub(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def load_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    import logging as _stdlog

    for pkg in [
        "nemo", "nemo.utils", "nemo.collections", "nemo.collections.asr",
        "nemo.collections.asr.parts", "nemo.collections.asr.parts.preprocessing",
        "nemo.collections.asr.parts.utils",
    ]:
        if pkg not in sys.modules:
            register_stub(pkg)
    sys.modules["nemo.utils"].logging = _stdlog

    register_stub(
        "nemo.collections.asr.parts.utils.audio_utils",
        db2mag=lambda db: 10.0 ** (db / 20.0),
    )

    seg_mod = load_from_path("nemo.collections.asr.parts.preprocessing.segment",
                             os.path.join(PREP, "segment.py"))
    AudioSegmentRef = seg_mod.AudioSegment

    class _NoOpAudioAugmentor:
        def __init__(self, *a, **k): ...
        def perturb(self, segment): ...
        def max_augmentation_length(self, length): return length
        @classmethod
        def from_config(cls, config): return cls()

    register_stub("nemo.collections.asr.parts.preprocessing.perturb",
                  AudioAugmentor=_NoOpAudioAugmentor)

    feat_mod = load_from_path("nemo.collections.asr.parts.preprocessing.features_ref",
                              os.path.join(PREP, "features.py"))
    FilterbankFeaturesRef = feat_mod.FilterbankFeatures

# Run reference exactly as normal NeMo (AudioToMelSpectrogramPreprocessor
# defaults, eval mode => no dither => deterministic)
ref_featurizer = FilterbankFeaturesRef(
    sample_rate=SR,
    n_window_size=int(0.02 * SR),
    n_window_stride=int(0.01 * SR),
    window="hann",
    normalize="per_feature",
    n_fft=None,
    preemph=0.97,
    nfilt=64,
    lowfreq=0,
    highfreq=None,
    log=True,
    log_zero_guard_type="add",
    log_zero_guard_value=2**-24,
    dither=1e-5,
    pad_to=16,
    frame_splicing=1,
    exact_pad=False,
    pad_value=0,
    mag_power=2.0,
    mel_norm="slaney",
)
ref_featurizer.eval()

segment = AudioSegmentRef.from_file(AUDIO, target_sr=SR, channel_selector=CS)
ref_signal = torch.tensor(segment.samples, dtype=torch.float).unsqueeze(0)
ref_length = torch.tensor([ref_signal.shape[1]], dtype=torch.long)

with torch.no_grad():
    mel_ref, len_ref = ref_featurizer(ref_signal, ref_length)
mel_ref = mel_ref.squeeze(0).numpy()

print(f"[nemo] mel {mel_ref.shape}, valid frames {len_ref.item()}")

# ----------------------------------------------------------------------------
# 2. C++ output
# ----------------------------------------------------------------------------
with open(CPP_BIN, "rb") as f:
    n_mels, total, seq_len = struct.unpack("<3i", f.read(12))
    mel_cpp = np.frombuffer(f.read(), dtype="<f4").reshape(n_mels, total)

print(f"[c++]  mel {mel_cpp.shape}, valid frames {seq_len}")

# ----------------------------------------------------------------------------
# 3. Compare
# ----------------------------------------------------------------------------
if mel_ref.shape != mel_cpp.shape or len_ref.item() != seq_len:
    print("\nFAIL: shape / seq_len mismatch")
    sys.exit(1)

diff = np.abs(mel_ref - mel_cpp)
rel = diff / np.maximum(np.abs(mel_ref), 1e-8)
atol_pass = diff.max() < 1e-4

print("\n================ C++ vs GENUINE NeMo ================")
print(f"bit-exact                  : {bool(np.array_equal(mel_ref, mel_cpp))}")
print(f"max |abs diff|             : {diff.max():.3e}")
print(f"mean |abs diff|            : {diff.mean():.3e}")
print(f"max relative diff          : {rel.max():.3e}")
print(f"within float32 rounding    : {atol_pass}  (threshold 1e-4)")
print("=====================================================")
if atol_pass:
    print("RESULT: C++ output matches genuine NeMo at float32-rounding level.")
else:
    print("RESULT: differences exceed float32 rounding - investigate.")
    sys.exit(1)
