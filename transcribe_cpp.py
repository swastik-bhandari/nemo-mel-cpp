#!/usr/bin/env python
"""Indic ASR via the compiled C++ mel front-end.

Pipeline:
    wav  --(./indic_mel, compiled C++)-->  mel [80,T]  --(encoder.onnx)-->
    encoder_output  --(ctc_decoder.onnx)-->  logprobs  --(mask + greedy CTC)-->  text

The mel is produced by the C++ binary (indic_mel), NOT numpy. onnxruntime runs
the model; the decode is a small numpy step. No torch / torchaudio / NeMo.

Usage:
    python transcribe_cpp.py <audio.wav> [lang]      # lang default: ne
"""
import os, sys, json, struct, subprocess, tempfile
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = 256
BIN = os.path.join(HERE, "indic_mel")
SRC = os.path.join(HERE, "indic_mel.cpp")


def ensure_binary():
    """Compile indic_mel.cpp if the binary is missing or out of date."""
    if not os.path.exists(BIN) or os.path.getmtime(SRC) > os.path.getmtime(BIN):
        print("[build] compiling indic_mel.cpp ...", file=sys.stderr)
        subprocess.run(["g++", "-O2", "-std=c++17", SRC, "-o", BIN],
                       cwd=HERE, check=True)


def cpp_mel(wav_path: str) -> np.ndarray:
    """Run the C++ binary; return mel [1, 80, T] float32."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run([BIN, wav_path, out], check=True,
                       stdout=subprocess.DEVNULL)
        with open(out, "rb") as f:
            n_mels, T = struct.unpack("<ii", f.read(8))
            mel = np.frombuffer(f.read(), dtype=np.float32).reshape(n_mels, T)
        return np.ascontiguousarray(mel[None], dtype=np.float32), T
    finally:
        os.unlink(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    wav_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ne"

    ensure_binary()

    # load model + per-language decode tables (once)
    A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")
    ENC = ort.InferenceSession(f"{A}/encoder.onnx", providers=["CPUExecutionProvider"])
    CTC = ort.InferenceSession(f"{A}/ctc_decoder.onnx", providers=["CPUExecutionProvider"])
    masks = json.load(open(f"{A}/language_masks.json"))
    vocabs = json.load(open(f"{A}/vocab.json"))
    if lang not in masks or lang not in vocabs:
        print(f"error: unsupported language '{lang}'", file=sys.stderr)
        sys.exit(1)
    mask = np.array(masks[lang], dtype=bool)
    vocab = vocabs[lang]

    # 1. wav -> mel  (compiled C++ binary)
    feat, T = cpp_mel(wav_path)
    length = np.array([T], dtype=np.int64)

    # 2. mel -> encoder -> ctc  (onnxruntime)
    enc_out, _ = ENC.run(["outputs", "encoded_lengths"],
                         {"audio_signal": feat, "length": length})
    logprobs = CTC.run(["logprobs"], {"encoder_output": enc_out})[0]   # [1, T, 5633]

    # 3. language mask + greedy CTC decode
    masked = logprobs[0][:, mask]
    idx = masked.argmax(axis=-1)
    collapsed = idx[np.insert(np.diff(idx) != 0, 0, True)]
    text = "".join(vocab[x] for x in collapsed if x != BLANK).replace("▁", " ").strip()

    print(text)


if __name__ == "__main__":
    main()
