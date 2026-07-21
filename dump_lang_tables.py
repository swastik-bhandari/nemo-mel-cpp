#!/usr/bin/env python
"""Build-time: dump compact per-language decode tables for the native C++ binary.

For each language emits  lang/<code>.tbl  (little-endian):
    int32   n_cols                 (== 257)
    int32   cols[n_cols]           column indices into the 5633-wide logprobs
    int32   blank_idx              (== 256)
    repeat n_cols times:
        int32 byte_len, <utf8 token bytes>   (vocab order == column order)

This removes any need for a JSON parser or SentencePiece lib in C++: the decode
is just gather -> argmax -> collapse -> map token -> replace U+2581 with space.
"""
import os, json, struct, sys
import numpy as np
from huggingface_hub import snapshot_download

OUT = sys.argv[1] if len(sys.argv) > 1 else "lang"
os.makedirs(OUT, exist_ok=True)
A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")
masks = json.load(open(f"{A}/language_masks.json"))
vocabs = json.load(open(f"{A}/vocab.json"))

n = 0
for lg, m in masks.items():
    if lg not in vocabs:
        continue
    cols = np.where(np.array(m, dtype=bool))[0].astype(np.int32)
    vocab = vocabs[lg]
    assert len(cols) == len(vocab), f"{lg}: {len(cols)} cols vs {len(vocab)} vocab"
    with open(os.path.join(OUT, f"{lg}.tbl"), "wb") as f:
        f.write(struct.pack("<i", len(cols)))
        f.write(cols.tobytes())
        f.write(struct.pack("<i", 256))            # blank_idx
        for tok in vocab:
            b = tok.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)
    n += 1
print(f"wrote {n} language tables to {OUT}/")
