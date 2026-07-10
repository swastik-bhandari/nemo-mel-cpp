#!/usr/bin/env python
"""CHECKPOINT 2 — dump the model's REAL preprocessor config + vocab. No assumptions."""
import json, os, re
import onnx, torch
from huggingface_hub import snapshot_download

SNAP = snapshot_download("ai4bharat/indic-conformer-600m-multilingual")
A = os.path.join(SNAP, "assets")
OUT = os.path.dirname(os.path.abspath(__file__))
LANG = "ne"

def onnx_io(path):
    m = onnx.load(path, load_external_data=False)
    def shp(vi):
        d = [(x.dim_param or x.dim_value) for x in vi.type.tensor_type.shape.dim]
        return f"{vi.name}{d}"
    return [shp(i) for i in m.graph.input], [shp(o) for o in m.graph.output]

print("=== encoder.onnx ===")
ei, eo = onnx_io(f"{A}/encoder.onnx")
print("  inputs :", ei)
print("  outputs:", eo)

print("=== ctc_decoder.onnx ===")
ci, co = onnx_io(f"{A}/ctc_decoder.onnx")
print("  inputs :", ci)
print("  outputs:", co)

# ---- preprocessor.ts (TorchScript NeMo featurizer): pull scalar constants + buffer shapes
print("=== preprocessor.ts constants ===")
pp = torch.jit.load(f"{A}/preprocessor.ts", map_location="cpu")
prep = {}
# named buffers reveal window length & mel-filterbank shape
for name, buf in pp.named_buffers():
    prep[f"buffer:{name}"] = list(buf.shape)
    print(f"  buffer {name}: shape={list(buf.shape)}")
# scalar attributes scraped from the scripted graph source
src = pp.code
for key in ["sample_rate","n_fft","win_length","hop_length","n_window_size","n_window_stride",
            "features","nfilt","dither","preemph","log","normalize","mag_power","frame_splicing",
            "lowfreq","highfreq","pad_to","pad_value","exact_pad","use_grads","center"]:
    m = re.search(rf"{key}\s*[:=]\s*([0-9eE.+\-]+|True|False|None|\"[a-zA-Z_ ]+\")", src)
    if m:
        prep[key] = m.group(1)
        print(f"  {key} = {m.group(1)}")

# ---- vocab + language mask
vocab = json.load(open(f"{A}/vocab.json"))
masks = json.load(open(f"{A}/language_masks.json"))
ne_vocab = vocab[LANG]
ne_mask = masks[LANG]
BLANK = 256
print("=== vocab / mask (ne) ===")
print("  languages in vocab.json:", list(vocab.keys()))
print("  len(vocab['ne']) :", len(ne_vocab))
print("  len(mask['ne'])  :", len(ne_mask))
print("  BLANK_ID (config):", BLANK, "-> vocab['ne'][256] =", repr(ne_vocab[256]) if len(ne_vocab)>256 else "N/A")
print("  first 20 ne tokens:", ne_vocab[:20])
print("  sample mid tokens :", ne_vocab[100:110])

# n_mels = encoder input feature dimension (2nd dim of audio_signal [B, n_mels, T])
enc_in_dims = onnx.load(f"{A}/encoder.onnx", load_external_data=False).graph.input[0].type.tensor_type.shape.dim
n_mels = enc_in_dims[1].dim_value or enc_in_dims[1].dim_param
print("\n########################################")
print(f"###  n_mels (encoder input dim) = {n_mels}")
print("########################################")

# ---- write artifacts for later checkpoints
json.dump(ne_vocab, open(f"{OUT}/indic_vocab_ne.json","w"), ensure_ascii=False, indent=0)
json.dump(ne_mask,  open(f"{OUT}/indic_langmask_ne.json","w"))
json.dump({"n_mels_from_encoder": n_mels, "BLANK_ID": BLANK,
           "encoder_inputs": ei, "encoder_outputs": eo,
           "ctc_inputs": ci, "ctc_outputs": co,
           "preprocessor": prep}, open(f"{OUT}/indic_preproc_config.json","w"),
          ensure_ascii=False, indent=2)
print("\nwrote: indic_vocab_ne.json, indic_langmask_ne.json, indic_preproc_config.json")
