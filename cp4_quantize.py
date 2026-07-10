#!/usr/bin/env python
"""CP4 (a) — int8-quantize the shipped encoder.onnx for the browser."""
import os, shutil, time
from huggingface_hub import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType

A = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"), "assets")
OUT = "/home/swastik/Downloads/ASR_voicetomelspectrogram"
work = os.path.join(OUT, "onnx_build")
os.makedirs(work, exist_ok=True)

# encoder.onnx references external weights by relative name -> copy graph + externals together
print("staging encoder.onnx + external weights ...", flush=True)
enc_src = os.path.join(A, "encoder.onnx")
enc_stage = os.path.join(work, "encoder.onnx")
shutil.copy(os.path.realpath(enc_src), enc_stage)
for f in os.listdir(A):
    if f.startswith("onnx__") or "pre_encode" in f or f.startswith("Constant"):
        dst = os.path.join(work, f)
        if not os.path.exists(dst):
            shutil.copy(os.path.realpath(os.path.join(A, f)), dst)
print("staged. quantizing (int8 dynamic) ...", flush=True)

t=time.time()
enc_q = os.path.join(OUT, "encoder.int8.onnx")
quantize_dynamic(enc_stage, enc_q, weight_type=QuantType.QInt8)
print(f"encoder.int8.onnx written in {time.time()-t:.0f}s : {os.path.getsize(enc_q)/1e6:.1f} MB", flush=True)

# ctc_decoder is small (23MB) but quantize too for consistency
ctc_stage = os.path.join(work, "ctc_decoder.onnx")
shutil.copy(os.path.realpath(os.path.join(A,"ctc_decoder.onnx")), ctc_stage)
ctc_q = os.path.join(OUT, "ctc_decoder.int8.onnx")
quantize_dynamic(ctc_stage, ctc_q, weight_type=QuantType.QInt8)
print(f"ctc_decoder.int8.onnx : {os.path.getsize(ctc_q)/1e6:.1f} MB", flush=True)
print("QUANT DONE", flush=True)
