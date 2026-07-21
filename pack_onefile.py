#!/usr/bin/env python3
# pack_onefile.py -- append the onnxruntime .so, models, external weights and
# language tables to the compiled stub, producing ONE self-contained executable.
#
#   python3 pack_onefile.py <stub_bin> <assets_dir> <lang_dir> <so_path> <out_bin>
#
# Container: [stub][file bytes ...][index][footer(32B)], all little-endian.
import os, shutil, struct, sys

stub, assets, langdir, so_path, out = sys.argv[1:6]

# names referenced as ONNX external data by encoder.onnx
with open("/tmp/enc_locations.txt") as f:
    ext_names = [l.strip() for l in f if l.strip()]

# (stored_name, source_path)
files = [
    ("libonnxruntime.so", so_path),
    ("encoder.onnx",      os.path.join(assets, "encoder.onnx")),
    ("ctc_decoder.onnx",  os.path.join(assets, "ctc_decoder.onnx")),
]
for n in ext_names:
    files.append((n, os.path.join(assets, n)))
for fn in sorted(os.listdir(langdir)):
    if fn.endswith(".tbl"):
        files.append((fn, os.path.join(langdir, fn)))

# sanity: unique names, all present
seen = set()
for name, path in files:
    if name in seen:
        sys.exit(f"duplicate embed name: {name}")
    seen.add(name)
    if not os.path.isfile(path):
        sys.exit(f"missing source file: {path}")

index = []  # (name, offset, length)
with open(out, "wb") as o:
    with open(stub, "rb") as s:
        shutil.copyfileobj(s, o, 1 << 20)
    for name, path in files:
        off = o.tell()
        with open(path, "rb") as src:
            shutil.copyfileobj(src, o, 1 << 20)
        index.append((name, off, o.tell() - off))

    idx_off = o.tell()
    buf = bytearray()
    buf += struct.pack("<I", len(index))
    for name, off, ln in index:
        nb = name.encode("utf-8")
        buf += struct.pack("<I", len(nb)) + nb + struct.pack("<QQ", off, ln)
    o.write(buf)
    o.write(b"IASRPK01" + struct.pack("<QQQ", idx_off, len(buf), 0))

os.chmod(out, 0o755)
total = os.path.getsize(out)
print(f"embedded {len(files)} files -> {out}  ({total/1e9:.2f} GB)")
