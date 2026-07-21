#!/usr/bin/env bash
# build_onefile.sh -- compile the stub and pack everything into ONE executable.
set -euo pipefail
cd "$(dirname "$0")"

ORT=${ORT:-/home/swastik/Downloads/onnxruntime-linux-x64-1.20.1}
ASSETS=indic_asr_offline/assets
LANG=indic_asr_offline/lang
SO=indic_asr_offline/lib/libonnxruntime.so.1.20.1
OUT=indic_asr_solo

echo "[1/2] compiling stub (onnxruntime dlopen'd at runtime, not linked)..."
# -static-libstdc++/-static-libgcc: no libstdc++ sibling needed for OUR code.
# NOT linking -lonnxruntime: it is loaded from an in-memory memfd at runtime.
g++ -O2 -std=c++17 indic_asr_onefile.cpp -o indic_asr_onefile.stub \
    -I"$ORT/include" \
    -static-libstdc++ -static-libgcc \
    -ldl -lpthread

echo "[2/2] packing blob (.so + models + 366 weights + lang tables)..."
python3 pack_onefile.py indic_asr_onefile.stub "$ASSETS" "$LANG" "$SO" "$OUT"
rm -f indic_asr_onefile.stub
echo "done -> ./$OUT   (single file, run: ./$OUT test_nepali.wav ne)"
