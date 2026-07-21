"""
Generic HF-dataset chunk fetcher for the ASR WER harness.

Reads <dir>/config.json:
  {
    "hf_dataset": "google/fleurs",
    "hf_config":  "ne_np",          # or null
    "split":      "test",
    "n":          100,
    "audio_field":"audio",
    "ref_fields": ["transcription","raw_transcription","text"]  # first non-empty wins
  }

Streams the split (no full download), decodes audio with soundfile via
Audio(decode=False) to avoid the torchcodec requirement in datasets>=5,
writes <dir>/clips/NNNN.wav (16k PCM_16 mono) + <dir>/manifest.json.

Usage:  python benchmarks/common/fetch_hf.py --dir benchmarks/fleurs_ne
"""
import os, io, json, argparse
import numpy as np, soundfile as sf
from datasets import load_dataset, Audio

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.dir, "config.json")))
outclips = os.path.join(args.dir, "clips")
os.makedirs(outclips, exist_ok=True)
N = cfg.get("n", 100)
audio_field = cfg.get("audio_field", "audio")
ref_fields = cfg.get("ref_fields", ["transcription", "raw_transcription", "text", "sentence"])

print(f"[fetch] streaming {cfg['hf_dataset']} {cfg.get('hf_config')} "
      f"{cfg['split']}, first {N} clips ...", flush=True)
ds = load_dataset(cfg["hf_dataset"], cfg.get("hf_config"),
                  split=cfg["split"], streaming=True)
ds = ds.cast_column(audio_field, Audio(decode=False))  # raw bytes, no torchcodec

def pick_ref(ex):
    for f in ref_fields:
        v = ex.get(f)
        if v:
            return v, f
    return "", None

manifest, kept = [], 0
for ex in ds:
    if kept >= N:
        break
    a = ex[audio_field]
    raw = a.get("bytes")
    if raw is None:            # some datasets give a path instead of bytes
        raw = open(a["path"], "rb").read()
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:            # resample handled downstream by the model, but keep 16k for the C++ binary
        import math
        # simple guard: our binary resamples internally, so just record sr and keep native
        pass
    ref, _ = pick_ref(ex)
    if not ref:
        continue               # skip unlabeled
    wav = os.path.join(outclips, f"{kept:04d}.wav")
    sf.write(wav, arr, sr, subtype="PCM_16")
    manifest.append({"id": kept, "wav": f"clips/{kept:04d}.wav",
                     "sr": int(sr), "dur_s": round(len(arr)/sr, 2), "ref": ref})
    kept += 1
    if kept % 20 == 0:
        print(f"  {kept} clips written", flush=True)

json.dump(manifest, open(os.path.join(args.dir, "manifest.json"), "w"),
          ensure_ascii=False, indent=1)
tot = sum(m["dur_s"] for m in manifest)
print(f"[done] {len(manifest)} clips, {tot:.0f}s (~{tot/60:.1f} min), sr={manifest[0]['sr'] if manifest else '?'}")
print("sample ref:", manifest[0]["ref"][:90] if manifest else "NONE")
