"""
Run THEIR official PyTorch model (ai4bharat/indic-conformer-600m-multilingual)
over a dataset dir's clips, in both CTC and RNNT decode modes.
Needs a gated HF token in ~/.cache/huggingface/token.
Usage:  python benchmarks/common/run_theirs.py --dir benchmarks/fleurs_ne --lang ne
Writes <dir>/hyp_theirs_ctc.json and <dir>/hyp_theirs_rnnt.json
"""
import json, time, os, argparse
import numpy as np, soundfile as sf, torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--lang", default="ne")
ap.add_argument("--modes", default="ctc,rnnt")
args = ap.parse_args()

print("[load] loading ai4bharat/indic-conformer-600m-multilingual ...", flush=True)
t0 = time.time()
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual",
                                  trust_remote_code=True)
model.eval()
print(f"[load] ready in {time.time()-t0:.0f}s", flush=True)

def load_wav(path):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as F
        arr = F.resample(torch.from_numpy(arr), sr, 16000).numpy()
    return torch.from_numpy(arr).unsqueeze(0)

def transcribe(wav, dec):
    with torch.no_grad():
        out = model(wav, args.lang, dec)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return str(out).strip()

man = json.load(open(os.path.join(args.dir, "manifest.json")))
for dec in args.modes.split(","):
    out, td = [], time.time()
    for m in man:
        wav = load_wav(os.path.join(args.dir, m["wav"]))
        ts = time.time()
        try:
            hyp = transcribe(wav, dec)
        except Exception as e:
            hyp = f"__ERROR__ {e}"
        out.append({"id": m["id"], "hyp": hyp, "sec": round(time.time()-ts, 2)})
        if (m["id"]+1) % 10 == 0:
            print(f"  [{dec}] {m['id']+1}/{len(man)} ({time.time()-td:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(args.dir, f"hyp_theirs_{dec}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[done {dec}] {len(out)} clips in {time.time()-td:.0f}s")
