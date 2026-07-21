"""
Run OUR offline binary (indic_asr_offline/indic_asr) over a dataset dir's clips.
Usage:  python benchmarks/common/run_ours.py --dir benchmarks/fleurs_ne --lang ne
Writes <dir>/hyp_ours.json
"""
import json, subprocess, time, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "indic_asr_offline", "indic_asr")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--lang", default="ne")
args = ap.parse_args()

man = json.load(open(os.path.join(args.dir, "manifest.json")))
out, t0 = [], time.time()
for m in man:
    wav = os.path.join(args.dir, m["wav"])
    ts = time.time()
    try:
        r = subprocess.run([BIN, wav, args.lang], capture_output=True, text=True, timeout=120)
        hyp = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
    except Exception as e:
        hyp = f"__ERROR__ {e}"
    out.append({"id": m["id"], "hyp": hyp, "sec": round(time.time()-ts, 2)})
    if (m["id"]+1) % 10 == 0:
        print(f"  ours {m['id']+1}/{len(man)} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(os.path.join(args.dir, "hyp_ours.json"), "w"), ensure_ascii=False, indent=1)
print(f"[done ours] {len(out)} clips in {time.time()-t0:.0f}s")
