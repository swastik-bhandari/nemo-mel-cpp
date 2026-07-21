"""
Score every hyp_*.json in a dataset dir against manifest refs (WER + CER),
plus a parity check between OURS and THEIRS-CTC.
Usage:  python benchmarks/common/score.py --dir benchmarks/fleurs_ne
Writes <dir>/scores.json
"""
import json, os, re, argparse, unicodedata
import jiwer

def norm(s):
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[।॥.,!?;:\"'`()\[\]{}<>/\\|@#%^&*_=+~—–\-]", " ", s)
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()

man = {m["id"]: m for m in json.load(open(os.path.join(args.dir, "manifest.json")))}

def load(name):
    p = os.path.join(args.dir, name)
    return {x["id"]: x["hyp"] for x in json.load(open(p))} if os.path.exists(p) else None

def score(hyps, label):
    refs, hys = [], []
    for i, m in man.items():
        r = norm(m["ref"])
        if not r or i not in hyps:
            continue
        refs.append(r); hys.append(norm(hyps[i]))
    W, C = jiwer.wer(refs, hys)*100, jiwer.cer(refs, hys)*100
    print(f"{label:24s}  n={len(refs):3d}  WER={W:6.2f}%   CER={C:6.2f}%")
    return {"label": label, "n": len(refs), "wer": round(W,2), "cer": round(C,2)}

print(f"=== {args.dir} ===")
results = []
for name, lab in [("hyp_ours.json","OURS (offline CTC)"),
                  ("hyp_theirs_ctc.json","THEIRS (PyTorch CTC)"),
                  ("hyp_theirs_rnnt.json","THEIRS (PyTorch RNNT)")]:
    h = load(name)
    if h is None:
        print(f"{lab:24s}  (not run yet)")
    else:
        results.append(score(h, lab))

o, t = load("hyp_ours.json"), load("hyp_theirs_ctc.json")
if o and t:
    ids = sorted(set(o) & set(t))
    nfc = lambda s: unicodedata.normalize("NFC", s.strip())
    exact = sum(nfc(o[i]) == nfc(t[i]) for i in ids)
    pw = jiwer.wer([nfc(t[i]) for i in ids], [nfc(o[i]) for i in ids])*100
    print(f"\nPARITY ours vs theirs-CTC: exact {exact}/{len(ids)} clips, WER={pw:.2f}%")

json.dump(results, open(os.path.join(args.dir, "scores.json"), "w"), ensure_ascii=False, indent=1)
