# ASR WER Benchmarks

Head-to-head WER/CER of our **offline binary** (`indic_asr_offline/indic_asr`, ONNX, no
PyTorch) vs the **official PyTorch model** (`ai4bharat/indic-conformer-600m-multilingual`),
scored against ground-truth transcripts from HuggingFace datasets.

## Layout
```
common/         generic, --dir-driven scripts (one set, reused per dataset)
  fetch_hf.py     stream a HF split -> clips/*.wav (16k) + manifest.json
  run_ours.py     our offline binary  -> hyp_ours.json
  run_theirs.py   their PyTorch model -> hyp_theirs_ctc.json + hyp_theirs_rnnt.json
  score.py        WER+CER for all hyps + ours-vs-theirs parity -> scores.json
<dataset>/      one folder per dataset run
  config.json     which HF dataset/config/split/lang/n + ref field names
  clips/          fetched wavs
  manifest.json   id, wav (relative), ref
  hyp_*.json      per-clip hypotheses
  scores.json     final WER/CER table
```

## Run a benchmark (from repo root, using indic_env)
```bash
V=indic_env/bin/python
D=benchmarks/fleurs_ne
$V benchmarks/common/fetch_hf.py  --dir $D            # 1. fetch a chunk
$V benchmarks/common/run_ours.py  --dir $D --lang ne  # 2. our binary
$V benchmarks/common/run_theirs.py --dir $D --lang ne # 3. their model (needs HF token)
$V benchmarks/common/score.py     --dir $D            # 4. score
```

## Datasets
- **fleurs_ne** — Google FLEURS Nepali, clean read speech, public (no terms). ✅ done.
- **indicvoices_ne** — AI4Bharat IndicVoices, spontaneous speech, the model's own
  benchmark. **Gated** — accept terms first (see its `config.json`).

## Results — Nepali, 100 clips each (2026-07-16)

**FLEURS** (clean read speech, public):
| System | Deps | WER | CER |
|---|---|---|---|
| OURS (offline ONNX CTC) | none | 27.16% | 8.78% |
| THEIRS PyTorch CTC | torch+HF | 27.16% | 8.78% |
| THEIRS PyTorch RNNT (their best) | torch+HF | 26.93% | 8.67% |

**IndicVoices** (AI4Bharat's own benchmark, `valid` split, strided sample):
| System | Deps | WER | CER |
|---|---|---|---|
| OURS (offline ONNX CTC) | none | 17.62% | 5.88% |
| THEIRS PyTorch CTC | torch+HF | 17.62% | 5.88% |
| THEIRS PyTorch RNNT (their best) | torch+HF | 15.97% | 5.57% |

**On both datasets: ours vs theirs-CTC = 100/100 clips byte-identical (0.00% WER).**
Our torch-free offline port reproduces their CTC path exactly, and trails their best RNNT
mode by only ~0.2 pts (FLEURS) / ~1.6 pts (IndicVoices). IndicVoices WER is lower than
FLEURS because its normalized transcripts avoid FLEURS's number-spelling/word-joining
inflation — so CER (~5.9%) there is the cleanest accuracy signal we have.
