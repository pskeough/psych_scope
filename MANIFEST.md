# MANIFEST: psych_scope

Source working folder: `C:\Research\PsychScope`. This repository is the public export.

| Path | Contents |
|---|---|
| `README.md` | results, pipeline diagram, setup |
| `pipeline/*.py` | scripts for the behavioural grids and causal pilots |
| `results/runs/*/` | per-run `summary.json`, `behavioral.csv`, `results.csv`, `report.json` |
| `results/analysis/*/summary.json` | one summary per experiment (format, direction, calibration, held-out, null cue, layer patching) |
| `results/analysis/layer_placement/`, `results/analysis/probes/` | layer-patching and probe outputs |
| `results/analysis/RECEIPTS.md` | number-to-source map for every README number |

Not included: raw completions (`completions.jsonl`), SAE activation tensors, model weights and caches. Raw completions are available on request.

Secrets scan (2026-10-19): no keys, tokens or personal paths.
