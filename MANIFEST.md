# MANIFEST — psych_scope

Provenance and allowlist for the public export. Source working folder:
`C:\Research\PsychScope` (read-only lab bench; this export is the cleaned exhibit).

## Staged (public)

| Path | Source | Notes |
|---|---|---|
| `paper/psychscope.tex` | `PsychScope/paper/` | paper source, mimesis research voice |
| `paper/psychscope.pdf` | `PsychScope/paper/` | compiled, 13 pp, 24 refs (all web-verified) |
| `paper/figures/fig*.pdf` | `PsychScope/figures/` | 5 paper figures |
| `pipeline/*.py` | `PsychScope/src/` | 17 scripts: behavioral grids, causal pilots, analysis, figures |
| `results/runs/*/` | `PsychScope/runs/*/` | per-run `summary.json`, `behavioral.csv`, `results.csv`, `report.json` only |
| `results/*.md` | `PsychScope/docs/` | powered results, ITT, multiplicity, methodology handbook, findings report |
| `results/fig*.pdf` | `PsychScope/figures/` | figures |

## Deliberately excluded (allowlist, not blocklist)

- `runs/*/completions.jsonl` — raw model generations (~thousands/run); processed CSVs staged instead. Available on request.
- `runs/*/feat_means.pt` — large SAE activation tensors; regenerable from scripts.
- Hugging Face caches, model weights, virtual environments.
- `_gauntlet/` — internal critique trail (three adversarial rounds + reviewer panel + decision logs). Underscore-prefixed, not uploaded per repo convention.
- No `.env`, tokens, or API receipts exist in this tree (secrets scan: clean).

## Verification

- Secrets scan: `_pipeline/scan-secrets.ps1 -Path psych_scope` -> clean.
- Bibliography: all 24 references web-verified against source (Nature, Lancet, ICLR, NeurIPS, arXiv).
- Numbers: independent 22/22 re-derivation from raw per-row files (see `results/FINDINGS_REPORT.md`).
