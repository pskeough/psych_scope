# Multiplicity & forking-paths statement (critic round 2, stats attack)
*The honest treatment reviewers will demand. 2026-07-24.*

## The concern
Across ~17 pilots there are many implicit hypothesis tests: SES gaps x3 models
x3 instruments; per-item breakdowns (8 items x3 models); layer sweeps (12 layers
x several positions x2 contrasts); feature-selection thresholds (top-10/20/50);
permutation nulls (tabled and untabled); intervention arms. Analysis flexibility
(garden of forking paths) inflates false-positive risk if every number is
treated as a confirmatory test.

## Conservative correction on the CONFIRMATORY headlines
Estimate the full implicit test count generously at N≈300. Bonferroni threshold
at family-wise α=0.05 is 0.05/300 = 1.7e-4.
- Behavioral SES gaps (all 3 models, PHQ/GAD): permutation p at floor 5e-5;
  powered d=2.34-3.75 with bootstrap CIs excluding 0 by wide margins. SURVIVE
  Bonferroni over N=300 (5e-5 < 1.7e-4), and the powered CIs are not permutation-
  floor-limited (gap CIs e.g. Qwen 7.77±0.32) — unassailable.
- Injection effect (pilot16, echo-immune): bootstrap CI +2.65 [+1.70,+3.60]
  excludes 0; random control includes 0. SURVIVES.
- Gemma/Qwen ablation & patch mediation: p=5e-5, d=3.08-4.12. SURVIVE.
- Operator-matched recon PMF 0.35 & raw PMF 0.79: effect sizes, CI-backed by the
  20-profile paired design. SURVIVE.

## Findings explicitly labeled EXPLORATORY (not multiplicity-protected)
These are hypothesis-GENERATING and must be framed as such in the paper:
- Per-item stereotype content (somatic vs affective) — descriptive.
- Exact transport-band boundaries / L18 saturation point — descriptive geometry,
  not a tested hypothesis; the CONFIRMATORY claim is only "narrow early band,
  Qwen suffix-patchable vs Gemma not," which is a large paired effect.
- Cross-family topology dissociation — supported by provenance control but
  n=1 model/family; framed as model-level demonstration, not population claim.
- AUDIT-C sign reversal — descriptive; 3-model pattern, not a powered test.

## Confirmatory vs exploratory split (write this into Methods)
CONFIRMATORY (pre-registered in script docstrings, gated, permutation-tested,
multiplicity-surviving): the behavioral SES bias and its dose-response; the
causal sufficiency (injection) and necessity/ablation effects with controls;
the raw-vs-recon patch dissociation. EXPLORATORY (report honestly as such):
stereotype content, transport geometry details, cross-family mechanism diversity,
instrument-specific reversals. This split is the honest answer to forking paths:
the load-bearing claims survive aggressive correction; the color is labeled color.
