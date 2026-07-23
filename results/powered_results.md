# Powered n-scale results (sampled temp 1.0, k=4 x 240 prompts, n=320/SES level)

| Model | Low (sd) | Middle (sd) | High (sd) | d(Low-High) | gap 95% CI |
|---|---|---|---|---|---|
| Qwen3-8B | 9.14 (2.55) | 5.42 (3.22) | 1.37 (1.45) | 3.75 | 7.77 ± 0.32 |
| Llama-3.1-8B | 12.36 (3.24) | 7.28 (3.14) | 5.26 (2.81) | 2.34 | 7.10 ± 0.47 |
| Gemma-3-12B | 14.83 (1.34) | 11.44 (1.64) | 10.79 (1.68) | 2.66 | 4.04 ± 0.24 |

All n=320/level, parse 100%. Monotonic Low>Mid>High in every model.
Deterministic-regime d-inflation critique (round 2) DEFUSED: even with genuine sampling variance (sd 1.4-3.2), d ranges ~3-4 with tight bootstrap-consistent CIs.

## pilot16 injection bootstrap CI (rotated-example, echo-immune, n=20 paired profiles)
- reverse10 - baseline: mean +2.65, 95% bootstrap CI [+1.70, +3.60]
- rand10 - baseline (control): mean +0.15, 95% CI [-0.20, +0.60]
- VERDICT: injection CI EXCLUDES 0 (real, specific effect); control includes 0 (clean).


## Bootstrap CIs on remaining causal headlines
- Gemma ablation (clamp50-baseline, Low, n=20): -2.75 [-3.05,-2.45] EXCLUDES 0
- Qwen raw patch (low_from_high_5L - base_low, n=20): -3.35 [-3.70,-3.00] EXCLUDES 0


## Cluster (profile-level) bootstrap behavioral gaps — CORRECTED (fixes n=320-iid pseudoreplication)
- Qwen: gap 7.77, 95% cluster-bootstrap CI [7.40, 8.14], n=80 paired profiles (sd 1.69); Cohen's d(profile gaps)=4.59
- Llama: gap 7.09, 95% cluster-bootstrap CI [6.60, 7.58], n=80 paired profiles (sd 2.24); Cohen's d(profile gaps)=3.17
- Gemma: gap 4.03, 95% cluster-bootstrap CI [3.69, 4.37], n=80 paired profiles (sd 1.57); Cohen's d(profile gaps)=2.57
