# Analysis suite report (A-series)

## A2 Monotonicity: fraction of paired cells with Low>Mid>High strictly / weakly
- Qwen PHQ: strict 61/80, weak 80/80
- Gemma PHQ: strict 32/80, weak 75/80
- Llama PHQ: strict 37/80, weak 79/80

## A3 Sampled-regime effect sizes (pilot5, temp 1.0)
- Low 11.19 (sd 0.58) vs High 2.80 (sd 1.80): Cohen's d = 6.31 (n=100/arm)
- FRAMING NOTE for paper: deterministic-regime paired d values (2-4+) reflect near-zero within-cell variance; the sampled-regime d above is the human-comparable one.

## A7 Consolidated BH-FDR ledger
| test | p | BH threshold (q=0.05) | significant |
|---|---|---|---|
| Qwen grid SES gap | 5e-05 | 0.004167 | YES |
| Gemma grid SES gap | 5e-05 | 0.008333 | YES |
| Llama grid SES gap | 5e-05 | 0.0125 | YES |
| Qwen injection | 5e-05 | 0.01667 | YES |
| Qwen patch5L | 5e-05 | 0.02083 | YES |
| Gemma ablate50 | 5e-05 | 0.025 | YES |
| Llama noise-inject | 5e-05 | 0.02917 | YES |
| Qwen ablate50x5 | 0.0005 | 0.03333 | YES |
| Llama inject | 0.5028 | 0.0375 | no |
| Qwen rand ctrl | 0.5031 | 0.04167 | no |
| Gemma inject | 0.6916 | 0.04583 | no |
| Gemma patch5L | 1 | 0.05 | no |

## A9 Completion audit (refusals / anomalies across all completions)
- 2506 completions scanned: refusal-language 0, empty 0, all-zero PHQ 49

## A10 Injection effect by demographic slice (Qwen pilot3 reverse10 - baseline, High)
- by race: Asian +5.8(n=4), Black +5.0(n=4), Hispanic +6.0(n=4), Multiracial +5.5(n=4), White +5.2(n=4)
- by gender: Cisgender Man +4.2(n=10), Cisgender Woman +6.8(n=10)

## A11: residual-space delta-norm comparison, real targets vs rand10 control
- L27: ||real_delta|| = 13.36, ||rand_delta|| = 11.48 (ratio 1.16); decoder-row norms real 1.00-1.00, rand 1.00-1.00
- L33: ||real_delta|| = 16.21, ||rand_delta|| = 15.97 (ratio 1.01); decoder-row norms real 1.00-1.00, rand 1.00-1.00
Interpretation: ratio ~1 => energy-matched control; real-target effect (+5.5) vs rand null (+/-0.1) cannot be a generic-perturbation artifact.


## ECHO-CORRECTED recomputation (excluding example-echo completions)
- pilot3_clamp/baseline: all-mean 2.40 (n=20), echo 1, NO-ECHO mean 2.05 (n=19)
- pilot3_clamp/reverse10: all-mean 7.90 (n=20), echo 16, NO-ECHO mean 3.50 (n=4)
- pilot4_patch/base_high: all-mean 2.40 (n=20), echo 1, NO-ECHO mean 2.05 (n=19)
- pilot4_patch/high_from_low_5L: all-mean 6.00 (n=20), echo 4, NO-ECHO mean 5.25 (n=16)
- pilot2 Qwen v0 echo-excluded SES gap: +7.95 (Low n=37, High n=39)
- pilot2 Qwen v1 echo-excluded SES gap: +9.10 (Low n=25, High n=40)

---
## SUPERSEDED-CLAIMS NOTE (2026-07-24)
The A10 per-slice injection numbers (+4.2 to +6.8) and the A11 "+5.5 real-target
effect" above are ECHO-ARTIFACT-CONFOUNDED (16/20 injected outputs were format
echoes). The clean, echo-immune injection effect is +2.65 [1.70,3.60] (pilot16,
rotated examples; see analysis_itt_report.md and FINDINGS_REPORT Finding 3). A11's
norm-matching conclusion (effect is direction-specific, not generic perturbation)
still holds; only the magnitude is superseded.
