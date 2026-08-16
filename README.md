# PsychScope: Same Bias, Different Mechanisms

*A counterfactual audit of socioeconomic-cue bias in LLM mental-health assessment, and why interpretability cannot replace it.*

This repository holds the code, data, and paper for a counterfactual audit of socioeconomic-cue bias in large language model mental-health assessment. We ask two questions. First, do open-weight models score the same patient as more depressed when the only change is a poorer income and insurance cue? Second, can sparse-autoencoder interpretability explain or repair that bias from the inside? The first answer is yes, sharply and across three model families. The second answer is no, and the reason it is no is the paper's contribution.

The behavioral finding is the spine. We built clinical-style vignettes in which the symptom presentation is held fixed and only the socioeconomic cue changes, so any score difference is an individual-fairness violation that needs no ground-truth severity. All three models we tested, Qwen3-8B, Gemma-3-12B-it, and Llama-3.1-8B-Instruct, raise assessed PHQ-8 and GAD-7 severity for the poorer patient, monotonically across three cue levels, with profile-clustered bootstrap intervals that clear zero by a wide margin. The gradient survives deleting the literal socioeconomic label, changing the framing, and sampled decoding.

The mechanistic finding is the twist. Cue-derived dictionary features do move scores when injected, but the mechanism behind the shared behavior is not shared. One model is injectable and not ablatable, another is the reverse, the stereotype content contradicts across models, and operator-matched patching shows that any single dictionary carries only about a third of the causal signal that the raw residual stream carries. A fairness auditor reading each model's internals would have reached three different conclusions about one identical harm. That is the argument for keeping behavioral counterfactual auditing as the load-bearing instrument, and it is why we propose a cue-swap invariance test as a concrete pre-deployment gate.

Every headline number in the paper was re-derived from the raw per-row files by an independent check, and the campaign ran behind a gate chain of determinism, parse-rate, reconstruction-fidelity, and permutation-null tests. One artifact was caught and corrected during that process, a format-echo effect under feature injection, and the correction is reported in the paper rather than hidden.

## Headline numbers

| Model | PHQ-8 gap (Low - High) | 95% CI | Sampled Cohen's d |
|---|---|---|---|
| Qwen3-8B | 7.77 | [7.40, 8.14] | 3.75 |
| Llama-3.1-8B | 7.09 | [6.60, 7.58] | 2.34 |
| Gemma-3-12B | 4.03 | [3.69, 4.37] | 2.66 |

Powered tier: n = 320 per level, temperature 1.0, 80 paired demographic profiles.
Causal (exploratory) tier: feature injection +2.65 [1.70, 3.60]; SAE dictionary carries ~35% of the causal signal vs ~79% for the raw stream.

## Layout

```
paper/        psychscope.tex, psychscope.pdf, figures/
pipeline/     the scripts that produced every result (pilots + analysis)
results/      compact processed data (per-run summaries, behavioral CSVs),
              stats reports, figures
```

## Reproduce

1. Python 3.14, PyTorch with CUDA, `transformers`, `bitsandbytes`, `huggingface_hub`, `scikit-learn`, `matplotlib`.
2. Accept the model and SAE licenses on Hugging Face (Gemma is gated).
3. Behavioral grids: `python pipeline/pilot15_nscale.py` (Qwen), `llama_pilot2.py`, `gemma_pilot2.py`.
4. Causal tier: `pilot3_clamp.py` (injection/ablation), `pilot4_patch.py` and `pilot13b_operator.py` (patching + operator control), `pilot19_p1b.py` (specificity), `pilot16_rotated.py` (echo-immune injection).
5. Figures and stats: `make_figures.py`, `analysis_itt.py`, `analysis_powered.py`.

Each script carries its pre-registration in the module docstring, hardcodes its seed, and smoke-tests before the full run.

## Citation

```
@misc{keough2026psychscope,
  title  = {Same Bias, Different Mechanisms: A Counterfactual Audit of
            Socioeconomic-Cue Bias in LLM Mental-Health Assessment,
            and Why Interpretability Cannot Replace It},
  author = {Keough, Patrick},
  year   = {2026}
}
```

## Limitations

The socioeconomic cue bundles income and insurance and is not a validated multidimensional construct. We test one checkpoint per family, so mechanistic claims are model-level, not family-general. The causal tier is greedy single-draw with several identification controls left for future work, and inference is quantized, so effect magnitude, unlike direction, depends on quantization. The demonstrated harm is a counterfactual-invariance violation in simulated self-report, not a measured clinical outcome, and every deployment implication is conditional.

## License

Code: MIT ([LICENSE](LICENSE)) · Paper text, figures and derived data: CC BY-NC-ND 4.0 ([LICENSE-DATA](LICENSE-DATA))
