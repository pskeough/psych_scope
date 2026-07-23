# PsychScope Methodology Handbook
*Everything needed to write the Methods (and defend them). 2026-07-23.*

## 1. How the "Scope" SAE suites actually work (write-ready explanation)
A sparse autoencoder (SAE) is trained to reconstruct a model's internal
activations (here: the post-layer residual stream) through a much wider, sparse
bottleneck. Encoder: `pre = x @ W_enc + b_enc`; a sparsity rule keeps only the
strongest latents; decoder: `x̂ = acts @ W_dec + b_dec`. Each latent ("feature")
is a direction in activation space that tends, empirically, to fire on an
interpretable pattern. The dictionaries are trained by the model vendors on
large corpora and released per layer:
- **Qwen-Scope** (Qwen3-8B-Base): 64K features/layer, TopK-style L0≈100.
  Reference encode = relu THEN top-k, NO input normalization, NO b_dec
  subtraction pre-encode (we matched their app.py exactly; getting this wrong
  cost recon 0.43→0.94 — see log 2026-07-21).
- **Gemma Scope 2** (Gemma-3 family): 16K width used here, JumpReLU
  (`acts = pre * (pre > threshold)`), trained on BOTH pretrained and
  instruction-tuned checkpoints — our arm used IT-trained SAEs (exact-checkpoint
  match, the methodological gold standard of the three).
- **Llama Scope** (Llama-3.1-8B-Base): 32K width, TopK k=50, per-layer repos.
Key concept for the paper: an SAE is a *lens*, not the model. Everything seen
"through features" is conditioned on (a) reconstruction fidelity at that layer
and (b) whether the dictionary's basis aligns with the behaviorally relevant
directions. We measured both and they became findings (illustrative fidelity ordering (confounded with model identity, n=3);
basis under-representation at L18 (~35% dictionary capture, operator-matched; ~44pp of removable causal mass outside the basis)).

## 2. Fidelity ledger (gates every feature-level claim)
| Arm | SAE provenance | recon cos (cue positions) | causal-tool validity |
|---|---|---|---|
| Gemma-3-12B-it | IT-trained (native) | 0.94-0.99 | clean (ablation works, dose-responsive) |
| Qwen3-8B | base-trained, transferred | 0.89-0.97 | partial (see artifact correction) |
| Llama-3.1-8B-it | base-trained, transferred | 0.31-0.87 | collapsed (interventions = noise) |
POST-HOC VALIDITY CRITERION (authored 2026-07-23 during bulletproofing, AFTER
Llama's null, in response to red-team F2 — NOT pre-registered; label honestly):
an arm's feature-level causal results are interpretable only if (i) recon cos
>= 0.85 at target-layer cue positions AND (ii) energy-matched random-feature
control is null. Llama fails (i). CAVEAT: because this criterion is post-hoc and
it is what reclassifies Llama's real d=3.08 noise-injection result as "tooling-
limited", the Finding-7 meta-claim resting on it is EXPLORATORY. Pair with the
0.70-0.95 threshold-sensitivity sweep to show the partition doesn't hinge on the
exact 0.85 cut.

## 3. Design skeleton (Methods section order)
1. Battery: PsychBench-derived matched vignettes; PHQ-8/GAD-7/AUDIT-C verbatim;
   demographic grid 5 races x 2 genders x 2 relationships x 3 SES; two framings
   (clinical/narrative) x two paraphrases (v0 with, v1 WITHOUT the "Low/High"
   label) = 240 unique prompts per model.
2. Decoding: greedy, fixed seed, thinking disabled (deterministic regime; each
   prompt = one observation). Sampled-regime replication (temp 1.0, n=100/arm).
3. Capture: teacher-forced pass; three positions (cue-span mean, last prompt
   token, answer mean). PRIMARY = cue-span (pre-response, best fidelity).
   Contamination control: answer-position features vs stimulus features overlap
   0.00-0.14 -> response-echo quantified and excluded.
4. Statistics: paired sign-flip permutation nulls (within-profile), 1k-20k
   shuffles; BH-FDR ledger over all tests; deterministic-regime d reported
   separately from powered sampled d=2.34-3.75 (n=320/level; the pilot5 d=6.31 was a tighter-variance n=100 check).
5. Interventions: (a) feature clamp via decoder-direction residual edit at all
   token positions; (b) raw residual suffix patching (positions aligned from
   prompt end; identical suffix text between pair members); (c) recon-then-patch
   control (pilot13). Controls: energy-matched random features (A11: norm ratio
   1.01-1.16), reverse direction, dose series, all-layer positive control
   (PMF 0.965-0.967 both patchable arms).
6. Verification chain: pre-registered gates (G1 determinism, G2 parse>=90%,
   G3 per-position recon, G4 paired permutation, G5 cue-span location);
   smoke test before every full run; independent re-derivation of all 22
   headline numbers from raw CSVs (22/22 match); completion audit (echo
   artifact discovered and corrected — see synthesis header).

## 4. Known artifacts & honest limitations (paper's limitations section)
- EXAMPLE-ECHO artifact (Qwen only): feature injection collapses output to the
  prompt's format example. Discovered by completion-diversity audit; all
  affected claims corrected/withdrawn (synthesis header). Recommendation for
  future batteries: rotate example arrays per prompt so echo is detectable and
  sums to zero.
- One checkpoint per family: say "model", not "family", for any
  mechanism/content claim (red-team MAJOR). Behavioral direction is 3/3 models.
- Quantization: 8-bit (Qwen), QAT-4bit (Gemma), 8-bit (Llama); recon gates
  passed under quantization; full-precision replication of key cells remains
  cheap insurance (unrun).
- SAE-provenance confound in the cross-model topology contrast (red-team F1):
  ablatability difference Gemma-vs-Qwen co-varies with SAE provenance/width.
  Defusal experiment (Gemma base-SAE arm) queued; until run, frame topology
  contrast as "model+toolchain" difference.
- Gemma clamp50x5 cell: n=11 valid (JSON key corruption under heavy clamping;
  parser fallback recovered scores where arrays survived).
- The battery's SES cue bundles label+income+insurance; v1 paraphrase isolates
  income+insurance (gap survives, +9.10 echo-excluded).

## 5. Reproducibility inventory
- All run scripts in src/ (pilot0-13 + analysis suite), pre-registration in
  docstrings, deviations in RESEARCH_LOG.md with timestamps.
- Every generation's full text in runs/*/completions.jsonl (~3,500 total).
- Feature matrices (SAE activations per prompt/layer/position) in
  runs/*/feat_means.pt. Raw residuals NOT retained (disk) — capture scripts
  regenerate deterministically.
- Figure-ready CSVs in figures/. Verification outputs in docs/workflow_extract.md.
- Seeds: every script hardcodes its seed; greedy decoding makes behavioral runs
  bit-reproducible on same hardware/quantization.

## 6. Terms to define in the paper (glossary the reviewers will hold you to)
BMF (Bias Mediation Fraction), PMF (Patch Mediation Fraction), stimulus-locked
vs response-contaminated capture, transport curve, read-vs-transport routing,
echo attractor, illustrative fidelity ordering (confounded with model identity, n=3), energy-matched control.
