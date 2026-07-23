# PsychScope — Findings Report with Proof Chains
*Prepared 2026-07-23, corrections propagated 2026-07-24. Headline numbers were
independently re-derived from raw CSVs at verification time (22/22); critic-round-2
corrections supersede specific numbers where noted (injection, Findings 4 & 5).
Confirmatory (behavioral, CI-backed causal) vs EXPLORATORY (single-model mechanism)
tiers are labeled. Proof pointers: runs/<dir>/{results,behavioral}.csv.*

## Finding 1 — Three models pathologize poverty, dose-responsively, on every
## internalizing instrument tested
| Model | PHQ-8 (L/M/H) | GAD-7 (L/M/H) | paired d | p |
|---|---|---|---|---|
| Qwen3-8B | 9.38 / 4.98 / 1.19 | 6.55 / 4.25 / 1.85 | +4.15 | 5e-05 |
| Gemma-3-12B-it | 15.14 / 11.49 / 10.84 | 13.20 / 10.95 / 9.05 | +2.17 | 5e-05 |
| Llama-3.1-8B-it | 13.90 / 5.56 / 3.66 | 15.15 / 10.25 / 9.90 | +3.92 | 5e-05 |
- Monotonic (weak) in 75-80/80 paired cells per model; robust to framing,
  label-free paraphrase (echo-corrected gaps: v0 +7.95, v1 +9.10 — the bias
  does NOT depend on the literal word "Low"), and sampling (temp 1.0:
  11.19/7.82/2.80, sd<=1.8, n=100/arm, d=6.31 with real variance).
- SES bias stacks SIMILARLY (not identically) across the race and gender levels
  tested (gaps 8.0-8.7; only Black/White and M/F were run).
- Direction is robust to quantization; absolute gap MAGNITUDE is quantization-
  dependent (pilot14: 4-bit gap 5.1 vs 8-bit 8.6).
- Powered/CI'd values (n=320/level, real variance): see powered_results.md.
- Proof: pilot2_qwen8b, gemma_pilot2, llama_pilot2, pilot15_*, pilot7*/8*.

## Finding 2 — The stereotype CONTENT differs by model
*(EXPLORATORY / descriptive — hypothesis-generating, not a powered/multiplicity-
protected test; scope: the three models tested.)*
- Qwen's poor patient is SOMATIC: psychomotor +1.49, sleep +1.34, concentration
  +1.23; anhedonia/appetite smallest (+0.51). Gemma's is AFFECTIVE:
  depressed_mood +1.01, appetite +0.84; psychomotor -0.02 (nil).
- Alcohol contradiction: Qwen assigns drinking to WEALTH (social pattern:
  freq/qty up, binge 0.00 for all); Gemma & Llama assign it to POVERTY
  (Gemma's binge item steepest, 1.00 vs 0.05). Same "biased" audit verdict,
  opposite substance stereotypes: cross-model consensus is instrument-dependent.
- Proof: analysis_notes_2026-07-22.md §1; analysis_gemma_items/auditc outputs.

## Finding 3 — Causal sufficiency of dictionary features (Qwen, echo-immune)
NOTE: these effects use DIFFERENT operators (feature-ADD at all positions vs
whole-residual OVERWRITE on 5 suffix layers) on DIFFERENT baselines — do NOT read
them as an additive "tiers of causal mass" partition (they are non-nested).
- **Feature injection (top-10 cue-derived): +2.65, bootstrap CI [+1.70, +3.60]**,
  echo-immune (pilot16 rotated example arrays, 0/20 echo), random control null
  [-0.20,+0.60]. Corroborated by narrative-frame +1.8 (zero echo) and held-out
  targets +1.3. SUPERSEDES the artifact-inflated +5.5 and collider-biased +1.45.
- Feature injection ALSO triggers a Qwen-specific format-echo attractor in the
  standard-example prompts (16/20 in clinical/v0) — a direction-specific
  disruption random features never cause (A11: delta norms matched 1.01-1.16).
- Raw-stream suffix patch transfers more (+3.2 echo-excluded, cos 0.921 to the
  natural item signature) — see Finding 5 for the operator-matched comparison.
- Necessity is bounded: ablating up to 250 features across 5 layers removes only
  ~8% of the gap (p=0.0005 — small but real).
- Proof: pilot16, pilot12_robustness, pilot3_clamp, pilot4_patch, analysis_itt.

## Finding 4 — Transport localization: the poverty signal enters in a narrow
## early band (L15-18), saturating by L18 (softened per critic round 2)
CUMULATIVE-from-L15 patch PMF (pilot17): L15 0.04, L15-16 0.38, L15-17 0.49,
**L15-18 0.88**, L15-19 0.92, L15-20 0.94, L15-22 0.965 = all-layer control.
So >90% of transferable causal mass is present by L18; all layers beyond L18
add ~0.081 combined. Single-layer sweep (pilot10*) peaks at L18 (0.79) and decays
to ~0 by L23. HONEST FRAMING (concedes the round-2 conceptual attack): NOT "one
transit layer" (that over-reifies argmax of a presence x leverage curve), but a
NARROW EARLY TRANSIT BAND that is essentially complete by L18; deeper layers
read rather than re-transport. Proof: pilot17_cumulative, pilot10_transport,
pilot10b_bracket, pilot10c_onset.

## Finding 5 — The SAE dictionary captures only ~1/3 of the causal signal
## (CORRECTED via operator control, pilot13b — critic round 2, M1/C2)
The naive pilot13 result (recon-patch PMF -5%, "dictionary fully blind") was
OPERATOR-CONFOUNDED, as the critic predicted. The operator control (pilot13b:
patch the recipient's OWN reconstruction back in — same encode-decode operator,
zero donor information) shows the recon operator is NOT neutral: it shifts the
Low prompt 11.0 -> 14.5 (+3.5, upward disruption). Operator-MATCHED reanalysis:
- self-recon (operator baseline, own recon): 14.5
- donor-High recon-patch: 11.45  ->  swapping own-recon for High-recon moves
  the score DOWN 3.05, i.e. the SAE reconstruction DOES carry High-SES info.
- Operator-matched recon PMF = (14.5 - 11.45)/8.6 = **0.35**.
- Raw-activation patch (clean operator; self-splice = identity): PMF **0.79**.
CORRECTED CLAIM: the SAE dictionary transfers ~35% of the causal bias; the raw
residual stream transfers ~79%; therefore **~44 percentage points of the
removable causal mass lives outside the dictionary basis** (not ~80%). A real,
quantified SAE-adequacy gap — but "the dictionary is blind" is FALSE and is
withdrawn. Secondary methods finding: SAE-reconstruction patching carries a
+3.5 operator artifact that any recon-patch study must control for.
SPECIFICITY CONTROL (pilot19/P1b — refutes the "generic corruption" alternative):
the SAME L18 recon operator transfers a NON-SES valence contrast (struggling vs
great) at **79%** (recipient 0 -> 14.35, donor 18.15), while it transfers SES at
only ~35%. A generically-corrupting operator could not carry any contrast at 79%;
it carries valence fine. Therefore the low SES-recon transfer is SES-SPECIFIC
dictionary under-representation, not operator failure. (Caveat: a valence
self-recon control would fully isolate the operator's directional push — the
valence donor is high-PHQ so the +3.5 operator artifact is transfer-aligned here;
that control is the one clean remaining upgrade, but the 79-vs-35 gap and the
refutation of generic corruption hold regardless.) Corroboration: dictionary
features carry the somatic stereotype component (signature cos 0.62 vs stream's
0.92). Proof: pilot13_reconpatch, pilot13b_operator, pilot19_p1b, pilot2 E1.

## Finding 6 — Causal handles differ by model+toolchain (scoped claim)
Gemma(+native IT-SAEs): ablation dose-responsive (BMF 0.31/0.41/0.46), injection
inert (+0.1, p=0.69); suffix patching zero (PMF -0.008) with perfect positive
control (0.967) -> CONSISTENT WITH cue-source reading but NOT uniquely identified
(the all-layer suffix patch DOES transfer 0.967, so the 5-layer zero is layer-
selection over 2 of >=4 possible routes, not proof of cue-only reading; Gemma's
transit band was never sweep-localized). Qwen(+base-SAEs): injection-active/
ablation-resistant. Llama(+low-fidelity SAEs): interventions collapse to noise
(+4.4 rise, d=3.08) -> POST-HOC validity criterion classifies arm as tooling-
limited (see caveat, Methodology §2).
PROVENANCE CONTROL (defuses the provenance HALF of red-team F1): Gemma ablation
with PT (base) SAEs preserves ablatability (clamp50 BMF 0.595 vs 0.46 IT, rand
-0.008) — so the dissociation is not a base-vs-IT-training artifact. BUT width
(16K vs 64K) and architecture (JumpReLU vs TopK) remain UNCONTROLLED and are
infeasible here (58GB/layer); a 64K dictionary spreading SES over 4x more
features predicts ablatable-vs-resistant on its own. SCOPE: provenance-controlled,
model+toolchain demonstration, n=1 model/family — EXPLORATORY, not a mechanism
proof. Proof: gemma_clamp, gemma_patch, llama_clamp, gemma_provenance.

## Finding 7 — Fidelity/usability ordering (EXPLORATORY, illustrative not dose-response)
Across the THREE models tested, higher recon fidelity co-occurred with cleaner
causal tooling: IT-trained SAEs (0.94-0.99) clean; base-transferred (0.89-0.97)
partial + artifact-prone (echo); weak-transfer (0.31-0.87) collapses to noise.
CAVEAT: fidelity is fully CONFOUNDED with model identity here (n=3), so this is
an illustrative ordering, NOT a dose-response or meta-finding. The operator-matched
recon result (Finding 5) shows the dictionary captures ~35% (it does not simply
"fail"). Proof: fidelity ledger (METHODOLOGY_HANDBOOK §2) + Findings 5-6.

## Critic-round-2 corrections (2026-07-24)
- RETRACTED: "switch-like/categorical dose-response" (pilot6/9) — every dose arm
  was echo-dominated; those runs make no valid mechanism claim.
- Feature-injection authentic effect, CLEAN estimate (pilot16, rotated example
  arrays defeat the echo attractor — 0/20 echo in all arms): reverse10 +2.65
  over baseline (5.75 vs 3.10), random control null (3.25 ~ 3.10). This is the
  defensible injection sufficiency number (~38% of the gap): real, specific, and
  echo-immune. Supersedes both the artifact-inflated +5.5 and the collider-
  biased echo-excluded +1.45. Corroborated by narrative-frame arm (pilot12 R1
  +1.8, zero echo) and held-out targets (pilot12 R3 +1.3).
- Transport "L18 transit layer" language softened to "peak of the patchability-
  by-depth curve" pending presence-vs-leverage decomposition (queued P5);
  the L15-18 cumulative 0.884 ≈ all-layer 0.965 point is noted.
- Recon-patch specificity control (self-recon + non-SES contrast) queued as P1
  before Finding 5 is called decisive in the paper.

## Self-corrections (report them — they are strengths)
- Example-echo artifact: discovered by our own verification chain; feature-
  injection sufficiency corrected from the artifact-inflated +5.5 to the
  echo-immune +2.65 [1.70,3.60] (pilot16 rotated examples); affected cells
  recomputed (synthesis header, analysis_itt_report.md).
- Response-contamination (Pilot 1): caught by redesign; contamination overlap
  quantified at 0.00-0.14.
- Pilot-2 methodology bugs (pooled-encoding, encoder mismatch) caught by
  fidelity gates before any results existed.

## What is NOT claimed
No claim of family-generality (one checkpoint per family). No claim that SAE
features suffice to reproduce authentic bias (withdrawn). No cross-family
mechanism universality — the mechanistic claims are per-model, and their
DIVERSITY is the point: behavioral audits cannot be replaced by single-model
interpretability.
