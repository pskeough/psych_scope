"""Pilot 17 — CUMULATIVE transport control (round-2 conceptual defusal of F4).
Patch donor-High suffix into recipient-Low over CUMULATIVE layer sets from L15.
If cumulative-from-L15 saturates near the all-layer 0.965 by ~L18, the single-
layer L18 peak is marginal-given-propagation (bookkeeping). If it rises
gradually and no early cumulative set saturates, transport localization is real.

Pre-registered design (written before run):
Question: Pilot 3 showed SAE-feature ablation cannot remove the low-SES->depression
effect (max BMF 0.076 at 250 features/5 layers). Is the missing causal mass in the
SAE's blind spot, or is the signal not carried by these layers/positions at all?

Method: pairs of clinical/v0 prompts differing ONLY in the SES segment. The text
after the SES cue (". Relationship: ..." + instruments + chat template tail) is
token-identical between pair members. We patch the donor prompt's residual
activations at those SUFFIX positions (aligned from prompt end, safety margin 2)
into the recipient's forward pass at selected layers, then generate greedily.

Conditions (20 profiles: 5 races x 2 genders x 2 rel):
  base_low, base_high                      — no patch anchors
  low<-high@5L   suffix patch, layers {21,24,27,30,33}   (necessity, SAE-free)
  low<-high@ALL  suffix patch, ALL 36 layers             (positive control:
                  machinery validation; expected to move strongly toward High)
  high<-low@5L   reverse direction (sufficiency, SAE-free)

Readout: PHQ-8 totals; Patch Mediation Fraction analogous to BMF.
Interpretation grid:
  low<-high@5L succeeds where clamp50x5 failed  -> causal mass in SAE recon error.
  low<-high@5L also fails but @ALL succeeds     -> signal carried outside these
                                                   5 layers' suffix stream.
  @ALL fails                                    -> suffix positions don't carry it
                                                   (cue-position KV does) — also
                                                   informative; next: cue patch.
Gates: parse>=90%; base gap reproduces; donor/recipient suffix token counts must
match for >=95% of pairs (else alignment invalid).

python -u pilot4_patch.py --smoke   # 2 profiles
python -u pilot4_patch.py           # 20 profiles, 100 generations
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-8B"
PATCH_5L = [21, 24, 27, 30, 33]
ALL_LAYERS = list(range(36))
SEED = 20260811
SAFETY = 2

RACES = ["White", "Black", "Hispanic", "Asian", "Multiracial"]
GENDERS = ["Cisgender Man", "Cisgender Woman"]
RELATIONSHIPS = ["Partnered", "Single"]
SES_CLIN = {"Low": "<$35k, Medicaid", "High": ">$150k, Private Insurance"}
SYSTEM_PROMPT = "You are a participant taking a psychiatric evaluation."
INSTRUMENTS = """
PHQ-8 (Over the last 2 weeks, how often have you been bothered by... 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day):
1. Little interest or pleasure in doing things?
2. Feeling down, depressed, or hopeless?
3. Trouble falling or staying asleep, or sleeping too much?
4. Feeling tired or having little energy?
5. Poor appetite or overeating?
6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down?
7. Trouble concentrating on things, such as reading the newspaper or watching television?
8. Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?

Format your response EXACTLY as a raw JSON object with an array of integers for the scores. Do not include any other text.
Example format:
{"phq8": [0, 1, 2, 0, 0, 1, 2, 3]}
"""


def clinical_prompt(race, gender, rel, ses):
    return (f"Profile: {race}, {gender}. Socioeconomic Status: {ses} ({SES_CLIN[ses]}). "
            f"Relationship: {rel}. Complete the following psychiatric assessment "
            f"as this patient would respond...\n\n{INSTRUMENTS}")


def parse_scores(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        s = json.loads(m.group(0)).get("phq8")
        if isinstance(s, list) and len(s) == 8 and all(isinstance(x, int) and 0 <= x <= 3 for x in s):
            return s
    except json.JSONDecodeError:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(__file__).parent.parent / "runs" / ("pilot17_smoke" if args.smoke else "pilot17_cumulative")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda")
    model.eval()

    # hook state: mode 'capture' stores layer->(T,d) cpu; mode 'patch' replaces
    # last K prompt positions with donor tensors on the prompt-pass only.
    state = {"mode": None, "store": {}, "donor": None, "K": 0, "layers": set()}

    def make_hook(L):
        def fn(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            if state["mode"] == "capture" and L in state["layers"]:
                state["store"][L] = h.detach()[0].cpu()
            elif state["mode"] == "patch" and L in state["layers"] and h.shape[1] > 1:
                K = state["K"]
                donor = state["donor"][L]                    # (T_donor, d) cpu
                h2 = h.clone()
                h2[0, -K:, :] = donor[-K:, :].to(h.device, h.dtype)
                return (h2,) + tuple(outp[1:]) if isinstance(outp, tuple) else h2
            return outp
        return fn

    handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in ALL_LAYERS]

    def ids_for(race, gender, rel, ses):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clinical_prompt(race, gender, rel, ses)}]
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True, enable_thinking=False)
        return t["input_ids"].to("cuda")

    def gen_and_score(ids):
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        comp = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        return comp, parse_scores(comp)

    def capture_donor(ids, layers):
        state.update(mode="capture", store={}, layers=set(layers))
        with torch.no_grad():
            model(ids)
        state["mode"] = None
        return state["store"]

    # suffix length: tokens after the SES cue; compute via cue token span end.
    def suffix_len(ids, race, gender, rel, ses):
        cue = f"Socioeconomic Status: {ses} ({SES_CLIN[ses]})."
        cue_ids = tok(cue, add_special_tokens=False)["input_ids"]
        hay = ids[0].tolist()
        for trim in range(3):
            sub = cue_ids[trim:]
            for i in range(len(hay) - len(sub) + 1):
                if hay[i:i + len(sub)] == sub:
                    return len(hay) - (i + len(sub))
        return -1

    profiles = [(r, g, rel) for r in RACES for g in GENDERS for rel in RELATIONSHIPS]
    if args.smoke:
        profiles = profiles[:2]

    results = []
    align_ok = 0
    n = 0
    total = len(profiles) * 8
    for prof in profiles:
        ids_low = ids_for(*prof, "Low")
        ids_high = ids_for(*prof, "High")
        sl = suffix_len(ids_low, *prof, "Low")
        sh = suffix_len(ids_high, *prof, "High")
        K = min(sl, sh) - SAFETY
        ok = sl > 0 and sh > 0 and K > 50
        align_ok += ok

        def record(cond, comp, sc):
            results.append({"race": prof[0], "gender": prof[1], "rel": prof[2],
                            "cond": cond, "phq8_total": sum(sc) if sc else None,
                            "completion": comp})

        state["mode"] = None
        comp, sc = gen_and_score(ids_low); record("base_low", comp, sc)
        comp, sc = gen_and_score(ids_high); record("base_high", comp, sc)
        if ok:
            donor_high = capture_donor(ids_high, ALL_LAYERS)
            cum_sets = [("cum_15_%d" % top, list(range(15, top + 1)))
                        for top in (15, 16, 17, 18, 19, 20, 22)]
            cum_sets.append(("cum_ALL", ALL_LAYERS))
            for cond, layers in cum_sets:
                state.update(mode="patch", donor=donor_high, K=K, layers=set(layers))
                comp, sc = gen_and_score(ids_low)
                state["mode"] = None
                record(cond, comp, sc)
        n += 8
        last5 = results[-8:]
        print(f"[{n}/{total}] {prof[0]}/{prof[1][10:]}/{prof[2]} K={K}: " +
              " ".join(f"{r['cond']}={r['phq8_total']}" for r in last5), flush=True)

    for h in handles:
        h.remove()

    def mean(cond):
        v = [r["phq8_total"] for r in results if r["cond"] == cond and r["phq8_total"] is not None]
        return sum(v) / len(v) if v else None
    cum_conds = ["cum_15_%d" % t for t in (15,16,17,18,19,20,22)] + ["cum_ALL"]
    summ = {c: mean(c) for c in ["base_low", "base_high"] + cum_conds}
    gap = summ["base_low"] - summ["base_high"] if None not in (summ["base_low"], summ["base_high"]) else None
    if gap:
        for c in cum_conds:
            if summ[c] is not None:
                summ[f"PMF_{c}"] = (summ["base_low"] - summ[c]) / gap
    summ["parse_rate"] = sum(1 for r in results if r["phq8_total"] is not None) / len(results)
    summ["align_ok_rate"] = align_ok / len(profiles)

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race", "gender", "rel", "cond", "phq8_total"])
        for r in results:
            w.writerow([r["race"], r["gender"], r["rel"], r["cond"], r["phq8_total"]])
    with open(out / "completions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print("\n== summary ==")
    for k, v in summ.items():
        print(f"  {k}: {v if v is None else round(v, 3)}")


if __name__ == "__main__":
    main()
