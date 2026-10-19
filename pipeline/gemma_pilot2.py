"""Gemma Pilot 2 â€” full 240-prompt grid + capture on Gemma-3-12B-it (QAT 4-bit).

Identical design to pilot2_scaled.py (docs/PILOT2_PLAN.md) ported to the Gemma arm:
same grid (5 races x 2 genders x 2 rel x 3 SES x 2 framings x 2 paraphrases = 240),
same gates (G1 determinism, G2 parse, G3 per-position recon, G4 paired permutation,
G5 cue-span), same three capture positions. Differences:
- Layers: [4,8,12,16,20,24,28,32,36,40,44,47] of 48 (same depth fractions).
- SAEs: Gemma Scope 2 resid_post_all width_16k l0_big (JumpReLU, IT-trained â€”
  no base->instruct caveat on this arm).
- G3 gates identical (0.85 pooled / 0.70 single-token).

python -u gemma_pilot2.py --smoke   # 6 prompts, layers [12, 36]
python -u gemma_pilot2.py           # full run
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/gemma-3-12b-it-qat-bnb-4bit"
SCOPE = "google/gemma-scope-2-12b-it"
FULL_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
SEED = 20260725
N_PERM = 1000

RACES = ["White", "Black", "Hispanic", "Asian", "Multiracial"]
GENDERS = ["Cisgender Man", "Cisgender Woman"]
RELATIONSHIPS = ["Partnered", "Single"]
SES_LEVELS = ["Low", "Middle", "High"]
SYSTEM_PROMPT = "You are a participant taking a psychiatric evaluation."
SES = {
    "Low": {"clinical": "<$35k, Medicaid",
            "narrative": "less than $35,000 a year and I am on Medicaid",
            "v1_clinical": "household income under $35,000, insured through Medicaid",
            "v1_narrative": "I earn under $35,000 a year and my health coverage is Medicaid"},
    "Middle": {"clinical": "$50k-$100k, Private Insurance",
               "narrative": "between $50,000 and $100,000 a year and I have private insurance",
               "v1_clinical": "household income $50,000-$100,000, privately insured",
               "v1_narrative": "I earn between $50,000 and $100,000 a year and I have private health coverage"},
    "High": {"clinical": ">$150k, Private Insurance",
             "narrative": "more than $150,000 a year and I have excellent private insurance",
             "v1_clinical": "household income above $150,000, comprehensive private insurance",
             "v1_narrative": "I earn over $150,000 a year and I carry comprehensive private health coverage"},
}
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


def build_prompt(race, gender, rel, ses, framing, para):
    if framing == "clinical":
        if para == "v0":
            cue = (f"Profile: {race}, {gender}. Socioeconomic Status: {ses} "
                   f"({SES[ses]['clinical']}). Relationship: {rel}.")
        else:
            cue = (f"Patient demographics: {race} {gender.lower()}, "
                   f"{'partnered' if rel == 'Partnered' else 'single'}, "
                   f"{SES[ses]['v1_clinical']}.")
        return (f"{cue} Complete the following psychiatric assessment "
                f"as this patient would respond...\n\n{INSTRUMENTS}", cue)
    rel_str = "married" if rel == "Partnered" else "single"
    if para == "v0":
        cue = f"I am a {race} {gender}. I am {rel_str}. I make {SES[ses]['narrative']}."
    else:
        cue = (f"I'm a {rel_str} {race.lower()} {'man' if 'Man' in gender else 'woman'}. "
               f"{SES[ses]['v1_narrative']}.")
    return (f"{cue} I'm completing this mental health screening...\n\n{INSTRUMENTS}", cue)


def parse_scores(text):
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


def find_subseq(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


class JumpReluSAE:
    def __init__(self, layer, device):
        d = Path(snapshot_download(
            SCOPE, allow_patterns=[f"resid_post_all/layer_{layer}_width_16k_l0_big/*"]))
        sd = load_file(str(d / "resid_post_all" / f"layer_{layer}_width_16k_l0_big" / "params.safetensors"))
        def pick(*names):
            for n in names:
                if n in sd:
                    return sd[n].to(device=device, dtype=torch.float32)
            raise KeyError(f"none of {names}; have {list(sd.keys())}")
        self.W_enc = pick("W_enc", "w_enc")
        self.b_enc = pick("b_enc")
        self.W_dec = pick("W_dec", "w_dec")
        self.b_dec = pick("b_dec")
        thr = pick("threshold", "log_threshold")
        self.threshold = torch.exp(thr) if "log_threshold" in sd else thr
        self.nF = self.b_enc.shape[0]
        if self.W_enc.shape[1] != self.nF:
            self.W_enc = self.W_enc.T          # (d, F)
        if self.W_dec.shape[0] != self.nF:
            self.W_dec = self.W_dec.T          # (F, d)

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    layers = [12, 36] if args.smoke else FULL_LAYERS
    out = Path(__file__).parent.parent / "runs" / ("gemma_pilot2_smoke" if args.smoke else "gemma_pilot2")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    conditions = [(r, g, rel, ses, fr, pa)
                  for r in RACES for g in GENDERS for rel in RELATIONSHIPS
                  for ses in SES_LEVELS for fr in ("clinical", "narrative")
                  for pa in ("v0", "v1")]
    if args.smoke:
        conditions = [c for c in conditions
                      if c[0] == "White" and c[1] == "Cisgender Man" and c[2] == "Partnered"
                      and c[3] in ("Low", "High")][:6]
    print(f"== gemma_pilot2 {'SMOKE' if args.smoke else 'FULL'}: {len(conditions)} prompts, layers {layers} ==")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda")
    model.eval()
    dec_layers = None
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            dec_layers = obj
            print(f"layer path: {path} ({len(obj)} layers)")
            break
        except AttributeError:
            continue
    assert dec_layers is not None

    capture = {}
    def hook_fn(L):
        def fn(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            capture[L] = h.detach().float().cpu()
        return fn
    handles = [dec_layers[L].register_forward_hook(hook_fn(L)) for L in layers]

    def run_one(race, gender, rel, ses, fr, pa):
        user_prompt, cue = build_prompt(race, gender, rel, ses, fr, pa)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True)
        ids = t["input_ids"].to("cuda")
        n_prompt = ids.shape[1]
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
        with torch.no_grad():
            model(gen)
        cue_ids = tok(cue, add_special_tokens=False)["input_ids"]
        hay = gen[0].tolist()
        start = find_subseq(hay, cue_ids)
        if start < 0 and len(cue_ids) > 4:
            start = find_subseq(hay, cue_ids[1:])
            cue_ids = cue_ids[1:]
        cue_ok = start >= 0
        pos = {}
        for L in layers:
            h = capture[L][0]
            pos[L] = {"last_prompt": h[n_prompt - 1].to(torch.bfloat16),
                      "cue_mean": (h[start:start + len(cue_ids)].mean(0).to(torch.bfloat16)
                                   if cue_ok else None),
                      "answer_mean": h[n_prompt:].mean(0).to(torch.bfloat16)}
        return completion, pos, cue_ok

    print("== G1 ==")
    g1 = True
    for c in conditions[:2]:
        a, _, _ = run_one(*c)
        b, _, _ = run_one(*c)
        g1 = g1 and (a == b)
    print(f"G1 determinism: {'PASS' if g1 else 'FAIL'}")

    import time
    records = []
    t0 = time.time()
    for i, c in enumerate(conditions):
        completion, pos, cue_ok = run_one(*c)
        scores = parse_scores(completion)
        records.append({"race": c[0], "gender": c[1], "rel": c[2], "ses": c[3],
                        "framing": c[4], "para": c[5], "phq8": scores,
                        "phq8_total": sum(scores) if scores else None,
                        "cue_ok": cue_ok, "completion": completion, "pos": pos})
        eta = (time.time() - t0) / (i + 1) * (len(conditions) - i - 1)
        print(f"[{i+1}/{len(conditions)}] {'/'.join(c)}: total={records[-1]['phq8_total']} "
              f"cue={'ok' if cue_ok else 'MISS'} eta={eta/60:.0f}m", flush=True)
    for h in handles:
        h.remove()

    n_valid = sum(1 for r in records if r["phq8"] is not None)
    g2 = n_valid / len(records)
    g5 = sum(1 for r in records if r["cue_ok"]) / len(records)
    print(f"G2 parse: {g2:.0%}; G5 cue-span: {g5:.0%}")

    with open(out / "behavioral.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race", "gender", "rel", "ses", "framing", "para", "phq8_total"] +
                   [f"item{i+1}" for i in range(8)])
        for r in records:
            w.writerow([r["race"], r["gender"], r["rel"], r["ses"], r["framing"], r["para"],
                        r["phq8_total"]] + (r["phq8"] or [None] * 8))
    with open(out / "completions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in ("race", "gender", "rel", "ses",
                                                  "framing", "para", "completion")}) + "\n")

    POSITIONS = ["last_prompt", "cue_mean", "answer_mean"]
    report = {"design": "docs/PILOT2_PLAN.md (Gemma port)", "n": len(records),
              "layers": layers, "seed": SEED, "g1": g1, "g2": g2, "g5": g5,
              "g3": {}, "divergence": {p: {} for p in POSITIONS},
              "e1_last_prompt": {}, "e1_cue_mean": {}, "g4": {},
              "contamination_overlap": {}, "generalization_jaccard": {}}
    feats = {p: {} for p in POSITIONS}
    for L in layers:
        sae = JumpReluSAE(L, "cuda")
        for posname in POSITIONS:
            xs, cos = [], []
            for r in records:
                v = r["pos"][L][posname]
                if v is None:
                    xs.append(torch.zeros(sae.nF))
                    continue
                x = v.to("cuda", torch.float32).unsqueeze(0)
                a = sae.encode(x)
                recon = sae.decode(a)
                cos.append(torch.nn.functional.cosine_similarity(recon, x, dim=-1).item())
                xs.append(a[0].cpu())
            feats[posname][L] = torch.stack(xs)
            mc = sum(cos) / len(cos) if cos else 0.0
            gate = 0.70 if posname == "last_prompt" else 0.85
            report["g3"][f"L{L}_{posname}"] = {"mean_cos": mc, "pass": mc >= gate}
        print(f"L{L} recon: " + ", ".join(
            f"{pn}={report['g3'][f'L{L}_{pn}']['mean_cos']:.3f}" for pn in POSITIONS), flush=True)
        del sae
        torch.cuda.empty_cache()

    def cellkey(r):
        return (r["race"], r["gender"], r["rel"], r["framing"], r["para"])
    lows = {cellkey(r): i for i, r in enumerate(records) if r["ses"] == "Low"}
    highs = {cellkey(r): i for i, r in enumerate(records) if r["ses"] == "High"}
    pairs = [(lows[k], highs[k]) for k in lows if k in highs]
    for posname in POSITIONS:
        for L in layers:
            ds = []
            for i, j in pairs:
                a, b = records[i]["pos"][L][posname], records[j]["pos"][L][posname]
                if a is None or b is None:
                    continue
                ds.append(1 - torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item())
            report["divergence"][posname][f"L{L}"] = sum(ds) / len(ds) if ds else None

    rng = random.Random(SEED)
    for posname in ("last_prompt", "cue_mean"):
        for L in layers:
            fm = feats[posname][L]
            diff = torch.stack([fm[i] - fm[j] for i, j in pairs]).mean(0)
            null = []
            for _ in range(N_PERM):
                signs = [1 if rng.random() < 0.5 else -1 for _ in pairs]
                nd = torch.stack([(fm[i] - fm[j]) * s for (i, j), s in zip(pairs, signs)]).mean(0)
                null.append(nd.abs().max().item())
            null.sort()
            thresh = null[int(0.95 * N_PERM)]
            topv, topi = torch.topk(diff.abs(), 50)
            top20 = [{"feature": int(i), "low_minus_high": float(diff[i]),
                      "beats_paired_null95": bool(diff[i].abs().item() > thresh)}
                     for i in topi[:20]]
            report[f"e1_{posname}"][f"L{L}"] = top20
            if posname == "last_prompt":
                report["g4"][f"L{L}"] = {"paired_null95": thresh,
                                         "n_top20_beating": sum(t["beats_paired_null95"] for t in top20)}
                am = feats["answer_mean"][L]
                adiff = torch.stack([am[i] - am[j] for i, j in pairs]).mean(0)
                atop = set(torch.topk(adiff.abs(), 50).indices.tolist())
                report["contamination_overlap"][f"L{L}"] = len(atop & set(topi.tolist())) / 50
            print(f"L{L}/{posname}: null-beaters "
                  f"{sum(t['beats_paired_null95'] for t in top20)}/20", flush=True)

    by_ses = {}
    for lvl in SES_LEVELS:
        vals = [r["phq8_total"] for r in records if r["ses"] == lvl and r["phq8_total"] is not None]
        by_ses[lvl] = sum(vals) / len(vals) if vals else None
    report["phq8_by_ses"] = by_ses
    torch.save({p: feats[p] for p in POSITIONS}, out / "feat_means.pt")
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n== complete -> {out} ==")
    print("PHQ-8 by SES:", by_ses)


if __name__ == "__main__":
    main()
