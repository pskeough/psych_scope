"""Gemma E3 — protocol-matched sufficiency/necessity clamp on Gemma-3-12B-it.

Mirrors pilot3_clamp.py at matched depth fractions: Qwen L27/L33 of 36 ->
Gemma L36/L44 of 48 (clamp), multi-layer arm L28/L32/L36/L40/L44.
Targets: top-k cue_mean SES-differential features from runs/gemma_pilot2.
Conditions identical to Pilot 3 (baseline/clamp10/20/50/clamp50x5/rand10 on Low;
baseline/reverse10/rand10 on High). Greedy, clinical/v0, 20 profiles.
Cross-family question: does feature injection cause depression scoring in a
second family, and does the necessity failure replicate?

python -u gemma_clamp.py --smoke
python -u gemma_clamp.py
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
CLAMP_LAYERS = [36, 44]
MULTI_LAYERS = [28, 32, 36, 40, 44]
SEED = 20260726
RUN_G2 = Path(__file__).parent.parent / "runs" / "gemma_pilot2"

RACES = ["White", "Black", "Hispanic", "Asian", "Multiracial"]
GENDERS = ["Cisgender Man", "Cisgender Woman"]
RELATIONSHIPS = ["Partnered", "Single"]
SES_LEVELS = ["Low", "Middle", "High"]
SES_CLIN = {"Low": "<$35k, Medicaid", "Middle": "$50k-$100k, Private Insurance",
            "High": ">$150k, Private Insurance"}
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
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        s = json.loads(m.group(0)).get("phq8")
        if isinstance(s, list) and len(s) == 8 and all(isinstance(x, int) and 0 <= x <= 3 for x in s):
            return s
    except json.JSONDecodeError:
        pass
    # fallback (clamp50x5 corrupts the KEY token, not the scores): first bare
    # array of exactly 8 in-range ints anywhere in the text.
    for arr in re.findall(r"\[([0-9,\s]+)\]", text):
        try:
            s = [int(x) for x in arr.split(",")]
        except ValueError:
            continue
        if len(s) == 8 and all(0 <= x <= 3 for x in s):
            return s
    return None


class JumpReluSAE:
    def __init__(self, layer, device):
        d = Path(snapshot_download(
            SCOPE, allow_patterns=[f"resid_post_all/layer_{layer}_width_16k_l0_big/*"]))
        sd = load_file(str(d / "resid_post_all" / f"layer_{layer}_width_16k_l0_big" / "params.safetensors"))
        def pick(*names):
            for n in names:
                if n in sd:
                    return sd[n].to(device=device, dtype=torch.float32)
            raise KeyError(f"none of {names}")
        self.W_enc = pick("W_enc", "w_enc")     # (d, F) after orient
        self.b_enc = pick("b_enc")
        self.W_dec = pick("W_dec", "w_dec")
        self.b_dec = pick("b_dec")
        thr = pick("threshold", "log_threshold")
        self.threshold = torch.exp(thr) if "log_threshold" in sd else thr
        self.nF = self.b_enc.shape[0]
        if self.W_enc.shape[1] != self.nF:
            self.W_enc = self.W_enc.T
        if self.W_dec.shape[0] != self.nF:
            self.W_dec = self.W_dec.T          # (F, d)


def gemma_condition_rows():
    return [(r, g, rel, ses, fr, pa)
            for r in RACES for g in GENDERS for rel in RELATIONSHIPS
            for ses in SES_LEVELS for fr in ("clinical", "narrative")
            for pa in ("v0", "v1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(__file__).parent.parent / "runs" / ("gemma_clamp_smoke" if args.smoke else "gemma_clamp")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    feats = torch.load(RUN_G2 / "feat_means.pt", weights_only=True)["cue_mean"]
    rows = gemma_condition_rows()
    assert len(rows) == next(iter(feats.values())).shape[0]
    idx_low = [i for i, c in enumerate(rows) if c[3] == "Low"]
    idx_high = [i for i, c in enumerate(rows) if c[3] == "High"]
    plan = {}
    for L in MULTI_LAYERS:
        fm = feats[L].float()
        low_m, high_m = fm[idx_low].mean(0), fm[idx_high].mean(0)
        diff = low_m - high_m
        top50 = torch.topk(diff.abs(), 50).indices.tolist()
        top500 = set(torch.topk(diff.abs(), 500).indices.tolist())
        rand10 = rng.sample([f for f in range(fm.shape[1]) if f not in top500], 10)
        plan[L] = {"top10": top50[:10], "top20": top50[:20], "top50": top50,
                   "rand10": rand10, "low_mean": low_m, "high_mean": high_m,
                   "magnitudes": [abs(float(diff[f])) for f in top50[:10]]}
        print(f"L{L} top10: {top50[:10]}")

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
            break
        except AttributeError:
            continue
    assert dec_layers is not None

    sae_slice = {L: JumpReluSAE(L, "cuda") for L in MULTI_LAYERS}
    active = {"edits": None}

    def make_hook(L):
        def fn(mod, inp, outp):
            if active["edits"] is None or L not in active["edits"]:
                return outp
            h = outp[0] if isinstance(outp, tuple) else outp
            fidx, tgt = active["edits"][L]
            s = sae_slice[L]
            hf = h.float()
            pre = hf @ s.W_enc[:, fidx] + s.b_enc[fidx]
            a = pre * (pre > s.threshold[fidx])
            delta = (tgt.view(1, 1, -1) - a) @ s.W_dec[fidx]
            h2 = (hf + delta).to(h.dtype)
            return (h2,) + tuple(outp[1:]) if isinstance(outp, tuple) else h2
        return fn

    handles = [dec_layers[L].register_forward_hook(make_hook(L)) for L in MULTI_LAYERS]

    def generate(race, gender, rel, ses, edits):
        active["edits"] = edits
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clinical_prompt(race, gender, rel, ses)}]
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True)
        ids = t["input_ids"].to("cuda")
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        active["edits"] = None
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    def edits_for(kind, direction, layer_set=None):
        e = {}
        for L in (layer_set or CLAMP_LAYERS):
            pl = plan[L]
            tgt_mean = pl["high_mean"] if direction == "toHigh" else pl["low_mean"]
            if kind in ("clamp10", "clamp20", "clamp50"):
                fidx = pl["top" + kind[5:]]
                tgt = torch.tensor([float(tgt_mean[f]) for f in fidx])
            else:
                fidx = pl["rand10"]
                sign = 1.0 if direction == "toHigh" else -1.0
                base = torch.tensor([float(pl["low_mean"][f]) for f in fidx])
                tgt = base + sign * torch.tensor(pl["magnitudes"])
            e[L] = (torch.tensor(fidx, device="cuda"), tgt.cuda())
        return e

    profiles = [(r, g, rel) for r in RACES for g in GENDERS for rel in RELATIONSHIPS]
    if args.smoke:
        profiles = profiles[:2]
    LOW_CONDS = [("baseline", None), ("clamp10", edits_for("clamp10", "toHigh")),
                 ("clamp20", edits_for("clamp20", "toHigh")),
                 ("clamp50", edits_for("clamp50", "toHigh")),
                 ("clamp50x5", edits_for("clamp50", "toHigh", MULTI_LAYERS)),
                 ("rand10", edits_for("rand10", "toHigh"))]
    HIGH_CONDS = [("baseline", None), ("reverse10", edits_for("clamp10", "toLow")),
                  ("rand10", edits_for("rand10", "toLow"))]

    results = []
    total = len(profiles) * (len(LOW_CONDS) + len(HIGH_CONDS))
    n = 0
    for prof in profiles:
        for ses, conds in (("Low", LOW_CONDS), ("High", HIGH_CONDS)):
            for cname, edits in conds:
                comp = generate(*prof, ses, edits)
                sc = parse_scores(comp)
                results.append({"race": prof[0], "gender": prof[1], "rel": prof[2],
                                "ses": ses, "cond": cname,
                                "phq8_total": sum(sc) if sc else None,
                                "completion": comp})
                n += 1
                print(f"[{n}/{total}] {prof[0]}/{prof[1][10:]}/{prof[2]}/{ses}/{cname}: "
                      f"{results[-1]['phq8_total']}", flush=True)

    def mean(cond, ses):
        v = [r["phq8_total"] for r in results if r["cond"] == cond and r["ses"] == ses
             and r["phq8_total"] is not None]
        return sum(v) / len(v) if v else None
    summ = {"base_low": mean("baseline", "Low"), "base_high": mean("baseline", "High"),
            "clamp10_low": mean("clamp10", "Low"), "clamp20_low": mean("clamp20", "Low"),
            "clamp50_low": mean("clamp50", "Low"), "clamp50x5_low": mean("clamp50x5", "Low"),
            "rand10_low": mean("rand10", "Low"), "reverse10_high": mean("reverse10", "High"),
            "rand10_high": mean("rand10", "High")}
    gap = (summ["base_low"] - summ["base_high"]) if None not in (summ["base_low"], summ["base_high"]) else None
    if gap:
        for c in ("clamp10", "clamp20", "clamp50", "clamp50x5"):
            if summ[f"{c}_low"] is not None:
                summ[f"BMF_{c}"] = (summ["base_low"] - summ[f"{c}_low"]) / gap
        if summ["reverse10_high"] is not None:
            summ["reverse_shift"] = summ["reverse10_high"] - summ["base_high"]
    summ["parse_rate"] = sum(1 for r in results if r["phq8_total"] is not None) / len(results)

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race", "gender", "rel", "ses", "cond", "phq8_total"])
        for r in results:
            w.writerow([r["race"], r["gender"], r["rel"], r["ses"], r["cond"], r["phq8_total"]])
    with open(out / "completions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"summary": summ, "targets": {str(L): {"top10": plan[L]["top10"]} for L in MULTI_LAYERS}},
        indent=2), encoding="utf-8")
    print("\n== summary ==")
    for k, v in summ.items():
        print(f"  {k}: {v if v is None else round(v, 3)}")


if __name__ == "__main__":
    main()
