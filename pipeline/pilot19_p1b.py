"""Pilot 19 (P1b) - non-SES specificity control for Finding 5.

Patch a strong NON-SES VALENCE contrast (struggling vs great) through the same
L18 raw and SAE-recon operators used for SES. Valence is a contrast SAEs
represent well (sentiment features are prototypical), so:
- if recon-patch TRANSFERS valence but SES-recon did not, dictionary-invisibility
  is SES-SPECIFIC -> Finding 5 (SES lives outside basis) HOLDS.
- if recon-patch ALSO fails for valence, recon failure is generic encode-decode
  corruption of any delicate read -> Finding 5's SES-specific claim DIES.

recipient = "great" prompt (expect LOW PHQ); donor = "struggling" (expect HIGH).
python -u pilot19_p1b.py --smoke   # 2 profiles
python -u pilot19_p1b.py           # 20 profiles
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-8B"
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"
LAYER = 18
SEED = 20260813
SAFETY = 2
RACES = ["White", "Black", "Hispanic", "Asian", "Multiracial"]
GENDERS = ["Cisgender Man", "Cisgender Woman"]
RELS = ["Partnered", "Single"]
VALENCE = {"struggling": "has been struggling badly, feeling exhausted, low, and hopeless lately",
           "great": "has been feeling wonderful, energetic, motivated, and genuinely content lately"}
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


def make_prompt(race, gender, rel, val):
    cue = f"Profile: {race}, {gender}, {rel}, who {VALENCE[val]}."
    return (cue + " Complete the following psychiatric assessment as this patient "
            "would respond...\n\n" + INSTRUMENTS), cue


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


def load_sae(device):
    p = hf_hub_download(SAE_REPO, f"layer{LAYER}.sae.pt")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    W_enc, b_enc = sd["W_enc"].float(), sd["b_enc"].float()
    W_dec, b_dec = sd["W_dec"].float(), sd["b_dec"].float()
    nF = b_enc.shape[0]
    if W_enc.shape[0] != nF:
        W_enc = W_enc.T
    if W_dec.shape[0] != nF:
        W_dec = W_dec.T
    W_enc, b_enc, W_dec, b_dec = (t.to(device) for t in (W_enc, b_enc, W_dec, b_dec))

    def recon(x):
        pre = x.float() @ W_enc.T + b_enc
        relu_x = torch.relu(pre)
        topv, topi = torch.topk(relu_x, 100, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        return (acts @ W_dec + b_dec).to(x.dtype)
    return recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(__file__).parent.parent / "runs" / ("pilot19_smoke" if args.smoke else "pilot19_p1b")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    dev = "cuda"
    sae_recon = load_sae(dev)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=dev)
    model.eval()

    state = {"mode": None, "cap": None, "donor": None, "K": 0}

    def hook(mod, inp, outp):
        h = outp[0] if isinstance(outp, tuple) else outp
        if state["mode"] == "capture":
            state["cap"] = h.detach()[0].cpu()
        elif state["mode"] == "patch" and h.shape[1] > 1:
            K = state["K"]
            h2 = h.clone()
            h2[0, -K:, :] = state["donor"][-K:, :].to(h.device, h.dtype)
            return (h2,) + tuple(outp[1:]) if isinstance(outp, tuple) else h2
        return outp
    model.model.layers[LAYER].register_forward_hook(hook)

    def ids_for(race, gender, rel, val):
        p, cue = make_prompt(race, gender, rel, val)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True, enable_thinking=False)
        return t["input_ids"].to(dev), cue

    def gen(ids):
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
        return parse_scores(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    def capture(ids):
        state["mode"] = "capture"
        with torch.no_grad():
            model(ids)
        state["mode"] = None
        return state["cap"]

    def suffix_len(ids, cue):
        cue_ids = tok(cue, add_special_tokens=False)["input_ids"]
        hay = ids[0].tolist()
        for trim in range(3):
            sub = cue_ids[trim:]
            for i in range(len(hay) - len(sub) + 1):
                if hay[i:i + len(sub)] == sub:
                    return len(hay) - (i + len(sub))
        return -1

    profiles = [(r, g, rel) for r in RACES for g in GENDERS for rel in RELS]
    if args.smoke:
        profiles = profiles[:2]
    results = []
    for prof in profiles:
        ids_g, cue_g = ids_for(*prof, "great")      # recipient
        ids_s, cue_s = ids_for(*prof, "struggling")  # donor
        sg, ss = suffix_len(ids_g, cue_g), suffix_len(ids_s, cue_s)
        K = min(sg, ss) - SAFETY
        if K < 40:
            continue
        state["mode"] = None
        base_great = gen(ids_g)
        base_strug = gen(ids_s)
        donor = capture(ids_s)
        state.update(mode="patch", donor=donor, K=K)
        raw = gen(ids_g)
        state["mode"] = None
        recon_donor = sae_recon(donor.to(dev)).cpu()
        state.update(mode="patch", donor=recon_donor, K=K)
        rec = gen(ids_g)
        state["mode"] = None
        row = {"prof": "/".join(prof),
               "base_great": sum(base_great) if base_great else None,
               "base_strug": sum(base_strug) if base_strug else None,
               "raw_patch": sum(raw) if raw else None,
               "recon_patch": sum(rec) if rec else None}
        results.append(row)
        print(row, flush=True)

    def mean(k):
        v = [r[k] for r in results if r[k] is not None]
        return sum(v) / len(v) if v else None
    bg, bs = mean("base_great"), mean("base_strug")
    raw, rec = mean("raw_patch"), mean("recon_patch")
    gap = (bs - bg) if None not in (bs, bg) else None
    summ = {"base_great": bg, "base_strug": bs, "raw_patch": raw, "recon_patch": rec,
            "valence_gap": gap}
    if gap:
        summ["raw_transfer_frac"] = (raw - bg) / gap if raw is not None else None
        summ["recon_transfer_frac"] = (rec - bg) / gap if rec is not None else None
    summ["parse_rate"] = sum(1 for r in results if r["recon_patch"] is not None) / max(len(results), 1)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prof", "base_great", "base_strug", "raw_patch", "recon_patch"])
        for r in results:
            w.writerow([r["prof"], r["base_great"], r["base_strug"], r["raw_patch"], r["recon_patch"]])
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n== summary ==", json.dumps(summ, indent=2))
    if gap:
        print("INTERPRETATION: recon_transfer_frac ~ raw_transfer_frac => recon works for "
              "valence => SES-recon failure is SES-SPECIFIC (F5 holds). recon_transfer_frac ~ 0 "
              "=> generic operator corruption (F5 SES-specific claim dies).")


if __name__ == "__main__":
    main()
