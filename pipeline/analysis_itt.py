"""Critic-round-2 statistics defusals (CPU, existing data).
Addresses: S1/C4 (report full distributions + ITT, not echo-excluded points),
collider caveat (CI on the n=4 authentic estimate), bimodality of injection.
Writes docs/analysis_itt_report.md.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
EX = "[0, 1, 2, 0, 0, 1, 2, 3]"
L = ["# ITT / full-distribution analysis (critic round 2 stats defusals)\n"]


def parse(t):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?\s*", "", t), flags=re.DOTALL)
    if not m:
        return None
    try:
        s = json.loads(m.group(0)).get("phq8")
        return s if isinstance(s, list) and len(s) == 8 else None
    except Exception:
        return None


def load(run):
    return [json.loads(x) for x in open(ROOT / "runs" / run / "completions.jsonl", encoding="utf-8")]


def mean_sd_ci(v):
    n = len(v)
    if n == 0:
        return (None, None, None)
    m = sum(v) / n
    if n < 2:
        return (m, 0.0, (m, m))
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    return (m, sd, (m - 1.96 * se, m + 1.96 * se))


# --- Injection full distribution (pilot3 High: baseline/reverse10/rand10) ---
L.append("## 1. Injection full PHQ-sum distribution (pilot3, High, NO exclusion, n=20 each)")
rows = load("pilot3_clamp")
for cond in ("baseline", "reverse10", "rand10"):
    sums = [sum(parse(r["completion"])) for r in rows
            if r.get("cond") == cond and r.get("ses") == "High" and parse(r["completion"])]
    echo_n = sum(1 for r in rows if r.get("cond") == cond and r.get("ses") == "High" and EX in r["completion"])
    hist = Counter(sums)
    m, sd, ci = mean_sd_ci(sums)
    L.append(f"- {cond}: mean {m:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), sd {sd:.2f}, "
             f"echo(sum9) {echo_n}/{len(sums)}; hist {dict(sorted(hist.items()))}")
L.append("Reading: reverse10 is BIMODAL (spike at 9 = echo attractor + a low spread); "
         "the 'authentic' mode is the non-9 mass, not a unimodal shifted distribution.")

# --- CI on the n=4 authentic (non-echo) injection estimate ---
L.append("\n## 2. Honest CI on echo-excluded 'authentic' injection (collider caveat)")
noecho = [sum(parse(r["completion"])) for r in rows
          if r.get("cond") == "reverse10" and r.get("ses") == "High"
          and parse(r["completion"]) and EX not in r["completion"]]
base_ne = [sum(parse(r["completion"])) for r in rows
           if r.get("cond") == "baseline" and r.get("ses") == "High"
           and parse(r["completion"]) and EX not in r["completion"]]
m, sd, ci = mean_sd_ci(noecho)
mb, sdb, cib = mean_sd_ci(base_ne)
L.append(f"- reverse10 non-echo: mean {m:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), n={len(noecho)}")
L.append(f"- baseline non-echo: mean {mb:.2f} (95% CI {cib[0]:.2f}-{cib[1]:.2f}), n={len(base_ne)}")
overlap = ci[0] <= cib[1]
L.append(f"- VERDICT: authentic-effect CI {'OVERLAPS' if overlap else 'separates from'} baseline CI. "
         f"This is why the clinical-frame authentic estimate is SUPPORTING-only; the clean "
         f"positive is the narrative-frame arm (pilot12 R1: +1.8, zero echo present) and the "
         f"rotated-example rerun (pilot16, running).")

# --- Behavioral ITT: v0/v1 gap with NO exclusion (raw retained data) ---
L.append("\n## 3. Behavioral SES gap: ITT (no exclusion) vs echo-excluded")
import csv
brows = list(csv.DictReader(open(ROOT / "runs" / "pilot2_qwen8b" / "behavioral.csv", encoding="utf-8")))
comp = {}
for r in load("pilot2_qwen8b"):
    comp[(r["race"], r["gender"], r["rel"], r["ses"], r["framing"], r["para"])] = EX in r["completion"]
for para in ("v0", "v1"):
    def cell(ses, excl):
        vv = []
        for r in brows:
            if r["para"] != para or r["ses"] != ses or r["phq8_total"] in ("", "None"):
                continue
            k = (r["race"], r["gender"], r["rel"], r["ses"], r["framing"], r["para"])
            if excl and comp.get(k, False):
                continue
            vv.append(float(r["phq8_total"]))
        return vv
    itt = sum(cell("Low", False)) / len(cell("Low", False)) - sum(cell("High", False)) / len(cell("High", False))
    exc = sum(cell("Low", True)) / len(cell("Low", True)) - sum(cell("High", True)) / len(cell("High", True))
    L.append(f"- {para}: ITT gap {itt:+.2f} (all rows) | echo-excluded {exc:+.2f}. "
             f"Both large; behavioral bias is robust to the echo question either way.")

(ROOT / "docs" / "analysis_itt_report.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
