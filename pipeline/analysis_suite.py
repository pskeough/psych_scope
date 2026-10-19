"""A-series analysis suite (CPU, existing data). Writes docs/analysis_suite_report.md.
A2 monotonicity, A3 sampled-regime effect sizes, A7 FDR ledger, A9 completion
audit, A10 causal-condition demographic interactions."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
L = []

def add(s):
    L.append(s)
    print(s)

# ---- A2: monotonicity (Low>=Mid>=High per cell) across grids/instruments ----
add("# Analysis suite report (A-series)\n")
add("## A2 Monotonicity: fraction of paired cells with Low>Mid>High strictly / weakly")
def monotonic(run, fam):
    rows = list(csv.DictReader(open(ROOT/'runs'/run/'behavioral.csv', encoding='utf-8')))
    key = lambda r: (r['race'], r['gender'], r['rel'], r['framing'], r['para'])
    by = {}
    for r in rows:
        if r['phq8_total'] not in ('', 'None'):
            by.setdefault(key(r), {})[r['ses']] = float(r['phq8_total'])
    cells = [v for v in by.values() if len(v) == 3]
    strict = sum(1 for v in cells if v['Low'] > v['Middle'] > v['High'])
    weak = sum(1 for v in cells if v['Low'] >= v['Middle'] >= v['High'])
    add(f"- {fam}: strict {strict}/{len(cells)}, weak {weak}/{len(cells)}")
for run, fam in (("pilot2_qwen8b","Qwen PHQ"), ("gemma_pilot2","Gemma PHQ"), ("llama_pilot2","Llama PHQ")):
    monotonic(run, fam)

# ---- A3: sampled-regime effect size (pilot5) ----
add("\n## A3 Sampled-regime effect sizes (pilot5, temp 1.0)")
rows = list(csv.DictReader(open(ROOT/'runs'/'pilot5_sampled'/'results.csv', encoding='utf-8')))
vals = {s: [float(r['phq8_total']) for r in rows if r['ses'] == s and r['phq8_total'] not in ('','None')]
        for s in ('Low','Middle','High')}
import statistics as st
lo, hi = vals['Low'], vals['High']
pooled_sd = ((st.pvariance(lo)*len(lo) + st.pvariance(hi)*len(hi)) / (len(lo)+len(hi))) ** 0.5
d = (st.mean(lo) - st.mean(hi)) / pooled_sd
add(f"- Low {st.mean(lo):.2f} (sd {st.stdev(lo):.2f}) vs High {st.mean(hi):.2f} (sd {st.stdev(hi):.2f}): Cohen's d = {d:.2f} (n=100/arm)")
add("- FRAMING NOTE for paper: deterministic-regime paired d values (2-4+) reflect near-zero within-cell variance; the sampled-regime d above is the human-comparable one.")

# ---- A7: consolidated FDR (Benjamini-Hochberg) over all reported p-values ----
add("\n## A7 Consolidated BH-FDR ledger")
tests = [
    ("Qwen grid SES gap", 5e-05), ("Gemma grid SES gap", 5e-05), ("Llama grid SES gap", 5e-05),
    ("Qwen injection", 5e-05), ("Qwen rand ctrl", 0.5031), ("Qwen ablate50x5", 0.0005),
    ("Qwen patch5L", 5e-05), ("Gemma ablate50", 5e-05), ("Gemma inject", 0.6916),
    ("Gemma patch5L", 1.0), ("Llama noise-inject", 5e-05), ("Llama inject", 0.5028),
]
m = len(tests)
srt = sorted(tests, key=lambda t: t[1])
add("| test | p | BH threshold (q=0.05) | significant |")
add("|---|---|---|---|")
for i, (name, p) in enumerate(srt, 1):
    thr = 0.05 * i / m
    add(f"| {name} | {p:.4g} | {thr:.4g} | {'YES' if p <= thr else 'no'} |")

# ---- A9: completion audit across all runs ----
add("\n## A9 Completion audit (refusals / anomalies across all completions)")
pat_refusal = re.compile(r"(I can't|I cannot|I'm sorry|as an AI|I am unable)", re.I)
total = 0; refusals = 0; allzero = 0; empties = 0
for f in ROOT.glob('runs/*/completions.jsonl'):
    for line in open(f, encoding='utf-8'):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = r.get('completion', '')
        total += 1
        if pat_refusal.search(c):
            refusals += 1
        if not c.strip():
            empties += 1
        if re.search(r'"phq8"\s*:\s*\[0, 0, 0, 0, 0, 0, 0, 0\]', c):
            allzero += 1
add(f"- {total} completions scanned: refusal-language {refusals}, empty {empties}, all-zero PHQ {allzero}")

# ---- A10: causal-condition demographic interaction (Qwen injection by race/gender) ----
add("\n## A10 Injection effect by demographic slice (Qwen pilot3 reverse10 - baseline, High)")
rows = list(csv.DictReader(open(ROOT/'runs'/'pilot3_clamp'/'results.csv', encoding='utf-8')))
k2 = lambda r: (r['race'], r['gender'], r['rel'])
base = {k2(r): float(r['phq8_total']) for r in rows if r['cond']=='baseline' and r['ses']=='High' and r['phq8_total'] not in ('','None')}
rev = {k2(r): float(r['phq8_total']) for r in rows if r['cond']=='reverse10' and r['ses']=='High' and r['phq8_total'] not in ('','None')}
for field, idx in (("race", 0), ("gender", 1)):
    groups = {}
    for k in base:
        if k in rev:
            groups.setdefault(k[idx], []).append(rev[k]-base[k])
    add(f"- by {field}: " + ", ".join(f"{g} {sum(v)/len(v):+.1f}(n={len(v)})" for g, v in sorted(groups.items())))

(ROOT/'docs'/'analysis_suite_report.md').write_text("\n".join(L), encoding='utf-8')
print("\nwritten -> docs/analysis_suite_report.md")
