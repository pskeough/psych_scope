import re
from collections import Counter
from pathlib import Path

tex = Path(r"C:\Research\PsychScope\paper\psychscope.tex").read_text(encoding="utf-8")
begins = re.findall(r"\\begin\{(\w+)\}", tex)
ends = re.findall(r"\\end\{(\w+)\}", tex)
b, e = Counter(begins), Counter(ends)
imbalance = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
print("environment imbalance:", imbalance or "NONE (balanced)")
print("brace balance:", "OK" if tex.count("{") == tex.count("}") else f"MISMATCH {tex.count('{')} vs {tex.count('}')}")
figdir = Path(r"C:\Research\PsychScope\figures")
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    print(f"  fig {f}: {'FOUND' if (figdir / f).exists() else 'MISSING'}")
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
print("refs without labels:", (refs - labels) or "NONE")
cites = set(re.findall(r"\\citet?\{([^}]+)\}", tex))
bibs = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", tex))
print("cites without bib entries:", (cites - bibs) or "NONE")
emdash = tex.count("—")
print("em-dashes (unicode) in source:", emdash)
print("word count:", len(tex.split()))
