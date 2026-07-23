"""Generate publication figures (PDF) for the PsychScope paper -> figures/.
Reads run summaries where possible; derived CIs from analysis are passed in.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})
# colorblind-safe
C = {"Low": "#d55e00", "Middle": "#e69f00", "High": "#0072b2",
     "Qwen": "#0072b2", "Gemma": "#009e73", "Llama": "#cc79a7"}


def js(run):
    p = ROOT / "runs" / run / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- F1: powered behavioral gradients, 3 models x 3 instruments ----
def fig1():
    models = [("Qwen3-8B", "pilot15_nscale", "pilot7_gad7", "pilot8_auditc_qwen"),
              ("Gemma-3-12B", "pilot15_gemma", "pilot7g_gad7_gemma", "pilot8g_auditc_gemma"),
              ("Llama-3.1-8B", "pilot15_llama", "llama_gad7", "llama_auditc")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    insts = ["PHQ-8", "GAD-7", "AUDIT-C"]
    for ax, (mname, phq, gad, aud) in zip(axes, models):
        for j, (inst, run) in enumerate(zip(insts, (phq, gad, aud))):
            s = js(run)
            if not s:
                continue
            xs = [0, 1, 2]
            ys = [s[k]["mean"] for k in ("Low", "Middle", "High")]
            es = [s[k].get("sd", 0) or 0 for k in ("Low", "Middle", "High")]
            ax.errorbar([x + j * 0.0 for x in xs], ys, yerr=es, marker="o", capsize=3,
                        label=inst, lw=1.5)
        ax.set_title(mname)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Socioeconomic status")
    axes[0].set_ylabel("Symptom score")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Poverty-graded symptom scoring across models and instruments (n=320/level, temp 1.0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gradients.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- F2: item signature, all 3 models (Qwen somatic; Gemma & Llama affective) ----
def fig2():
    sigs = json.loads((FIG / "item_sigs.json").read_text())
    items = ["anhed", "mood", "sleep", "fatig", "appet", "self", "conc", "psychmot"]
    order = ["anhedonia","depressed_mood","sleep","fatigue","appetite","self_worth","concentration","psychomotor"]
    x = range(len(items))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, (fam, lab) in enumerate((("Qwen","Qwen (somatic-led)"),("Gemma","Gemma (affective-led)"),("Llama","Llama (affective-led)"))):
        ys = [sigs[fam][k] for k in order]
        ax.bar([i + (j-1)*0.27 for i in x], ys, 0.27, label=lab, color=C[fam])
    ax.set_xticks(list(x)); ax.set_xticklabels(items, rotation=30, ha="right")
    ax.set_ylabel("Low-High per-item gap"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Stereotype content differs by model (PHQ-8 item decomposition, all 3 models)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_item_signature.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- F3: transport curve (single-layer + cumulative PMF by depth, Qwen) ----
def fig3():
    single = {15: 0.035, 16: 0.39, 17: 0.42, 18: 0.79, 19: 0.64, 20: 0.50, 21: 0.40, 22: 0.23, 23: 0.01}
    cum = {15: 0.035, 16: 0.378, 17: 0.494, 18: 0.884, 19: 0.924, 20: 0.936, 22: 0.965}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(list(single), list(single.values()), "o-", label="single-layer patch", color=C["Low"])
    ax.plot(list(cum), list(cum.values()), "s--", label="cumulative from L15", color=C["High"])
    ax.axhline(0.965, color="gray", ls=":", lw=1, label="all-layer control (0.965)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Patch mediation fraction")
    ax.set_title("SES signal patchable in a narrow early band (Qwen suffix)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_transport.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- F4: causal intervention effects with CIs ----
def fig4():
    labels = ["Injection\n(Qwen, High)", "Ablation\n(Gemma, Low)", "Raw patch\n(Qwen 5L)"]
    means = [2.65, -2.75, -3.35]
    lo = [1.70, -3.05, -3.70]; hi = [3.60, -2.45, -3.00]
    err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.errorbar(range(3), means, yerr=err, fmt="o", capsize=5, color=C["Qwen"], ms=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PHQ-8 shift (95% bootstrap CI)")
    ax.set_title("Causal intervention effects (all CIs exclude 0)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_causal_ci.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- F5: SAE adequacy (raw vs recon vs valence-recon transfer at L18) ----
def fig5():
    labels = ["Raw patch\n(SES)", "Recon patch\n(SES, op-matched)", "Recon patch\n(valence, non-SES)"]
    vals = [0.79, 0.35, 0.79]
    cols = [C["High"], C["Low"], C["Gemma"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(range(3), vals, color=cols)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Transfer fraction at L18")
    ax.set_title("Dictionary carries valence (79%) but SES only ~35%:\nSES-specific under-representation, not operator failure", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 0.95)
    fig.tight_layout(); fig.savefig(FIG / "fig5_sae_adequacy.pdf", bbox_inches="tight")
    plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5):
    f()
    print("wrote", f.__name__)
print("figures ->", [p.name for p in sorted(FIG.glob("fig*.pdf"))])
