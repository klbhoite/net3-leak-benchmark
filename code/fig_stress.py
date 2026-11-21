"""Fig. 8: robustness to additional sensor noise (fold-0 test scenarios)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import FOCUS, color_of, FIG
S = json.load(open("../results/stress.json"))
lv = list(S["noise"].keys()); x = np.arange(len(lv))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for n in [m for m in FOCUS if m in S["noise"][lv[0]]]:
    for ax, met in zip(axs, ["roc_auc", "specificity"]):
        ax.plot(x, [S["noise"][l][n][met] for l in lv], marker="o", ms=4, color=color_of(n), lw=2.2 if n.startswith("HyEns") else 1.4, label=n)
for ax, t in zip(axs, ["(a) ROC-AUC", "(b) Specificity at threshold 0.5"]):
    ax.set_xticks(x); ax.set_xticklabels(["none", "+0.25 m\n+1 %", "+0.5 m\n+2 %", "+1.0 m\n+3 %"], fontsize=7)
    ax.set_xlabel("Additional pressure / flow noise (s.d.)"); ax.set_title(t, fontsize=9, loc="left")
axs[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig8_stress.png"); fig.savefig(f"{FIG}/fig8_stress.pdf")
for n, r in S["smallleak"].items():
    print(n, {k: round(r[k], 3) for k in ["f1", "roc_auc", "recall", "specificity", "precision"]})
