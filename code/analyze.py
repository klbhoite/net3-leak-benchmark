"""
Aggregate out-of-fold predictions -> tables, scenario-level alarm statistics,
cluster-bootstrap confidence intervals and figures.
Usage: python analyze.py
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, roc_auc_score
from common import load, metrics, scenario_alarms

RES, FIG = "../results", "../figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
                     "savefig.dpi": 600, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5})
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8c8b86"
X, Y, LF, CH, META = load()
FLOW = np.array([s.get("mean_leak_flow_Ls", 0.0) for s in META["scenarios"]])
KIND = np.array([s.get("kind", "none") for s in META["scenarios"]])
ENSEMBLE = json.load(open("ensemble.json")) if os.path.exists("ensemble.json") else None


def add_mass_balance_rule(dd, fold, rep, seed=7, k=5):
    """Non-learning physical baseline: 12-h mean of the reference-day residual of
    the mass-balance net demand; alarm threshold = 95th percentile over normal
    windows of the training scenarios of the same fold."""
    from sklearn.model_selection import StratifiedKFold
    from common import net_demand, residual_series
    global _MB
    if "_MB" not in globals():
        _MB = residual_series(net_demand(X, CH))[..., 0]
    leak = np.array([s["leak"] for s in META["scenarios"]]).astype(int)
    tr, _ = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(leak)), leak))[fold]
    W = 48
    def score(s, e): return _MB[s, e - W + 1:e + 1].mean()
    ref = [score(s, e) for s in tr for e in range(143, X.shape[1], 4) if Y[s, e] == 0]
    sc = np.array([score(s, e) for s, e in zip(dd["sid"], dd["ends"])])
    thr = np.percentile(ref, 95); scale = np.std(ref)
    dd["P"]["Mass-balance rule"] = 1 / (1 + np.exp(-(sc - thr) / scale))


def load_oof(rep):
    folds = sorted(glob.glob(f"{RES}/oof_{rep}_fold*.npz"))
    out = []
    for f in folds:
        d = np.load(f, allow_pickle=True)
        dd = {"y": d["y"], "sid": d["sid"], "ends": d["ends"], "P": dict(zip(list(d["names"]), d["probs"]))}
        extra = sorted(glob.glob(f.replace(f"oof_{rep}_", f"oof_{rep}_*_")))
        for g in glob.glob(f"{RES}/oof_{rep}_*_fold{f[-5]}.npz"):
            e = np.load(g, allow_pickle=True)
            assert np.array_equal(e["sid"], d["sid"]) and np.array_equal(e["ends"], d["ends"])
            dd["P"].update(dict(zip(list(e["names"]), e["probs"])))
        add_mass_balance_rule(dd, int(f[-5]), rep)
        if ENSEMBLE:
            for name, members in ENSEMBLE.items():
                if all(m in dd["P"] for m in members):
                    dd["P"][name] = np.mean([dd["P"][m] for m in members], 0)
        out.append(dd)
    return out


def table(folds, order=None):
    names = order or list(folds[0]["P"].keys())
    rows = []
    for n in names:
        per = pd.DataFrame([metrics(f["y"], f["P"][n]) for f in folds])
        r = {"model": n}
        for c in per.columns:
            r[c] = per[c].mean(); r[c + "_sd"] = per[c].std(ddof=1)
        # scenario-level alarm statistics pooled over folds
        al = {}
        for f in folds:
            al.update(scenario_alarms(f["P"][n], f["sid"], f["ends"], Y))
        leak = [v for v in al.values() if v["leak"]]; free = [v for v in al.values() if not v["leak"]]
        r["scen_detect_rate"] = np.mean([v["detected"] for v in leak])
        r["scen_false_alarm_rate"] = np.mean([v["false_alarm"] for v in al.values()])
        r["free_scen_false_alarm_rate"] = np.mean([v["false_alarm"] for v in free])
        dl = [v["delay_h"] for v in leak if v["detected"]]
        r["median_delay_h"] = float(np.median(dl)); r["mean_delay_h"] = float(np.mean(dl))
        rows.append(r)
    return pd.DataFrame(rows)


def pooled(folds, n):
    return (np.concatenate([f["y"] for f in folds]), np.concatenate([f["P"][n] for f in folds]),
            np.concatenate([f["sid"] for f in folds]), np.concatenate([f["ends"] for f in folds]))


def _auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_bootstrap(folds, a, b, B=1000, seed=0):
    """Scenario-level (cluster) bootstrap of the difference in F1 and ROC-AUC, a - b."""
    y, pa, sid, _ = pooled(folds, a); _, pb, _, _ = pooled(folds, b)
    scen = np.unique(sid); idx = {s: np.where(sid == s)[0] for s in scen}
    rng = np.random.default_rng(seed); dF, dA = [], []
    for _ in range(B):
        pick = np.concatenate([idx[s] for s in rng.choice(scen, len(scen))])
        yy = y[pick]
        dF.append(f1_score(yy, pa[pick] >= .5) - f1_score(yy, pb[pick] >= .5))
        dA.append(_auc(yy, pa[pick]) - _auc(yy, pb[pick]))
    dF, dA = np.array(dF), np.array(dA)
    return {"dF1": float(f1_score(y, pa >= .5) - f1_score(y, pb >= .5)), "dF1_lo": float(np.percentile(dF, 2.5)),
            "dF1_hi": float(np.percentile(dF, 97.5)), "p_F1": float(min(1, 2 * min((dF <= 0).mean(), (dF >= 0).mean()))),
            "dAUC": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), "dAUC_lo": float(np.percentile(dA, 2.5)),
            "dAUC_hi": float(np.percentile(dA, 97.5)), "p_AUC": float(min(1, 2 * min((dA <= 0).mean(), (dA >= 0).mean())))}


def operational_alarms(folds, n, q=99, persist=3):
    """Cross-fitted alarm threshold: for test fold k the threshold is the q-th
    percentile of normal-window scores in the out-of-fold predictions of the
    other folds (no information from fold k is used)."""
    al = {}
    for k, f in enumerate(folds):
        ref = np.concatenate([g["P"][n][g["y"] == 0] for j, g in enumerate(folds) if j != k])
        thr = np.percentile(ref, q)
        al.update(scenario_alarms(f["P"][n], f["sid"], f["ends"], Y, thr=thr, persist=persist))
    leak = [v for v in al.values() if v["leak"]]; free = [v for v in al.values() if not v["leak"]]
    dl = [v["delay_h"] for v in leak if v["detected"]]
    det_big = [v["detected"] for s_, v in al.items() if v["leak"] and FLOW[s_] > 20]
    return {"model": n, "q": q, "persist": persist, "detect_rate": np.mean([v["detected"] for v in leak]),
            "detect_rate_gt20Ls": np.mean(det_big), "false_alarm_free_weeks": np.mean([v["false_alarm"] for v in free]),
            "false_alarm_any": np.mean([v["false_alarm"] for v in al.values()]),
            "median_delay_h": float(np.median(dl)) if dl else np.nan}


def by_leak_size(folds, names, bins=((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9))):
    rows = []
    for n in names:
        y, p, sid, ends = pooled(folds, n)
        for lo, hi in bins:
            m = (y == 1) & (FLOW[sid] > lo) & (FLOW[sid] <= hi)
            rows.append({"model": n, "bin": f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}", "recall": (p[m] >= .5).mean(), "n": int(m.sum())})
        for k in ["abrupt", "incipient"]:
            m = (y == 1) & (KIND[sid] == k)
            rows.append({"model": n, "bin": k, "recall": (p[m] >= .5).mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def fmt(df, cols):
    out = df[["model"]].copy()
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df[c], df[c + "_sd"])]
    return out


if __name__ == "__main__":
    summary = {}
    for rep in ["hyd", "raw"]:
        folds = load_oof(rep)
        if not folds:
            continue
        t = table(folds)
        t.to_csv(f"{RES}/table_{rep}.csv", index=False)
        print(f"\n=== {rep} ===")
        print(t[["model", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc",
                 "scen_detect_rate", "free_scen_false_alarm_rate", "median_delay_h"]].round(3).to_string(index=False))
        summary[rep] = t
    json.dump({k: v.to_dict(orient="records") for k, v in summary.items()}, open(f"{RES}/summary.json", "w"), indent=1)


# ----------------------------------------------------------------------------- figures
ORDER = ["Mass-balance rule", "IsolationForest", "ConvAE", "LogReg", "RandomForest", "XGBoost", "LightGBM",
         "P-CNN (baseline)", "1D-CNN", "CNN-BiLSTM", "TCN", "Transformer", "HyEns (soft vote)"]
FOCUS = ["P-CNN (baseline)", "LogReg", "LightGBM", "TCN", "Transformer", "HyEns (soft vote)"]


def color_of(n):
    return {"P-CNN (baseline)": GRAY, "LogReg": PAL[2], "LightGBM": PAL[3], "TCN": PAL[0],
            "Transformer": PAL[6], "HyEns (soft vote)": PAL[1], "CNN-BiLSTM": PAL[4]}.get(n, "#bdbcb6")


def make_figures(folds, t_hyd, t_raw):
    # Fig 4: ROC and PR
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for n in FOCUS:
        y, p, _, _ = pooled(folds, n)
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        lw = 2.2 if n.startswith("HyEns") else 1.4
        axs[0].plot(fpr, tpr, color=color_of(n), lw=lw, label=f"{n} ({roc_auc_score(y, p):.3f})")
        axs[1].plot(rc, pr, color=color_of(n), lw=lw, label=n)
    axs[0].plot([0, 1], [0, 1], color="#d0cfca", lw=0.8, ls=":")
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].legend(fontsize=6.5, frameon=False, loc="lower right", title="ROC-AUC (pooled)", title_fontsize=6.5)
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0.3, 1.0)
    axs[0].set_title("(a) ROC", fontsize=9, loc="left"); axs[1].set_title("(b) Precision–recall", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_roc_pr.png"); fig.savefig(f"{FIG}/fig4_roc_pr.pdf"); plt.close(fig)

    # Fig 5: F1 and AUC for all models, both representations
    names = [n for n in ORDER if n in set(t_hyd.model)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    yy = np.arange(len(names))
    for ax, met, ttl in zip(axs, ["f1", "roc_auc"], ["(a) F1-score", "(b) ROC-AUC"]):
        th = t_hyd.set_index("model").loc[names]
        ax.barh(yy + 0.2, th[met], height=0.38, xerr=th[met + "_sd"], color=PAL[0], label="Hydraulically informed input",
                error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"))
        if t_raw is not None:
            tr_ = t_raw.set_index("model").reindex(names)
            ax.barh(yy - 0.2, tr_[met], height=0.38, xerr=tr_[met + "_sd"], color="#bdbcb6", label="Raw min–max input",
                    error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"))
        ax.set_title(ttl, fontsize=9, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
    axs[0].set_yticks(yy); axs[0].set_yticklabels(names, fontsize=7.5); axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/fig5_models.png"); fig.savefig(f"{FIG}/fig5_models.pdf"); plt.close(fig)

    # Fig 6: recall by leak size
    bl = by_leak_size(folds, FOCUS)
    bl.to_csv(f"{RES}/recall_by_leak.csv", index=False)
    bins = [b for b in bl.bin.unique() if b not in ("abrupt", "incipient")]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(bins)); wbar = 0.8 / len(FOCUS)
    for k, n in enumerate(FOCUS):
        v = bl[(bl.model == n) & bl.bin.isin(bins)].set_index("bin").loc[bins, "recall"]
        ax.bar(x - 0.4 + wbar * (k + 0.5), v, width=wbar * 0.92, color=color_of(n), label=n)
    ns = bl[(bl.model == FOCUS[0]) & bl.bin.isin(bins)].set_index("bin").loc[bins, "n"]
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={m})" for b, m in zip(bins, ns)], fontsize=7)
    ax.set_xlabel("Mean leak outflow (L/s)  [n = leak windows]"); ax.set_ylabel("Window-level recall"); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_leaksize.png"); fig.savefig(f"{FIG}/fig6_leaksize.pdf"); plt.close(fig)

    # Fig 7: detection delay distributions (scenario level)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    data, labs = [], []
    for n in FOCUS:
        al = {}
        for f in folds:
            al.update(scenario_alarms(f["P"][n], f["sid"], f["ends"], Y))
        data.append([v["delay_h"] for v in al.values() if v["leak"] and v["detected"]]); labs.append(n)
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#0b0b0b", lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, n in zip(bp["boxes"], labs):
        patch.set_facecolor(color_of(n)); patch.set_alpha(0.85); patch.set_edgecolor("white")
    ax.set_yticks(range(1, len(labs) + 1)); ax.set_yticklabels(labs, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Detection delay after leak onset (h)"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_delay.png"); fig.savefig(f"{FIG}/fig7_delay.pdf"); plt.close(fig)


def make_tables(t_hyd, t_raw, folds):
    names = [n for n in ORDER if n in set(t_hyd.model)]
    th = t_hyd.set_index("model").loc[names].reset_index()
    win = fmt(th, ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc"])
    win.to_csv(f"{RES}/table_window_hyd.csv", index=False)
    alarm = th[["model", "scen_detect_rate", "free_scen_false_alarm_rate", "scen_false_alarm_rate", "median_delay_h", "mean_delay_h"]].round(3)
    alarm.to_csv(f"{RES}/table_alarm_hyd.csv", index=False)
    if t_raw is not None:
        tr_ = t_raw.set_index("model").reindex(names).reset_index()
        ab = pd.DataFrame({"model": names,
                           "F1 raw": [f"{m:.3f} ± {s:.3f}" for m, s in zip(tr_.f1, tr_.f1_sd)],
                           "F1 informed": [f"{m:.3f} ± {s:.3f}" for m, s in zip(th.f1, th.f1_sd)],
                           "AUC raw": [f"{m:.3f} ± {s:.3f}" for m, s in zip(tr_.roc_auc, tr_.roc_auc_sd)],
                           "AUC informed": [f"{m:.3f} ± {s:.3f}" for m, s in zip(th.roc_auc, th.roc_auc_sd)],
                           "dAUC": (th.roc_auc.values - tr_.roc_auc.values).round(3)})
        ab.to_csv(f"{RES}/table_ablation.csv", index=False)
    ops = [operational_alarms(folds, n, q, p) for n in names for q, p in [(95, 3), (99, 3), (99, 6)]]
    pd.DataFrame(ops).to_csv(f"{RES}/table_alarm_operational.csv", index=False)
    print(pd.DataFrame(ops).round(3).to_string(index=False))
    bs = []
    for n in names:
        if n == "P-CNN (baseline)": continue
        r = cluster_bootstrap(folds, n, "P-CNN (baseline)"); r["model"] = n; bs.append(r)
    pd.DataFrame(bs).to_csv(f"{RES}/table_bootstrap_vs_pcnn.csv", index=False)
    if "HyEns (soft vote)" in names:
        bs2 = []
        for n in names:
            if n.startswith("HyEns"): continue
            r = cluster_bootstrap(folds, "HyEns (soft vote)", n); r["model"] = n; bs2.append(r)
        pd.DataFrame(bs2).to_csv(f"{RES}/table_bootstrap_ens_vs_all.csv", index=False)


if __name__ == "__main__":
    fh = load_oof("hyd"); fr = load_oof("raw") if glob.glob(f"{RES}/oof_raw_fold4.npz") else None
    th = pd.read_csv(f"{RES}/table_hyd.csv"); tr_ = pd.read_csv(f"{RES}/table_raw.csv") if fr else None
    make_figures(fh, th, tr_)
    make_tables(th, tr_, fh)
    print("figures and tables written")
