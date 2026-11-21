"""
5-fold scenario-grouped cross-validation of all detectors.
Saves out-of-fold window probabilities for every model so that ensembles,
alarm logic and statistics can be computed afterwards (analyze.py).

Usage: python run_experiments.py --rep hyd      (hydraulically informed input; main study)
       python run_experiments.py --rep raw      (min-max scaled raw input; ablation)
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import common
import lightgbm as lgb
from common import *
from models import DEEP, ConvAE

torch.set_num_threads(4)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def train_deep(cls, Xtr, ytr, Xva, yva, seed, max_epochs=25, patience=4, bs=256, lr=1e-3):
    set_seed(seed)
    m = cls(Xtr.shape[2], Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)
    best, best_state, wait, hist = 1e9, None, 0, []
    for ep in range(max_epochs):
        m.train(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); l = lossf(m(Xt[idx]), yt[idx]); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl += l.item() * len(idx)
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(m(Xv[i:i + 2048]), yv[i:i + 2048]).item() * len(Xv[i:i + 2048]) for i in range(0, len(Xv), 2048)]) * (len(range(0, len(Xv), 2048))) / len(Xv))
        hist.append((tl / len(Xt), vl))
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)
    return m, hist


def predict_deep(m, X, bs=2048):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(m(torch.tensor(X[i:i + bs]))).numpy())
    return np.concatenate(out)


def train_ae(Xn, Xva_n, seed, max_epochs=30, patience=4, bs=256):
    set_seed(seed)
    m = ConvAE(Xn.shape[2], Xn.shape[1]); opt = torch.optim.AdamW(m.parameters(), 1e-3)
    Xt = torch.tensor(Xn); Xv = torch.tensor(Xva_n); best, wait, bstate = 1e9, 0, None
    for ep in range(max_epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = Xt[perm[i:i + bs]]; opt.zero_grad(); l = ((m(b) - b) ** 2).mean(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = ((m(Xv) - Xv) ** 2).mean().item()
        if vl < best - 1e-5:
            best, wait, bstate = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    m.load_state_dict(bstate); return m


def ae_score(m, X):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            b = torch.tensor(X[i:i + 2048]); out.append(((m(b) - b) ** 2).mean((1, 2)).numpy())
    return np.concatenate(out)


def to_prob_like(score, ref_normal):
    """Map an anomaly score to (0,1) with threshold 0.5 at the 95th percentile of
    validation leak-free scores (so 'p>=0.5' means 'alarm')."""
    thr = np.percentile(ref_normal, 95)
    scale = np.std(ref_normal) + 1e-9
    return 1 / (1 + np.exp(-(score - thr) / scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="hyd", choices=["hyd", "raw"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--folds_to_run", default="all", help="e.g. 3,4 to resume")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    import common; common.W = a.win
    X, Y, LF, ch, meta = load()
    N, T, C = X.shape
    S = build_series(X, ch, a.rep); C = S.shape[2]
    tod = calendar_channels(meta, T)
    scen_leak = np.array([s["leak"] for s in meta["scenarios"]]).astype(int)
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    os.makedirs(a.out, exist_ok=True)
    timing = {}
    want = None if a.models == "all" else set(a.models.split(","))
    run_f = None if a.folds_to_run == "all" else {int(x) for x in a.folds_to_run.split(",")}
    tjson = f"{a.out}/timing_{a.rep}.json"
    if run_f and os.path.exists(tjson):
        timing = json.load(open(tjson))
    for fold, (tr_s, te_s) in enumerate(skf.split(np.zeros(N), scen_leak)):
        if run_f is not None and fold not in run_f:
            continue
        rng = np.random.default_rng(a.seed + fold)
        tr_s = rng.permutation(tr_s); nva = int(0.15 * len(tr_s))
        va_s, tr_s = tr_s[:nva], tr_s[nva:]
        # normalisation fitted on training scenarios only
        if a.rep == "hyd":   # z-score with training-scenario statistics
            flat = S[tr_s][:, DAY:].reshape(-1, C); mu = flat.mean(0); sd = flat.std(0) + 1e-6
        else:  # min-max to [0,1] like the reference study
            flat = S[tr_s].reshape(-1, C); mu = flat.min(0); sd = flat.max(0) - flat.min(0) + 1e-6
        Sn = np.concatenate([(S - mu) / sd, tod], 2).astype(np.float32)
        Xtr, ytr, _, _ = make_windows(Sn, Y, tr_s, STRIDE_TRAIN)
        Xva, yva, _, _ = make_windows(Sn, Y, va_s, STRIDE_TEST)
        Xte, yte, sid, ends = make_windows(Sn, Y, te_s, STRIDE_TEST)
        Ftr, Fva, Fte = window_features(Xtr), window_features(Xva), window_features(Xte)
        print(f"fold {fold}: train {Xtr.shape} pos={ytr.mean():.3f} val {Xva.shape} test {Xte.shape}", flush=True)
        P = {}

        def rec(name, t_train, f):
            t0 = time.time(); p = f(); t_inf = (time.time() - t0) / len(Xte) * 1000
            P[name] = p.astype(np.float32)
            timing.setdefault(name, []).append({"train_s": t_train, "infer_ms_per_1000": t_inf * 1000 / 1000})
            r = metrics(yte, p); print(f"  {name:22s} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} MCC={r['mcc']:.4f} ({t_train:.0f}s)", flush=True)

        def use(n): return want is None or n in want
        if use("LogReg"):
            sc = StandardScaler().fit(Ftr); t0 = time.time()
            lr_ = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Ftr), ytr)
            rec("LogReg", time.time() - t0, lambda: lr_.predict_proba(sc.transform(Fte))[:, 1])
        if use("RandomForest"):
            t0 = time.time(); rf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=4, random_state=a.seed).fit(Ftr, ytr)
            rec("RandomForest", time.time() - t0, lambda: rf.predict_proba(Fte)[:, 1])
        if use("XGBoost"):
            t0 = time.time()
            xg = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=a.seed,
                                   early_stopping_rounds=50, eval_metric="logloss")
            xg.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
            rec("XGBoost", time.time() - t0, lambda: xg.predict_proba(Fte)[:, 1])
        if use("LightGBM"):
            t0 = time.time()
            lg = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, n_jobs=4, random_state=a.seed, verbose=-1)
            lg.fit(Ftr, ytr, eval_set=[(Fva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
            rec("LightGBM", time.time() - t0, lambda: lg.predict_proba(Fte)[:, 1])
        if use("IsolationForest"):
            t0 = time.time(); isf = IsolationForest(n_estimators=300, random_state=a.seed, n_jobs=4).fit(Ftr[ytr == 0])
            ref = -isf.score_samples(Fva[yva == 0])
            rec("IsolationForest", time.time() - t0, lambda: to_prob_like(-isf.score_samples(Fte), ref))
        if use("ConvAE"):
            t0 = time.time(); ae = train_ae(Xtr[ytr == 0], Xva[yva == 0], a.seed + fold)
            ref = ae_score(ae, Xva[yva == 0])
            rec("ConvAE", time.time() - t0, lambda: to_prob_like(ae_score(ae, Xte), ref))
        for name, cls in DEEP.items():
            if not use(name): continue
            t0 = time.time(); m, hist = train_deep(cls, Xtr, ytr, Xva, yva, a.seed + fold)
            nparam = sum(p.numel() for p in m.parameters())
            rec(name, time.time() - t0, lambda: predict_deep(m, Xte))
            timing[name][-1].update(params=int(nparam), epochs=len(hist))
            if fold == 0:
                json.dump(hist, open(f"{a.out}/hist_{a.rep}_{name.split()[0]}.json", "w"))
                torch.save(m.state_dict(), f"{a.out}/model_{a.rep}_{name.split()[0]}_fold0.pt")
        tag = "" if want is None else "_" + "_".join(sorted(n.split()[0] for n in want))
        np.savez_compressed(f"{a.out}/oof_{a.rep}{tag}_fold{fold}.npz", y=yte, sid=sid, ends=ends,
                            names=np.array(list(P.keys())), probs=np.stack(list(P.values())))
        json.dump(timing, open(f"{a.out}/timing_{a.rep}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
