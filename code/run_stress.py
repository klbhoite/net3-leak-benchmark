"""
Robustness tests (fold-0 training scenarios of the main dataset):
  (a) additional sensor noise injected into the fold-0 test scenarios,
  (b) transfer to an independent small-leak dataset (0.8-5 cm orifices, 400 new scenarios).
Usage: python run_stress.py
"""
import json, numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common
from common import *
from models import DEEP
from run_experiments import train_deep, predict_deep

torch.set_num_threads(4)
SEED = 7
X, Y, LF, ch, meta = load()
N, T, C0 = X.shape
scen_leak = np.array([s["leak"] for s in meta["scenarios"]]).astype(int)
tr_s, te_s = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(N), scen_leak))
rng = np.random.default_rng(SEED); tr_s = rng.permutation(tr_s); nva = int(0.15 * len(tr_s)); va_s, tr_s = tr_s[:nva], tr_s[nva:]
S = build_series(X, ch, "hyd"); C = S.shape[2]
flat = S[tr_s][:, DAY:].reshape(-1, C); mu = flat.mean(0); sd = flat.std(0) + 1e-6
cal = calendar_channels(meta, T)


def prep(Xr, Yr, cal_, idx):
    Sn = np.concatenate([(build_series(Xr, ch, "hyd") - mu) / sd, cal_], 2).astype(np.float32)
    return make_windows(Sn, Yr, idx, STRIDE_TEST)


Xtr, ytr, _, _ = prep(X, Y, cal, tr_s); Xva, yva, _, _ = prep(X, Y, cal, va_s)
models = {}
sc = StandardScaler().fit(window_features(Xtr))
lr = LogisticRegression(max_iter=2000).fit(sc.transform(window_features(Xtr)), ytr)
models["LogReg"] = lambda W: lr.predict_proba(sc.transform(window_features(W)))[:, 1]
for n in ["P-CNN (baseline)", "TCN", "Transformer", "CNN-BiLSTM"]:
    m, _ = train_deep(DEEP[n], Xtr, ytr, Xva, yva, SEED)
    models[n] = (lambda mm: (lambda W: predict_deep(mm, W)))(m)
    print("trained", n, flush=True)
ENS = json.load(open("ensemble.json")) if __import__("os").path.exists("ensemble.json") else {}


def evaluate(Wx, wy, tag):
    P = {n: f(Wx) for n, f in models.items()}
    for en, mem in ENS.items():
        if all(m in P for m in mem):
            P[en] = np.mean([P[m] for m in mem], 0)
    return {n: {**metrics(wy, p), "tag": tag} for n, p in P.items()}


out = {"noise": {}, "smallleak": {}}
ip = [i for i, c in enumerate(ch) if c.startswith("P_")]; iq = [i for i, c in enumerate(ch) if c.startswith("Q_")]
for k, (sp, sq) in enumerate([(0.0, 0.0), (0.25, 0.01), (0.5, 0.02), (1.0, 0.03)]):
    r = np.random.default_rng(100 + k); Xn = X.copy()
    Xn[:, :, ip] += r.normal(0, sp, Xn[:, :, ip].shape)
    Xn[:, :, iq] *= 1 + r.normal(0, sq, Xn[:, :, iq].shape)
    W, wy, _, _ = prep(Xn, Y, cal, te_s)
    out["noise"][f"+{sp}m/+{int(sq*100)}%"] = evaluate(W, wy, "noise"); print("noise", sp, flush=True)
Xs, Ys, _, _, ms = load("../data/net3_smallleak.npz")
W, wy, sid, ends = prep(Xs, Ys, calendar_channels(ms, T), np.arange(len(Xs)))
out["smallleak"] = evaluate(W, wy, "smallleak")
json.dump(out, open("../results/stress.json", "w"), indent=1, default=float)
for k, v in out["noise"].items():
    print(k, {n: round(r["f1"], 3) for n, r in v.items()})
print("small", {n: (round(r["f1"], 3), round(r["roc_auc"], 3)) for n, r in out["smallleak"].items()})
