"""Shared data handling: windowing, reference-day residuals, features, metrics."""
import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             confusion_matrix)

DAY = 96          # samples per day (15-min)
W = 48            # window length = 12 h
STRIDE_TRAIN = 4  # 1 h
STRIDE_TEST = 4


def load(path="../data/net3_leak.npz"):
    d = np.load(path, allow_pickle=True)
    meta = json.load(open(path.replace(".npz", "_meta.json")))
    return d["X"], d["Y"], d["leak_flow"], list(d["channels"]), meta


def tod_channels(T):
    t = np.arange(T) % DAY
    return np.stack([np.sin(2 * np.pi * t / DAY), np.cos(2 * np.pi * t / DAY)], 1).astype(np.float32)


def calendar_channels(meta, T):
    """Time-of-day (sin, cos) and weekend flag for every scenario: [N,T,3]."""
    tod = tod_channels(T)
    out = []
    for sc in meta["scenarios"]:
        wd = (sc.get("start_weekday", 0) + np.arange(T) // DAY) % 7
        out.append(np.concatenate([tod, (wd >= 5).astype(np.float32)[:, None]], 1))
    return np.stack(out).astype(np.float32)


def residual_series(X):
    """Reference-day residual: subtract, for each time-of-day slot, the value
    observed in the first (commissioning / known leak-free) day of the record.
    The reference day is smoothed with a 3-sample moving average to damp noise."""
    ref = X[:, :DAY, :].copy()
    k = np.array([1, 1, 1], np.float32) / 3
    for s in range(X.shape[0]):
        for c in range(X.shape[2]):
            v = np.concatenate([ref[s, -1:, c], ref[s, :, c], ref[s, :1, c]])
            ref[s, :, c] = np.convolve(v, k, mode="valid")
    T = X.shape[1]
    reps = int(np.ceil(T / DAY))
    R = X - np.tile(ref, (1, reps, 1))[:, :T, :]
    return R.astype(np.float32)


TANK_AREA = np.array([527.18, 182.41, 1962.49], np.float32)  # m^2, Net3 tanks 1-3


def net_demand(X, ch):
    """Mass-balance channel (L/s): metered source inflow minus rate of change of
    tank storage = consumption + leakage. Tank-level derivative is taken over a
    1-h centred difference to limit noise amplification."""
    iq = [ch.index("Q_in_River"), ch.index("Q_in_Lake")]
    il = [i for i, c in enumerate(ch) if c.startswith("L_tank")]
    L = X[:, :, il]
    dL = np.zeros_like(L)
    dL[:, 2:-2] = (L[:, 4:] - L[:, :-4]) / (4 * 900.0)
    dL[:, :2] = dL[:, 2:3]; dL[:, -2:] = dL[:, -3:-2]
    return (X[:, :, iq].sum(2) - (dL * TANK_AREA * 1000.0).sum(2))[..., None].astype(np.float32)


def build_series(X, ch, rep):
    """rep='raw': the 16 SCADA channels as measured (reference-study style input).
       rep='hyd': hydraulically informed input = raw channels + net-demand channel
                  + reference-day residuals of all of these."""
    if rep == "raw":
        return X.astype(np.float32)
    D = net_demand(X, ch)
    base = np.concatenate([X, D], 2)
    return np.concatenate([base, residual_series(base)], 2).astype(np.float32)


def make_windows(S, Y, scen_idx, stride, first_end=None):
    """S: [N,T,C] series. Returns windows [n,W,C], labels (label of last sample),
    scenario id, end index."""
    xs, ys, sid, ends = [], [], [], []
    T = S.shape[1]
    if first_end is None:
        first_end = DAY + W - 1
    for s in scen_idx:
        for e in range(first_end, T, stride):
            xs.append(S[s, e - W + 1:e + 1]); ys.append(Y[s, e]); sid.append(s); ends.append(e)
    return (np.stack(xs).astype(np.float32), np.array(ys, np.int64),
            np.array(sid), np.array(ends))


def window_features(Xw):
    """Hand-crafted statistics per channel for tree / shallow models."""
    n, w, c = Xw.shape
    t = np.arange(w, dtype=np.float32); t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    q = w // 6
    feats = [Xw.mean(1), Xw.std(1), Xw.min(1), Xw.max(1), Xw[:, -1], Xw[:, -q:].mean(1),
             Xw[:, -q:].mean(1) - Xw[:, :q].mean(1), np.einsum("nwc,w->nc", Xw, t),
             np.abs(np.diff(Xw, axis=1)).mean(1)]
    return np.concatenate(feats, 1).astype(np.float32)


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh, zero_division=0), "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, yh, zero_division=0), "mcc": matthews_corrcoef(y, yh),
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
    }


def scenario_alarms(p, sid, ends, Y, thr=0.5, persist=3):
    """Online alarm logic: raise an alarm when `persist` consecutive window
    scores exceed thr. Returns per-scenario dicts (detected, delay_h, false alarm)."""
    out = {}
    for s in np.unique(sid):
        m = sid == s
        e = ends[m]; o = np.argsort(e); e = e[o]; ps = p[m][o]
        above = ps >= thr
        run = 0; alarm_t = None
        for k in range(len(e)):
            run = run + 1 if above[k] else 0
            if run >= persist:
                alarm_t = e[k]; break
        onset = np.argmax(Y[s] > 0) if Y[s].any() else None
        if onset is None:
            out[int(s)] = {"leak": False, "false_alarm": alarm_t is not None}
        else:
            # an alarm before onset is a false alarm for that scenario
            fa = alarm_t is not None and alarm_t < onset
            det = alarm_t is not None and alarm_t >= onset
            if fa:  # look for a later valid alarm too
                run = 0; alarm2 = None
                for k in range(len(e)):
                    if e[k] < onset:
                        continue
                    run = run + 1 if above[k] else 0
                    if run >= persist:
                        alarm2 = e[k]; break
                det = alarm2 is not None; alarm_t = alarm2
            out[int(s)] = {"leak": True, "false_alarm": bool(fa), "detected": bool(det),
                           "delay_h": float((alarm_t - onset) * 0.25) if det else None}
    return out
