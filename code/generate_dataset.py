"""
Generate a reproducible, simulated leak/anomaly dataset for a looped water
transmission/distribution network (EPANET 'Net3' benchmark) with WNTR.

Every scenario is a 7-day extended-period simulation at 15-min resolution
under pressure-driven demand (PDD). Realism is added through
  * scenario-level demand scaling and per-junction base-demand perturbation,
  * per-junction, temporally correlated (AR(1)) demand noise,
  * randomised initial tank levels (pump controls in Net3 then react),
  * leaks modelled with the orifice equation (WNTR leak model, Cd = 0.75),
    either abrupt (burst) or incipient (area grows linearly over 6-48 h),
  * Gaussian sensor noise on SCADA-like channels.

Channels (16): pressure at 10 junctions, inflow from the two sources
(River main, pipe 60; Lake pump, link 10), flow through booster pump 335,
and water level in the 3 tanks.

Usage: python generate_dataset.py --n_leak 500 --n_free 200 --out ../data/net3_leak.npz
"""
import argparse, json, os, warnings
import numpy as np
import wntr
from multiprocessing import Pool

warnings.filterwarnings("ignore")
INP = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp")
if not os.path.exists(INP):
    import glob
    INP = glob.glob("/usr/**/wntr/library/networks/Net3.inp", recursive=True)[0]

STEP = 900                 # 15 min
DAYS = 7
T = DAYS * 24 * 3600 // STEP   # 672 samples
DMIN, DMAX = 0.015, 0.07   # leak orifice diameter range (m); set by --dmin/--dmax
FLOW_LINKS = ["60", "10", "335"]   # River main, Lake pump, booster pump


def select_sensors(n=10, seed=0):
    """Pick n junctions spread over the network: k-means on node coordinates,
    the junction closest to each centroid is the sensor (deterministic)."""
    wn = wntr.network.WaterNetworkModel(INP)
    names = [j for j in wn.junction_name_list if wn.get_node(j).base_demand >= 0]
    xy = np.array([wn.get_node(j).coordinates for j in names], dtype=float)
    rng = np.random.default_rng(seed)
    c = xy[rng.choice(len(xy), n, replace=False)]
    for _ in range(100):
        lab = np.argmin(((xy[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([xy[lab == k].mean(0) if np.any(lab == k) else c[k] for k in range(n)])
    chosen = []
    for k in range(n):
        d = ((xy - c[k]) ** 2).sum(1)
        for idx in np.argsort(d):
            if names[idx] not in chosen:
                chosen.append(names[idx]); break
    return chosen


def build_model(rng):
    wn = wntr.network.WaterNetworkModel(INP)
    wn.options.time.duration = DAYS * 24 * 3600
    wn.options.time.hydraulic_timestep = STEP
    wn.options.time.report_timestep = STEP
    wn.options.time.pattern_timestep = STEP
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.required_pressure = 20.0   # m
    wn.options.hydraulic.minimum_pressure = 0.0

    global_scale = rng.uniform(0.85, 1.15)
    # weekend demand is lower; the record starts on a random weekday
    start_wd = int(rng.integers(7))                 # 0 = Monday
    wd = (start_wd + np.arange(T) // 96) % 7
    week = np.where(wd >= 5, rng.uniform(0.85, 0.97), 1.0)
    for jn in wn.junction_name_list:
        j = wn.get_node(jn)
        for d in j.demand_timeseries_list:
            if d.base_value <= 0:
                continue
            pat = d.pattern if d.pattern is not None else wn.get_pattern(wn.options.hydraulic.pattern or "1")
            hourly = np.array(pat.multipliers, dtype=float)
            base = np.tile(np.repeat(hourly, 3600 // STEP), int(np.ceil(T / (len(hourly) * 4))) + 1)[:T]
            # AR(1) demand noise, phi = 0.9, marginal sd ~ 8 %
            e = rng.normal(0, 0.08 * np.sqrt(1 - 0.81), T)
            ar = np.zeros(T)
            for t in range(1, T):
                ar[t] = 0.9 * ar[t - 1] + e[t]
            mult = np.clip(base * week * (1 + ar), 0, None)
            pname = f"p_{jn}"
            wn.add_pattern(pname, mult.tolist())
            d.pattern_name = pname
            d.base_value = d.base_value * global_scale * rng.uniform(0.85, 1.15)
    for tn in wn.tank_name_list:
        tk = wn.get_node(tn)
        lo, hi = tk.min_level, tk.max_level
        tk.init_level = float(np.clip(tk.init_level + rng.uniform(-0.15, 0.15) * (hi - lo), lo + 0.1, hi - 0.1))
    wn._start_weekday = start_wd
    return wn


def run_scenario(args):
    sid, has_leak, seed, sensors = args
    rng = np.random.default_rng(seed)
    wn = build_model(rng)
    meta = {"id": sid, "leak": bool(has_leak), "seed": int(seed), "start_weekday": wn._start_weekday}
    leak_area_ts = np.zeros(T)
    if has_leak:
        cand = [j for j in wn.junction_name_list]
        node = cand[rng.integers(len(cand))]
        diam = float(np.exp(rng.uniform(np.log(DMIN), np.log(DMAX))))  # log-uniform equivalent orifice diameter
        area = np.pi * diam ** 2 / 4
        start = int(rng.integers(24 * 4, 120 * 4))       # sample index, day 2-5
        kind = "incipient" if rng.random() < 0.5 else "abrupt"
        ramp = int(rng.integers(6 * 4, 48 * 4)) if kind == "incipient" else 0
        meta.update(node=node, diameter_m=diam, area_m2=area, start=start, kind=kind, ramp=ramp)
        j = wn.get_node(node)
        if kind == "abrupt":
            j.add_leak(wn, area=area, discharge_coeff=0.75, start_time=start * STEP)
            sim = wntr.sim.WNTRSimulator(wn)
            res = sim.run_sim()
            P = res.node["pressure"]; Q = res.link["flowrate"]; H = res.node["head"]; LD = res.node["leak_demand"][node]
            leak_area_ts[start:] = area
        else:
            # piecewise-constant growth in 1-h increments, simulated in chunks
            j.add_leak(wn, area=1e-9, discharge_coeff=0.75, start_time=0)
            sim = wntr.sim.WNTRSimulator(wn)
            parts = []
            j._leak_area = 1e-9
            wn.options.time.duration = start * STEP
            parts.append(sim.run_sim())
            n_chunks = int(np.ceil(ramp / 4))
            for k in range(1, n_chunks + 1):
                a = area * min(1.0, k * 4 / ramp)
                j._leak_area = a
                end = min(start + k * 4, T)
                leak_area_ts[start + (k - 1) * 4:end] = a
                wn.options.time.duration = end * STEP
                parts.append(sim.run_sim())
                if end >= T:
                    break
            if start + ramp < T:
                j._leak_area = area
                leak_area_ts[start + ramp:] = area
                wn.options.time.duration = DAYS * 24 * 3600
                parts.append(sim.run_sim())
            import pandas as pd
            P = pd.concat([r.node["pressure"] for r in parts]); P = P[~P.index.duplicated(keep="last")]
            Q = pd.concat([r.link["flowrate"] for r in parts]); Q = Q[~Q.index.duplicated(keep="last")]
            H = pd.concat([r.node["head"] for r in parts]); H = H[~H.index.duplicated(keep="last")]
            LD = pd.concat([r.node["leak_demand"][node] for r in parts]); LD = LD[~LD.index.duplicated(keep="last")]
    else:
        res = wntr.sim.WNTRSimulator(wn).run_sim()
        P = res.node["pressure"]; Q = res.link["flowrate"]; H = res.node["head"]; LD = None

    P = P.iloc[:T]; Q = Q.iloc[:T]; H = H.iloc[:T]
    if len(P) < T:
        return None
    tanks = wn.tank_name_list
    level = H[tanks].values - np.array([wn.get_node(t).elevation for t in tanks])
    X = np.concatenate([P[sensors].values, Q[FLOW_LINKS].values * 1000.0, level], axis=1)  # m, L/s, m
    # sensor noise
    n_p = len(sensors)
    X[:, :n_p] += rng.normal(0, 0.15, (T, n_p))
    nf = len(FLOW_LINKS)
    X[:, n_p:n_p + nf] *= (1 + rng.normal(0, 0.01, (T, nf)))   # 1 % flow-meter noise
    X[:, n_p + nf:] += rng.normal(0, 0.02, (T, len(tanks)))      # 2 cm level noise
    if LD is not None:
        ld = LD.values[:T] * 1000.0
        y = (leak_area_ts > 1e-8).astype(np.int8)
        meta["mean_leak_flow_Ls"] = float(ld[y == 1].mean()) if y.sum() else 0.0
        meta["final_leak_flow_Ls"] = float(ld[-1])
    else:
        ld = np.zeros(T); y = np.zeros(T, np.int8)
    return sid, X.astype(np.float32), y, ld.astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_leak", type=int, default=500)
    ap.add_argument("--n_free", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="../data/net3_leak.npz")
    ap.add_argument("--dmin", type=float, default=0.015)
    ap.add_argument("--dmax", type=float, default=0.07)
    a = ap.parse_args()
    global DMIN, DMAX
    DMIN, DMAX = a.dmin, a.dmax
    sensors = select_sensors(10)
    rng = np.random.default_rng(a.seed)
    flags = np.array([1] * a.n_leak + [0] * a.n_free)
    rng.shuffle(flags)
    seeds = rng.integers(0, 2**31 - 1, len(flags))
    jobs = [(i, int(f), int(s), sensors) for i, (f, s) in enumerate(zip(flags, seeds))]
    with Pool(a.workers) as pool:
        out = [o for o in pool.imap(run_scenario, jobs, chunksize=4) if o is not None]
    out.sort(key=lambda o: o[0])
    X = np.stack([o[1] for o in out]); Y = np.stack([o[2] for o in out]); LD = np.stack([o[3] for o in out])
    metas = [o[4] for o in out]
    wn = wntr.network.WaterNetworkModel(INP)
    channels = [f"P_{s}" for s in sensors] + ["Q_in_River", "Q_in_Lake", "Q_pump_335"] + [f"L_tank_{t}" for t in wn.tank_name_list]
    np.savez_compressed(a.out, X=X, Y=Y, leak_flow=LD, channels=np.array(channels))
    with open(a.out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({"channels": channels, "step_s": STEP, "days": DAYS, "scenarios": metas}, f, indent=1)
    print("saved", X.shape, "positive fraction", Y.mean())


if __name__ == "__main__":
    main()
