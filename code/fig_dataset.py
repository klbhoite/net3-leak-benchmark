"""Dataset figures: network layout with sensors/leak sites, example scenario, leak-size distribution."""
import json, os, numpy as np, wntr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, net_demand, DAY
from generate_dataset import INP
from analyze import PAL, GRAY, FIG  # shared style

X, Y, LF, CH, META = load()
sc = META["scenarios"]
wn = wntr.network.WaterNetworkModel(INP)
sensors = [c[2:] for c in CH if c.startswith("P_")]

# Fig. 1 network
fig, ax = plt.subplots(figsize=(6.3, 4.6))
for ln in wn.link_name_list:
    l = wn.get_link(ln); a = wn.get_node(l.start_node_name).coordinates; b = wn.get_node(l.end_node_name).coordinates
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#bdbcb6", lw=0.8, zorder=1)
leak_nodes = [s["node"] for s in sc if s["leak"]]
u, cnt = np.unique(leak_nodes, return_counts=True)
xy = np.array([wn.get_node(n).coordinates for n in u])
ax.scatter(xy[:, 0], xy[:, 1], s=4 + 3 * cnt, color=PAL[1], alpha=0.45, lw=0, zorder=2, label="Leak locations (size = count)")
sxy = np.array([wn.get_node(n).coordinates for n in sensors])
ax.scatter(sxy[:, 0], sxy[:, 1], marker="^", s=60, color=PAL[0], edgecolor="white", lw=0.8, zorder=4, label="Pressure sensors (10)")
for t in wn.tank_name_list:
    c = wn.get_node(t).coordinates; ax.scatter(*c, marker="s", s=60, color=PAL[2], edgecolor="white", zorder=4)
    ax.annotate(f"Tank {t}", c, xytext=(5, 4), textcoords="offset points", fontsize=7)
for r in wn.reservoir_name_list:
    c = wn.get_node(r).coordinates; ax.scatter(*c, marker="H", s=80, color=PAL[6], edgecolor="white", zorder=4)
    ax.annotate(r, c, xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.scatter([], [], marker="s", color=PAL[2], label="Tanks (level sensors)")
ax.scatter([], [], marker="H", color=PAL[6], label="Sources (inflow meters)")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
ax.legend(loc="lower left", fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_network.png"); fig.savefig(f"{FIG}/fig1_network.pdf"); plt.close(fig)

# Fig. 2 example scenario (abrupt leak with ~25 L/s)
cand = [s for s in sc if s["leak"] and s["kind"] == "abrupt" and 20 < s["mean_leak_flow_Ls"] < 35]
s = cand[0]; i = s["id"]; t = np.arange(X.shape[1]) / DAY
D = net_demand(X[i:i + 1], CH)[0, :, 0]
ip = CH.index("P_" + sensors[0])
il = CH.index("L_tank_3")
fig, axs = plt.subplots(3, 1, figsize=(6.3, 5.2), sharex=True)
for ax, v, lab in zip(axs, [X[i, :, ip], D, X[i, :, il]],
                      [f"Pressure P_{sensors[0]} (m)", "Net demand (L/s)", "Tank 3 level (m)"]):
    ax.plot(t, v, color=PAL[0], lw=0.9)
    ax.axvspan(s["start"] / DAY, t[-1], color=PAL[1], alpha=0.08, lw=0)
    ax.axvline(s["start"] / DAY, color=PAL[1], lw=1.2, ls="--")
    ax.set_ylabel(lab, fontsize=8)
axs[0].annotate(f"leak onset ({s['mean_leak_flow_Ls']:.1f} L/s, node {s['node']})", (s["start"] / DAY, axs[0].get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points", fontsize=7, color="#52514e")
axs[0].axvspan(0, 1, color=GRAY, alpha=0.10, lw=0); axs[0].annotate("reference day", (0.05, axs[0].get_ylim()[0]), xytext=(0, 4), textcoords="offset points", fontsize=7, color="#52514e")
axs[-1].set_xlabel("Time (days)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_example.png"); fig.savefig(f"{FIG}/fig2_example.pdf"); plt.close(fig)

# Fig. 3 leak size distribution by type
fig, ax = plt.subplots(figsize=(4.5, 3))
bins = np.logspace(np.log10(1), np.log10(120), 25)
for k, c in [("abrupt", PAL[0]), ("incipient", PAL[1])]:
    ax.hist([x["mean_leak_flow_Ls"] for x in sc if x.get("kind") == k], bins=bins, color=c, alpha=0.75, label=k, edgecolor="white", lw=0.5)
ax.set_xscale("log"); ax.set_xlabel("Mean leak outflow (L/s)"); ax.set_ylabel("Scenarios"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_leaksize.png"); fig.savefig(f"{FIG}/fig3_leaksize.pdf"); plt.close(fig)

# dataset statistics for the paper
dem = net_demand(X, CH)[..., 0]
fl = np.array([x["mean_leak_flow_Ls"] for x in sc if x["leak"]])
stats = {"n_scen": len(sc), "n_leak": int(sum(x["leak"] for x in sc)), "n_incipient": int(sum(x.get("kind") == "incipient" for x in sc)),
         "mean_net_demand_Ls": float(dem.mean()), "leak_flow_pct": np.percentile(fl, [0, 25, 50, 75, 100]).round(2).tolist(),
         "leak_flow_pct_of_demand_median": float(np.median(fl) / dem.mean() * 100),
         "diam_range_cm": [round(min(x["diameter_m"] for x in sc if x["leak"]) * 100, 2), round(max(x["diameter_m"] for x in sc if x["leak"]) * 100, 2)],
         "pos_fraction": float(Y.mean()), "sensors": sensors, "example": s}
json.dump(stats, open("../results/dataset_stats.json", "w"), indent=1); print(stats)
