"""Dataset inventory + graph stats + plots.  usage: python scripts/eda.py <Student folder> <out_dir>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.data import load_trip, list_trips

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
for part in ["train", "student_test"]:
    for d in list_trips(os.path.join(root, part)):
        t = load_trip(d)
        tr = t.track; dt = np.diff(tr[:, 2])
        V, A = len(t.nodes), len(t.arcs)
        g = coo_matrix((np.ones(A), (t.arcs[:, 0], t.arcs[:, 1])), shape=(V, V))
        ncomp, _ = connected_components(g, directed=False)
        deg = np.bincount(t.arcs.ravel(), minlength=V)
        seg = np.linalg.norm(np.diff(t.pxy, axis=0), axis=1)
        rows.append(dict(
            partition=part, trip_id=t.trip_id, n_points=len(tr),
            duration_s=tr[-1, 2] - tr[0, 2], track_len_m=seg.sum(),
            frac_dt_1s=np.mean(dt == 1), max_dt_s=dt.max(), n_nonmonotonic_t=int((dt <= 0).sum()),
            n_nan=int(np.isnan(tr).sum()), n_nodes=V, n_arcs=A, n_components=ncomp,
            isolated_nodes=int((deg == 0).sum()), route_len=len(t.route),
            route_connected_frac=float(np.mean(t.arcs[t.route[:-1], 1] == t.arcs[t.route[1:], 0])),
            lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
            max_jump_m=seg.max()))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "dataset_inventory.csv"), index=False)
print(df.groupby("partition")[["n_points", "n_nodes", "n_arcs", "route_len", "max_dt_s", "n_nan", "n_components"]].describe().T.to_string())

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    s = df[df.partition == part]
    ax[0, 0].hist(s.n_points, bins=20, alpha=.6, label=part, color=c)
    ax[0, 1].hist(s.max_dt_s, bins=20, alpha=.6, label=part, color=c)
    ax[1, 0].scatter(s.n_arcs, s.route_len, alpha=.7, label=part, color=c)
    ax[1, 1].hist(s.track_len_m / 1000, bins=20, alpha=.6, label=part, color=c)
ax[0, 0].set_title("GPS points per trip"); ax[0, 1].set_title("Max time gap (s)")
ax[1, 0].set_title("Road arcs vs route length"); ax[1, 1].set_title("Trip length (km)")
for a in ax.ravel(): a.legend()
plt.tight_layout(); plt.savefig(os.path.join(out, "eda_overview.png"), dpi=150)

# GPS trajectories + route for 4 sample trips
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for a, d in zip(ax.ravel(), list_trips(os.path.join(root, "train"))[:4]):
    t = load_trip(d)
    p = t.nxy[t.arcs[t.route]]  # (R,2,2)
    for s in p: a.plot(s[:, 0], s[:, 1], "g-", lw=2)
    a.plot(t.pxy[:, 0], t.pxy[:, 1], "b.", ms=2); a.set_title(f"train trip {t.trip_id}"); a.set_aspect("equal")
plt.tight_layout(); plt.savefig(os.path.join(out, "eda_trajectories.png"), dpi=150)
print("saved to", out)
