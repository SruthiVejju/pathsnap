"""Neighbour-point ("context") features added on top of pathsnap.candidates.

The base features describe one GPS point in isolation. These describe the SAME candidate arc as seen
from the neighbouring GPS points of the same trip, which is what separates:
  * an arc from its opposite-direction twin   (does the vehicle move forward along the arc?)
  * an arc from a parallel / neighbouring arc  (does the arc stay close over a window of points?)

Like build_candidates this NEVER reads the reference route, so it is safe at inference time.
Neighbours across a time gap larger than MAX_GAP_S are ignored (their values stay NaN; the gradient
boosting models handle NaN natively).
"""
import numpy as np
import pandas as pd

from pathsnap.candidates import FEATURE_COLUMNS, RADIUS

MAX_GAP_S = 10.0
WINDOW = 3                      # points on each side for the windowed heading

CONTEXT_COLUMNS = [
    "dist_prev1", "dist_next1", "dist_prev2", "dist_next2",
    "along_prev1_m", "along_next1_m", "along_span2_m",
    "win_support", "win_mean_dist", "win_rank", "win_gap_to_best",
    "cos_heading_win",
]
EXTENDED_COLUMNS = [*FEATURE_COLUMNS, *CONTEXT_COLUMNS]


def _neighbour(df, offset, t, max_gap):
    """dist_m / frac_along of the same arc at point_idx + offset (NaN if absent or across a time gap)."""
    nb = df[["point_idx", "arc_id", "dist_m", "frac_along"]].copy()
    nb["point_idx"] = nb["point_idx"] - offset          # row for point i now carries point i+offset's values
    nb = nb.rename(columns={"dist_m": "d_nb", "frac_along": "f_nb"})
    out = df[["point_idx", "arc_id"]].merge(nb, on=["point_idx", "arc_id"], how="left")
    j = np.clip(df["point_idx"].to_numpy() + offset, 0, len(t) - 1)
    gap = np.abs(t[j] - t[df["point_idx"].to_numpy()]) > max_gap * abs(offset)
    d, f = out["d_nb"].to_numpy(float).copy(), out["f_nb"].to_numpy(float).copy()
    d[gap], f[gap] = np.nan, np.nan
    return d, f


def _window_heading(pxy, t, w=WINDOW, max_gap=MAX_GAP_S):
    """Unit heading from point i-w to i+w (clipped at the ends); zero vector when undefined or across a gap."""
    n = len(pxy)
    i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
    v = pxy[hi] - pxy[lo]
    norm = np.linalg.norm(v, axis=1)
    ok = (norm > 3.0) & ((t[hi] - t[lo]) <= max_gap * 2 * w)
    return np.where(ok[:, None], v / np.maximum(norm, 1e-9)[:, None], 0.0)


def add_context(df: pd.DataFrame, trip) -> pd.DataFrame:
    """Return df (candidate table of ONE trip, point_idx/arc_id/base features) plus CONTEXT_COLUMNS."""
    df = df.reset_index(drop=True)
    if df.empty:
        return df.assign(**{c: pd.Series(dtype=float) for c in CONTEXT_COLUMNS})
    t = np.asarray(trip.track[:, 2], dtype=float)
    L = df["arc_len_m"].to_numpy(float)

    d_p1, f_p1 = _neighbour(df, -1, t, MAX_GAP_S)
    d_n1, f_n1 = _neighbour(df, +1, t, MAX_GAP_S)
    d_p2, f_p2 = _neighbour(df, -2, t, MAX_GAP_S)
    d_n2, f_n2 = _neighbour(df, +2, t, MAX_GAP_S)
    f0 = df["frac_along"].to_numpy(float)

    out = df.copy()
    out["dist_prev1"], out["dist_next1"], out["dist_prev2"], out["dist_next2"] = d_p1, d_n1, d_p2, d_n2
    out["along_prev1_m"] = (f0 - f_p1) * L                 # metres moved forward along the arc since the previous point
    out["along_next1_m"] = (f_n1 - f0) * L
    out["along_span2_m"] = (f_n2 - f_p2) * L

    stack = np.column_stack([df["dist_m"].to_numpy(float), d_p1, d_n1, d_p2, d_n2])
    out["win_support"] = np.isfinite(stack).sum(1)
    filled = np.where(np.isfinite(stack), stack, RADIUS)    # an absent neighbour counts as "far" (= radius)
    out["win_mean_dist"] = filled.mean(1)
    g = out.groupby("point_idx")["win_mean_dist"]
    out["win_rank"] = g.rank(method="first")
    out["win_gap_to_best"] = out["win_mean_dist"] - g.transform("min")

    heading = _window_heading(trip.pxy, t)
    a, b = trip.nxy[trip.arcs[:, 0]], trip.nxy[trip.arcs[:, 1]]
    vec = (b - a)[out["arc_id"].to_numpy()]
    uvec = vec / np.maximum(np.linalg.norm(vec, axis=1), 1e-9)[:, None]
    out["cos_heading_win"] = (uvec * heading[out["point_idx"].to_numpy()]).sum(1)
    return out
