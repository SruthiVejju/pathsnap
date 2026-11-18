"""Candidate generation, label alignment and features. 
build_candidates() NEVER uses the route -> safe to call at inference time."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADIUS = 60.0   # metres, tune with candidate miss rate
MAX_CAND = 25   # 25 matched the student-test recall of 30 with fewer candidates
FEATURE_COLUMNS = [
    "dist_m",
    "frac_along",
    "arc_len_m",
    "cos_heading",
    "dist_rank",
    "n_cand",
    "min_dist_m",
    "dist_gap_to_best",
    "speed_mps",
    "dt_prev_s",
    "moving",
]


def _seg_geom(trip):
    a = trip.nxy[trip.arcs[:, 0]]
    b = trip.nxy[trip.arcs[:, 1]]
    return a, b


def point_segment(p, a, b):
    """Distance, fraction along segment, for points p (M,2) vs segments a,b (M,2)."""
    ab = b - a
    L2 = (ab ** 2).sum(1)
    t = np.where(L2 > 0, ((p - a) * ab).sum(1) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    proj = a + ab * t[:, None]
    return np.sqrt(((p - proj) ** 2).sum(1)), t


def point_heading(pxy, timestamps, max_gap_s=5.0):
    """Estimate heading and speed without using GPS steps across large time gaps."""
    n = len(pxy)
    heading = np.zeros((n, 2), dtype=float)
    step = np.zeros(n, dtype=float)
    speed = np.zeros(n, dtype=float)
    intervals = np.diff(timestamps)

    for i in range(n):
        has_prev = i > 0 and 0 < intervals[i - 1] <= max_gap_s
        has_next = i < n - 1 and 0 < intervals[i] <= max_gap_s
        start = i - 1 if has_prev else i
        end = i + 1 if has_next else i
        if start == end:
            continue

        vector = pxy[end] - pxy[start]
        step[i] = np.linalg.norm(vector)
        elapsed = timestamps[end] - timestamps[start]
        speed[i] = step[i] / elapsed
        if step[i] > 1.0:
            heading[i] = vector / step[i]

    return heading, step, speed


def build_candidates(trip, radius=RADIUS, max_cand=MAX_CAND):
    a, b = _seg_geom(trip)
    mid = (a + b) / 2
    half = np.linalg.norm(b - a, axis=1) / 2
    tree = cKDTree(mid)
    rmax = radius + half.max()
    hit = tree.query_ball_point(trip.pxy, r=rmax)
    heading, step, speed = point_heading(trip.pxy, trip.track[:, 2])
    t = trip.track[:, 2]
    dt_prev = np.r_[1.0, np.diff(t)]

    rows = []
    for i, cand in enumerate(hit):
        if not cand:
            continue
        cand = np.array(cand)
        d, frac = point_segment(np.repeat(trip.pxy[i][None], len(cand), 0), a[cand], b[cand])
        keep = d <= radius
        if not keep.any():
            continue
        cand, d, frac = cand[keep], d[keep], frac[keep]
        order = np.argsort(d)[:max_cand]
        cand, d, frac = cand[order], d[order], frac[order]
        vec = b[cand] - a[cand]
        L = np.linalg.norm(vec, axis=1)
        uvec = vec / np.maximum(L, 1e-9)[:, None]
        cosh = uvec @ heading[i]
        for r, (arc, dd, ff, ll, ch) in enumerate(zip(cand, d, frac, L, cosh)):
            rows.append((trip.trip_id, i, int(arc), dd, ff, ll, ch, r, len(cand),
                         d[0], dd - d[0], speed[i], dt_prev[i], step[i] > 1.0))
    df = pd.DataFrame(
        rows,
        columns=["trip_id", "point_idx", "arc_id", *FEATURE_COLUMNS],
    )
    df["moving"] = df["moving"].astype(int)
    return df


def align_labels(trip):
    """Monotonic alignment of GPS points to the ORDERED reference route arcs.
    dp[i,j] = d[i,j] + min_{k<=j} dp[i-1,k]. Returns true arc id per point (len N)."""
    a, b = _seg_geom(trip)
    ra, rb = a[trip.route], b[trip.route]
    N, M = len(trip.pxy), len(trip.route)
    back = np.zeros((N, M), dtype=np.int32)
    prev, _ = point_segment(
        np.repeat(trip.pxy[0][None], M, axis=0), ra, rb
    )
    for i in range(1, N):
        cm = np.minimum.accumulate(prev)
        arg = np.zeros(M, dtype=np.int32)
        best = 0
        for j in range(M):
            if prev[j] < prev[best]:
                best = j
            arg[j] = best
        back[i] = arg
        distance, _ = point_segment(
            np.repeat(trip.pxy[i][None], M, axis=0), ra, rb
        )
        prev = distance + cm

    j = int(np.argmin(prev))
    idx = np.zeros(N, dtype=int)
    for i in range(N - 1, -1, -1):
        idx[i] = j
        j = back[i, j]
    true_arc = trip.route[idx]
    true_distance, _ = point_segment(
        trip.pxy, a[true_arc], b[true_arc]
    )
    return true_arc, true_distance


def add_labels(df, trip):
    true_arc, true_d = align_labels(trip)
    df = df.copy()
    df["true_arc"] = true_arc[df["point_idx"].values]
    df["label"] = (df["arc_id"] == df["true_arc"]).astype(int)
    return df.drop(columns="true_arc"), true_arc, true_d
