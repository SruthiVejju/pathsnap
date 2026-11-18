"""Diagnose why some training trips have a high candidate miss rate.

For each trip it reports (training partition, per-trip UTM projection):
  A. distance from every GPS point to the nearest arc of the REFERENCE route
  B. miss rate under radius-only / top-K-cap-only / both rules (a proxy for candidate misses)
  C. for ~400 evenly spaced points >60 m from the route: distance to the nearest ANY arc, and
     the mean point->route offset vector; then the median distance again after removing that
     constant shift (tests "same shape, displaced" vs "different route").

NOTE: uses the reference route, so this is an offline data-quality diagnostic only. It must never
feed the prediction pipeline or the evaluator-facing code.

Usage: python scripts/diagnose_offset.py 00000033 00000054 00000002
"""
import sys

import numpy as np

from explore_route import load_trip, point_seg_dist
from pathsnap.data.geo import make_projector, project

RADIUS_M, CAP_K = 60.0, 15


def diagnose(tid: str, partition: str = "train"):
    t, n, a, r = load_trip(partition, tid)
    tf = make_projector(t[:, :2])
    N, P = project(tf, n), project(tf, t[:, :2])
    A, B = N[a[:, 0]], N[a[:, 1]]

    dr = np.empty(len(P))
    rank = np.empty(len(P), int)        # arcs strictly closer than the closest route arc
    for s in range(0, len(P), 100):
        d = point_seg_dist(P[s:s + 100], A, B)
        best = d[:, r].min(1)
        dr[s:s + 100] = best
        rank[s:s + 100] = (d < best[:, None]).sum(1)
    bad = (dr > RADIUS_M) | (rank >= CAP_K)
    idx = np.where(bad)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []

    print(f"== trip {tid} ({partition}): points={len(P)} arcs={len(a)} route_arcs={len(r)}")
    print(f"   A. dist to nearest route arc: median={np.median(dr):.1f} p90={np.percentile(dr, 90):.1f} max={dr.max():.1f} m")
    print(f"   B. miss: radius {RADIUS_M:.0f} m only={(dr > RADIUS_M).mean():.1%}, top-{CAP_K} only={(rank >= CAP_K).mean():.1%}, "
          f"both={bad.mean():.1%}; {len(runs)} contiguous runs, longest={max((len(x) for x in runs), default=0)}")

    sample = np.arange(0, len(P), max(1, len(P) // 400))
    Ar, Br = A[r], B[r]
    far, vecs, d_any = [], [], []
    for i in sample:
        if dr[i] <= RADIUS_M:
            continue
        d_all = point_seg_dist(P[i:i + 1], A, B)[0]
        d_rt = point_seg_dist(P[i:i + 1], Ar, Br)[0]
        j = d_rt.argmin()
        ab = Br[j] - Ar[j]
        tt = np.clip(((P[i] - Ar[j]) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        vecs.append(Ar[j] + tt * ab - P[i])
        d_any.append(d_all.min())
        far.append(i)
    if not far:
        print("   C. no points farther than the radius in the sample\n")
        return
    V = np.array(vecs)
    m = V.mean(0)
    shifted = P[far] + m
    d_after = [point_seg_dist(shifted[k:k + 1], Ar, Br)[0].min() for k in range(len(far))]
    print(f"   C. sampled {len(sample)} pts, {len(far) / len(sample):.0%} beyond {RADIUS_M:.0f} m of the route")
    print(f"      distance to nearest ANY arc (those points): median={np.median(d_any):.1f} m")
    print(f"      offset point->route: mean=({m[0]:.0f},{m[1]:.0f}) m (east,north), |mean|={np.linalg.norm(m):.0f} m, "
          f"std=({V[:, 0].std():.0f},{V[:, 1].std():.0f}) m")
    print(f"      median dist to route: before shift removal={np.median(dr[far]):.1f} m, after={np.median(d_after):.1f} m\n")


if __name__ == "__main__":
    for tid in sys.argv[1:] or ["00000033", "00000054", "00000002"]:
        diagnose(tid)
