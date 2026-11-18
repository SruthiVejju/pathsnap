# Data-quality note: GPS vs reference-route offsets in training trips

**Partition analysed:** training (trips 00000033, 00000054, 00000002). Student-test trips were checked only via
`results/explore_route_student_test.csv` (see "Student-test partition").
**Reproduce:** `python scripts/diagnose_offset.py 00000033 00000054 00000002`
(output saved in `results/offset_diagnosis.txt`). Projection: per-trip UTM (`src/pathsnap/data/geo.py`).
**Status:** findings are measured; the handling decision (section 5) is **proposed, not final**.

> This diagnostic uses the reference route and is an offline data-quality check only.
> It is not part of the prediction pipeline and must never be used as a feature or at inference.

## 1. Problem
The candidate table shows a high candidate miss rate for a few training trips (mean over trips 2.74%;
trip 2 = 11.8%, trip 33 = 66.8%, trip 54 = 90.4%). The question is whether this is a candidate-generation
problem (radius/cap too small) or a property of the data.

## 2. Measurements

| Trip | Median dist. to nearest route arc | Miss at 60 m / top-15 | Contiguous miss runs | After removing one constant shift |
|---|---|---|---|---|
| 00000033 | 122.4 m (p90 139.7) | 66.6% | 14 (longest 3,036 pts) | median 135.3 m -> **6.8 m** |
| 00000054 | 138.6 m (p90 196.8) | 88.3% | 42 (longest 9,098 pts) | median 149.4 m -> 23.3 m |
| 00000002 | 5.7 m (p90 97.8) | 11.7% | 7 (longest 541 pts) | 207.6 m -> 64.6 m (no constant shift) |

(Miss rates here are a proxy computed as "nearest reference-route arc is beyond 60 m, or ranked below the
15th closest arc"; they match the candidate-table figures to within 0.2 points for trip 33.)

Offset between each affected GPS point and its nearest route arc (sample of about 400 points per trip):

| Trip | Mean offset (east, north) | Spread (std, east/north) | Distance from affected GPS points to nearest *any* arc |
|---|---|---|---|
| 00000033 | (6, 131) m | (12, 23) m | median 6.2 m |
| 00000054 | (14, -139) m | (61, 22) m | median 87.6 m |
| 00000002 | (-75, 199) m | (65, 173) m | median 57.2 m |

## 3. Interpretation
- **Trips 33 and 54: the GPS track and the reference route have the same shape and size but are displaced by
  about 130-140 m, mostly north-south, with opposite signs.** The mean offset is much larger than its spread,
  and removing one constant shift brings the median distance down to 6.8 m (trip 33) and 23.3 m (trip 54).
  Bounding boxes of track and route agree within about 50 m. This points to a systematic displacement between
  the track and the road network/route, not random GPS noise.
- **This is not a candidate-generation problem.** The true arc is about 130 m away, so a larger radius would
  add many wrong candidates without recovering the reference arc.
- **Which side is displaced is not established.** In trip 54 the GPS is about 88 m from *any* road, which
  suggests the track is displaced. In trip 33 the GPS lies on real roads (6 m away), probably parallel streets,
  so the evidence does not say whether the track or the route is displaced.
- **Only parts of trip 33 are affected** (the first 626 points are fine, misses come in blocks).
- **Trip 2 is a different pattern:** most points are close to the route (median 5.7 m) but there are blocks of
  large, non-constant deviations (spread 173 m), consistent with GPS jumps or dropouts. Separately, 8.5% of its
  points have the reference arc ranked below the 15th closest arc: almost every road appears as two directed
  arcs, so a cap of 15 covers only about 7 distinct roads in dense areas. A larger cap may help trip 2 but not 33 or 54.

## 4. Limitations
- Shift statistics use about 400 evenly spaced points per trip, not all points.
- The route is the supplied reference, not independently verified ground truth; this analysis cannot say
  whether the reference or the GPS is the erroneous one in trip 33.
- Three trips were analysed in detail; the other training trips were not checked for blocks of offset.
- Constant-shift removal is a diagnostic only and is not applied to any data.

## 5. Proposed handling (to confirm with the instructor)
1. Keep trips 33, 54 and 2 in the EDA and miss-rate records; do not exclude them silently.
2. Report training/CV metrics **both with and without** these trips, so the effect is visible.
3. Do not correct the shift: estimating it needs the reference route, which is unavailable at inference.
4. Ask the instructor whether to exclude trips 33 and 54 from model fitting.
5. Try a larger candidate cap (e.g. 25-30) and report the miss-rate change for trip 2 and the others.

## 6. Student-test partition
In `results/explore_route_student_test.csv` the worst median GPS-to-route distance is 6.8 m and the largest
time gap is 210 s, so none of the offset behaviour above appears there. The hidden evaluator trips have not
been examined (and must not be used during development).
