## 2. Dataset understanding and exploratory data analysis

**Partitions.** The dataset has three partitions: **training** (80 trips, used for fitting, cross-validation and tuning), **student-test** (10 trips, used only for development-stage comparison) and **evaluator** (10 reserved trips, never used in development, not analysed here). Each trip has a GPS track (`.track`: longitude, latitude, time), a road graph (`.nodes`, `.arcs`) and a reference route (`.route`: an ordered list of arcs, with **no per-point labels**).

### 2.1 How the data was preprocessed

1. **Per-trip projection to metres.** Trips come from {{data.utm_zones}} different UTM zones (longitude range of the raw tracks), so no single projection works. Each trip is projected with its own UTM zone (`src/pathsnap/data/geo.py`) so distances and headings are in metres.
2. **Candidate generation** (`src/pathsnap/candidates.py`, never reads the route): for every GPS point, the road arcs within **60 m** are found with a KD-tree on arc midpoints followed by exact point-to-segment distance; the **25 nearest** are kept.
3. **Labels for training only.** Because `.route` has no per-point labels, a monotonic dynamic-programming alignment of the GPS points to the *ordered* reference route assigns each point its true arc; a candidate is labelled 1 if it equals that arc. These are *derived* labels, not independent ground truth.
4. **Heading and speed** are estimated from neighbouring points but are not bridged across time gaps larger than 5 s.
5. **Data-quality handling.** Trips whose GPS track is displaced from the reference route are documented in `docs/data_quality/trip_offsets.md`, and the training tables can be rebuilt excluding them (`--exclude-trip`).

<div class="team" markdown="1">
**TEAM TO WRITE (Shwetha S):** in your own words, confirm or correct steps 1-5 above and add anything else you did to read, inspect and clean the raw data (for example the trips or rows you removed and why). Explain why a 60 m radius and a cap of 25 candidates were chosen (what evidence did you use?).
</div>

### 2.2 Missing-value and integrity analysis (training and student-test partitions)

{{@data_table}}

No missing, infinite or out-of-order values were found in either partition, and every arc and route entry refers to an existing node or arc. The data's real "gaps" are in **time**, not in values: most trips are sampled every second, but {{data.train.trips_with_gap_over_30s:.0f}} of 80 training trips contain at least one pause longer than 30 s (up to {{data.train.max_dt_s:,.0f}} s), which the decoder handles by cutting the chain (Section 5). The other kind of "missing" is a **candidate miss**: the reference arc is not among a point's candidates. On the student-test partition this happens for {{m.hgb_base.candidate_miss_rate:.2%}} of points.

### 2.3 Patterns observed and why these features

<div class="fig"><img src="../../results/eda/training_observations_per_trip.png" style="width:78%"/><div class="cap">Figure 2.1. GPS observations per trip (training partition). Most trips have a few thousand points; a few exceed 15,000.</div></div>

<div class="fig"><img src="../../results/eda/training_timestamp_intervals.png" style="width:78%"/><div class="cap">Figure 2.2. Time between consecutive GPS observations (training partition; the upper 1% is clipped for display). The sampling interval is almost always 1 s, with rare longer gaps.</div></div>

<div class="fig"><img src="../../results/eda/training_trip_00000001_trajectory.png" style="width:55%"/><div class="cap">Figure 2.3. Raw GPS trajectory of training trip 00000001 (longitude/latitude).</div></div>

Road-network geometry and GPS observations together on a map are in Section 6 (Figures 6.1-6.3, student-test partition).

**Offsets between GPS and reference route.** For training trips 00000033 and 00000054 the GPS track and the reference route have the same shape but are displaced by about 130-140 m (`docs/data_quality/trip_offsets.md`, reproduced by `scripts/diagnose_offset.py`). These two trips cause more than half of the candidate misses in the training set and are reported separately where relevant.

**Features.** The classifier uses 23 features per (GPS point, candidate arc) pair.

| Group | Feature | What it captures |
|---|---|---|
| Base | `dist_m` | distance from the GPS point to the arc (m) |
| Base | `frac_along` | position of the projected point along the arc (0 to 1) |
| Base | `arc_len_m` | arc length (m) |
| Base | `cos_heading` | agreement between the vehicle's heading and the arc direction |
| Base | `dist_rank`, `n_cand`, `min_dist_m`, `dist_gap_to_best` | how the arc compares with the other candidates of the same point |
| Base | `speed_mps`, `dt_prev_s`, `moving` | vehicle speed, time since the previous point, whether it is moving |
| Context | `dist_prev1`, `dist_next1`, `dist_prev2`, `dist_next2` | distance of the *same arc* from the neighbouring GPS points (NaN if it is not a candidate there or across a time gap) |
| Context | `along_prev1_m`, `along_next1_m`, `along_span2_m` | how far the vehicle moved *along* the arc between neighbouring points: positive on the arc it is driving along, negative on its opposite-direction twin |
| Context | `win_support`, `win_mean_dist`, `win_rank`, `win_gap_to_best` | how consistently the arc stays close over a 5-point window, and its rank within that window |
| Context | `cos_heading_win` | heading over a 7-point window, less noisy than the single-point heading |

The context features were added after an error analysis of the first system (Section 5.3): {{err.shares.reverse:pct}}% of its decoder errors were opposite-direction twin arcs and {{err.shares.adjacent:pct}}% were neighbouring arcs, and in {{err.predicted_arc_not_farther_than_label_arc_share:pct}}% of the errors the predicted arc was as close to the GPS point as the labelled one, which no single-point distance feature can separate.

<div class="team" markdown="1">
**TEAM TO WRITE (Shwetha S for the base features, Subramani for the context features):** explain in your own words why each group of features was chosen and what pattern in the data motivated it. Point to evidence (a plot, a number) for at least two features.
</div>
