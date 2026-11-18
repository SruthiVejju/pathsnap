## 5. HMM and Viterbi route reconstruction

Code: `src/pathsnap/hmm/viterbi.py` (decoder), `src/pathsnap/hmm/graph.py` (road graph), `src/pathsnap/hmm/metrics.py` (metrics), `scripts/tune_decoder.py` and `scripts/train_context_model.py` (tuning).

**1. Hidden states.** For each GPS point the state is one of that point's *candidate road arcs* (directed segments, at most 25 within 60 m). Because states are arcs, the candidate-to-road-segment correspondence is preserved: the decoder outputs an arc id per GPS point.

**2. Emission.** The log of the classifier's probability that the candidate is the true arc: `emit(t, j) = emission_weight * log P(arc j is true | features)`. The classifier gives a *posterior*, not a likelihood, so this is a practical scoring rule rather than a strictly Bayesian emission model.

**3. Transition.** The cost of moving from arc *i* at point *t-1* to arc *j* at point *t* depends on directed arc-steps `h` through the road graph: staying on the same arc costs 0; a reachable arc costs `hop_penalty * h` (up to `max_hops`); an arc that is not reachable costs `unreachable_penalty`; and turning around onto the opposite-direction twin arc costs `uturn_penalty` (the breadth-first search used for `h` never takes a free turnaround step). The log-transition score is the negative cost. These are costs, not normalised probabilities.

**Viterbi recurrence.** `delta_t(j) = emit(t, j) + max_i [ delta_{t-1}(i) - cost(i, j) ]`, with back-pointers `argmax_i`; the best path is recovered by backtracking from `argmax_j delta_T(j)`.

**4. Missing candidates and GPS gaps.** A GPS point with no candidate is skipped and gets the output −1; the chain bridges over it. If the time between consecutive points exceeds `gap_s` the chain is cut and each piece is decoded independently, so no transition is imposed across a long pause.

**5. Disconnected road segments.** Disconnection is a *soft* penalty, never a hard ban. If consecutive observations suggest arcs that are not connected within `max_hops`, the decoder pays `unreachable_penalty` and can still choose them. This keeps noisy or offset data (for example trips 00000033 and 00000054) from making the decoder fail, at the price of occasionally accepting a jump.

**Decoder settings.** Chosen by Bayesian optimisation (Optuna TPE, seed 42, {{hmm_run.trials}} trials) on the *training* trips using **out-of-fold** classifier probabilities, so the decoder is tuned against probabilities that were not produced by a model fitted on the same trips. Objective: mean over trips of pointwise accuracy on points whose true arc is among the candidates.

{{@hmm_params_table}}

<div class="fig"><img src="figures/fig_decoder_trials.png" style="width:80%"/><div class="cap">Figure 5.1. Decoder-tuning trials (training partition, out-of-fold probabilities). The first trial is the default setting.</div></div>

### 5.1 Did HMM decoding improve the route?

{{@hmm_table}}

*Pointwise accuracy* is the share of GPS points whose predicted arc equals the arc the point aligns to on the reference route, computed on points whose true arc is among the candidates, then averaged over trips. *Route continuity* is the share of consecutive predicted-arc changes that follow the directed road network.

<div class="fig"><img src="figures/fig_hmm_stages.png" style="width:96%"/><div class="cap">Figure 5.2. Classifier only versus classifier + HMM (default and tuned settings), on the training trips (out-of-fold) and the student-test trips.</div></div>

<div class="fig"><img src="figures/fig_per_trip_student.png" style="width:82%"/><div class="cap">Figure 5.3. Per-trip pointwise accuracy on the student-test partition: classifier only (blue) versus classifier + tuned HMM (orange).</div></div>

Yes. On the student-test trips the HMM raised accuracy in {{hurt.student_better:d}} of {{hurt.student_of:d}} trips and lowered it in {{hurt.student_hurt:d}}; the largest gain is trip {{hurt.student_best}} ({{hurt.student_best_gain:+.1f}} points). On the training trips it lowered accuracy in {{hurt.n:d}} of {{hurt.of:d}} trips. The classifier alone often picks a neighbouring or opposite-direction road for single points; the HMM removes isolated flips because they require expensive jumps.

### 5.2 What each component added

Accuracy points are cumulative; the 11-feature runs used a 60,000-row classifier fit for the out-of-fold probabilities, the 23-feature runs used all rows (a direct check showed this alone changes the final accuracy by under 0.1 points).

{{@ablation_table}}

### 5.3 Where decoding did not help

- **Weakest trips after decoding** (training, out-of-fold): {{hurt.weak}}. Trips 00000033 and 00000054 have a constant 130-140 m GPS-to-route offset (Section 2.3): the candidate set often does not contain the labelled arc at all, which no decoder can repair.
- **The four smallest changes** from adding the HMM: {{hurt.worst}}. The HMM is least useful on trips where the classifier is already near-perfect.
- **Remaining error types.** On the first (11-feature) system, a script-based error analysis (`scripts/error_analysis.py`, `results/report/error_analysis.json`) shows {{err.shares.reverse:pct}}% of wrong points are the opposite-direction twin of the labelled arc, {{err.shares.adjacent:pct}}% are a neighbouring arc within three steps, {{err.shares.other:pct}}% other; in {{err.predicted_arc_not_farther_than_label_arc_share:pct}}% of the errors the predicted arc is at least as close to the GPS point as the labelled one (within 3 m), so distance alone cannot separate them. That motivated the context features and the U-turn penalty.
- **Bridging across missed road geometry** (Figure 6.3): where the true route takes a short loop that the decoder skips, the predicted route draws a straight shortcut between non-adjacent arcs.
- **Sensitivity of the decoder settings.** {{sens.n}} of the {{sens.of}} tuning trials are within 0.1 accuracy point of the best, yet they span `gap_s` from {{sens.gap.0:.0f}} to {{sens.gap.1:.0f}} s and `uturn_penalty` from {{sens.uturn.0:.0f}} to {{sens.uturn.1:.0f}}. The objective is flat in those directions, so individual trips with long pauses can react to `gap_s` even though the training average does not.
- **Labels are derived.** Pointwise accuracy is measured against the monotonic alignment, not against independent ground truth, so it may be slightly optimistic or pessimistic for points near arc boundaries.

<div class="team" markdown="1">
**TEAM TO WRITE (Subramani B M):** in your own words, explain the Viterbi recurrence and the three cost terms (hop, unreachable, U-turn) as you would to the evaluator, and say which failure case in 5.3 you would fix first and how.
</div>
