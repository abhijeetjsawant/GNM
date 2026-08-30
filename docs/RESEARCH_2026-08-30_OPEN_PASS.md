# Open research pass — Sol, unconstrained, 2026-08-30

**Tag integrity, stated by Sol up front and recorded here unaltered: this pass
contains no `[W]` claims — nothing was opened.** Every literature item is `[R]`:
recall, named precisely enough that Fable can verify each in minutes. An `[R]` that
survives with a real citation graduates to `[W]` **then, not now.**

`[T]` grounded in this repo · `[R]` recalled, unverified · `[S]` speculative/novel

This pass was deliberately run **without** the previous brief's "refuse anything
that would not change a decision" constraint, which had suppressed exploration.

---

## 1. Camera count is the unpriced lever — and it is [T] from our own numbers

Four independent threads converge on the same variable, and **not one document in
this lane treats camera count as a design parameter**:

- CRLB is **33.2 mm** single-frame at 4 clean views;
- **35.3% of slots have only 2 eligible views**, where the inlier gate — our one
  proven tail defence — is provably helpless;
- association margins **halve** under contact;
- hands die at ~20 px crops, the one place resolution still mattered.

Every one of those improves superlinearly with 2–4 more cameras, **and we are
building an owned rig anyway.** `[R]`: 4 views is the *minimum* configuration in the
literature, not the typical one.

**MEASURED THE SAME DAY — `tools/swap-harness/camera_count.py`. Camera count is a
real lever of a specific and unspectacular size, and Sol's "superlinear" is refuted.**

Synthetic views placed on the ring fitted to the real rig, corrected noise
(clean 2.75 px / 7% bad at 36 px), single-frame CRLB:

| views | all clean | with outliers | vs 4 |
|---:|---:|---:|---:|
| **4 (real rig)** | **16.60 mm** | 17.66 mm | — |
| 4 *(synthetic — CONTROL)* | 16.66 mm | 18.08 mm | **PASS, 0%** |
| 6 | 13.58 mm | 14.26 mm | −18% |
| 8 | 11.76 mm | 12.32 mm | −29% |
| 12 | 9.60 mm | 10.00 mm | −42% |

*Control:* a synthetic 4-camera ring must reproduce the real rig's bound — it lands
within **0%**, so the placement is representative and the rows above are readable.

**It is exactly 1/√N, not superlinear.** 16.60·√(4/6) = 13.55 against 13.58
measured; 16.60·√(4/8) = 11.74 against 11.76. **This is textbook independent
averaging and nothing more** — the "improves superlinearly" claim is **REFUTED**.
Six cameras buy 18%, eight buy 29%. Real, worth having in a rig spec, not
transformative.

**And the redundancy half of the argument cannot be extrapolated at all — the model
fails its own control.** Measured per-camera eligibility is **p = 0.890** over
19,499 joint-camera slots. Assuming independence, that predicts only **6.2%** of
slots at ≤2 eligible views on four cameras. **The measured figure is 35.3% — a 5.7×
miss on the one case we can check.** So occlusion is strongly correlated across
views, the independence bound is not an estimate, and **"more cameras fixes the
starved-slot problem" is unsupported.** The correlation is the thing to measure, and
only owned multi-camera footage can measure it.

## 2. Epipolar heatmap fusion — attacks the measured bulk, no retraining

`[T]`: our SOMA-77 worker decodes heatmaps itself (`_decode`, the UDP nudge) — **the
heatmaps are already in hand.** `[R]`: AdaFuse (IJCV 2020) and Epipolar Transformers
(CVPR 2020) fuse per-view heatmap evidence *along epipolar lines* **before**
committing to an argmax.

Our pipeline commits to `(x, y, conf)` per view — an information bottleneck at
exactly the stage the remaining gap lives (**bulk 2.2×**, not tail).

**This is not the fusion door we closed.** That verdict was on fusing per-view *3D
MHR fits*; this is pre-commitment **2D evidence sharing**, a different mechanism.
The veto closure does not apply either — fusion *improves* estimates rather than
rejecting observations.
*Blindness:* helpless against view-consistent appearance bias — "agree and be wrong
together".
*Discriminating measurement:* A/B through the existing substitution harness, raw
decode vs fused decode, same frames, same denominator. Moderate plumbing
(crop-to-camera warps), **zero training**.

## 3. The unknown unknown: rolling shutter — a sixth absent term

`[T]`: the plan's absent-by-construction list carries **five** terms everywhere it
appears — calibration, distortion, sync, soft tissue, joint definition. **Rolling
shutter is in none of them**, and the Battle 2 rig spec's "fast-readout sensors"
line shows someone knew obliquely without it ever becoming a tracked error term.

`[R]`: consumer 4K readout runs ~8–25 ms, against our **≤5 ms sync budget**. Per-row
skew means genlocked cameras still sample head and feet at different instants; at
p95 1.9 m/s that is **tens of mm of pose-correlated, systematic distortion** — the
worst kind, because it does not look like noise.

*Discriminating measurement:* look up readout time for the actual camera models, then
one LED-strip or fan photograph per camera measures it directly.

## 4. Functional joint calibration (SCoRE / SARA) — feeds the schema work

`[R]`: the biomechanics standard — estimate joint **centres and axes** from a
per-performer range-of-motion trial rather than from a detector's joint definitions.
It yields bone lengths *and* joint centres **in our own convention**, attacking the
joint-definition half of the 18.6 px coherent bias, and it is exactly the input the
per-performer skeleton variant needs.

`[R]`: **Pose2Sim** is our architecture — detector → triangulate → model scaling →
IK — done in gait labs with published marker validation. The closest published
cousin we have; pull its accuracy tables and its licence.

## 5. Pre-register the marker-to-joint mapping NOW, or Battle 2 discovers definition offset as error

`[T]`+`[R]`: raw marker trajectories are **surface** positions. Joint centres come
from a regression protocol (Plug-in-Gait lineage) carrying its own 5–10 mm error and
a **third** joint convention. Our checklist correctly asks for raw trajectories — but
**nothing pre-registers how markers map to our 19 joints.**

This lane's signature defect, waiting at the finish line, on the one instrument that
was supposed to settle everything. **One page in the plan now is the whole fix.**

## 6. Sync and calibration from the humans themselves

`[R]`: spatiotemporal bundle adjustment (Vo et al., CMU, ~2016) estimates per-camera
**sub-frame time offsets** and extrinsics jointly from moving subjects. That means
our own footage can *measure* the sync residual we have never measured, and refine an
owned calibration — with **bone-length constancy** as the gate, which we already
pre-registered elsewhere.

## 7. Architecture verdict — and one tension nobody has framed

The shape — per-view 2D → associate → triangulate → solve → retarget — matches the
reference system's shape. **Nothing says tear it up.** Two honest deltas: the argmax
bottleneck (§2 fixes it in place), and this:

> `[T]` **Our sequence solve retreated to recovery-only because it moved good
> geometry 11–14 mm. MAMMA's fit moves *its* triangulation 9–11 mm — and ships at
> 13.5 mm.** Same phenomenon, opposite verdicts.

A **constrained full re-estimator** — bone lengths, temporal priors, all views,
`weight_before_loss` finally live — is unexplored. And the synthetic fixture **can**
price it: its documented blindness was specific to the *recovery-only path never
executing*, and a full re-estimator touches every slot. **That also finally makes the
uncommitted flag measurable.**

## 8. Target calibration, honestly

`[R]`: published markerless-vs-marker systems mostly land **25–40 mm**. Low-teens is
top-of-field, and at 13.5 mm the marker reference's *own* soft-tissue and placement
error is first-order — we would be comparing two noisy instruments. **The checklist's
ask for the vendor's own residual is load-bearing, not politeness.**

---

## Standing gates, unchanged and not displaced by any of this

Booking the marker session · performer releases covering ML training · archiving the
NVIDIA OML text. All three are the user's, and all three remain rank 0.
