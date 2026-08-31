# Head and neck — measured, 2026-08-31

**Status: the head solve is in the pipeline, does NOT uniformly pass, and the delivered
artifacts have NOT been regenerated.** Performer 0 clears all four bands; performer 1
misses P1's p95 by **1.18°** (§6g). That gap grew from 0.28° when a sharper reference frame
stopped hiding it. **§6h decomposes it**: our head scores 9.91° / 9.80° p95 against a
perfect head in the same frame — *identical on both performers* — so the pass/fail split is
the reference frame, not the head. On the shipped
configuration (§6f) performer 0 passes all four bands; performer 1 passes P1's median, P2,
P3 and P4 and misses **P1's p95 by 0.28°**. A capture *would now* ship a moving `Head`; the
`subject-*.body-track.npz` on disk predate the stage and still carry the identity
quaternion on every frame. **Regenerating them is the outstanding delivery step**, and
until it happens no user has received a solved head. The earlier "passes on both" was measured
on the prototype's rotations, not the pipeline's — corrected in §6f. Both degenerate controls fail, the oracle passes, and the fit's
head-on-torso spread matches the reference to within 2.5°. The solve is a pipeline stage —
`src/autoanim_gnm/head_orientation.py` — and `reconstruct_multiview` threads it into the
delivered track, so a capture now ships a head that moves. §9a records what wiring cost.
No
*shipping* head code was changed; the fit in §6a is a new instrument in `tools/head/`,
not on the delivery path. This is the
instrument pass the goal asked for — score MAMMA's head on this footage first, measure
`HeadEnd`, enumerate the integrated adapters on the same denominator — plus one finding
that was not in the plan and reframes the region.

Read `docs/BODY_LANE_PLAN.md` §0–2 and `docs/HEAD_FEET_HANDS_PLAN.md` before this.
Everything below is on the SOMA-77 four-camera clip: 150 frames, 30 fps, 1280×720,
two performers at ~5 m. Scripts are in `tools/head/`; outputs in
`artifacts/head-lane/`. §8 reproduces all of it.

---

## 0. The finding that was not in the plan

**The delivered head is a constant. It is welded to the torso, in one line, by the
retarget — and it passes a jitter comparison against MAMMA.**

`src/autoanim_gnm/commercial_multiview.py:1350`:

```python
for name in ("Spine", "Chest", "UpperChest", "Neck", "Head"):
    _set_world(local, world, frame, name, torso_world)
for name in ("LeftEye", "RightEye"):
    _set_world(local, world, frame, name, torso_world)
```

`torso_world` is built from pelvis→neck and the shoulder axis. The head, the neck, the
whole chest chain and both eyes are assigned it verbatim, so in the shipped
`subject-*.body-track.npz` the local rotations of `Chest`, `UpperChest`, `Neck`, `Head`,
`LeftEye` and `RightEye` are **exactly the identity quaternion on every frame of both
subjects**. The three head landmarks the pipeline does triangulate — `nose`, `left_eye`,
`right_eye` — are **never read** by `positions_to_body_track`.

Head orientation relative to the thorax, measured on the delivered track: **0.000000°**
median spread, maximum 2.7×10⁻⁶° — floating point, not motion.

**And here is why this matters more than the number.** Scored on frame-to-frame world
head rotation — the obvious head-quality statistic — the delivered head is
indistinguishable from MAMMA:

| frame-to-frame world head rotation | median | p95 | max |
|---|---:|---:|---:|
| MAMMA, **our** subject 0 | 1.32° | 3.44° | 4.80° |
| **ours (delivered), subject 0** | 2.22° | 6.87° | 13.93° |
| MAMMA, **our** subject 1 | 1.93° | **6.62°** | 11.47° |
| **ours (delivered), subject 1** | 2.56° | **6.64°** | 17.93° |

**On subject 1 our p95 is within 0.3% of MAMMA's** — 6.64° against 6.62°. A head that
carries no information whatsoever scores at parity on the metric a reasonable person
would reach for first. On subject 0 the same constant reads 2.0× MAMMA's p95, **so the
naive gate's verdict depends on which performer you happened to score** — which is worse
than a gate that fails cleanly. This is `BODY_LANE_PLAN.md` §1's *"no gate a constant can
pass"* rule arriving as a measurement rather than a warning: **the constant is what we
ship today**, and the negative control the head gate needs is not a construction — it is
`artifacts/commercial-multiview-soma77/subject-*.body-track.npz` as it stands.

> **⚠ MAMMA's subject indices are not ours, and the first version of this table had them
> crossed.** `body_id-00` is our subject **1**. Nothing in either output says so — both
> are two-element lists indexed 0 and 1 — and the swap is invisible in every per-subject
> statistic taken *separately*. It corrupts only the **pairings**, which is precisely
> what a parity table is. It was caught by §1's corroboration returning noise, not by
> inspection. Correspondence is now derived from 3D pelvis agreement in
> `tools/head/subject_map.py`: MAMMA's world frame **is** the camera-rig world frame
> (matched pairs agree to 41–55 mm; crossed pairs sit 1.36–1.38 m apart, a 25× margin),
> and the resolver asserts that margin rather than trusting it. **Any future comparison
> against MAMMA must go through it.**

> **Correction to `HEAD_FEET_HANDS_PLAN.md` §1c, which is not wrong but is about a
> different arm.** Its `c_head` 4.48° / p95 43.79° is a **local, parent-relative**
> rotation read from `fitted_0.gltf`, the momentum fit. The **delivered** path is the
> table above. The two must not be quoted as one pipeline: one is noisy, the other is
> frozen, and the fix for each is different.

*Instrument:* `tools/head/our_head_bar.py` composes the delivered `DETAILED_HUMANOID`
chain Root→Hips→Spine→Chest→UpperChest→Neck→Head from `local_rotations_xyzw`, using the
**same** composition and geodesic functions as the MAMMA bar. Confirmed independently by
reading the raw quaternions: identity on 150/150 frames, both subjects.
*Blind to:* accuracy. Both sides are tracking statistics of an estimate. What the shared
instrument makes legitimate is the *comparison*, not either number as truth.

**The feet carry the same defect and it is recorded here, not acted on.** Two lines
below, `for foot in ("LeftFoot", "RightFoot"): _set_world(..., torso_world)` — foot
world orientation is the torso frame too. Head first, per the plan's ordering.

---

## 1. The bar — MAMMA's own head on this footage

`HEAD_FEET_HANDS_PLAN.md` §7 requires the bar be measured rather than assumed. MAMMA's
retained fit carries `smplx_pose` (150, 165) = 55 joints × 3 axis-angle, so its head
*rigid* orientation reads out directly.

Keyed by **our** subject index throughout, via `subject_map.py` — see the warning in §0.

| | our subject 0 | our subject 1 |
|---|---:|---:|
| *(MAMMA's own label)* | *body_id-01* | *body_id-00* |
| head frame-to-frame, median / p95 | 1.32° / 3.44° | 1.93° / 6.62° |
| neck frame-to-frame, median / p95 | 1.62° / 3.82° | 2.12° / 6.52° |
| thorax (spine3) frame-to-frame, median / p95 | 1.86° / 4.15° | 2.17° / 5.82° |
| **head relative to thorax — spread about the take mean** | **16.33° median, 35.79° p95** | **13.26° median, 25.68° p95** |
| head relative to thorax — offset from identity, median | 20.71° | 21.91° |
| head relative to thorax — frame-to-frame travel, median | 1.06° | 1.46° |
| eye axis vs shoulder axis, median | 12.30° | 9.76° |
| **frames where those axes oppose** | **0 / 150** | **0 / 150** |

**The load-bearing row is head-relative-to-thorax spread.** MAMMA's *estimate* of this
footage contains **13–16° of median, ~29° of range**, head-on-torso motion. A head locked
to the chest explains 0.0° of it, so **the locked head is falsifiable against MAMMA on
this clip** and a tracking gate here is not vacuous. That had to be measured: had MAMMA
shown 2°, no head gate worth writing would have been possible on this fixture.

### Corroborating that spread with a second instrument — weakly positive, and it cost the ear axis its recommendation

The row above is MAMMA's *estimate*, produced with a temporal prior, so it was left
uncorroborated in the first draft. The corroboration has now been run: the **signed yaw
of the head's lateral axis relative to the shoulder axis, about world up**, computed
twice — once from MAMMA's eye axis and SMPL-X fit, once from **Apple Vision's ear axis
and independent triangulation**. Different detector, different estimator.

| | our subject 0 | our subject 1 |
|---|---:|---:|
| frames compared | 104 | 100 |
| **Pearson \|r\|** | **0.337** | **0.397** |
| shuffled-permutation control, \|r\| p95 | 0.197 | 0.215 |
| MAMMA head-yaw sd | 10.7° | 10.2° |
| **Apple Vision ear-yaw sd** | **21.2°** | **41.0°** |

**Both correlations clear their own shuffled control, so this is not nothing** — two
unrelated instruments do co-vary on head yaw, and MAMMA's spread is therefore not pure
fit drift. But r ≈ 0.34–0.40 is **11–16 % of shared variance**, which is corroboration of
the *existence* of head motion and not of its magnitude. Read the 13–16° as
*"independently supported that a signal exists"*, still not as a measured quantity.

**The second row is the one that changes a recommendation.** Apple Vision's ear *yaw*
scatters at **21–41°**, two to four times MAMMA's head yaw and **larger than the entire
head-on-torso spread it would be used to explain**. Driving head yaw from the ear axis
today would inject more noise than signal.

> **The instrument lesson, and it generalises past the head.** §4 ranks the ear axis on
> the stability of its *length*, and **length is nearly blind to the error that destroys
> direction.** For an axis roughly perpendicular to the camera's depth direction, a
> differential depth error δ between its two endpoints rotates it by ≈ `atan(δ/L)` —
> **first order** — while changing its length only by ≈ `L·(δ/L)²/2` — **second order**.
> At L = 160 mm, a 50 mm differential depth error is 17° of yaw and 8 mm of length. So a
> segment can hold its length to 11 % and still be useless as a direction. This sits
> alongside the lane's standing *"reprojection cannot score depth"*: **a length invariant
> cannot score direction.** Any axis proposed for orientation must be scored on its
> direction, against an independent reference — which is what this subsection is.
> *Instrument:* `tools/head/corroborate_bar.py`.
> *Blind to:* common-mode error. If both instruments inherit the same bias from the
> shared camera geometry they agree and are both wrong.

**MAMMA estimates no jaw.** `run_args.json` names the body model `smplx_locked_head`;
SMPL-X joint 22 is identically zero on all 300 subject-frames. Its eyes move ~2°, which
is regularisation, not gaze. MAMMA's head output is **rigid pose only** — see §5.

*Instrument:* `tools/head/mamma_head_bar.py`. The chain
pelvis(0)→spine1(3)→spine2(6)→spine3(9)→neck(12)→head(15) is verified against
`kintree_table` in the local `SMPLX_NEUTRAL.npz`, not assumed.
*Self-check:* the world inter-eye vector expressed in the composed head frame must be
constant, since eye joint positions depend only on the head's world rotation and a fixed
rest offset. Residual **0.00016 mm median**. The composition is right.
*Blind to:* accuracy, and to its own prior. MAMMA fits with temporal smoothing, so a
smooth head can be smoothly wrong. This bar is *tracking statistics of an estimate*, and
until the marker session no number here can be called correct.
*Never scored against anything named `gt_` in that tree* — CLAUDE.md records those as
byte-copies of `pred_`.

---

## 2. `HeadEnd` — measured, and it does not rescue the head

`HEAD_FEET_HANDS_PLAN.md` §2 calls `HeadEnd` (SOMA-77 index 7) *"the single most
promising head lead on this page"* and marks it UNMEASURED. It is now measured, on the
harness shape `FINGER_TRIANGULATION_GATE.md` established: **reprojection flatters, the
physical invariant decides.**

Segment length stability over the take, triangulated by production `triangulate_point`:

| segment | subject 0 sd | subject 1 sd | mean length | anatomical |
|---|---:|---:|---:|---|
| **Head → HeadEnd** | **66.5 %** | **115.0 %** | 194 / 201 mm | ~110–140 |
| Head → Jaw | 203.9 % | 272.2 % | 54 / 46 mm | ~60–90 |
| Neck2 → Head | 166.4 % | 169.7 % | 86 / 69 mm | ~80–110 |
| Chest → Neck1 | 6.2 % | 6.9 % | 309 / 267 mm | — |
| CONTROL upper arm L | 4.2 % | 16.6 % | 289 / 288 mm | ~270–290 |
| CONTROL forearm L | 8.9 % | 14.7 % | 275 / 267 mm | ~240–270 |
| CONTROL shin L | **2.6 %** | **2.5 %** | 395 / 406 mm | ~390–410 |
| INCUMBENT eye baseline | 142.1 % | 215.4 % | 70 / 83 mm | ~63 |

*Convention robustness, independently recomputed:* under an all-valid-pairs frame
convention rather than this table's common-frame mask, `Head → HeadEnd` reads **72.0 % /
125.6 %** and the shin control reads **2.57 % / 2.54 %**. The verdict does not turn on the
convention; the third significant figure does. Quote the range, not a single percent.

**The strongest statement here needs no control and no ratio, which is why it leads: a
rigid bone measures 194, 201 and 254 mm on three frame sets of the same two people,
against an anatomical 110–140, with a standard deviation that exceeds the anatomical
quantity itself.** That is not a measurement of a skull; it is the phalanx pattern from
`FINGER_TRIANGULATION_GATE.md` — 68–196 mm where anatomy says 25–45, standard deviations
exceeding means — reappearing one length scale up. `HeadEnd` is roughly twice as good as
the eye baseline and five to twenty-five times worse than the body controls, but those
multiples move with the control chosen and the frame set, and §4 explains why they should
not be leaned on.

**Do not read the rotation medians as quality.** On the shared frame set the SOMA skull
axis turns a median of only 2.22°/6.73° per frame — smoother than several controls —
while failing the invariant at 66 %/124 %. A slowly-varying bias looks smooth. Every
rotation number in this document travels with its invariant for that reason.

### Which landmark is at fault, and why the segments disagree with the points

A cross-view disagreement probe — triangulate each landmark from every supporting camera
*pair* and measure how far those independent solutions land from each other, which is
**blind to real motion** because it is a within-frame statistic.

> **The first version of this probe used the MAX over pairs and had to be replaced.**
> Max-over-pairs grows with the *number* of pairs, and camera support differs by
> landmark and by detector — 14 pairs per frame on subject 0 against 7–10 on subject 1.
> At identical noise the better-observed landmark reads worse. The median pair distance
> and the RMS about the centroid do not have that bias, and they are what is quoted.
> The old max column is kept below so the correction is traceable, and **the numbers it
> produced supported a stronger claim than the unbiased ones do.**

| SOMA-77, median pair distance (mm) | subject 0 | subject 1 | | RMS s0 | RMS s1 | | p95 s0 | p95 s1 | | old max s0 |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|
| Hips | 45.9 | 49.3 | | 35.7 | 31.3 | | 71 | 104 | | 91 |
| Chest | 45.4 | 41.2 | | 35.6 | 28.5 | | 76 | 104 | | 87 |
| Neck1 | 57.8 | 40.1 | | 59.9 | 31.4 | | 124 | 159 | | 153 |
| Neck2 | 57.3 | 46.7 | | 68.6 | 33.3 | | 122 | 212 | | 184 |
| Head | 62.1 | 48.6 | | 67.5 | 31.5 | | 146 | 283 | | 177 |
| **HeadEnd** | **69.5** | **51.1** | | 76.1 | 38.9 | | 203 | **607** | | 198 |
| Jaw | 74.2 | 49.0 | | 72.1 | 30.0 | | 176 | **667** | | 176 |
| eyes | 74.4 / 61.1 | 59.1 / 49.2 | | 71.9 / 47.5 | 39.6 / 34.9 | | 193 / 165 | **1697 / 1805** | | 175 / 128 |
| CONTROL LeftArm | 43.1 | 55.5 | | 46.7 | 37.5 | | 74 | 117 | | 123 |
| CONTROL forearm | 48.6 | 49.7 | | 40.1 | 36.5 | | 99 | 141 | | 100 |
| CONTROL hand | 38.3 | 59.6 | | 27.5 | 37.9 | | 106 | 94 | | 68 |
| CONTROL shin | 34.8 | 32.4 | | 24.2 | 22.1 | | 51 | 81 | | 57 |
| CONTROL foot | 29.1 | 28.2 | | 19.8 | 18.5 | | 43 | 76 | | 46 |

**The corrected reading, and it is weaker than the max column suggested: the head
chain's cross-view disagreement is modestly worse than the limbs at the median on
subject 0 (1.3–2.2×) and *indistinguishable from them on subject 1*. The separation is
almost entirely in the tail** — subject 1's eyes reach a p95 of 1697/1805 mm against
76–141 mm on every limb control. That is consistent with this lane's standing finding
that *the detector's error tail is the story* (`BODY_LANE_PLAN.md` §3), and it means the
head chain is not uniformly bad — it is occasionally catastrophic, which is worse for an
L2 estimator and is exactly what the segment-length variance in the table above records.

**And the apparent contradiction — Chest→Neck1 is 6.2 % stable while Neck1's own points
disagree by 153 mm — is the most useful mechanical fact on this page.** Both endpoints
share a **common-mode depth error**: rays from a camera cluster are near-parallel
relative to the feature, so the pair slides along the viewing direction *together*, and
the error cancels in their difference. Two consequences, and they point opposite ways:

- a **segment length or an axis direction is differential**, so common-mode depth noise
  cancels — which is exactly why an axis can carry orientation even when its endpoint
  positions are poor. This is the argument *for* §4's ear axis.
- therefore a segment that is **still** unstable, as every segment touching `Head`,
  `HeadEnd`, `Jaw` or an eye is, has error that is **not** common-mode. That is real,
  uncorrelated, per-landmark error, and no differencing removes it.

*Instruments:* `tools/head/headend_gate.py`, `tools/head/head_landmark_noise.py`.
*Blind to:* bias. A landmark can be cross-view consistent and rigid-body constant while
sitting systematically 30 mm from where it is named — the convention-offset term
`FITTER_PLAN.md` §4 says dominates on real footage.
*Denominator caution, stated because it limits the probe:* the pair spread exists only on
frames with ≥3 camera support, and that subset differs per landmark and per detector. It
ranks landmarks **within a detector**; it does not rank detectors against each other, and
§4 does not use it for that.

> **One column of `head_landmark_noise.py` is void by the script's own criterion, and is
> not quoted anywhere above.** Its second probe measures distance from `Neck1` to each
> head landmark, on the stated condition that `Neck1` first prove clean. `Neck1` does not:
> its own pair spread is 57.8 mm, worse than Hips and Chest. The `distance_to_Neck1`
> figures in `artifacts/head-lane/landmark-noise.json` are therefore **not per-landmark
> error** and must not be read as such. The mechanism is the common-mode one below, which
> is the finding that column was reaching for and states honestly.

### 2b. Where the head's failure actually lives — the 2D are fine

§7.3 flagged a **measured disanalogy** with the fingers and made the fit a hypothesis
rather than a route: the fingers *passed* cross-view reprojection at 1.2× the body
control, which localised their failure in depth, whereas §2's probe showed head
landmarks disagreeing across views. **That probe measures 3D spread, which cannot
separate a 2D disagreement from a depth ambiguity.** The 2D-native test — symmetric
epipolar distance per camera pair, which is *blind to depth by construction* — settles it.

One-sided epipolar distance at 1280 width, per landmark, on the pipeline's own
association, with each detector's own body landmarks as control:

| | subject 0 | subject 1 | | subject 0 | subject 1 |
|---|---:|---:|---|---:|---:|
| | **SOMA-77 median px** | | | **ratio to body control** | |
| Head | 1.58 | 2.48 | | **0.53×** | **0.77×** |
| HeadEnd | 2.09 | 2.36 | | **0.70×** | **0.74×** |
| Jaw | 1.56 | 2.30 | | 0.53× | 0.72× |
| eyes | 1.75 / 1.61 | 2.49 / 2.37 | | 0.59× / 0.54× | 0.78× / 0.74× |
| *body control* | *2.97* | *3.20* | | *1.00×* | *1.00×* |
| | **Apple Vision median px** | | | | |
| nose | 3.65 | 2.58 | | **0.57×** | **0.35×** |
| ears | 4.85 / 3.35 | 3.11 / 2.08 | | 0.76× / 0.52× | 0.43× / 0.29× |
| *body control* | *6.39* | *7.30* | | *1.00×* | *1.00×* |

**Every head landmark on both detectors is more epipolar-consistent than that detector's
own body landmarks — 0.29–0.78×, never above 1.0.** The four cameras agree on a single
3D head point *better* than they agree on a knee.

**So the head's failure is not in the detector. It is entirely depth.** Rays to a
120–160 mm feature at 5 m are near-parallel relative to that feature, so the 2D can be
excellent while independent triangulation scatters along the viewing direction — which is
precisely what §2's segment-length instability and §1's ear-yaw scatter are made of, and
precisely what a model-constrained multi-view fit repairs. The fingers passed this test at
1.2×; **the head passes it at 0.29–0.78×, more cleanly than the case that motivated the
fit in the first place.**

> **This retires the objection that made §7.3 a hypothesis.** A reviewer's warning that
> "cross-view-inconsistent 2D is a detector failure a constrained fit cannot fit through"
> was correct in principle and rested on the 3D probe, which conflates the two failures.
> The 2D-native test disagrees, and it is the one that can tell them apart.

*Instrument:* `tools/head/head_epipolar_gate.py`. `_epipolar_distance_px` in the pipeline
returns the **symmetric** distance — CLAUDE.md records the ratio at 1.962 — so every
figure above is halved to a one-sided pixel distance.
*Blind to:* error **along** the epipolar line, which is the depth direction. That is the
design: a failure here would mean the 2D disagree in the one direction depth cannot
explain. There is no such failure.

---

---

## 3. `landmarks_soma77` is already on disk — the re-run is not needed

`HEAD_FEET_HANDS_PLAN.md` §2 caution 2 states that answering this question needs *"a
**worker re-run**, not a re-read"*, because the retained observations were written after
the adapter dropped 60 of 77 joints. **That is false, and it is the load-bearing
correction of this session.**

`artifacts/soma77-full/work/{A001,B001,C001,D001}-observations.jsonl` carry
`landmarks_soma77` — **all 77 points, every person, every frame, all four cameras** —
alongside the 17-joint dict, exactly as `soma77_pose.py`'s docstring says they should.

- 600 person-frames, every array exactly 77 long.
- **20,043 identity checks, 0 mismatches**: every one of the 17 mapped joints is
  byte-identical to `landmarks_soma77[index]`, which is what proves the array is in
  SOMA-77 order rather than merely 77 long.
- The 2D values are **bit-identical** (60,129 values, max difference exactly 0.0) to
  `artifacts/commercial-multiview-soma77/work/*-soma77-observations.jsonl`, the file that
  produced the shipped tracks — which lacks `landmarks_soma77`. Same detector run, richer
  export.
- Running the real `reconstruct_multiview` on them reproduces the retained
  `raw_triangulated_world_positions_z_up_m` at **0.000000 mm with an identical NaN
  pattern**, and reproduces `valid_joint_fraction` and the median reprojection error in
  `run-report.json` exactly.

**So toes (70, 71, 75, 76), `Neck1` (4), `Jaw` (8), `HeadEnd` (7) and all 30 finger
joints are available for measurement right now**, on the shipped association, with no
detector re-run. §2 above is the first use of it.

*Instrument:* `tools/head/verify_soma77_retention.py`, `tools/head/associate.py`.
*Blind to:* whether the values are *correct*. This establishes indexing and provenance.

> **A trap banked while recovering the association.** Person `index` is **not** a
> cross-camera identity: taking person 0 in each camera as one subject lands **1.84 m**
> from the retained track on frame 0. And a hand-replication of `reconstruct_multiview`'s
> association loop — same associator, same gates, written from the source — drifted from
> the retained tracks by a median worst-joint **9–19 mm**, which would have been invisible
> in every head number downstream. It was discarded. `associate.py` wraps the **real**
> function instead and refuses to write unless the replay reproduces the retained track at
> 0.0 mm. *Replicating a pipeline to instrument it is a defect generator; wrap it.*

---

## 4. The three integrated adapters, same denominator

`HEAD_FEET_HANDS_PLAN.md` §2 requires this before anyone proposes a new detector.

| adapter | head landmarks on this footage |
|---|---|
| `soma77_pose.py` | `Head`(6), `HeadEnd`(7), `Jaw`(8), both eyes. **No ears** — SOMA-77 has none; `left_ear`/`right_ear` are schema-only, 0 frames. |
| `apple_vision_pose.swift` | nose, both eyes, **both ears** (lines 59–60). Already run for this window — its detections are the boxes SOMA-77 was driven from. |
| `mediapipe_pose.py` | nose, both eyes, **both ears** (indices 7/8). **Run here for the first time** on this footage, from the cached `pose_landmarker_heavy.task`. |

Each head axis is normalised against **both** of its own detector's body controls,
because the two normalisations disagree and quoting only the flattering one is this
lane's recurring defect.

**Apple Vision vs SOMA-77, 95 frames** — the stronger set, since it does not need
MediaPipe's coverage:

| axis | subject 0 sd | ÷ own shoulder | ÷ own shin | subject 1 sd | ÷ shoulder | ÷ shin |
|---|---:|---:|---:|---:|---:|---:|
| **AV ear axis** (160/185 mm) | **11.0 %** | **0.76×** | 2.08× | **33.3 %** | 1.41× | 4.01× |
| AV eye baseline | 29.3 % | 2.04× | 5.53× | 18.9 % | 0.80× | 2.28× |
| AV control — shoulder width | 14.4 % | — | — | 23.6 % | — | — |
| AV control — shin | 5.3 % | — | — | 8.3 % | — | — |
| **SOMA skull axis** (184/254 mm) | **61.4 %** | **5.96×** | **23.6×** | **134.0 %** | **5.19×** | **58.3×** |
| SOMA eye baseline | 20.9 % | 2.03× | 8.04× | 230.6 % | 8.94× | 100× |
| SOMA control — shoulder width | 10.3 % | — | — | 25.8 % | — | — |
| SOMA control — shin | 2.6 % | — | — | 2.3 % | — | — |

**All three detectors, 42 frames (subject 0) and 14 frames (subject 1)** — the full
intersection, which MediaPipe's coverage collapses. **Subject 1's 14 frames are too few
to conclude from and are printed only so nobody re-derives them as new:**

| axis, subject 0 | sd | ÷ shoulder | ÷ shin | rot p95 |
|---|---:|---:|---:|---:|
| **AV ear axis** | **7.6 %** | **0.57×** | **1.89×** | 14.0° |
| AV eye baseline | 24.2 % | 1.82× | 6.05× | 41.8° |
| MediaPipe ear axis | 24.8 % | 2.13× | 3.28× | **99.1°** |
| MediaPipe eye baseline | 26.5 % | 2.28× | 3.51× | 56.1° |
| SOMA skull axis | 65.9 % | 8.60× | **24.7×** | 4.5° |
| SOMA eye baseline | 18.4 % | 2.40× | 6.88× | 10.0° |

### The comparison that needs no control at all

**Ratios to a body control are a weak instrument and should not carry this section.**
Three reasons, and each was checked: `sd_pct` shrinks with segment length at fixed
millimetre noise, so a 184 mm skull axis is handicapped against a 326 mm shoulder width
before any physics; the control is not noise-matched across detectors — Apple Vision's
shoulders are among its *noisiest* landmarks, which inflates its denominator and flatters
its ratio; and biacromial width genuinely changes 2–4 cm with arm elevation, so part of
the control's own 10–26 % is biology rather than instrument.

The two head axes in question are **close to the same length**, so they can be compared
in millimetres directly, with no control and no normalisation:

| 95-frame set | segment length | **sd, mm** | sd % |
|---|---:|---:|---:|
| Apple Vision ear axis, subject 0 | 160 mm | **17.6** | 11.0 % |
| SOMA-77 skull axis, subject 0 | 184 mm | **112.8** | 61.4 % |
| Apple Vision ear axis, subject 1 | 185 mm | **61.7** | 33.3 % |
| SOMA-77 skull axis, subject 1 | 254 mm | **340.9** | 134.0 % |

**On segments within 15–37 % of the same length, Apple Vision's ear axis is 6.4× and 5.5×
tighter than SOMA-77's skull axis.** That is the cross-detector claim, and it survives
without choosing a control.

> **⚠ And it is a claim about LENGTH, which §1's corroboration shows is nearly blind to
> the error that matters for orientation.** Scored on *direction* against MAMMA, the same
> ear axis scatters at **21–41° of yaw** — larger than the whole head-on-torso signal.
> Everything in this section ranks *conditioning*, and conditioning is a reason to
> prefer the ears as an **input to a fit**; it is not a licence to drive head yaw from
> them directly. §7 carries the revised recommendation.

**What the ratios add, and all they add:** *Apple Vision's ear axis is the only head axis
here that stays inside its own detector's body-control range; MediaPipe's ears and
SOMA-77's skull axis both sit outside theirs.* The exact multiple depends on which control
is chosen — 0.57–0.76× by shoulders and 1.89–4.01× by shin for the ears — so no single
multiple is quotable.

> **A comparison that was drafted and removed: "the same class as the fingers' 4.4×".**
> It does not hold. `FINGER_TRIANGULATION_GATE.md`'s 4.4× came from a deliberately
> degraded harness whose own text says *"absolute numbers are not comparable to
> production"*, and its control was a pooled body mean, not a shoulder width. Carrying a
> ratio across two harnesses to make a class argument is numerology. Worse, under the
> shin normalisation the Apple Vision ear axis reaches 4.0× on subject 1 — so the
> "fingers class" would swallow the candidate this section recommends. **The
> finger gate's relevance here is its *method* — invariant over reprojection, null over
> plausibility — not its numbers.**

### The ears against a null that ships no ear evidence

`FINGER_TRIANGULATION_GATE.md`'s lesson is that temporally coherent and confidently
predicted is not the same as image-measured. Accepting the ears without testing that
would be §4's own caution ignored, so: per camera and per subject, predict each ear's 2D
from `nose`, both eyes and `neck` by least squares — a pure face template with no ear
evidence in it — fit on even frames and predicted on odd and vice versa, then push the
**predicted** ears through the identical triangulation and axis statistics.

| | subject 0 | subject 1 |
|---|---:|---:|
| held-out 2D residual, ears, median | **7.1 / 7.7 px** | **16.4 / 34.0 px** |
| ear axis, **observed** | 158.9 mm, sd **13.1 mm (8.3 %)**, rot p95 13.4° | 195.4 mm, sd **70.0 mm (35.8 %)**, rot p95 19.5° |
| ear axis, **null template** | 425.0 mm, sd **716.5 mm (168.6 %)**, rot p95 142.0° | 353.6 mm, sd **462.8 mm (130.9 %)**, rot p95 94.7° |

**The null is 20× worse on subject 0 and 6.6× worse on subject 1**, and the face template
misses the ear by 7–34 px — 42–201 mm at this depth's 5.92 mm/px. The detected ear is not
the face landmarks in disguise. The null is deliberately *generous*: it may learn a
different ear offset per camera and per subject, which a real crop-conditioned detector
could not. Beating a generous null is the strong direction.
*Instrument:* `tools/head/ear_null_template.py`.
*Blind to:* whether the ear is anatomically an ear. A detector measuring some real image
feature that is not an ear passes this.

**Three cautions, all load-bearing:**
1. **This does not say Apple Vision is the better detector.** It is not: SOMA-77's shin
   control is 2.3–2.7 % against Apple Vision's 5.3–12.1 %, and SOMA-77 was adopted in
   Battle 1 for exactly that reason. The finding is regional — **SOMA-77's body with
   somebody's ears**, not a detector swap.
2. **Coverage and the tail are Apple Vision's weaknesses here, and the tail is not
   small.** Its ears resolve on 107–127 of 150 frames at **2.35–2.73 cameras** of support
   and confidence 0.66–0.75, against SOMA-77's head chain at 3.5 cameras and 0.88–0.93.
   Their cross-view pair spread reaches a **p95 of 1129–1692 mm** — this lane's doctrine
   is that the tail is the story, and quoting only the medians would hide it. A
   well-conditioned axis on two thirds of frames is a promising input, not a head solve.
3. **The cross-view probe could not interrogate the ears as hard as it interrogated
   `HeadEnd`.** At 2.35–2.73 cameras of support the pair statistic exists on only 40 of
   115 and 93 of 127 ear frames, against nearly every frame for SOMA-77's head chain at
   3.5 cameras. So the instrument that rejected `HeadEnd` was **structurally blind on
   roughly half the ear frames**, which is the asymmetry `av_ear_probe.py`'s own docstring
   promised to avoid and did not fully achieve. The null-template gate above is what
   carries the positive claim instead, because it runs on every frame. The remaining
   untested route is a 2D-native one — symmetric epipolar distance per camera pair, which
   works on two-view frames and needs no support selection at all. Not run.

*Instruments:* `tools/head/same_denominator_head.py`, `tools/head/three_detector_head.py`,
`tools/head/adapter_head_comparison.py` (per-detector, superseded for cross-detector use).
*Blind to:* joint convention. An Apple Vision ear, a MediaPipe ear and a SOMA-77 `Head`
are different points under different definitions. Stability compares across conventions;
position does not. Nothing here says which is closer to anatomical truth.
*A sign test that was removed rather than reported:* an earlier pass scored the skull
**long** axis against the **shoulder** axis and produced "36/95 opposing". Those axes are
near-perpendicular, so the sign of their dot product is meaningless. The skull axis is
tested against neck-up instead, where opposition is genuinely impossible: 1/42 and 2/14.

---

## 5. Who owns head rigid pose — settle before building

`soma77_pose.py:79` says *"the head is GNM's anyway"*. That is the assumption this
session's §0 finding turns into a defect: the body lane emits a head orientation, it is a
constant, and nothing downstream complains.

**The measurement that bears on it:** MAMMA — the reference implementation of this
architecture — estimates head and neck **rigid** pose and estimates **no jaw at all**
(`smplx_locked_head`, joint 22 identically zero). Its face-lane equivalent is absent by
construction. That is direct evidence for the split:

> **SETTLED 2026-08-31 by the user, and this is now the rule:** the **body lane owns head
> and neck rigid world orientation**; the **face lane owns expression and gaze within
> that frame** (`a2f*.py`, `visual_face_retarget.py`, `visual_track.py`). **One writer
> per quantity.**

Three consequences that follow immediately, and each is now a defect rather than a
question:

1. **`soma77_pose.py:79`'s comment — *"the head is GNM's anyway"* — is no longer true**
   and should not be relied on by anything downstream. The body lane owns this.
2. **`commercial_multiview.py:1350` is the body lane declining to do its own job.** The
   `Head` and `Neck` channels in `BodyTrack` are the body lane's to fill, and it fills
   them with a constant.
3. **The face lane must not write head world orientation.** If it currently does, that is
   a second writer on an owned quantity and the disagreement is silent by construction.
   Not audited this session — named here so it is audited before the head solve lands.

This was a build blocker, not a measurement blocker: every number above stands either
way. It is now unblocked.

---

## 6. The gate — pre-registered, with two named failing controls

`BODY_LANE_PLAN.md` §1: no gate a constant can pass, and every band ships with a
demonstration that a degenerate solution fails it. §0 showed that **one** control is not
enough here: the locked head fails a spread test but a *noisy* head passes it. So the
head gate is pre-registered with **two** controls, both of which must demonstrably fail.

**Primary metric — head orientation relative to the thorax, tracked against MAMMA as
instrument.** Per frame, compose our world head and thorax rotations, form the relative
rotation, and score the geodesic angle against MAMMA's relative rotation on the same
frame, both computed by the same functions.

**Reference frames, amended 2026-08-31 — read this before reproducing the gate.**
*Thorax frame:* built from the pipeline's own **smoothed** torso positions
(`triangulated_world_positions_z_up_m`, joints `neck`/`root`/shoulders), **not** raw
triangulation. The first version used raw and it dominated P4 — see §6a. MAMMA's thorax
uses the same positional construction on its own `pred_joints`.
*Head frame:* ours from the fitted rotation, MAMMA's from its composed SMPL-X chain.
*P2's 8° derivation:* half of MAMMA's spread **as the gate computes it**, 14.96° / 15.99°
with the positional thorax — not the 13.3° / 16.3° chain-thorax figures quoted in §1,
which are a different construction. The two happen to give the same band; do not
re-derive it from the §1 pair.
*Fourth arm:* an **ORACLE** — MAMMA's head through our thorax — is mandatory, per §6a.

**Every threshold below is per-subject, and the subject labels are OUR indices.** MAMMA's
`body_id-00` is our subject 1; a gate spec with crossed labels mis-scores a candidate
just as silently as a crossed parity table did in §0. Resolve through `subject_map.py`.

| must pass | |
|---|---|
| P1 | agreement with MAMMA's head-relative-to-thorax, **median ≤ 8°, p95 ≤ 20°** — set at roughly half MAMMA's own measured median spread, **16.3° on subject 0 and 13.3° on subject 1**, so a candidate must explain more than half the signal |
| P2 | relative-pose **spread about the take mean ≥ 8°**, against MAMMA's **16.3° / 13.3°** (subject 0 / subject 1) |
| P3 | **0 frames** where the head axis opposes its own neck-up direction |
| P4 | frame-to-frame relative travel p95 **≤ 3× MAMMA's** — **2.40° on subject 0, 4.15° on subject 1** |

| must fail, and be shown to fail in the same report | |
|---|---|
| **C1 — the locked head** | today's delivered track. Spread 0.000000°, so **P2 fails by construction**, while its frame-to-frame world rotation *passes* a naive jitter gate at 6.87° against MAMMA's 6.62°. |
| **C2 — the noisy head** | independently triangulated SOMA-77 eye/skull axes. **P1 and P4 fail**: the eye axis alone turns 90–117° at p95 and the skull invariant fails at 66–134 %. |

| must PASS, or the gate is miscalibrated | |
|---|---|
| **ORACLE — MAMMA's own head through our thorax frame** | *"No gate a constant can pass"* has a dual: **a gate no oracle can pass is miscalibrated.** P1 compares two mean-removed relative rotations built on different thorax definitions, so a floor exists that no head can beat. If the oracle fails, the bands are measuring frame mismatch rather than head quality and must be re-derived — not the candidate's fault. Measured: it **passes**, at P1 5.46° / 4.87° median. |

A gate reporting P1–P4 without C1 and C2 beside them is not this gate. **C1 and C2 are
real artefacts on disk, not constructions** — the whole point of §0.

**What this gate cannot do, stated before anyone builds to it.** MAMMA is an
*instrument*, not truth, so P1 measures **agreement with MAMMA**, never accuracy. If
MAMMA's head is smoothly wrong, a candidate that matches it scores well and is equally
wrong. Until the Battle 2 marker session no head accuracy claim is supportable in either
direction (`BODY_LANE_PLAN.md` §1), and this gate must never be quoted as one. It is a
*parity* gate, which is exactly what the goal asked for and no more.

---

## 6a. The gate, run — it rejects both controls **and** the candidate

A first estimator was built and scored, because a gate nobody has run against a real
candidate is a specification, not an instrument.

**The candidate: a rigid-head multi-view, multi-frame fit.** Per subject, over the whole
take at once, solve for one rigid head **template** (head-local positions of SOMA-77's
`Head`, `HeadEnd`, `Jaw` and both eyes — the skull is rigid) plus a per-frame **rotation**
and **position**, minimising robust reprojection into all four calibrated cameras with a
temporal prior. **No head landmark is ever triangulated on its own**, which is §2b's
verdict turned into an estimator and the same architecture MAMMA uses.

| subject 0 · MAMMA spread 14.96°, travel p95 2.38° | P1 med | P1 p95 | P2 spread | P3 | P4 travel | verdict |
|---|---:|---:|---:|---:|---:|---|
| **ORACLE — MAMMA's head, our thorax** | 5.46° | 17.22° | 17.51° | — | 7.08° | **PASS** |
| **candidate — neck-anchored fit** | **7.54°** | **19.35°** | **17.45°** | **0** | **3.72°** | **PASS** |
| **C2 — noisy, per-frame triangulated** | 6.52° | 65.62° | 18.66° | — | 86.93° | **FAIL** (P1, P4) |
| **C1 — locked head, as delivered** | 14.96° | 30.76° | **0.00°** | — | 0.00° | **FAIL** (P1, P2) |
| *bands* | *≤ 8* | *≤ 20* | *≥ 8* | *= 0* | *≤ 7.14* | |

| subject 1 · MAMMA spread 15.99°, travel p95 4.17° | P1 med | P1 p95 | P2 spread | P3 | P4 travel | verdict |
|---|---:|---:|---:|---:|---:|---|
| **ORACLE — MAMMA's head, our thorax** | 4.87° | 17.59° | 15.88° | — | 7.32° | **PASS** |
| **candidate — neck-anchored fit** | **6.40°** | **16.07°** | **16.51°** | **0** | **4.66°** | **PASS** |
| **C2 — noisy, per-frame triangulated** | 7.60° | 88.68° | 18.84° | — | 98.65° | **FAIL** (P1, P4) |
| **C1 — locked head, as delivered** | 15.99° | 27.18° | **0.00°** | — | 0.00° | **FAIL** (P1, P2) |
| *bands* | *≤ 8* | *≤ 20* | *≥ 8* | *= 0* | *≤ 12.51* | |

## **The head passes the gate on both performers — read §6d before quoting it.**

Both subjects, all four bands, **at the same temporal weight (100) chosen by the same rule
— no per-subject tuning.** An independent adversarial audit (§6d) confirms the pass is
**not** an artifact of iterating against the gate: the pass region spans two orders of
magnitude of weight and the verdict is invariant across every sane selection rule. It also
lands four qualifications that belong in the same breath as the headline:

1. **P2 and P4 are knob settings for the candidate**, because the fit regularises exactly
   the quantity P4 measures, in the gate's own frame. They discriminate the *controls*;
   they are not evidence about the candidate. **The substance of the pass is P1.**
2. **Subject 0's P1 margin is a coin flip** — moving-block bootstrap 90 % CI
   [5.66°, 13.15°], P(median > 8°) = 0.48. The oracle's is 0.02, so the fragility is ours.
3. **P1's p95 barely discriminates** — the oracle sits at 17.2°/17.6° against a 20° band.
   The load-bearing number is the median.
4. **The neck anchor is decoration** — ablating it still passes. Beside them, both degenerate controls fail and fail
*differently*, and the oracle passes, so the bands are neither vacuous nor unreachable.

**And the fit's spread now matches the reference rather than approximating it:** 17.45°
against MAMMA's 14.96°, and 16.51° against 15.99°. It is explaining the head motion, not
damping it — which is the failure P2 exists to catch, and which earlier over-smoothed
versions of this fit were sliding toward. *The other reading of subject 0's 2.5°
overshoot is that some of it is residual noise dressed as motion; P1 passing bounds how
much, but the overshoot is not proof of extra signal.*

**Read the headline with the non-uniformity below.** A split-half check shows the fit
sits within 1.7° of the oracle on three of four half-takes and **8° away on subject 0's
first half**. The pass is a whole-take result and it is not uniform across the take.

**One number that looks anomalous and is not:** the candidate's P4 (3.72° / 4.66°) is
*lower* than the oracle's (7.08° / 7.32°). Expected — both inherit the same thorax frame,
but ours is smoothed against it while MAMMA's head meets it raw. P4 is largely a statistic
about the reference frame, per §6a.

> **What this is, stated exactly.** *Head-orientation parity with MAMMA on this footage,
> on a tracking metric, with both degenerate solutions demonstrably rejected and the
> instrument floor measured.* It is **not** an accuracy claim: MAMMA is an instrument, not
> truth, and `BODY_LANE_PLAN.md` §1 forbids reading any of this as one. If MAMMA's head is
> smoothly wrong, this fit is wrong with it. **And it is not shipped** — the fit lives in
> `tools/head/`, while `commercial_multiview.py:1350` still welds the delivered head to
> the torso. Passing the gate and delivering the head are two different things.

### What it took, in the order it mattered

| change | why it was made | effect |
|---|---|---|
| **fit, don't triangulate** | §2b: head 2D are epipolar-clean at 0.29–0.78× the body control, so the failure is depth | P1 p95 65–89° → 21–22° |
| **smooth the neck, not the world head** | anatomy: the head in world carries the torso's motion too | P1 p95 → 18.8° / 17.3°; P4 falls an order of magnitude |
| **anchor the head to the neck** | anatomy — but it is a *soft* prior at 10 mm that the data overrides by 12–18 mm median, so it damps rather than constrains | subject 1 p95 → 12.4° — **but ablation still passes (§6d), so this is decoration** |
| **fix the selection rule** | it was choosing a fit 2.935 px when 2.719 px sat on the same curve | subject 0 P1 8.91° → 7.54° |

**Three of the four are anatomy, and none is a tuned parameter.** The one knob in the
model — the temporal weight — is chosen by a rule that consults only our own reprojection.
Of the four, the audit finds **two load-bearing** (fit-not-triangulate, and the neck-space
prior), **one forced and verified** (the thorax-frame repair, which the oracle independently
condemns the old version of), **one that matters only for subject 0's margin** (the rule
fix), and **one decoration** (the anchor).

### A self-attack on the metric, and what it exposed

P1 removes each take's own mean relative pose, which is a **gauge fix** — our head-template
frame and our thorax definition each differ from MAMMA's by a constant that nothing
observes. If that offset is genuinely constant, estimating it from half the take and
applying it to the other half must work. So: fit the mean on one half, score the other.

| | in-sample | fit 1st → score 2nd | fit 2nd → score 1st |
|---|---:|---:|---:|
| **ORACLE**, subject 0 | 5.46 / 17.22 | 8.12 / 9.00 | 6.30 / **24.71** |
| candidate, subject 0 | 7.54 / 19.35 | 9.00 / 16.34 | **14.29** / 23.80 |
| **ORACLE**, subject 1 | 4.87 / 17.59 | **8.74** / 11.43 | 5.81 / **23.00** |
| candidate, subject 1 | 6.40 / 16.07 | 8.12 / 11.98 | 7.48 / 19.49 |
| *bands* | *≤ 8 / ≤ 20* | | |

**The oracle fails it too, on both subjects and both splits.** A test that a *perfect head*
cannot pass is not measuring the head — it is measuring the assumption that the gauge is
constant, and refuting it. The offset between the two frame conventions **drifts within a
take**, so the whole-take mean is the correct construction and the split is not a valid
alternative. **That is the oracle arm earning its place a fourth time**: without it this
table reads as "the pass is an overfitting artifact", which is the wrong conclusion.

**But it does expose something real, and it qualifies the pass.** Compare candidate to
oracle *within* each split — the gauge drift cancels, since both suffer it equally:

| | candidate − oracle, median |
|---|---:|
| subject 0, second half | +0.88° |
| **subject 0, first half** | **+7.99°** |
| subject 1, second half | −0.62° (candidate better) |
| subject 1, first half | +1.67° |

**On three of the four half-takes our head sits within 1.7° of a perfect one. On subject
0's first half it sits 8° away.** The pass is real and it is **not uniform**: most of this
footage is solved close to the instrument floor, and roughly a quarter of it is not. That
belongs beside the headline, because a whole-take median can hide exactly this.

*Instrument:* the split above, run on all four arms.
*Blind to:* which half is at fault in absolute terms — the oracle bounds the *frame*
mismatch, not our error's sign. And "first half" is a crude window; the right follow-up is
to localise the excess frame-by-frame against camera support, which §6a's next-steps
already name.

### 6d THE AUDIT — what survives, and three things that do not

An independent reviewer was given the whole change chain and one instruction: **attack the
claim that the PASS is an artifact of iterating against the gate.** It re-solved and
re-scored every row of the sweep in an isolated harness. The verdict is that the pass is
**not** a gate-directed artifact — and four of its findings materially qualify what the
pass means. All four are adopted.

**Why the pass is not an artifact, in the reviewer's terms:**
- **The pass region is a plateau, not a knife-edge.** Subject 0 passes at *every* interior
  weight from 100 to 10,000 — two orders of magnitude — and fails only at the extremes.
- **The verdict is rule-invariant.** Argmin picks 100 (pass); the *old* rule on its
  *original* pre-extension grid picks 3,000 (also a pass). The failure that triggered the
  rule change existed only because a gate-blind grid extension had put 30,000 in reach.
- **The rule is anti-tuned in operation.** Argmin selected the **worst-margin** passing
  weight when 3,000 sat 0.024 px away with far better numbers — and it **degraded subject
  1** (p95 12.40° → 16.07°), which was kept anyway. A gate-tuner takes neither.
- MAMMA never enters the solver or the selection; both consult only our own reprojection.

**Finding 1 — and it is the one that changes how to read the table. P2 and P4 are knob
settings for the candidate, not evidence.** After the neck-space prior, the temporal term
regularises frame-to-frame change of head-relative-to-thorax **in the gate's exact
reference frame** — so P4 *is* the regularised quantity and P2's ceiling side is its
mirror. `solve_head.py:thorax_frames()` says so in its own docstring. **They keep their
force only as discriminators against the controls** (C2 fails P4 at 87–99°, C1 fails P2 at
exactly 0.00°). **The substance of the candidate's pass is P1 alone** — the one band the
solver cannot optimise, because MAMMA never enters it — plus P2's *floor* showing the knob
was not driven to the C1 corner.

**Finding 2 — subject 0's P1 pass has no statistical margin.** Per-frame agreement has
lag-1 autocorrelation **0.99**, so a moving-block bootstrap is the right resampler. It puts
subject 0's median of 7.54° at a 90 % CI of **[5.66°, 13.15°], with P(median > 8°) = 0.48**
— a coin flip. Paired against the oracle on identical draws, the oracle's is 0.025/0.013,
so **the fragility is the candidate's, not the frame-mismatch floor's**; the paired excess
over the oracle is 2.36° [0.53, 6.62] and 1.65° [0.64, 2.65]. This is the same structure
the split-half self-attack found. **"Passes on both performers" is a whole-take median
over one 150-frame take, and on subject 0 it is not robust to resampling that take's
segments.**

**Finding 3 — P1's p95 arm barely discriminates.** The oracle sits at 17.22° / 17.59°
against a 20° band, so the frame mismatch nearly saturates it. **The load-bearing number
is the P1 median**, and the p95 should not be quoted as independent evidence.

**Finding 4 — the neck anchor is decoration, and its stated justification was thin.**
Ablating it and re-scoring at the chosen weight **still passes** subject 0 (P1 7.79° /
18.40°, P2 16.67°, P4 3.67°). Its acceptance margin was 0.005 px, inside the curve's own
0.02–0.04 px weight-to-weight jitter, and its causal role in the rule change is refuted
above. It is retained because it is anatomically right and costs nothing, **not because it
earned the pass**. Related: the argmin's *location* moves under a ≤0.03 px perturbation,
so the argmin deserves no significance — **the plateau carries the verdict, not the
minimum**.

**Also adopted:** the reviewer found that this document's pass-declaring commit had
**deleted the very sweep table an earlier commit promised as auditable evidence**, along
with §6b and §6c. That was a transparency regression, and all three are restored above and
below. A code nit — the gate fills invalid frames with identity before mean-removal,
slightly diluting C2's mean — is immaterial here (candidate and oracle have 150/150 valid
frames) and is recorded rather than silently fixed.

> **So the honest headline is narrower than "the head passes".** It is: *the head clears a
> pre-registered parity gate on both performers; the substance of that is P1, one of whose
> two subjects has a coin-flip margin under resampling; the controls and the oracle make
> the result interpretable; and none of it is shipped.*

### 6b PRE-REGISTRATION — flagging under-observed frames

**Written and committed before it was measured.** The measured cause of the residual P1
gap is under-observation, not under-modelling (−0.60 correlation with camera support), and
no prior recovers information that was never captured. The pipeline's existing rule for
that situation is a flag, not a guess: *"missing or untrusted hand observations remain
explicitly `review_required`; they do not fall back to an unreported canned gesture"*
(`FINGER_TRIANGULATION_GATE.md`). This applies it to the head. **Reading the gate after
choosing a threshold would be the tuning trap wearing a product hat**, so the rules are
fixed here first.

1. **Threshold.** A frame is `head_review_required` when **fewer than 3 cameras** see at
   least 3 of the 5 head landmarks at confidence ≥ 0.25. *Chosen from geometry, not from
   any score:* two rays fix a 3D point but leave the rotation of a ~120 mm object at 5 m
   badly conditioned, and three is the smallest count that carries any redundancy.
2. **The verdict does not move.** Pass/fail is taken from the **ungated** arm, over all
   frames, exactly as already reported. The gated arm describes what a flagged output
   would deliver; it cannot rescue a failing candidate.
3. **Same denominator.** Both controls and the oracle are re-scored on the identical
   gated population. A candidate compared against controls scored on all frames would be
   the composition shift this lane has been caught by twice.
4. **The flagged fraction is reported, and it is itself a result.** If it exceeds **25 %**
   the flag is doing too much work, and the honest reading is *"this fixture is
   under-observed"* rather than *"the estimator works"*. Stated before the number is known.
5. **Nothing is tuned to the outcome.** The fit, its weights and the bands are unchanged.

### 6c MEASURED — ⚠ THIS SECTION WAS WRONG, AND THE ERROR WAS MINE

**Retracted and rewritten 2026-08-31 after an independent verifier found a defect in
`head_gate.py`.** What this section originally said — *"the flag makes every arm worse,
including the oracle, so it is removing frames where the two thorax frames happen to
agree"* — was an artifact of my own code, and the elaborate reasoning I built on top of it
about facing direction and composition shift was explaining a bug.

**The defect.** `mean_removed` had no population argument, and the caller injected the
identity on every excluded frame *before* calling it. The take-mean the gated arms were
referenced to was therefore a mixture of real deviations and injected identities, dragged
toward identity, which **inflated every gated arm by roughly 2×**. Fixed by estimating the
gauge on the scored population — for both sides of the comparison — which is what a gauge
fix has to mean.

**Containment, checked first and it holds:** the **ungated arms were never affected**,
because their masks are all-true, so the injection was a no-op there. The headline pass
(7.54° / 19.35° and 6.40° / 16.07°) is byte-identical before and after the fix. **The
verdict never rested on the defect.**

**What the flag actually does**, recomputed. Flagged 19/150 (12.7 %) and 23/150 (15.3 %),
both under §6b's pre-registered 25 % ceiling:

| P1 median, subject 0 / 1 | all 150 frames | reported frames only |
|---|---:|---:|
| ORACLE | 5.46° / 4.87° | **4.35° / 4.31°** |
| candidate | 7.54° / 6.40° | **5.73° / 6.65°** |
| C2 noisy | 6.52° / 7.60° | 5.27° / 6.57° — **still FAILS** (p95 53°/71°, P4 20°/82°) |
| C1 locked | 14.96° / 15.99° | 15.29° / 16.33° — **still FAILS** (P2 exactly 0.00°) |

**So the flag is a real improvement on subject 0** — the candidate's P1 median falls from
7.54° to 5.73°, which is most of the way to the oracle — and roughly neutral on subject 1.
**And it does not launder a failure:** both degenerate controls still fail on the reported
frames, C1 on spread and C2 on jitter, so the flag cannot be used to smuggle a bad head
through.

**What stands from §6b, unchanged:** the verdict is taken from the **ungated** arm, and it
passes there. The flag is therefore *not needed* for the pass — it is an available quality
improvement whose effect is now correctly measured rather than a rejected idea.

> **Two things I got wrong here, and the second is worse than the first.** The arithmetic
> was a code defect, which is ordinary. The failure of process is that I found a result
> that flattered a tidy narrative — *"only the oracle could have caught this"* — and wrote
> it up at length **without recomputing it**. The oracle *did* behave oddly; I read that
> as signal when it was my own contamination showing through the one arm whose true value
> I could predict. **An anomalous oracle is a reason to audit the instrument first, not to
> theorise.** That is the lesson, and it is more useful than the section it replaces.

### The oracle arm, and why a gate without one is only half-checked

*"No gate a constant can pass"* has a dual that this lane had not written down: **a gate
no oracle can pass is miscalibrated.** The two controls show that degenerate solutions
fail. They cannot show that a *passing* candidate exists — and P1 compares two
mean-removed relative rotations whose thorax frames come from different definitions
(ours landmark-derived, MAMMA's from its fitted chain). Mean-removal kills the constant
offset between them; the **pose-dependent** part survives and lands straight in P1. That
imposes a floor no head, however perfect, can get under.

So the oracle arm feeds **MAMMA's own head rotations** through **our** thorax frame and
the identical scoring path. It **passes on both subjects**, which settles two things at
once:

1. **The bands are achievable.** P1 median 5.46° / 4.87° against a ≤ 8° band. The gate is
   not asking for something the instrument pair cannot deliver.
2. **The candidate's P1 gap is real, and it is ours.** Against the floor, the candidate
   is **+3.93° / +2.26°** on median and **+4.14° / +4.88°** on p95. That is estimator
   error, not frame-definition mismatch, and it is what the next build has to close.

> **An oracle arm passing is not "MAMMA passes, so ship MAMMA".** MAMMA is a measuring
> instrument and never enters a delivered artifact (CLAUDE.md). The arm exists to
> calibrate the gate, and for nothing else.

**And it re-reads P4.** *(Superseded by the audit below: after step 4 the temporal prior
regularises exactly the quantity P4 measures, so for the candidate P4 is a knob setting,
not evidence. It retains force only against the controls.)* The oracle's P4 is
7.08° / 7.32°; the candidate's, after the neck-space prior, is 3.72° / 4.66°. Both are
dominated by our thorax frame's residual jitter rather than the head's.

**The gate works, and it discriminates.** The locked head dies on spread (P2, exactly
0.00°), the noisy head dies on jitter (P4, at 87–99°), and both the oracle and the
candidate pass. No arm passes by accident, and no constant can reach P2.

### An instrument defect in the gate itself, found and fixed — the third this session

An early run reported the candidate failing **P4** at 13.13° and 37.48°. It was the gate's
own reference frame, not the head.

Our thorax came from **raw triangulated** `Neck2`, `Hips` and shoulders while MAMMA's came
from its **fitted** chain. A relative rotation carries the noise of whichever frame is
noisier, and the tell was available: the candidate head's *own world* travel is p95
**1.16° / 4.68°**, so nearly all of the 13°/37° was the reference wobbling underneath it.
**A smooth quantity scored against a noisy reference reads as noisy** — a same-denominator
violation between the two arms' *references* rather than their populations.

**Independently confirmed by the audit, and it shows the fix was forced rather than
convenient:** scored through the *original raw* thorax, the **ORACLE itself fails** —
subject 0 on P4 (13.38° against a 7.15° ceiling), subject 1 on P1 p95 *and* P4 (28.17°,
36.91° against 12.52°). A perfect head could not pass the old frame, so the gate's own
dual condemns it, using an arm that owes nothing to our head landmarks. The oracle's
raw-frame P4 of 13.38°/36.91° also matches the 13.13°/37.48° that had been charged to the
candidate.

### Adding the ears made it worse — and the reason is the lane's own founding finding

§7.4 recommended, and §6a's own next-step list repeated, that Apple Vision's ears belong
**in the objective of a multi-frame fit**: widest head baseline on this footage, most
epipolar-consistent landmarks measured (§2b), survivors of a null template (§4). The fit
was extended to seven landmarks and re-run under the identical selection rule. Apple
Vision's subjects were matched to ours by 3D root agreement — straight, 32× margin, 43 mm
median — never by index, per §0's trap.

**It did not help, and on one subject it clearly hurt:**

| P1 median / p95 | subject 0 | subject 1 |
|---|---:|---:|
| SOMA-77 head points only | **9.11° / 22.38°** | **7.05° / 28.93°** |
| plus Apple Vision's ears | 9.29° / 23.61° | 10.36° / 34.67° |

**One thing the ears did do, and it is worth keeping.** The fit recovers an ear
separation of **153.2 mm and 153.6 mm** on the two subjects — anatomically right and
consistent between them, where independent triangulation of the same points gave
160 / 185 mm with standard deviations of 17.6 / 61.7 mm. The fit *can* see the ears. It
just cannot use them.

**Why, measured rather than guessed.** In the joint fit the ears carry the **worst**
per-landmark reprojection residual — 3.8–4.5 px median against 2.3–3.5 px for the SOMA-77
head joints, with a p95 to 24 px — despite §2b showing them to be the *most* cross-view
consistent landmarks in the entire study. Those two facts are only compatible one way:
**the ears agree across cameras within a frame and do not sit rigidly on the skull across
frames.**

> **This is `soma77_pose.py`'s own founding argument, arriving at the head.** That
> adapter's docstring says SOMA-77 was adopted because *"surface landmarks move relative
> to the underlying bone as a subject turns, which is the mechanism behind the
> limb-length instability. Interior joint centres should not."* An ear is a surface
> landmark: as the head turns, the detected point slides over the skull. Every camera
> sees the same *apparent* point, so epipolar consistency is excellent and triangulation
> is well-conditioned — and the point it converges on is **not a fixed place on the
> skull**. A rigid-template fit across a whole take is precisely the estimator that
> exposes that, and it is the first instrument in this lane that could have.
>
> **So my own stated assumption was wrong and is now refuted:** the fit's docstring
> claimed rigidity to the skull was *"true of ears and of skull joints alike."* It is
> true of the joints and false of the ears. Cross-view consistency was never evidence of
> it — that is a within-frame property, and rigidity is a between-frame one.

**Consequence for §4 and §7.4, both of which now carry a correction.** The ear axis's
conditioning advantage is real and its usefulness is not. Recovering the head does not
need a wider baseline; it needs landmarks that stay put on the bone, which is what
SOMA-77 already emits. **The SOMA-77-only fit is the reported candidate**, and the
ear-augmented arm is retained at `head-solve-with-ears.npz` as the measurement that
closed the question.

### The selection rule was replaced, after it failed, and the reason is structural

The first fit chose its temporal weight by **held-out-camera cross-validation** and chose
**zero** on both subjects; the gate then rejected it with a travel p95 of 58–77°. The
rule was mis-specified: **detector noise is correlated across cameras within a frame** —
all four views are one detector at one instant — so dropping a view does not create an
independent test of *temporal* behaviour. An unsmoothed fit chasing per-frame noise still
predicts the held-out camera well. **Held-out-camera CV can select a spatial model and is
blind to a temporal one by construction.**

The replacement is an **L-curve on our own reprojection**, which never touches MAMMA:
sweep the weight, take the largest whose in-frame reprojection stays within 10% of the
unsmoothed fit. Recorded because it matters: the L-curve is nearly **flat** — reprojection
moves only 2.72 → 3.24 px across five orders of magnitude of weight — so *the 2D barely
constrain the head's temporal behaviour at all*. That is itself a finding about this
framing, and the first result to fold into the next attempt.

**One suspect ruled out.** The fitted template's per-landmark radii are lopsided
(96.5 mm against 50.7 mm for the two eyes on subject 1), which looked like the gauge
prior distorting the shape. It is not: dropping the prior tenfold changes in-frame
reprojection by 0.01 px and the template by under 2 mm, and the meaningful quantity —
**eye separation — comes out at 56.1–56.6 mm on both subjects**, consistent and
anatomically sane for joint centres inside the skull. The radii are lopsided only because
the gauge centres the template on five arbitrary points, not on the skull. **The P1 gap
is not the template.**

> **A weight that scores better on the gate exists, and was not chosen.** Subject 0 at
> weight 3,000 gives P1 8.36 / 20.89 and travel 13.17 — closer to passing than the
> 10,000 the rule selects. **Selecting it would be gate-tuning**, which is the
> overlay-flatterer degenerate solution one level up, so the rule's answer stands and the
> better number is recorded here rather than reported as the result.

*Instruments:* `tools/head/solve_head.py` (the fit and the superseded held-out-camera
rule), `tools/head/select_smoothing.py` (the L-curve), `tools/head/head_gate.py`.
Superseded outputs retained as `head-*-heldout-camera.json`.
*Blind to:* everything §6 already lists — this is a **parity** gate against an
instrument, never an accuracy claim. Also blind to a constant heading error, removed by
the mean-removal that makes the comparison a tracking comparison at all.

---

## 6e DELIVERY — what wiring it cost, and the two defects that only appeared there

The gate is a measurement; delivery is a different problem, and it exposed two things the
gate is structurally incapable of noticing.

**1. A rigid fit is correct only up to a constant, and the gate removes exactly that
constant.** Nothing in the objective observes where the skull's zero is — the template's
local frame is wherever the optimiser landed. P1 mean-removes each take and scores
*tracking*, so it never saw the offset. **Delivery cannot**: shipped raw, the solve put an
**80–176° constant offset** on the `Head` joint. A head pointing sideways, smoothly, on
every frame.

The zero is now fixed from the template's own anatomy — `HeadEnd` above `Head` for the
skull's long axis, the eyes for the lateral one. **Never from the reference**, because
aligning to a research fitter's head would be a shipped constant fitted on a
reference-derived artifact, which `BODY_LANE_PLAN.md` forbids outright. Absent those
landmarks the solve **refuses** rather than delivering an arbitrary orientation.
*Delivered local head rotation fell from 80° / 112° median to* **26.5° / 20.7°**, *max
40.2° / 37.6° — the right order for real head-on-torso motion.*

**2. Minimum reprojection has no notion of anatomy.** An unsmoothed solve delivered a
**140° single-frame head flip** that reprojected perfectly well — because much of the flip
lay along the viewing rays, which is the same blindness §2b exploits in the other
direction. Solutions whose frame-to-frame **neck** travel exceeds 60° are now discarded
before the reprojection rule chooses. 60° at 30 fps is 1800°/s against a human peak near
500–800°/s: a hard physical reject, not a tuning knob.

**Measured end to end, it fires and it is not vacuous.** On subject 1 it discarded the
unsmoothed solve the reprojection rule had chosen, moving the selected weight **0 → 100**
and the delivered maximum **140.2° → 37.6°**. Subject 0 was untouched at weight 30. So the
filter changed exactly the take that needed it and left the other alone — and the same
contrast is asserted in the tests, where lifting the bound restores the impossible step.

> **Measured at the neck, not in world, and the distinction is load-bearing.** A head on a
> turning body travels in world while the neck is perfectly still. Scoring world rotation
> would reject that honest motion *and* accept a genuine neck flip during a still moment —
> both errors at once.

**What delivery does not include, stated so it is not assumed.** The whole head-on-torso
rotation is placed on `Head`; **`Neck` keeps the torso frame**. Distributing it across the
neck chain would look better on a mesh and is **unmeasured** — the gate scores the head,
and the obvious source for a split, the reference's own neck/head ratio, is barred by the
same rule as above. Splitting it is a named next step, not a guess to make here.

**And the temporal weight is swept per take, not hardcoded.** A weight chosen once on this
fixture and then shipped is a constant calibrated on this fixture. It costs a sweep per
subject and buys the right not to have fitted a delivered constant on one clip.

*Instruments:* `src/autoanim_gnm/head_orientation.py`, `tests/test_head_orientation.py`
(7 tests, 24 green with the existing suite).
*Blind to:* everything §6 is blind to. Wiring changes what ships; it changes nothing about
what has been measured, and no accuracy claim follows from it.

---

## 6f THE SHIPPED CONFIGURATION, SCORED — and what the last 0.28° actually is

**Scored on the rotations the pipeline delivers, not the prototype's.** They are different
estimators with different selection paths, and until this was run *"the head passes the
gate"* and *"the head is on the delivery path"* were two true statements about **two
different heads** — this lane's recurring defect, one sentence away. `head_gate.py` now
takes a path, so from here the gate scores what ships.

| shipped fit | P1 median | P1 p95 | P2 spread | P3 | P4 travel | verdict |
|---|---:|---:|---:|---:|---:|---|
| **performer 0** | **7.09°** | **17.75°** | 17.18° | 0 | 3.69° | **PASS** |
| **performer 1** | **7.00°** | **20.28°** | 16.57° | 0 | 4.76° | **FAIL** — p95 by 0.28° |
| *oracle* | *5.46 / 4.87°* | *17.22 / 17.59°* | | | | *PASS* |
| *bands* | *≤ 8* | *≤ 20* | *≥ 8* | *= 0* | *≤ 7.14 / 12.51* | |

**A defect of mine caused the first divergence, and it is worth naming precisely.** Porting
the solver into the library I replaced *"average each camera's median residual"* with *"one
median over all observations."* A camera that sees more of the head contributes more
observations, so pooling is weighted by coverage and lets the best-placed view decide the
temporal weight — a **same-denominator violation**, this lane's own standing rule, inside
the selection rule. Restored, both performers select the same weight and performer 0 moved
from **fail to pass** (8.48 → 7.09 median, 22.25 → 17.75 p95).

### The remaining 0.28° is not in the head, and five measurements say so

| ruled out | measurement |
|---|---|
| **the optimiser had not converged** | reprojection at `max_nfev` 120 and 400 is identical to four decimals on both performers |
| **the weight grid was too coarse** | on a fine grid (50, 70, 100, 150, 200, 300, 500) weight 100 is the true argmin for both — the 3×-spaced grid was not hiding a better fit |
| **a better thorax frame exists** | five constructions scored **by the oracle alone**, an arm containing none of our head: the current one is the **best of the five** (17.22 / 17.59 p95) and every alternative is worse, up to 32.33 |
| **the robust loss was mis-specified** | it *is* mis-specified — it applies ρ(\|r\|), not soft-L1's ρ(r²) — and correcting it makes the fit **worse on our own reprojection**: 2.71 / 2.63 px as written, against 2.82 / 3.03 with a MAD-scaled soft-L1 and 2.98 / 3.16 with a wider one. The head detector's residuals have a fat tail, and a transform that compresses hard everywhere suppresses it better than one quadratic on a 2.7 px bulk. Reverted **on the reprojection criterion, not the gate's**, and the reason is now in the code |
| **the fit is stuck in a poor basin** | the objective is non-convex and had only ever been seeded one way. Three seeds — triangulation, a torso-aligned head, and a constant mean pose — at two weights each: the existing seed wins on both performers (2.7148 / 2.6311 px against 2.79–2.87 and 2.69–2.76) |

**What is left is the instrument floor.** The oracle — *a perfect head* — sits at
**17.59°** of performer 1's 20° band, consuming **88 %** of it. Our candidate at 20.28°
adds about **10.1°** of its own in quadrature. An independent audit had already reached
this from the other direction and it is recorded above in §6d: *"P1's p95 arm is nearly
saturated by frame mismatch and discriminates almost nothing; the load-bearing number is
the P1 median."* **On that number the shipped head passes on both performers**, 7.09° and
7.00° against a ≤ 8° band, as do P2, P3 and P4.

> **This is a report of a failure, not an argument for excusing one.** The band is
> pre-registered and is **not being moved**; performer 1 fails and the boards say so. What
> the three measurements establish is *where the work is*: closing 0.28° on a statistic
> 88 % consumed by torso-frame mismatch is a **better-torso** problem, not a better-head
> one — and a better torso is `FITTER_PLAN.md`'s parametric body fit, a different build
> with its own identifiability wall. Any re-derivation of the p95 band must be argued and
> written **before** anything is re-scored against it, or it is band-moving with extra
> steps.

*Instruments:* `tools/head/gate_the_shipped_head.py`, `tools/head/head_gate.py <path>`.
*Blind to:* everything §6 is blind to. The oracle bounds the frame mismatch; it does not
tell us whether the mismatch is our error or the reference's, and nothing here can.

---

## 6g A SHARPER INSTRUMENT — and it corrects §6f's conclusion, in the unwelcome direction

§6f concluded that performer 1's remaining 0.28° was *"the instrument floor"*, on the
grounds that the oracle consumed 88 % of the band. **That conclusion was wrong, and a
sixth measurement is what shows it.**

**The thorax frame was jittery in a way five earlier checks did not reach.** Our torso
frame is built per frame from landmark geometry, and **a frame is a nonlinear function of
its landmarks** — so smoothing the positions, which the pipeline already does, leaves the
*frame* jittery. MAMMA's comes from a fitted kinematic chain and is smooth by
construction. Comparing the two charged our reference's own wobble to the head.

Smoothing our thorax **as a rotation**, with the window chosen by the **oracle arm alone**:

| window | oracle P1 median | oracle P1 p95 |
|---|---|---|
| none | 5.46 / 4.87 | 17.22 / 17.59 |
| 5 | 5.42 / 4.87 | 17.00 / 17.21 |
| **15 — chosen** | **5.46 / 4.89** | **14.14 / 16.30** |
| 21 | 5.87 / 4.96 | 14.12 / 17.19 |
| 31 | 6.52 / 4.88 | 13.78 / 16.39 |

**It has an interior optimum, which is what separates matching a real property from
degenerating toward a constant.** Past 15 frames — half a second — the p95 stops improving
while the median degrades, because real torso motion is being lost. That turn is the
evidence the window is not a knob.

### And the candidate got worse

| shipped fit, smoothed thorax | P1 median | P1 p95 | P2 | P3 | P4 | verdict |
|---|---:|---:|---:|---:|---:|---|
| performer 0 | **6.69°** | **17.77°** | 16.84° | 0 | 3.76° | **PASS** |
| performer 1 | 7.20° | **21.18°** | 16.63° | 0 | 5.09° | **FAIL** — p95 by 1.18° |
| *oracle* | *5.46 / 4.89°* | *14.14 / 16.30°* | | | | *PASS* |

**Performer 1 now misses by 1.18° where it missed by 0.28°.** The floor fell 1.29° and the
candidate rose 0.90°, so the gap between our head and a perfect one widened from **+2.69°
to +4.88°**.

> **This is the gate working, and the result is kept even though it reads worse.** The
> window was chosen on the oracle — an arm with none of our head in it — and the
> candidate's own reprojection *improved* slightly under it (2.715/2.631 → 2.704/2.615 px).
> A sharper reference did not make our head worse; it stopped hiding how wrong our head
> already was. **Reverting to the jittery frame because it flatters us would be
> gate-tuning with the sign flipped**, and it is the exact move this document has refused
> five times in the other direction.

**So §6f's "the remaining gap is the instrument floor" is withdrawn.** With the floor at
16.30° and the candidate at 21.18°, **+4.88° of performer 1's p95 is ours.** There is real
estimator work here after all, and the five exclusions in §6f narrow *where* it is rather
than proving it absent: not convergence, not the grid, not the thorax definition, not the
loss, not the seed — and now, not the floor either.

**One structural fix landed with it.** The gate had been rebuilding its own copy of the
torso frame instead of importing the pipeline's. Two copies of one definition is how they
drift, and this lane was already caught once scoring one head while shipping another
(§6f). There is now one `_thorax_frames`, imported by both.

---

## 6h THE DECOMPOSITION — our head is the same on both performers; the frame is not

Seven hypotheses had been excluded by guessing at mechanisms. This asks the data instead:
score **our head against the oracle's head in the SAME frame**, which cancels the frame
mismatch and leaves only our own error.

| p95, performer 0 / 1 | |
|---|---:|
| what the gate scores (candidate vs the reference in *its own* frame) | **17.77° / 21.18°** |
| the oracle floor (a *perfect* head through *our* frame) | 14.14° / **16.30°** |
| **our head against a perfect one, same frame — purely our error** | **9.91° / 9.80°** |
| *…median* | *5.26° / 4.90°* |

**Our head estimator performs identically on the two performers — 9.91° against 9.80° at
p95, 5.26° against 4.90° at median.** The gate passes one and fails the other, and the
difference between them is almost entirely the frame: the floors differ by **2.16°**
(14.14 vs 16.30) while our own error differs by **0.11°**.

> **So performer 1 does not fail because our head is worse there. It is not worse there.**
> It fails because the reference-frame mismatch is 2.2° larger on that performer, and the
> band has no room for it. That is a sharper statement than either §6f's *"the gap is the
> floor"* — withdrawn as too strong — or §6g's *"+4.88° is ours"* — true of the total, but
> it conflates a term that is constant across performers with one that is not.

**Where our own 9.8° lives.** It is **episodic**, not scattered: the worst decile occupies
**3–4 contiguous runs** on each performer, a handful of moments rather than broadband
noise. On performer 1 it correlates with detector confidence (−0.386) and torso angular
speed (+0.329); on performer 0 with camera support (−0.363) and trunk tilt (+0.281). Those
are different mechanisms per performer, which is consistent with a handful of hard moments
rather than one systematic defect.

### Two more exclusions, both from this pass

| ruled out | measurement |
|---|---|
| **the jaw is not rigid to the skull** | it is, on this footage. Its per-landmark residual is the *lowest* of the five (2.38 / 2.40 px against 1.78–2.83), and removing it makes the fit **worse** on the four landmarks both fits share (+0.079 / +0.048 px, same denominator). The ears' failure mode does not repeat here |
| **the thorax smoother's chart is bad** | it is a tangent-space linearisation about the take's mean, only valid for modest excursions, and performer 1 goes supine — so a chart-free quaternion smoother should differ. It does not: 14.14 / 16.29 against 14.14 / 16.30, identical to two decimals |

**That is eight exclusions.** Not convergence, not grid resolution, not the thorax landmark
choice, not the robust loss, not the seed, not the jaw, not the smoothing chart — and not
"the floor" alone either, since our head contributes a real 9.8°.

**What this says to build, and it is not more head work.** Our head is as good on the
failing performer as on the passing one. The lever that separates them is the torso frame,
and the torso frame is landmark geometry where the reference's is a fitted kinematic
chain. Closing it means fitting the body, which is `FITTER_PLAN.md`'s parametric body fit —
**the lane's largest structural gap, named long before this session** — and not another
head estimator.

---

## 7. What the measurements say to build — in order, none of it started

0. ~~**Put the solve on the delivery path.**~~ **DONE** — §6e. Two defects appeared only
   at delivery: the gauge and the anatomical reject. Both are fixed and tested; both were
   invisible to the gate by construction.

1. ~~**Settle §5's ownership question.**~~ **SETTLED** — body lane owns head and neck
   rigid world orientation, face lane owns expression within that frame. See §5 for the
   three defects that follow from it. **The remaining audit:** confirm the face lane does
   not also write head world orientation, because it is now a second writer if it does.
2. **Read the head landmarks at all.** `positions_to_body_track` triangulates `nose`,
   `left_eye`, `right_eye` and then ignores them. Whatever the head solve becomes, the
   line at `commercial_multiview.py:1350` has to stop being an unconditional assignment.
   *This is not the fix* — driving a 59 mm eye baseline into the head is control C2,
   which the gate rejects. It is the plumbing the fix needs.
3. **A model-constrained multi-view fit is the route to *test next* — not, on this
   evidence, the established route.** The pull toward it is real: MAMMA recovers a
   stable, never-opposing head from **these exact four videos** (§1), so the information
   is in the footage and our estimator discards some of it; and `FITTER_PLAN.md` §5
   records momentum + MHR already integrated, MHR carrying `c_neck`, `c_head`, `c_jaw`,
   `l_eye`, `r_eye`, and `CameraKeypointData` taking multi-camera 2D keypoints with
   confidences directly — so the head chain is a locator-set extension of a fitter that
   already runs.

   **The break in the fingers analogy has been tested and does not hold — §2b.** The
   first draft of this item withheld the conclusion because §2's 3D probe showed head
   landmarks disagreeing across views, and cross-view-inconsistent 2D is a detector
   failure no fit can repair. The 2D-native epipolar test says the opposite: head
   landmarks are **more** consistent than body landmarks on both detectors, 0.29–0.78×.
   **The head's failure is pure depth, which is the estimator's job.** This is now the
   route, not a hypothesis.
3a. **What the built fit says to do next — three items, in order, all named by its own
   failure mode.** §6a's candidate fails on P4, and more smoothing does not help, so the
   remaining travel is a minority of badly-conditioned frames rather than broadband noise.
   - ~~**Add Apple Vision's ears to the objective.**~~ **DONE, and it does not work** —
     §6a. The ears fit a rigid skull 1.3–1.6× worse than SOMA-77's joints despite being
     the most cross-view-consistent landmarks measured, because an ear is a *surface*
     landmark that slides over the skull as the head turns. **The question is closed:
     the head does not need a wider baseline, it needs landmarks that stay on the bone.**
   - **Find and gate the jumping frames — now the first item.** P4 is what defeats the
     candidate and more smoothing does not touch it, so the residual is a minority of
     badly-conditioned frames. Report travel against per-frame camera support and
     confidence; if the tail concentrates where the head is seen by two cameras, a
     support-conditioned prior is the fix rather than a stronger global one. Untried.
   - **Reconsider P4 itself — but only against a stated argument, never against the
     score.** The band was pre-registered at 3× MAMMA's travel p95, and MAMMA's travel is
     small *because it carries a strong temporal prior of its own*. A band set from a
     smoothed reference may be asking a per-frame estimator for smoothness the reference
     gets for free. That is an argument to be made and recorded **before** any candidate
     is re-scored against a changed band, or it is gate-tuning.
   - **Then, and only then, momentum + MHR.** `FITTER_PLAN.md` §5's fitter already carries
     `c_neck`, `c_head`, `c_jaw`, `l_eye`, `r_eye` and takes multi-view 2D directly. The
     bespoke fit here exists to establish whether the *architecture* recovers the head at
     this framing before that integration is paid for. It says: mostly, not yet.

4. **Ears are an input to the fit, not a driver of the head — revised after §1's
   corroboration ran.** The draft of this item read *"wire the ear axis to head yaw"*.
   **That is now refused by measurement:** the ear axis's yaw scatters at 21–41°,
   two to four times MAMMA's head yaw and larger than the entire signal it would
   explain. The ears remain the **best-conditioned head observation on this footage** and
   they survive the null-template gate, so they belong in the objective of a
   multi-frame, multi-view fit — where a temporal prior and a rigid-skull constraint can
   average that scatter down — and **not** on a direct per-frame path to a rotation
   channel. This is the finger gate's shape a third time: a good observation, a bad
   estimator.
5. **Then feet.** The same `torso_world` assignment freezes them (§0), and toe landmarks
   70/71/75/76 are already on disk (§3). Not before the head holds.

---

## 8. Regenerating every number here

```bash
python3 tools/head/verify_soma77_retention.py     # §3 -- 20,043 index checks
python3 tools/head/associate.py                   # §3 -- association, 0.0 mm provenance gate
python3 tools/head/mamma_head_bar.py              # §1 -- the bar
.venv/bin/python tools/head/our_head_bar.py       # §0 -- the delivered head
python3 tools/head/headend_gate.py                # §2 -- HeadEnd vs the invariant
python3 tools/head/head_landmark_noise.py         # §2 -- cross-view disagreement
python3 tools/head/av_ear_probe.py                # §4 caution 2 -- AV support, confidence, tail
python3 tools/head/same_denominator_head.py       # §4 -- AV vs SOMA, 95 frames
python3 tools/head/ear_null_template.py           # §4 -- the ears against a no-ear null
python3 tools/head/subject_map.py                 # §0 -- MAMMA body_id <-> our subject
python3 tools/head/corroborate_bar.py             # §1 -- MAMMA head yaw vs AV ear yaw
python3 tools/head/head_epipolar_gate.py          # §2b -- the head's 2D vs the body's
python3 tools/head/select_smoothing.py            # §6a -- the fit, L-curve weight (~10 min)
python3 tools/head/head_gate.py                   # §6a -- the gate, candidate + both controls
.venv/bin/python tools/head/three_detector_head.py  # §4 -- all three, 42/14 frames
```

MediaPipe detections for §4 are regenerated by
`workers/commercial_multiview/mediapipe_pose.py` against
`.cache/autoanim_gnm/mediapipe/pose_landmarker_heavy.task` and the frames already in
`artifacts/commercial-multiview-soma77/work/frames/`; they land in
`artifacts/head-lane/mediapipe/`. `artifacts/` is gitignored, so all of it regenerates.

`associate.py` and `three_detector_head.py` cache to `artifacts/head-lane/*.npz`; delete
the cache to force a recompute. `our_head_bar.py` and `three_detector_head.py` need
`.venv` for `autoanim_gnm.body` and `mediapipe` respectively.

---

## 9. Review — what two independent reviewers changed, recorded inline

Both reviewers saw the same scripts and JSON and were asked to attack, not agree. Their
corrections are already applied above; they are listed here because *what was wrong before
review* is the useful part, and because this lane's recurring defect is a correct
measurement carrying a claim it does not support.

**Sol (advisor).** Three repairs, all taken:
- The bar compared MAMMA's **composed world** head rotation against §1c's **local**
  `c_head` from the momentum fit. Different quantities, different pipelines — the
  same-denominator rule broken inside the headline. Repair: compose our own chain with
  the same functions. **That repair is what found §0.** Without it the session's largest
  finding would not have surfaced.
- Report **both** normalisations, because the shoulder and shin controls disagree and
  only one flatters the candidate.
- MediaPipe was enumerated but not run, while the goal required it scored. Repair: run
  it. It is now §4's third row.

**Fable.** Six corrections, five taken, one declined:
- *"Same class as the fingers' 4.4×"* is cross-harness numerology and under the shin
  control would swallow the ear axis too. **Taken** — removed, with the reasoning kept
  in §4 so it is not re-derived.
- *"The route is a model-constrained fit"* overclaims: the fingers **passed** cross-view
  reprojection, isolating their failure in depth; the head landmarks do not, and
  cross-view-inconsistent 2D is a detector failure a fit cannot repair. **Taken** — §7.3
  is now a hypothesis with its two missing tests named.
- The cross-view probe's **max-over-pairs statistic scales with pair count**, and support
  differs by landmark and detector. **Taken** — replaced with median pair distance and
  RMS about the centroid, which *weakened the finding*: subject 1's head landmarks are
  indistinguishable from its limbs at the median, and only the tail separates them.
- Probe B was **void by its own stated criterion** — its `Neck1` anchor is not clean —
  and published anyway. **Taken** — marked void in §2, not quoted.
- The ear claim rested on medians while the ear tail reaches 1129–1692 mm, and the probe
  was blind on roughly half the ear frames. **Taken** — §4 cautions 2 and 3.
- *"MAMMA's ~29° is real head motion"* is an estimate with a temporal prior and no second
  instrument. **Taken** — §1 now says the reference asserts a signal, and names the
  corroboration that has not been run.
- **Declined, with reason:** Fable read the `HeadEnd` verdict as unsupported because the
  *direction* statistic (rotation p95 6.62° on 95 frames) disagrees with the *length*
  statistic. It does disagree, and §2 now says so explicitly — but the disagreement is
  itself diagnostic rather than exculpatory: a slowly-varying bias produces exactly a
  smooth direction with an unstable length, which is why this document pairs every
  rotation number with its invariant. The verdict rests on the anatomical argument in
  §2, which needs neither statistic.

**Codex, third reviewer, asked only to re-derive four numbers from primary data without
reading my outputs.** All four **CONFIRMED**, with one useful qualification and one bonus:
- the MAMMA chain, its frame-to-frame statistics, and the jaw being identically zero:
  confirmed. It strengthened the self-check by adding a control I had not run —
  expressing the inter-eye vector in the **head joint's frame alone** leaves ~32 mm RMS,
  against 0.0002 mm for the composed chain. That is the difference between the right
  chain and a plausible one.
- the 77-landmark retention: confirmed exact across all **1,179 person-records**.
- `Head → HeadEnd` at 66 % / 115 %: **confirmed, and flagged frame-selection-sensitive.**
  Under a segment-relevant four-endpoint mask it reads **72.0 % / 126.0 %**; the shin
  control is 2.57 % / 2.54 % under both. Recorded in §2 as a range rather than a figure.

**What the user's chosen next step then found, recorded because it corrected two of the
conclusions above.** Running §1's corroboration first — rather than building on the bar —
surfaced a defect neither reviewer nor I had looked for: **MAMMA's subject indices are
crossed relative to ours**, which had silently swapped the performers in every parity
pairing in §0 and §1. It announced itself only as a correlation of |r| = 0.21/0.03
against shuffled controls of 0.18/0.22 — i.e. as *no signal*. Corrected, the same data
gives 0.34/0.40, above both controls. And the corroboration then **revoked the ear axis's
recommendation** (§7.4), because a length invariant is second-order blind to the
differential-depth error that is first-order in direction. *Two conclusions changed by
running the cheap corroboration before the build, which is the argument for running it
in that order.*

**Fable's own summary sentence, adopted as this document's §0:** *the delivered pipeline
ships a head rigidly locked to the chest, that constant currently passes the only
smoothness comparison available, and no gate exists yet that it fails.*
