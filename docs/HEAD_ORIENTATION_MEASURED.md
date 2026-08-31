# Head and neck — measured, 2026-08-31

**Status: measured, and a first estimator built and scored — it does not pass.** No
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

| subject 0 · MAMMA spread 14.96°, travel p95 2.38° | P1 med | P1 p95 | P2 spread | P4 travel | verdict |
|---|---:|---:|---:|---:|---|
| **candidate — multi-view fit** | 9.11° | 22.38° | **18.76°** | 13.13° | **FAIL** (P1, P4) |
| **C2 — noisy, per-frame triangulated** | 6.73° | 69.01° | 18.18° | 88.95° | **FAIL** (P1, P4) |
| **C1 — locked head, as delivered** | 14.96° | 30.76° | **0.00°** | 0.00° | **FAIL** (P1, P2) |
| *bands* | *≤ 8* | *≤ 20* | *≥ 8* | *≤ 7.14* | |

| subject 1 · MAMMA spread 15.99°, travel p95 4.17° | P1 med | P1 p95 | P2 spread | P4 travel | verdict |
|---|---:|---:|---:|---:|---|
| **candidate — multi-view fit** | **7.05°** | 28.93° | **16.54°** | 37.48° | **FAIL** (P1, P4) |
| **C2 — noisy, per-frame triangulated** | 8.35° | 94.00° | 19.55° | 109.62° | **FAIL** (P1, P4) |
| **C1 — locked head, as delivered** | 15.99° | 27.18° | **0.00°** | 0.00° | **FAIL** (P1, P2) |
| *bands* | *≤ 8* | *≤ 20* | *≥ 8* | *≤ 12.51* | |

**The gate works, and that is the result worth keeping.** Three arms, three *different*
failure signatures: the locked head dies on spread (P2), the noisy head dies on jitter
(P4, at 89–110°), and the candidate dies on jitter too but **3.2–3.3× tighter on P1 p95
and 2.9–6.8× tighter on P4** than the noisy control, while holding a healthy spread. No
arm passes by accident, and no constant can reach P2.

**The candidate does not pass, and it is reported as failing.** It gets most of the way:
subject 1's P1 median clears the band at 7.05°, spread is right for both, and the
estimator is unambiguously the right direction. What defeats it is P4 — residual jitter
that **more smoothing does not remove**. Subject 0 was fitted at ten times subject 1's
temporal weight and its travel barely moved (13.17° → 13.13°) while P1 got *worse*. So
the remaining travel is not high-frequency noise the prior can absorb; it is a smaller
number of poorly-conditioned frames where the rotation jumps.

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

## 7. What the measurements say to build — in order, none of it started

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
   - **Add Apple Vision's ears to the objective.** This is §7.4's recommendation and it
     is still untried: the ears are the widest and best-conditioned head baseline on this
     footage (§4), and a fit is exactly where their per-frame yaw scatter can be averaged
     down (§1). It needs a cross-detector subject match, which `subject_map.py` shows how
     to do from 3D positions.
   - **Find and gate the jumping frames.** Report travel against per-frame camera support
     and confidence; if the tail is concentrated where the head is seen by two cameras, a
     support-conditioned prior is the fix, not a stronger global one.
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
