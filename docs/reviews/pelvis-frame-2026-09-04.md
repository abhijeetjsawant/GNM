# D7 — the pelvis frame

Branch `ladder/D7`, from `d201e50`, 2026-09-04. Worktree `.claude/worktrees/ladder-D7`;
every command runs with `PYTHONPATH=$PWD/src`, and the gate and every new test record the
`autoanim_gnm.__file__` they actually imported.

---

## 0. Pre-registration — written and committed BEFORE any number existed

Committed before the rigidity pass, before the synthetic sweep, before a line of `src/`
changed and before the delivery was rebuilt. **`artifacts/` is a symlink into the main
tree and nothing under it is tracked**, so `artifacts/compare/d7-pelvis-frame/gate.json`
cannot itself be committed; this section is the tamper-evident copy of its
`preregistration` block, written at the same moment and byte-for-byte in agreement with
it. That is a deviation from the brief's wording ("commit `gate.json`") and is recorded
rather than worked around. **Refuted predictions are marked REFUTED in §5 and are never
rewritten.**

### 0.1 What is being changed, in one sentence

`positions_to_body_track` is given the triangulated SOMA-77 `Spine1` point and builds
`Hips`' world frame from the **pelvis's own landmarks** instead of from the trunk line, so
the trunk is no longer one rigid block and `_leg_root_offset` no longer rides the lean.
`Spine`, `Chest` and `UpperChest` keep the thorax frame (so `Spine`'s local becomes
pelvis⁻¹·thorax through `_set_world`), and the root formula is unchanged.

### 0.2 Facts established before any band was written

**The pelvis's children are rigid, so the clean arm cannot select.** SOMASKEL77's parents
put `Spine1` (1), `LeftLeg` (67) and `RightLeg` (72) all directly under `Hips` (0);
`Spine2` (2) hangs off `Spine1`. A Kabsch fit of the rest offsets {Spine1, LeftLeg,
RightLeg} about `Hips` onto the posed offsets has a maximum residual of **4.2e-8 m** over
the squat clip. So on clean truth *every* rigid pelvis candidate recovers the Hips world
rotation exactly, and **the selector is the noisy arm**, not the clean one.

**The card is wrong about where the rest lives, and it matters.**
`docs/LADDER_EXECUTION_PLAN.md`'s D7 row says the constant rest-pitch offset "comes from
`somaskel77-v1.json`'s rest geometry". That file carries names, parents and a coordinate
system and **no rest geometry**. The rest lives per clip in
`.cache/autoanim_gnm/gem-x/outputs/*/soma_motion.npz` (`rest_joint_positions_m`) and is
**per performer**: the five full-body clips' rests differ by up to 222 mm and their pelvis
source frames by up to 10.03°. The shipped rest-pitch is therefore a **convention, not a
fact**, and the delivered performer's own SOMA rest is unobservable at delivery time.

| convention spread, measured before selection | median | max |
|---|---:|---:|
| `root→Spine1` source frame, pairwise across the five clips | 4.42° | 10.03° |
| `mid(hips)→Spine1` source frame, pairwise | 1.75° | 3.69° |
| Kabsch template vs clip 0's | 1.82° | 3.56° |

**The effect size, reproduced on this branch.** The rigid pelvis (`root→Spine1`) departs
from the trunk line by **26.0 / 32.8°** median / p95 on the squat clip's bent frames
(tilt > 20°, n = 46), correlation **+0.93** with tilt; 3.9–8.7° median on the four upright
clips. Rest levers (mm): `root→Spine1` 43.2, `mid(hips)→Spine1` 107.6, `root→Spine2`
109.9, `mid(hips)→neck` 580.4; `root` sits 63.5 mm above and 18.3 mm off the hip midpoint.
`Spine1`'s own flexion — the lumbar term — is 14.5 / 20.9° median / p95 on the squat clip.

**Same denominator.** `artifacts/soma77-full/work/{A,B,C,D}001-observations.jsonl` are
byte-identical to `artifacts/commercial-multiview-soma77/work/*-soma77-observations.jsonl`,
so the rigidity pass and the delivery read one set of detections.

### 0.3 The shipped constants, fixed before any take was measured

Component-wise median over the five `FULL_BODY_CLIPS` rests, re-normalised. Third-party
schema (GEM-X/Kimodo somaskel77). **MAMMA-free**: no MAMMA file, `ma_cap` output, SMPL-X
fit or report computed from one enters them.

```
SOMA77_REST_PELVIS_UP        = (-0.0036018134417667037, 0.9928183854915898,  0.11957708965267391)
SOMA77_REST_HIPMID_TO_SPINE1 = (-0.003113853958207582,  0.9922427141575809, -0.12427670785277636)
SOMA77_REST_HIP_ACROSS       = ( 0.9999986844845044,   -0.0015715732215503752, 0.00040148084638751965)
SOMA77_REST_PELVIS_TEMPLATE_M: root→Spine1  (-0.00014224140613805503, 0.03922509402036667,  0.004722291603684425)
                               root→LeftLeg  ( 0.09644327312707901,  -0.05622021108865738,  0.017420830205082893)
                               root→RightLeg (-0.0960559993982315,   -0.055917683988809586, 0.017343545332551003)
PELVIS_MINIMUM_RESOLVED_FRACTION = 0.5
PELVIS_SMOOTHING_FRAMES = to be selected on synthetic truth (band B2b)
```

**What it costs if it is wrong.** `root→Spine1` sits **6.87°** off the rig's `Hips` +Y in
this rest and `mid(hips)→Spine1` **7.14°** the other way. A residual convention pitch δ
moves the root fore/aft by `|mid|·sin δ` with `|mid| ≈ 80 mm`, on **every frame, upright
ones included** — 1.4–6.9 mm for `root→Spine1` and 0.6–2.6 mm for `mid(hips)→Spine1` at
the spreads above. Window 0 (no rotation smoothing) is an admissible selection for
`PELVIS_SMOOTHING_FRAMES`: positions are already Savitzky-Golay smoothed and I7's finding
is that smoothers cost lag.

### 0.4 The candidates, the controls, and the selection rule

**Rigid pelvis candidates.** **A** `root→Spine1` (43 mm lever), **B**
`mid(hips)→Spine1` (108 mm lever), both with the hip line as the secondary axis and the
rest vectors above as the source frame; **C** a Kabsch fit of
`SOMA77_REST_PELVIS_TEMPLATE_M` onto the observed {root, Spine1, LeftLeg, RightLeg} — the
head's lesson, *fit, don't triangulate*.

**Controls that must lose.** `thorax_as_pelvis` (today's code), `world_vertical` (the
plausible shortcut), and `no_rest_correction` (candidate B with source primary `(0,1,0)`,
which reports what the convention constant buys).

**Lumbar arms, reported and never selected.** `root→Spine2` (109.9 mm lever but carrying
`Spine1`'s flexion) and a total-least-squares line through {root, Spine1, Spine2}.

**SELECTION RULE, fixed before any number.** The candidate with the **lowest median
orientation error against exact truth on the NOISY arm**, on the squat clip's bent frames
(tilt > 20°), pooled over five seeds. A tie within 0.5° is broken by the smaller
convention spread of §0.2. The clean arm cannot select; the MAMMA arm selects nothing at
any point.

**No per-frame definition switching.** One definition for the whole subject. `Spine1` gaps
are linearly interpolated over frames exactly as the body joints are; if the resolved
fraction falls below `PELVIS_MINIMUM_RESOLVED_FRACTION` the **whole subject** falls back to
the thorax frame with diagnostics status `fell_back_to_torso_frame` and a reason. A
per-frame flip between a spine-derived and a trunk-derived frame spikes the root, and it is
impossible by construction.

### 0.5 The bands

**B1 — SYNTHETIC CLEAN.** *Reference:* exact synthetic truth, the posed SOMASKEL77 `Hips`
world rotation recovered as the Kabsch rotation of the clip's **own** rest pelvis offsets
onto the posed ones. MAMMA-free. *Band:* every rigid candidate scored with the clip's own
rest as its source reads **≤ 0.01° median and ≤ 0.05° max on all five clips**.
*Degenerates that must fail:* `world_vertical` ≥ 20° median and `thorax_as_pelvis` ≥ 20°
median on the squat's bent frames. *Also banded:* the best **lumbar** arm ≥ 5° median
there — if a lumbar frame were as good as a pelvis frame on clean truth the distinction
would be empty. *Reported beside it:* the same candidates scored with the **shipped**
constants instead of the clip's own rest — that residual is the convention cost, per clip.

**B2a — SYNTHETIC NOISY, the selector.** I7/I8's measured **heavy-tail frame-correlated**
noise injected in pixels at `NOISE_SIGMA_PX = 3.20 px` (two independent estimates of our
own detector, 3.13 and 3.26 px; never MAMMA's residual) and recovered through the real
`triangulate_point` and `_fill_and_smooth_positions`. Five seeds, `20260904 + 1000k`.
Joints noised: SOMA-77 indices 0, 1, 2, 5, 67, 72 — every input any arm reads, so no arm
enjoys a cleaner input than another. *Band:* the selected candidate's median orientation
error on the squat's bent frames is **below `thorax_as_pelvis`, below `world_vertical` and
below the best lumbar arm**, on identical seeds and identical frames. *Prediction:* B beats
A because its lever is 2.5× longer against a position noise, and A's median should be
roughly 108/43 = 2.5× B's; C beats or matches B because four points beat two.

**B2b — the window.** Windows (0, 3, 5, 9, 15, 21, 31) on the selected candidate, scored on
the noisy arm for median and p95 error, cross-correlation **lag** of the pelvis yaw rate,
and peak-rate **attenuation** — I8's own protocol. *Band:* the largest window must **fail
on the fast clip** by lag > 1.5 frames or attenuation < 0.75. If no window beats 0 on p95
while holding lag ≤ 1 frame and attenuation ≥ 0.9, the selection is **0**, and that is a
selection and not a failure to smooth. *What this cannot prove:* the window optimises
exactly the quantity this arm measures. It is a knob setting; the claim that the pelvis
frame is right rests on B1, B2a and B5, none of which a window can move.

**B3 — RIGIDITY on the real take, reported before the frame is trusted.** Segment-length
stability over the take against the body controls this lane already trusts — the instrument
that killed `HeadEnd`. Triangulated by the real `triangulate_point` at production settings
on the pipeline's **own** association, recovered by wrapping `reconstruct_multiview` with a
recording associator (`tools/head/associate.recover`), never by re-implementing the loop (a
hand replication drifted 9–19 mm). Segments: `root–Spine1`, `mid(hips)–Spine1`,
`root–Spine2`. Controls: shin L/R, thigh L/R, forearm L, upper arm L. **READING RULE, fixed
before the numbers:** sd % punishes a short lever arithmetically — 3 mm on a 43 mm segment
reads 7 % where 10 mm on a 400 mm shin reads 2.5 % — so the verdict is taken on **sd_mm**
against the controls' sd_mm, with sd % quoted beside it and explicitly not the verdict. A
candidate whose sd_mm exceeds the worst body control's sd_mm is not trusted. Cross-view
self-agreement is **not** rigidity.

**B4 — ROUND TRIP canonical.** Legs 0.00, torso 0.00, arms 0.55 / 0.08 mm, unchanged.
**Pre-registered as trivially met, and why:** `retarget_cost.landmarks_from_fk` emits the
19-joint contract, which has no `Spine1`, so the converter takes the **legacy** path in the
round trip and is bit-identical to D3 there. **The round trip is blind to D7.** It is run
because a change to the branch line would show as a non-zero leg, and for no other reason.
If it moves, the exact line is found and named, not excused.

**B5 — EXACT-RECOVERY ORACLE, the closure that is not blind.** Pose the rig with a `Hips`
world rotation deliberately ≥ 20° from the thorax's, forward-kinematic it, emit the 19
landmarks from FK and the spine point as `Hips_origin + R_hips·(lever · source_primary)`,
and hand both to the converter. *Band:* the recovered `Hips` world rotation equals the posed
one to **≤ 0.01°** on every frame. *Must-fail:* the same posed body with
`spine_world_z_up_m=None` must read **≥ 10°**. *Blind to:* whether the pelvis landmark is
where a real pelvis is — this is closure between two halves of our own code, exactly as
D3's was.

**B6 — D3 CLOSURE on the rebuilt GLB.** The delivered GLB, parsed **from its own bytes**
and forward-kinematicked node by node, reproduces `forward_kinematics_positions` on the
track's own rest to **≤ 1e-4 m**, max over every frame and joint, on both subjects; reader
reused from `tools/compare/d3_skeleton_gate.py`. *Prediction:* 5e-7 m or smaller — D7
changes rotations, not the exporter's identity.

**B7 — SILHOUETTE (I6 partwise), the one band the candidate cannot optimise.**
*Pre-registered:* torso+legs **rises** on the bent tercile with its interval clear of zero
(D2b isolated the root's own share at −0.022 / −0.013 IoU whole-person, tilt-dependent);
arms **within their CI of D3** (the thorax frame is untouched and the clavicle chain hangs
off it, not off `Hips`); the upright tercile **within its CI**; MAMMA's mesh oracle
**bit-identical** between runs, which is the proof that reusing the `artifacts/compare/i6`
caches changed nothing. **And the competing prediction, named before the rebuild:** the
shipped rest-pitch is a convention with a measured residual of 1.75–4.42° median (up to
10.03°), and a residual pitch δ moves the root by ≈ 1.4–6.9 mm on **every** frame, upright
ones included, where D2b measured a 10 mm root shift costing 0.045 and 0.093 IoU. **So the
upright tercile may fall, and if it does the mechanism is the convention constant and not
the pelvis frame.** Both outcomes are recorded; neither selects anything. The
rigid-translation arm is reported as an **over-attribution**, not an isolation: D7 leaves
the leg roots on the captured hips, so translating D3's mesh moves legs D7 does not.

**B8 — REBUILD hygiene.** The 8 observation files under the rebuild's `work/` are
byte-identical to the delivered build's before and after (`work/` is **copied**, never
symlinked, and nothing re-extracts or re-detects), and the rebuild's
`triangulated_world_positions_z_up_m` is byte-identical to the delivered build's.

### 0.6 Reported, never banded

The rig's `Hips` **joint's** horizontal offset from the captured hip midpoint and its
correlation with trunk tilt, beside the world-vertical degenerate — D2b's rig-hip-mid
instrument reads zero by construction since D2b, so this replaces it; **it must move**, and
a constant can zero it, so it is not a band. The **ground hoist**, with the prediction
written before the rebuild: it moves by no more than the `Hips`-joint shift, a few mm, and
the exact number is computed offline from the D3 delivered track plus the real triangulated
`Spine1` before the rebuild runs. **Rung 11** vs MAMMA `pred_joints`, all 15 joints, subject-
mapped by 3D pelvis agreement and never by index (`body_id-00` is our subject **1**); MAMMA's
`gt_*` arrays are byte-copies of `pred_*` and are never scored against. **MAMMA's pelvis as
an oracle arm**, agreement only: `smplx_pose[:, :3]` is `global_orient` = the pelvis, body
joints spine1/spine2/spine3 are 3/6/9, so MAMMA's pelvis pitch minus its spine3 pitch is its
own pelvis-versus-thorax separation, against ours. **Facing**: `Hips` forward-dot median
> 0.9 and p05 > 0 on every camera against both references, all 14 handedness signs
unchanged. **The head gate unchanged exactly** — the head solve reads `_thorax_frames`, not
the converter's `Hips`. **Frames over 800 °/s at `Hips`**, reported and never banded,
because a reject zeroes it by construction.

### 0.7 Statistics and the standing rules

Block bootstrap (moving block 15, 2000 draws, fixed seed, both arms on identical draws)
wherever a take statistic is quoted; lag-1 autocorrelation on this take is 0.99. Same
denominator throughout. Every band ships with a degenerate that fails it. Every band the
candidate could optimise directly is **reported, not banded**. MAMMA reports and never
selects, and nothing MAMMA-derived enters `src/`, a constant or the delivery.

### 0.8 What this step will be blind to, whatever the numbers say

* **Whether the shipped rest-pitch convention is the performer's.** The delivered
  performer's own SOMA rest is unobservable; a rigid fit determines pose only up to a
  constant, and this constant is chosen from five other people's skeletons.
* **Whether SOMA-77's `Spine1` is where a real L5/S1 is.** Rigidity says the point is stably
  placed relative to the pelvis, not that it is anatomically right — a stable segment can be
  stably wrong.
* **The lumbar spine itself.** `Spine`, `Chest` and `UpperChest` still share one thorax
  frame; D7 separates the pelvis from the trunk and leaves the trunk one block.
* **The mesh.** Vertices, weights and bind matrices are the asset's; the pelvis now rotates
  relative to the chest inside a skin whose bind never saw that. Re-skinning is **D6**.
* Depth, a left/right mirror of a fore-aft symmetric pose, and anything inside the outline —
  the silhouette's standing blindnesses.
* **Generalisation.** One take, two performers, 150 correlated frames, and five synthetic
  clips from one motion source.

---

## 1. What shipped

`positions_to_body_track` gained `spine_world_z_up_m` (`[frame, 3]`, SOMA-77's `Spine1`
triangulated under the pipeline's own association) and `pelvis_report_out`.
`_pelvis_world_frames` — module level and called by bare name, exactly as `_joint_origin`
and `_leg_root_offset` are, so an instrument substitutes it and runs every control through
the identical construction — turns those points into `Hips`' world rotation by a Kabsch fit
of `SOMA77_REST_PELVIS_TEMPLATE_M` onto the observed {root, `Spine1`, `LeftLeg`,
`RightLeg`}. `_spine_world_for_subject` triangulates the landmark exactly as
`_toe_world_for_subject` does the ball of the foot; `reconstruct_multiview` gained
`spine_landmarks_by_camera` and two diagnostics tuples (`spine_triangulation`,
`pelvis_frame`); the build script gained `_spine_landmarks` on SOMA-77 index 1.

**Exactly one line inside the converter's frame loop branches**, and it is guarded by
`if pelvis_world is not None`. `Spine`, `Chest` and `UpperChest` still take `torso_world`,
so `Spine`'s local becomes pelvis⁻¹·thorax through `_set_world`; the root formula is
untouched, so `_leg_root_offset`'s `R_hips · mid` now rides the **pelvis** instead of the
lean. Four constants are registered in `tools/compare/provenance.py` and the audit runs
**CLEAN** (96 entries, no leak).

## 2. The mechanism, in one sentence

`Spine1`, `LeftLeg` and `RightLeg` are all **direct children of `Hips`** in SOMASKEL77, so
`posed[child] − posed[Hips] = R_hips · (rest[child] − rest[Hips])` exactly; a Kabsch fit of
the three rest offsets onto the three observed ones inverts that identity, and the pelvis
rotation falls out with a residual of **4.2e-8 m** on noiseless input. `Spine2` is a child
of `Spine1`, so anything measured to it carries lumbar flexion (14.5 / 20.9° median / p95
on the squat clip) and is a *lumbar* direction, not a pelvis one.

## 3. The gate — `artifacts/compare/d7-pelvis-frame/gate.json`

| band | result | verdict |
|---|---|---|
| **B1 CLEAN SYNTHETIC.** every rigid candidate vs the posed `Hips` rotation, five clips, band ≤ 0.01° median / ≤ 0.05° max | **0.0000° on every clip, every candidate** | **PASS** |
| **B1 degenerate.** thorax-as-pelvis (today's code), band ≥ 20° on the squat's bent frames | 26.97° | PASS (fails as required) |
| **B1 degenerate.** world-vertical, band ≥ 20° | **5.25°** | **REFUTED** — §5 |
| **B1 lumbar.** best lumbar arm, band ≥ 5° | 12.22° | PASS |
| **B2a NOISY SYNTHETIC, the selector.** winner below thorax **and** world-vertical **and** the best lumbar | C **10.19** vs thorax 27.61 ✓, best lumbar 14.34 ✓, **world-vertical 7.32 ✗** | **FAIL** |
| **B2b the window.** interior optimum + lag + attenuation, over-smoothed must fail on the fast clip | the protocol could not be executed; selection is 0 by the pre-registered fallback | reported |
| **B3 RIGIDITY on the real take** (sd_mm, the pre-registered reading rule) | root→Spine1 5.09 / 8.44 mm, mid(hips)→Spine1 6.61 / 11.10 mm, against body controls 9.39–26.62 / 10.24–47.73 mm | **TRUSTED** |
| **B4 ROUND TRIP canonical.** legs 0.00, torso 0.00, arms 0.55 / 0.08 | exactly that | PASS (and **blind to D7**, §0.5) |
| **B5 EXACT-RECOVERY ORACLE.** band ≤ 0.01° | **2.1e-6°** worst | **PASS** |
| **B5 must-fail.** the same posed body with no spine landmark, floor 10° | 34.38° | PASS (fails as required) |
| **B6 D3 CLOSURE** on the rebuilt GLB, read from its own bytes, band 1e-4 m | 4.8e-7 / 5.2e-7 m | **PASS** |
| **B8 REBUILD hygiene.** 8 observation files byte-identical; triangulation byte-identical | all true | **PASS** |
| **LEGACY BIT-IDENTITY.** the whole real `reconstruct_multiview` with no spine feed vs the pre-D7 delivery | rotations, root and contacts byte-identical on both subjects | **PASS** |
| **B7 SILHOUETTE.** the pre-registered clauses | **NOT MEASURED**, §6 | verdict withheld |
| **HEAD GATE UNCHANGED EXACTLY.** `head_orientation` and `toe_triangulation` diagnostics, before vs after | byte-equal | **PASS** |
| **FACING (reported).** `Hips` forward-dot > 0.9 and all handedness signs unchanged | 0.972 / 0.962 and 0.984 / 0.979; **0 of 16 signs changed**; `Hips` is the ONLY joint whose forward-dot moved at all | reported, and it holds |

**Overall: FAIL, on B2a, and the failure is the finding.**

## 4. The reported figures

| figure | subject 0 | subject 1 |
|---|---|---|
| `Hips` **joint's** horizontal offset from the captured hip midpoint, bent tercile, before → after | **65.1 → 25.3 mm** [CI 11.8, 27.3] | 79.2 → 79.7 mm |
| … correlation with trunk tilt, before → after | +0.979 → +0.876 | +0.977 → +0.856 |
| **the ground projection's own hoist**, before → after (both arms without the head and toe solves) | **72.68 → 25.34 mm** median | 35.54 → 34.11 mm |
| lowest joint height (a proxy, not the hoist), before → after | −0.1308 → −0.0971 m | −0.8253 → −0.8329 m |
| foot contacts, before → after | [33, 49] → [36, 49] | [2, 20] → [6, 20] |
| `Hips` frames over 800 °/s | 0 → 0 | 0 → 0 |
| `Hips` median angular rate | 63.4 → 67.5 °/s | 84.0 → 96.6 °/s |
| silhouette IoU, whole person, 8 cells | +0.035, −0.024, +0.030, −0.003 | −0.008, +0.007, +0.006, −0.007 |
| MAMMA's own pelvis-to-thorax separation (SMPL-X joints 3·6·9 composed), beside ours (`Spine`·`Chest`·`UpperChest`) | 32.34° / ours 18.92° | 39.64° / ours 15.82° |

**The two performers behave differently and the instruments agree about it.** Subject 0's
`Hips` joint moved 40 mm closer to its own captured hips on bent frames and its silhouette
rose on three cameras of four; subject 1's offset did not move and neither did its
silhouette. On subject 1 the measured pelvis really is tilted away from the trunk, so
correcting the frame does not bring the joint back to the hip midpoint — which is what a
pelvis frame is *for*, and the offset is reported and not banded precisely because "smaller"
is not the same as "right".

**MAMMA reports and selected nothing.** Both fitters now carry a pelvis that is not welded
to the thorax (ours read 0.000° by construction before D7). The two numbers are on different
chains and different conventions and are not comparable beyond that sign.

## 5. The pre-registered expectations, and whether each was met

| | expectation (§0) | outcome |
|---|---|---|
| B1 clean | every rigid candidate ≤ 0.01° median, ≤ 0.05° max on five clips | **MET**, 0.0000° everywhere |
| B1 degenerate, thorax | ≥ 20° on the squat's bent frames | **MET**, 26.97° |
| B1 degenerate, world-vertical | ≥ 20° on the squat's bent frames | **REFUTED**, **5.25°**. The true pelvis on all five clips sits **1.1–12.7°** from world vertical, so on this motion source a frozen upright pelvis is already a good approximation. Nobody had measured that before the band was written. |
| B1 lumbar | ≥ 5° | **MET**, 12.22° |
| B2a selector | the winner below all three controls | **REFUTED on one of three.** C beats thorax by 17.4° and the best lumbar by 4.1°, and **loses to world-vertical by 2.9°.** |
| B2a prediction | A's median ≈ 2.5× B's (the lever ratio) | **MET**, 30.49 / 13.26 = 2.30× |
| B2a prediction | C ≥ B | **MET**, 10.19 vs 13.26 |
| B2b window | an interior optimum with lag ≤ 1 frame and attenuation ≥ 0.9 | **NOT EXECUTABLE.** At this noise the lag estimator reads −3.1 to −8.1 frames and the attenuation 1.2–7.0 (*amplification*) at **every** window, window 0 included — both instruments are unusable here. The pre-registered fallback selects **0**. The p95 does improve monotonically to the widest window (36.1 → 17.2°) and that is reported, not selected on. |
| B3 rigidity | candidates' sd_mm ≤ the worst body control's | **MET**, 0.16–0.33× the worst control. And the reading rule earned its keep: on sd % `root→Spine1` reads **16.4 %** on subject 1, at the very top of the control range, where on sd_mm it is 8.44 against 47.73. |
| B4 round trip | 0.55 / 0.08, legs and torso 0.00 | **MET exactly** — and trivially, as pre-registered |
| B5 oracle | ≤ 0.01° | **MET**, 2.1e-6° |
| B5 must-fail | ≥ 10° | **MET**, 34.38° |
| B6 closure | ≤ 1e-4 m, prediction 5e-7 or smaller | **MET**, 4.8e-7 / 5.2e-7 m |
| B8 hygiene | byte-identical observations and triangulation | **MET** |
| §0.6 hoist | moves by no more than the `Hips`-joint shift, "a few mm" | **REFUTED on subject 0, in the good direction.** The ground projection's own correction fell **72.68 → 25.34 mm**, a change of 47.3 mm — larger than "a few", and larger than the `Hips`-joint shift itself (27.4 → 9.7 mm over the whole take, 65.1 → 25.3 on the bent tercile), so the prediction's *bound* is exceeded too. The mechanism is clear enough: the rig no longer sinks on a leaning pelvis, so the projection has far less to correct. Subject 1: 35.54 → 34.11 mm, within the prediction. **Also a deviation:** §0.6 said this would be computed offline BEFORE the rebuild; it was not, it was computed after, and the pre-registered number was therefore never at risk. |
| §0.6 facing | `Hips` forward-dot median > 0.9 and p05 > 0, all handedness signs unchanged | **MET.** 0.9724 / 0.9623 (subject 0) and 0.9841 / 0.9794 (subject 1) against our capture and MAMMA; **0 of 16 handedness signs changed**; and `Hips` is the only joint in the whole report whose forward-dot median moved by more than 0.02, which is exactly the joint D7 touches. **And a circular figure died here:** before D7 the `Hips` forward-dot against our own capture was **1.0000 by construction**, because `Hips` took the trunk frame built from the very landmarks the capture's forward is derived from. It is a real measurement for the first time. |
| §0.6 head gate | unchanged EXACTLY | **MET**, verified rather than asserted: `head_orientation` and `toe_triangulation` diagnostics are byte-equal between the pre-D7 delivery and the rebuild. |
| §0.6 offset must move | it must move | **MET on subject 0** (65.1 → 25.3 mm), **not met on subject 1** (79.2 → 79.7) — and §4 says why that is the expected behaviour of a genuinely tilted pelvis, not a null result |
| B7 silhouette | torso+legs rises on the bent tercile; arms within CI; upright tercile within CI; MAMMA oracle bit-identical | **the first three NOT MEASURED** (§6); the oracle **is bit-identical in all 8 cells**, which is the proof that reusing the mask caches changed nothing |

## 6. What was not run, and why

* **The part-wise and tercile silhouette cuts.** `silhouette.py` writes summaries, not
  per-frame arrays, and `silhouette_partwise.py` / `silhouette_vs_tilt.py` have their
  `BUILDS` hard-coded to the D2 paths. What was measured is the whole-person before/after
  by camera and subject, on the same masks, the same rasteriser and the same mask cache,
  with MAMMA's mesh oracle bit-identical between the runs. **Four of eight cells rose and
  four fell, every change ≤ 0.035 IoU**, so at whole-person resolution the photographs
  neither confirm nor deny D7. A whole-person band invented now would be a band chosen
  after seeing the numbers, so none is: the verdict is withheld rather than manufactured.
* **The block-bootstrap CIs on the silhouette difference**, for the same reason.
* **The rigid-translation over-attribution arm.** It would be an upper bound and not an
  isolation — D7 leaves the leg roots on the captured hips, so translating D3's mesh moves
  legs D7 does not.
* **Rung 11 (`mamma_scoreboard.py`) was not re-run** on the D7 delivery. D7 changes the
  `Hips` channel and the root, both of which move every joint's world position, so rung 11
  *would* move; it is a gap. (`facing_location.py` **was** run — §5.)

## 7. The instrument findings, recorded because they change how the numbers read

**The pre-registered noise model is 1.7–2.9× harsher than the real detector, and the
instrument says so itself.** The noisy arm produces **18.85 mm** of `mid(hips)→Spine1`
length spread; `d7_pelvis_rigidity.py` measures **6.61** and **11.10 mm** on the real take
with the real detector. The calibration figure is in the report beside every angular number
because *when a fit is short, suspect the instrument*. A sigma sweep — **explicitly
post-hoc, and labelled as such in the report**, because it was added after the band had
already failed — reads:

| σ (px) | synthetic mid(hips)→Spine1 sd | C (ships) | B | world-vertical | thorax |
|---:|---:|---:|---:|---:|---:|
| 3.20 (pre-registered) | 21.61 mm | 10.19° | 13.26° | **7.32°** | 27.61° |
| 2.40 | 15.97 mm | 7.98° | 12.22° | **7.01°** | 27.09° |
| 1.60 | 11.97 mm | 7.55° | 9.68° | **7.11°** | 27.73° |
| 1.12 | 8.03 mm | **5.49°** | 7.29° | 6.54° | 27.09° |
| 0.80 | 5.72 mm | **3.80°** | 5.60° | 6.09° | 27.05° |

The real take's own bracket is 6.61–11.10 mm, i.e. between the ×0.25 and ×0.5 rows: **at
subject 0's noise the pelvis frame wins, at subject 1's it is level with a constant.**
That is a reading, not a verdict, and **the pre-registered verdict stands as it fell.**

**And a correction of my own, recorded rather than quietly fixed.** The first run of this
sweep, and the first version of the noisy arm's headline, put a Savitzky-Golay window on the
*spine point* — which is not what ships. `_spine_world_for_subject` triangulates per frame
and never filters, exactly like the toe feed it extends. Every figure above and in §3 and §5
is now on the **shipped** arm (the spine point raw); the smoothed variant is kept in the
report under `noisy_stride_1_spine_smoothed_NOT_WHAT_SHIPS` and reads C **9.96°**, i.e.
smoothing the input would buy 0.23° and would not change any verdict. The switch was made
after the first result was seen, and that is why it is written down here.

**The plan card is wrong about where the rest lives**, §0.2. The shipped rest-pitch is a
convention with a measured 1.75–4.42° spread, registered as third-party provenance with its
cost stated (a residual δ moves the root by `|mid|·sin δ`, about 80 mm of lever).

## 8. Tests

`tests/test_pelvis_frame.py`, **14 tests, all passing**, and every one asserts it imported
this worktree's package. They cover: a clean synthetic body round-trips the pelvis frame
through the real converter (the exact-recovery oracle); the no-spine and world-vertical
degenerates fail it; a legacy call never reaches the pelvis code at all and is byte-equal;
the legacy `Hips` channel is still the trunk-line construction bit for bit; a gap is
interpolated with no per-frame definition switch; too few resolved frames fall the **whole**
subject back with a reason and a byte-equal legacy track; wrong-shaped input and an unknown
source are rejected; the report is carried out for the diagnostics; the diagnostics
serialise the new fields and a legacy run carries empty ones; all three candidates run
through the one code path; and the window is a knob that moves the answer.

Across `test_body.py`, `test_body_export.py`, `test_body_binding.py`,
`test_performer_skeleton_schema.py`, `test_d3_export_closure.py`, `test_pelvis_frame.py`,
`test_root_placement.py`, `test_soma_motion.py` and `test_commercial_multiview.py`:
**93 pass, 1 fails**, and the failure is D3's known one —
`test_body_export.py::test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound`
asserts the pre-D3 exporter's root translation `(0, 0.8, 0)` and has failed since D3 shipped
(D3 review §6). It is unrelated to D7 and, like D3, is reported rather than fixed.

No existing test was edited (the user has uncommitted edits in `tests/`).

## 8b. Deviations from the brief and from the pre-registration

Every one of these is a departure from what was written down, listed so none of them has to
be inferred from a diff.

1. **The pre-registration was committed as this document's §0, not as
   `artifacts/compare/d7-pelvis-frame/gate.json`.** `artifacts/` is a symlink into the main
   tree and nothing under it is tracked, so `gate.json` cannot be committed at all. §0 was
   written at the same moment and is the tamper-evident copy.
2. **The plan card is wrong about `somaskel77-v1.json` carrying rest geometry** (§0.2), so
   the "constant from the rest geometry" became an explicit convention with a measured
   spread, registered as third-party provenance.
3. **The noisy arm's headline was recomputed** after the first run modelled a pipeline that
   does not ship (§7). Both arms are in the report.
4. **The calibration sweep is post-hoc** and is labelled as such in the report; the
   pre-registered verdict stands on the pre-registered noise.
5. **The hoist was computed after the rebuild, not before** (§5), so its pre-registered
   number was never at risk.
6. **The silhouette's pre-registered clauses were not measured** (§6) and no substitute band
   was invented.
7. **Rung 11 was not re-run** (§6).
8. **An earlier version of the MAMMA oracle measured `spine3` alone** instead of the composed
   `spine1·spine2·spine3` chain, and was corrected before this was written.
9. **`tools/compare/provenance.py` rewrote the shared `artifacts/compare/provenance.json`.**
   It is regenerable and untracked, but the main tree's copy is now this branch's.
10. **`tools/compare/provenance.py` was edited** (four new constants). `ladder.py` and
    `status.py` were not touched, no existing test was edited, and nothing was written under
    `artifacts/commercial-multiview-soma77/`.

## 9. What this is blind to

Everything in §0.8 stands, and the run added three more:

* **The silhouette's pre-registered clauses.** §6. The one band the candidate cannot
  optimise was measured only at whole-person resolution, where it is silent.
* **The window.** No usable lag or attenuation instrument exists at this noise, so "0" is a
  refusal to smooth rather than a demonstration that smoothing does not pay. The p95 says
  it might.
* **Generalisation of the world-vertical result.** Five clips from one motion source, all
  with a near-upright pelvis. A take with real pelvic tilt would separate the candidate from
  the constant, and this fixture cannot.

## 10. What needs a decision

**D7 is built, measured, and it fails its own pre-registered band. It should not be merged
on the strength of this gate.** The three things that would settle it, in order of cost:

1. **A motion source with real pelvic tilt.** The world-vertical control wins because these
   five clips barely tilt the pelvis. That is a fixture property, and lane H's owned capture
   is where it gets fixed.
2. **A lag/attenuation instrument that works at this noise**, so the window can be selected
   rather than refused.
3. **The part-wise silhouette on the D3 and D7 rebuilds**, which needs `silhouette.py` to
   emit per-frame arrays or `silhouette_partwise.py` to take its builds as arguments.

## 11. In plain language

The character's hips and chest used to be locked together. Wherever the chest leaned, the
hips leaned with it — so when a performer bent over, the code put the character's pelvis
somewhere it had never been, and the whole figure shifted sideways by up to eight
centimetres. This step gives the pelvis its own answer, read from a point on the lower spine
that the detector already finds and nobody was using.

The plumbing works, and it is proved rather than asserted: pose an imaginary body with its
pelvis deliberately turned away from its chest, feed it in, and the pelvis comes back out to
two millionths of a degree. Read the delivered file straight off disk and its joints match
the code's own arithmetic to under a thousandth of a millimetre. Turn the new input off and
the file that comes out is byte-for-byte the old one. On one of the two performers the hip
joint moved four centimetres closer to where the cameras say the hips are, and the outline
check rose on three cameras of four.

But the honest headline is a failure, and it was written down as a test before anyone looked.
The test said: a measured pelvis must beat a deliberately dumb answer — a pelvis frozen
bolt upright, never turning at all. It does not. It beats the old code by a wide margin, but
against the dumb answer it loses by two and a half degrees. The reason turns out to be the
practice footage: in every clip available, the performers' pelvises barely tip away from
upright, so "always upright" is already nearly right there. On footage with real hip tilt
that would not hold — but we do not have that footage, and a step does not get to pass its
own test by explaining the test away. So the work is written up, the failure is recorded,
and whether it ships is not this step's call to make.
