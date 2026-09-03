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
