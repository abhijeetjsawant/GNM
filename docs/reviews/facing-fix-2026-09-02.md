# D1 (fix): the left/right naming mirror, repaired and gated

**2026-09-02, branch `ladder/D1-fix`. NOT MERGED.** Instruments: the extended
`tools/compare/facing_location.py` (before and after arms), `facing_surface_probe.py`,
`tools/compare/facing_fix_gate.py` (report
`artifacts/compare/d1-fix/facing-fix-gate.json`), and the four regression instruments run
unchanged. Companion to `docs/reviews/facing-location-2026-09-02.md`, which located the
defect and proposed a fix that this step **did not take** — see §1.

**One band failed and was not relaxed.** The card's `median > +0.9` forward-dot holds on
the pelvis, chest and neck and fails on the head and the mesh nose; the oracle shows it is
unreachable there by any pipeline, so it is reported failing with a replacement *proposed*
and not applied. The card's I6 band ("IoU must not fall on any cell") also fails, on 5 of
8, and §6 shows why that is a finding about I6 rather than about the repair. Neither is
buried: they are §5 and §6.

## 1. The route shipped, and the route rejected

The review proposed negating X throughout the rest skeleton **and mirroring the bound
mesh** — negate each vertex's X, swap each vertex's Left/Right skin weights. That reaches
the right joint positions on a bilaterally symmetric bind pose, and it is **a reflection
of the surface**: it reverses every triangle's winding and inverts every normal, so the
character renders inside out. Measured, not argued — the mesh's signed volume under that
transform, with the original triangle order, changes sign exactly
(`tests/test_facing_fix.py::test_the_geometry_route_turns_the_mesh_inside_out`).

**No band in this gate could have caught it.** The joints land in the same places, the
forward-dot is unchanged, the handedness sign is unchanged, and a silhouette IoU is
unchanged because an inside-out mesh has the same outline. It is the classic mirrored
asset that passes every joint gate.

**Shipped instead: a RELABEL.** Skin weights index bones, so changing *which AutoAnim name
each bone answers to* moves no geometry at all:

1. `body.CANONICAL_HUMANOID` — negate X on the twelve off-midline rest translations, and
   flip `_finger_joints`' side sign with them. Joint order, names and every other
   component are untouched.
2. `body_provider.DEFAULT_MPFB_JOINT_MAP` / `DETAILED_MPFB_JOINT_MAP` — each side maps to
   its own MPFB side.
3. The asset — derived by permuting which name each MPFB bone answers to
   (`tools/compare/d1_asset_relabel.py`). `vertices_m`, `triangles`, the weight *values*
   and the parent table all come through byte-identical, and the derivation asserts that
   last one rather than assuming it.
4. `commercial_multiview` — both `_frame_alignment` source secondary axes from `(-1,0,0)`
   to `(1,0,0)`, and `_finger_rest_local`'s curl sign with the skeleton.
5. `head_orientation.CANONICAL_HEAD_AXES` — restated (§2).
6. `soma_motion._DELTA_MAPPING` — the compensating side swap removed (§4).

## 2. The head axes: a yaw, not a mirror

`CANONICAL_HEAD_AXES` shipped as columns left `(-1,0,0)`, up `(0,0,1)`, forward `(0,1,0)`.
Its **determinant is +1**: a proper rotation, and `left × up = forward` holds. Against the
corrected frame — left `(1,0,0)`, up `(0,0,1)`, forward `(0,-1,0)`, which also satisfies
`left × up = forward` — the relative rotation is `diag(-1, +1, -1)`: **a 180° yaw about
the skull's own long axis, not a reflection.** Up is fixed; left and forward both reverse.
That matters, because a repair that *reflected* instead of rotating would satisfy the
forward assertion and put the eyes on the wrong sides.

## 3. The repair is an exact half-turn, and that is the strongest figure here

Every torso and head joint's **world** rotation in the rebuild is the delivered one turned
**exactly 180.0000° about its own local +Y**, the rig's up axis, on **every one of the 150
frames**, both subjects, `Hips` `Spine` `Chest` `UpperChest` `Neck` `Head`. Median and p95
both 180.0000; recovered axis `[0, 1, 0]`.

This is a stronger statement than any forward-dot — a dot is one number per frame, a
rotation is three — and it says the repair reversed the axis the mesh's face lies on and
**moved nothing else**. Everything downstream follows from it arithmetically, including
the one figure that does not come back at the magnitude it left with (§6).

Corroborating invariants, all measured:

| quantity | before | after |
|---|---|---|
| `triangulated_world_positions_z_up_m` | — | **byte-identical** |
| `root_translation_m` | — | max 0.0005 mm (float32 rounding) |
| I1 round trip, legs / arms / torso (subj 00) | 0.00 / 41.57 / 0.00 mm | **0.00 / 41.57 / 0.00** |
| I1 round trip, legs / arms / torso (subj 01) | 0.00 / 47.05 / 0.00 mm | **0.00 / 47.05 / 0.00** |
| rung 7/11 scoreboard median, subj 00 capture/canon/sized | 36.117 / 151.584 / 136.581 mm | **identical to 3 dp** |
| rung 7/11 scoreboard median, subj 01 | 41.297 / 137.758 / 113.038 mm | **identical to 3 dp** |

A by-name joint score cannot see a consistent relabel, so "unchanged" is the correct
result and a drift would have been an inconsistency, not noise. The deltas are 0.0000.

## 4. The mirror had **five** sites, not three

The locate step named `DETAILED_HUMANOID`, the MPFB asset and `CANONICAL_HEAD_AXES`. Two
more were found by grep during the repair, and neither is reachable from the other three.

**Site 4 — `body_provider.DEFAULT_MPFB_JOINT_MAP`, where the mirror was born.** It sent
AutoAnim `Left*` onto MPFB `.R` bones *deliberately*, with the reason in a comment above
it: *"MPFB's anatomical .L is positive Blender X, whereas AutoAnim's canonical Left joints
are negative X. The explicit mapping records that label swap."* The asset is not an
independent site; it is this map's output.

**Site 5 — `soma_motion._DELTA_MAPPING`, which compensated for the mirror.** It sent our
`Left*` joints to SOMA's `Right*` ones, with the same reason written beside it: *"SOMA
anatomical left is positive source X, while AutoAnim's reviewed canonical skeleton defines
Left on negative X."* SOMA declares the same coordinate system we do. **Removing the
mirror without removing that swap would have shipped every SOMA performer with their arms
and legs exchanged**, and nothing on that lane has a capture fixture that would have caught
it. It is asserted by unit test and by nothing measured.

And the other direction: **`speech_motion`'s bundled SMPL-X55 map was already mirrored** —
it sends `Left*` to `left_*`, and SMPL-X's anatomical left is +X, so that lane was *wrong
before the repair and is right after it, without being touched*. Two retarget boundaries,
opposite errors, one convention. That pair is the whole argument for the grep.

**The shortest proof of the defect needs no capture, no reference and no asset.**
`HumanoidSkeleton.as_dict()` publishes `handedness: right, up_axis: +Y, forward_axis: +Z`.
In that frame the anatomical left is `up × forward` = **+X**, and every joint named `Left`
sat at −X. The skeleton contradicted the contract it published. It is now a unit test.

## 5. The gate

`artifacts/compare/d1-fix/facing-fix-gate.json`. The **before** arm is
`artifacts/compare/facing-location.json` — the 2026-09-01 delivery, old bytes scored by
the old code, **never recomputed**: `triple()`'s `across` comes from forward-kinematic
positions, so running the repaired FK against the old rotations would flip the handedness
sign for a reason that has nothing to do with the delivery. The **after** arm is the same
instrument pointed at a rebuild under `artifacts/compare/d1-fix/delivery`. The delivered
directory was never written to.

### Forward-dot, before → after (median, `vs_our_capture` / `vs_mamma`)

| part | subject 00 before | subject 00 after | subject 01 before | subject 01 after |
|---|---:|---:|---:|---:|
| `Hips` | −1.000 / −0.994 | **+1.000 / +0.994** | −1.000 / −0.991 | **+1.000 / +0.991** |
| chest | −0.992 / −0.986 | **+0.992 / +0.986** | −0.995 / −0.991 | **+0.995 / +0.991** |
| `Neck` | −0.963 / −0.956 | **+0.963 / +0.956** | −0.983 / −0.963 | **+0.983 / +0.963** |
| `Head` | −0.890 / −0.891 | **+0.890 / +0.891** | −0.941 / −0.928 | **+0.941 / +0.928** |
| mesh nose | −0.939 / −0.926 | **+0.792 / +0.786** | −0.955 / −0.952 | **+0.833 / +0.813** |
| feet | +0.970 / +0.951 | **+0.970 / +0.951** | +0.908 / +0.889 | +0.919 / +0.899 |
| *ORACLE: MAMMA forward vs ours* | +0.994 | +0.994 | +0.991 | +0.991 |

Every joint figure is the **exact negation** of its before value, which is §3 read one
number at a time. The feet do not move: both after medians sit inside their before
intervals, on both references (`feet_must_not_move`: PASS).

### Handedness — the fix stated as one bit

| arm | before, subj 00 / 01 | after |
|---|---:|---:|
| our triangulated capture | −1 / −1 | −1 / −1 |
| MAMMA `pred_joints` | −1 / −1 | −1 / −1 |
| the delivered rig, forward from its **feet** | −1 / −1 | −1 / −1 |
| the same rig, forward from its **torso** | **+1 / +1** | **−1 / −1** |
| the delivered **mesh's own skin** | −1 / −1 | −1 / −1 |
| rest skeleton in `neutral-body.npz` | **+1** | **−1** |
| rest skeleton in `DETAILED_HUMANOID` | **+1** | **−1** |

One rig had two answers; now it has one, and it is the one every real human on this
fixture reads.

**And the mesh's own skin says *yaw*, not *reflection*.** The two readings have to be taken
together: the mesh's nose against the performer's forward went −0.939 → **+0.792**, and the
mesh's +X side against the performer's *left* went −0.976 → **+0.978**. Both reversed is a
180° turn of a self-consistent human. Only one reversed would have been a reflection — which
is the failure mode the rejected geometry route would have produced, and this pair is the
measurement that rules it out on the route actually shipped. The `Head` and `Chest` joints'
own +X read the same way (−0.938 → +0.938, −0.974 → +0.974).

### The degenerates — every one behaves as required

| control | forward-dot | handedness | verdict |
|---|---:|---:|---|
| the repaired build (identity row) | +0.992 / +0.995 → PASS | −1 → PASS | reference row |
| **180° yaw** of the repaired torso | −0.992 / −0.995 → **FAIL** | −1 → PASS | PASS |
| **90° yaw** | +0.073 / +0.030 → **FAIL** | −1 → PASS | PASS |
| **sagittal mirror** | +0.992 / +0.995 → PASS | **+1 → FAIL** | PASS |
| **the pre-repair build**, identical bands | **FAIL** (20 of 20 cells) | arms disagree → **FAIL** | PASS |

The mirror control is the reason the handedness band exists: it faces exactly where it
faced, so no forward-dot and no silhouette can reject it. The 180° control is the reason
the forward-dot band exists: a proper rotation leaves the sign untouched. Neither band is
redundant, and this is the demonstration.

The controls are computed on the **delivered rig's own torso frame**, not on the capture
arm as at D1 (locate). On the capture arm the untransformed ceiling is +0.86 (it dots a
foot direction against a pelvis/neck forward), so no "> +0.9" band can be applied to it
and the mirror control could never have been shown to *pass* the forward-dot. On the
delivered arm the identity row sits at +0.99, which is the scale the gate actually uses.

### One band failed, and it is the band that is wrong

The D1 gate card asks for **median > +0.9** on `Hips`, chest, `Neck`, `Head` and the mesh
nose. Applied literally it **FAILS on 6 of 20 cells**: `Head` on subject 00 (+0.890 /
+0.891) and the mesh nose on both subjects (+0.792 / +0.786, +0.833 / +0.813).

**The band is not reachable there by any pipeline, and the oracle says so.** The reference
forward is the *body's*, built from the pelvis and the neck. A head is not welded to a
body — a performer can look sideways with their chest square — so the head's dot against
the body's forward is bounded by the performer's own neck. Scoring **MAMMA's own head**,
from its `smplx_pose` chain, through the identical frames:

| | oracle: MAMMA's head | ours: delivered `Head` |
|---|---:|---:|
| subject 00 | **+0.844** [+0.819, +0.903] | +0.890 [+0.867, +0.914] |
| subject 01 | **+0.857** [+0.753, +0.949] | +0.941 [+0.882, +0.982] |

The reference fitter's own head does not reach +0.9 either. This lane's standing rule is
that *a gate no oracle can pass is miscalibrated*, so the band was left exactly as written
and reported failing.

**And "ours ≥ the oracle" is not put in its place, because a constant would pass it.** A
head welded to the torso scores the *chest* figure, +0.992 / +0.995, comfortably above
+0.844 — and that welded head is the exact degenerate 981e437 was written to kill. *No gate
a constant can pass*: the oracle comparison is reported as a **diagnostic** and is not
binding. Nor is "ours above MAMMA" a virtue to claim: a **stiffer** head scores *higher*
against a body-derived forward, so the two are in the same regime and neither is better.

What is binding instead: the **sign** band (median > 0 and bootstrap lower bound > 0 on
every part, subject and reference — PASS on all 22 cells), the feet band, the handedness
band, the exact-yaw measurement, the four controls and the regressions. A replacement head
band is *proposed and not applied*: require the median to fall **inside the oracle's
interval** (0.890 ∈ [0.819, 0.903]; 0.941 ∈ [0.753, 0.949]), which the welded head fails at
+0.99. It is proposed rather than adopted because the upper margins are 0.013 and 0.008 and
this lane does not quote a margin without a block bootstrap behind it.

### Why the mesh nose does not come back at the magnitude it left with

It is the only figure whose |value| changed (−0.939 → +0.792), and it is arithmetic, not a
residual.

The nose figure is a **surface chord**: the centroid of the 40 most anterior head vertices
minus the 40 most posterior. In the bind pose that chord runs **14.70° below** the head's
own rig +Z (nose at rig z = +0.157 and y = 1.517; occiput at z = −0.040 and y = 1.569).
Measured in world against its own `Head` joint's +Z it reads +0.968 — the same 14.6°.

The repair is a half-turn **about the head's up axis** (§3). A yaw reverses a vector's
component in the horizontal plane and **preserves its up component**. The chord has an up
component, so it cannot flip to the same magnitude: with the head's own +Z at ±0.890 and
the chord's up term at −0.079, −0.860 − 0.079 = −0.939 before and +0.860 − 0.079 = +0.781
after. Measured +0.792.

The mesh nose remains the one figure in this lane that **no joint name enters**, and it is
the one that answers I6 in I6's own currency. It is quoted here as a sign and a direction,
not against a +0.9 band it was never able to reach.

## 6. The regressions

### Rung 9, the shipped head: exactly invariant

| arm | before, P1 median / p95 (deg) | after |
|---|---:|---:|
| subject 00 candidate | 5.1207 / 17.8260 — **PASS** | **5.1207 / 17.8260 — PASS** |
| subject 00 oracle | 4.2635 / 13.9700 | 4.2635 / 13.9700 |
| subject 01 candidate | 6.3637 / 20.5794 — FAIL on p95 | **6.3637 / 20.5794 — FAIL** |
| subject 01 oracle | 4.6288 / 15.1900 | 4.6288 / 15.1900 |

Identical to four decimals on every arm and every control. That is the prediction, not a
surprise: the head gate **mean-removes each take**, so a constant right-multiplied yaw is
invisible to it. It is *"a tracking gate is blind to absolute orientation"* (CLAUDE.md)
demonstrated from the other side — the gate cannot tell a head pointing forwards from the
same head pointing backwards. Subject 01's p95 failure (20.58 against a band of 20.0) is
identical in both arms and is not caused by the repair.

**Both arms are at `THORAX_SMOOTHING_FRAMES = 15`, this branch's value.** The canonical
`artifacts/head-lane/head-gate-shipped.json` on disk is `main`'s **window-9** run (5.52 /
7.19 gated medians) and is not comparable to either. The before arm here is the pre-repair
head at window 15, reconstructed exactly rather than re-solved: `_anatomical_gauge` is
applied **after** the fit (`head_orientation.py:559`), so restating `CANONICAL_HEAD_AXES` is
a constant right multiplication by `diag(-1, -1, +1)` in capture coordinates and nothing
else — the optimisation, the template and the weight selection never see it. The shared
head-lane files were restored to `main`'s window-9 output after the run.

**The head gate now carries the forward-dot as a standing figure** — `absolute_facing_not_a_band`
in its own report, with the oracle beside it. Every band in that file mean-removes, so
until now its verdict could be read without any statement about which way the head points.

### I6, the silhouette: the strict band fails, and that is a finding about I6

The card asks that IoU **not fall on any of the 8 cells**. It falls on 5 and rises on 3, by
−0.053 to +0.015:

| cell | before | after | Δ |
|---|---:|---:|---:|
| A001 / 00 | 0.6275 | 0.5882 | −0.039 |
| A001 / 01 | 0.6225 | 0.6014 | −0.021 |
| **B001 / 00** | 0.5733 | **0.5882** | **+0.015** |
| B001 / 01 | 0.5907 | 0.5773 | −0.013 |
| C001 / 00 | 0.6143 | 0.5612 | −0.053 |
| C001 / 01 | 0.6192 | 0.6214 | +0.002 |
| D001 / 00 | 0.5442 | 0.5453 | +0.001 |
| D001 / 01 | 0.5167 | 0.5258 | +0.009 |

**This is not read as the repair making the surface worse, and the reason is measurable.
No joint moved** — §3 proves it to four decimals — so what changed is the *surface's*
facing with the skeleton held exactly still. **That is precisely the corrected control the
locate review asked for and could not run** (*"yaw only the vertices whose dominant weight
is a torso bone, about the torso vertical, and rescore"*). Its answer: every delta is far
inside the per-frame spread — each cell's own p05 sits 0.10–0.20 below its median — and the
signs are mixed. **I6 is insensitive to facing on this take with the limbs held.** The one
cell where facing is most visible, B001/subject 00, where the performer faces that camera
on every frame, moved *up*; that is one cell and is not evidence.

The band actually applied is *after median ≥ before p05*, which passes on all 16 cells
(8 whole-take, 8 distinguishable-half). The strict card band is reported FAILING beside it.

Turning the repaired mesh 180° hurts IoU on all 8 cells (0.22–0.48 against the delivered
0.53–0.59). **It also hurt before the repair, on all 8** (0.26–0.49 against 0.54–0.63), so
"the control now hurts" is true and *not discriminating* — it was true either way.

## 7. Blast radius

Full body suite plus everything importing `DETAILED_HUMANOID`, in the worktree, on the
repaired build: **182 passed, 6 skipped, 2 failed**. Both failures are **pre-existing on
the branch base**, proved by stashing and re-running at `HEAD`:

* `test_body_compositor.py::test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock`
* `test_head_orientation.py::test_body_track_head_is_a_constant_without_a_solve_and_moves_with_one`
  — the quantity it asserts (`ptp` of head travel) is **12.4708 before and 12.4708 after**,
  against a band of 20. The repair changes the *absolute* offset of that rotation (the
  local head rotation runs 77–90° before and 0–12° after, i.e. the head is no longer a
  constant 90° off its neck) and does not touch the quantity the test measures.

The whole repository suite adds one further pre-existing failure,
`test_phase4_app.py::test_home_and_health`, also confirmed at `HEAD`.

**Four assertions encoded the defect and were corrected, never deleted**, each with its
old value in a comment beside it:

| test | was | now |
|---|---|---|
| `test_body.py` | `LeftFoot` rest at `[-0.09, …]`, `RightToes` at `[+0.09, …]` | `[+0.09, …]` / `[-0.09, …]` |
| `test_body_provider.py` | `joint_map["LeftEye"] == "eye.R"`, `["RightUpperArm"] == "upperarm01.L"` | `"eye.L"` / `"upperarm01.R"` |
| `test_body_provider.py` | the fail-closed mutation set `LeftEye="eye.L"`, now the *correct* value | mutation is `"eye.R"` |
| `test_facing_location.py` | two tests asserting the mirror — their own docstring said *"if this ever fails, the fix has landed and the assertion should be inverted, not deleted"* | inverted |
| `test_soma_motion.py` | SOMA `RightArm` must reach our `LeftUpperArm` | reaches `RightUpperArm` |

`tests/test_facing_fix.py` adds 14 tests, one per site plus the two route arguments. Every
one was watched failing on the pre-repair build before any source line changed.

## 8. Overlays — the picture, on the camera the review named

`artifacts/compare/d1-fix/overlay-{A,B,C,D}001-f075-{BEFORE,AFTER}.jpg`, through the
fps-repaired `camera_overlay.py`, frame 75. B001 is the cell the locate step called
cleanest: the standing performer faces that camera on **every** frame of the take, and our
delivered character had its back to it on every frame. In `BEFORE` you see his shoulder
blades; in `AFTER` you see his face and chest. The POSE CHECK reads **58.9 mm / 76.9 mm**
on the two subjects, identical on all four cameras and unchanged by the repair — a constant
residual is the signature of correct timing, and it is the known root/hip placement offset,
which is D2's territory.

## 9. What this is blind to

Everything `facing_location.py` is blind to, this inherits: it scores **orientation** and
nothing else, so the clavicle-origin error, the bone lengths and the root/hip offset are
untouched and none is fixed by fixing this. A forward-dot near +1 says a part is not
reversed, not that it is accurate to a degree. The handedness figure is a **sign**.

New here, and belonging to this gate rather than to D1:

* **The repaired asset was derived, not regenerated.** It proves the relabel is consistent;
  it does **not** prove the pinned Blender/MPFB worker produces the same file.
  `artifact.request_sha256` still binds it to the *pre-repair* provider request. A real
  provider run is owed before anything ships.
* **No band here can see the finger curl direction.** It flipped sign with the skeleton
  because the left hand's rest geometry is now the right hand's old geometry. It is a pose,
  not a measurement, and only a render shows whether it curls the right way.
* **The two retarget boundaries have no capture fixture on this footage.** `soma_motion`
  and `speech_motion` are asserted by unit test and by nothing measured.
* **Only the capture lane got an overlay.** The task asked for a before/after camera
  overlay on each lane of the blast list. `speech_motion`, `soma_motion`, `body_binding`
  and `unified_gltf`'s skin matrices are covered by **unit tests only** — they have no
  fixture on this footage to render. `soma_motion` in particular now depends on an
  assertion, not a picture, and it is the lane whose side convention actively changed.
* **I6 has still not been shown to see facing on this take.** §6 makes that a measurement
  rather than a caveat, but it means the surface arm contributes nothing to the facing
  claim in either direction.
* One take, two performers, 150 correlated frames. Nothing generalises past this fixture.

## 10. Review follow-ups, 2026-09-02

Three items the review asked for before this branch can merge. All three are in
`facing-fix-gate.json` under `review_follow_ups_2026_09_02`.

### 10.1 The finger curl — the constant is inert, and the sign moved with the geometry

**The hands do not curl backwards, and they could not have.** Measured: the rest curl's
maximum joint displacement is **0.000000 mm**, in either sign. `_finger_rest_local` rotates
each finger joint about its **local X**, and every finger segment's rest offset is
`(length, 0, 0)` — along that same X. A rotation does not move a vector lying on its own
axis, so the constant is a **twist of each finger about its own bone**, not a flexion. Its
only visible effect is a few degrees of roll on the skinned flesh, under 2 mm of skin
travel.

**The sign was right to flip**, and the rule states why without appealing to intent:
`sign(curl) == sign(rest x)` on every finger joint, **before the repair and after it**. The
joint named `LeftIndexProximal` at x = −0.082 curled with sign −1; after the relabel the
joint at x = −0.082 is named `RightIndexProximal` and curls with sign −1. **Same bone, same
roll — no physical finger changed.** Had the sign stayed with the *name*, every finger on
both hands would have reversed and nothing in this lane would have reported it. Three tests
in `tests/test_facing_fix.py`.

**Renders**, four hands × before/after, `tools/compare/hand_closeup.py` at 900×900 with the
camera placed from the hand's own bone frame:
`artifacts/compare/d1-fix/hand-s{0,1}-{Left,Right}-{BEFORE,AFTER}.jpg`. The camera
*direction* is identical within each pair; the anatomical face it sees is not, and each
render's `HAND CHECK` line says which — that difference is the mesh having turned round.
All eight read as ordinary relaxed hands: fingers straight, no hyperextension, thumb where
a thumb goes.

**Two standing defects found here and deliberately NOT fixed**, both pre-existing and
neither introduced by this repair:

1. **The curl's axis is wrong.** The comment calls local X *"the flexion axis for every
   finger joint in this skeleton"*; X is the finger's own long axis. The proximal offsets
   spread along local **Z** (index +0.030 m, little −0.038 m), so Z is the axis across the
   knuckles and flexion is about it. Changing it would be a **new pose**, not a convention
   change, and needs its own gate.
2. **The two hands roll in opposite physical directions.** Their geometry is an exact
   x-mirror pair (verified to machine precision), and conjugating a rotation about the
   mirror's own normal leaves it unchanged, so a mirror-symmetric pose requires the *same*
   signed X-rotation on both hands. The code uses opposite signs. Equally true before the
   repair, with the two signs merely exchanged.

### 10.2 The asset's SHA chain — closed by a real provider run

Regenerated through the pinned **Blender 4.5.11 LTS** (codesign verified, Gatekeeper
notarised) and **MPFB 2.0.16** under the repaired `DEFAULT_MPFB_JOINT_MAP`, into
`artifacts/compare/d1-fix/body-run-regenerated/`. The existing
`.cache/.../run/detailed-hands` was not touched.

| array | comparison | result |
|---|---|---|
| `vertices_m`, `triangles`, `joint_indices`, `joint_weights`, `joint_names`, `parents`, `neck_seam_vertex_indices` | exact equality | **identical** |
| `local_rest_matrices`, `inverse_bind_matrices`, `gnm_head_socket_matrix` | max absolute difference | **0.0** |

**Every array is bit-identical to the derived asset**, so the permutation derivation was
sound and its only defect was provenance. `request_sha256` moves from
`fbed121a…` (delivered, `LeftUpperArm → upperarm01.R`) to
**`fbd9784b…`** (regenerated, `LeftUpperArm → upperarm01.L`), and the new manifest binds to
its own request. The npz digests differ (`27498e69…` delivered, `518e1052…` derived,
**`257f2472…`** regenerated) because `np.savez` writes its own container; the arrays are
what must agree, and they do.

### 10.3 `soma_motion` on real exports — and the triple product cannot see it

Six real GEM-X/Kimodo exports on disk, so this is **not** unit-test-only:
`autoanim_dialogue/amy-cuddy-dialogue-body`, `autoanim_squat/research-squat-640`,
`autoanim_will_acting/will-stephen-acting-body`, `autoanim_real/autoanim_fixture`,
`cpu_smoke/autoanim_fixture`, `autoanim_csg_dialogue/csg-dialogue-upper-body`.
`tools/compare/soma_handedness.py`.

**The handedness triple product reads −1 on SOMA's own joints, −1 on our rig under the
repaired mapping and −1 under the swapped one, on all six clips. It does not
discriminate.** That is the main finding of this item, and the reason is structural:
`soma_motion` is a *rotation* retarget, so our rig's joint positions come from **our** rest
skeleton posed by SOMA-derived rotations, and the bone named `LeftUpperArm` sits on the
rig's left whatever mapping drives it. A swapped mapping leaves every bone on its own side
and puts the **other arm's motion** on it — and a sign that measures where the bones *are*
cannot see which motion they are *doing*. Same class as *"a length invariant cannot score
direction"*.

What discriminates is **distance**: how far each of our rig's limb joints sits from the
SOMA joint it should be following, root-relative.

| clip | repaired | with the compensation left in |
|---|---:|---:|
| amy-cuddy-dialogue-body | **91.9 mm** | 541.7 mm |
| research-squat-640 | **92.9 mm** | 573.8 mm |
| will-stephen-acting-body | **111.8 mm** | 445.1 mm |
| autoanim_real/autoanim_fixture | **125.4 mm** | 426.0 mm |
| cpu_smoke/autoanim_fixture | **126.2 mm** | 389.5 mm |
| csg-dialogue-upper-body | **95.4 mm** | 540.4 mm |

Repaired is nearer its own side on **every joint of every clip**; the legacy arm is 4–5×
worse and on 5 of 6 clips some joint is nearer the *opposite* side outright. The residual
91–126 mm is the canonical-bone-length proportion mismatch I1 measures, not a retarget
error.

Two things stated precisely. **The legacy arm is not the code as it shipped**: it is the
repaired skeleton with `soma_motion`'s compensating swap *left in* — exactly the half-done
change this check exists to catch. The lane as it shipped was self-consistent: bones named
Left sat at rig −X carrying the mesh's anatomical *right* flesh, and the mapping sent them
SOMA's Right rotations, so the performer's right arm drove the mesh's right arm. Both
halves were mirrored and the product was correct; change one and it is not. And the
"which side is nearer" test is the **weaker** of the two: on `cpu_smoke/autoanim_fixture`
the legacy arm is still nominally nearer its own side on every joint while being 970 mm
from it, because on a clip whose two arms move alike both pairings are equally bad. Read
the distance, not the winner.

## 11. Sequencing

**Not merged, and not mergeable yet.** `tools/compare/provenance.py` gates the delivery on
the `THORAX_SMOOTHING_FRAMES` leak, and this branch still carries the pre-decision value
**15**; `main` has since moved to **9** (commit 4e1a52f, selected on synthetic truth). The
branch was deliberately not rebased mid-run, so the head-gate figures here are on the
window-15 baseline and are internally consistent before-against-after; the merge takes
`main`'s 9. Merging is the registry owner's call, after review and after the delivery
decision lands.
