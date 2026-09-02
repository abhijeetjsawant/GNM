# D1 (fix): the left/right naming mirror, repaired and gated

**2026-09-02, branch `ladder/D1-fix`. NOT MERGED.** Instruments: the extended
`tools/compare/facing_location.py` (before and after arms), `facing_surface_probe.py`,
`tools/compare/facing_fix_gate.py` (report
`artifacts/compare/d1-fix/facing-fix-gate.json`), and the four regression instruments run
unchanged. Companion to `docs/reviews/facing-location-2026-09-02.md`, which located the
defect and proposed a fix that this step **did not take** — see §3.

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

### The degenerates — every one behaves as required

| control | forward-dot | handedness | verdict |
|---|---:|---:|---|
| the repaired build (identity row) | +0.992 / +0.995 → PASS | −1 → PASS | reference row |
| **180° yaw** of the repaired torso | −0.992 / −0.995 → **FAIL** | −1 → PASS | PASS |
| **90° yaw** | +0.073 / +0.030 → **FAIL** | −1 → PASS | PASS |
| **sagittal mirror** | +0.992 / +0.995 → PASS | **+1 → FAIL** | PASS |
| **the pre-repair build**, identical bands | **FAIL** (10 of 20 cells) | arms disagree → **FAIL** | PASS |

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

The reference fitter's own head does not reach +0.9 either, and ours is **above** it on
both subjects. This lane's standing rule is that *a gate no oracle can pass is
miscalibrated*, so the band was left exactly as written, reported failing, and replaced in
the binding set by two bands that are claims about the pipeline rather than about the
take: **sign** (median > 0 and bootstrap lower bound > 0 on every part — PASS on all 22
cells) and **head against the oracle** (ours ≥ MAMMA's — PASS on both subjects).

## 6. Why the mesh nose does not come back at the magnitude it left with

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
* One take, two performers, 150 correlated frames. Nothing generalises past this fixture.

## 10. Sequencing

**Not merged, and not mergeable yet.** `tools/compare/provenance.py` gates the delivery on
the `THORAX_SMOOTHING_FRAMES` leak, and this branch still carries the pre-decision value
**15**; `main` has since moved to **9** (commit 4e1a52f, selected on synthetic truth). The
branch was deliberately not rebased mid-run, so the head-gate figures here are on the
window-15 baseline and are internally consistent before-against-after; the merge takes
`main`'s 9. Merging is the registry owner's call, after review and after the delivery
decision lands.
