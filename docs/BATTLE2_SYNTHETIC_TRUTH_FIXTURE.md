# Battle 2 side-quest — the synthetic ground-truth fixture

**Status:** design, not yet built. Read-only synthesis of four scouting passes over this repo (`mhr-blender`, `camera-rig`, `motion-and-realism`, `licence-and-traps`). Every path, function name and number below was checked against the working tree unless explicitly flagged as unverified.

**Companion documents:** `docs/OWNED_BODY_CAPTURE_PLAN.md` (battle structure), `docs/OWNED_BODY_CAPTURE_RESEARCH.md` §8/§9/§9a/§9b (asset licences, motion corpora, error budget), `docs/BATTLE1_INCREMENT4_SOMA77_DETECTOR.md`, `docs/BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md`, `docs/BATTLE1_INCREMENT6_UNCERTAINTY_WEIGHTING.md`, `docs/FINGER_TRIANGULATION_GATE.md`, `docs/TEST_FIXTURES.md`.

---

## 1. What this measures, and what it cannot

### 1.1 The problem

We have no reference. Every accuracy number this project has produced is one of three things:

- **reprojection** — `median_reprojection_error_px` = 2.913, `p95` = 6.334, `max` = 18.228 in `artifacts/commercial-multiview-soma77/run-report.json`, measured against the same detections the track was fitted to;
- **held-out reprojection** — the same quantity with a view withheld;
- **agreement with MAMMA's retained output** — which is itself an estimate, and which `docs/OWNED_BODY_CAPTURE_RESEARCH.md` §4.1 already establishes cannot be our internal quality reference.

None of these is error against truth. We cannot presently say whether our body track sits at 20 mm or 50 mm. The 18–25 mm of per-frame spread we measured against MAMMA is unattributed: some of it is detector noise, some is estimator noise, and we cannot separate them.

### 1.2 What the fixture does

Pose a body model through the pipeline's own forward kinematics, project or render it through a calibrated virtual rig, run the pipeline on the result, and compare against the joints we posed. The joints we posed **are** truth, exactly, to float precision. Two things follow immediately:

1. **The estimator is measured exactly.** Triangulation, the sequence solve, association, the gating behaviour — all of it is measured against a reference with zero error, in millimetres, today.
2. **The detector is measured only up to a domain gap** whose size is unknown until measured, and which is structurally biased in the flattering direction (see §7.6).

### 1.3 What it structurally cannot measure

State this on every artefact the fixture emits. These terms are **absent by construction**, not merely small:

| Absent term | Why it is absent | Where it is real |
|---|---|---|
| **Calibration error** | The fixture's camera model and the pipeline's `CalibratedCamera` are the same code. `camera_from_dict` (`src/autoanim_gnm/commercial_multiview.py:246`) takes exactly `[fx, fy, cx, cy]`. | §9b Ledger A — a first-order term |
| **Lens distortion** | The rig schema has nowhere to put coefficients. The MAMMA iPhone rig happens to be `[0,0,0,0]`, so dropping it is lossless *there*; Vicon-calibrated rigs in the same directory carry nonzero `vicon_radial_2`. | Battle 2's own rig, unknown |
| **Sync error** | All virtual cameras sample the same instant. | §9b names this the *hard physical gate*: median P95 body-joint speed 1.76 m/s → 1 ms of slip is 1.8 mm |
| **Soft-tissue artefact** | The body model has no skin sliding over bone. | Ledger B: >30 mm thigh, 28.7 ± 4.0 mm acromion |
| **Joint-definition error** | GT and detector share the SOMA rig by construction. This is the term that dominated Battles 0 and 1 (10.4% limb-length drift with pose; 29–53 mm systematic hip/knee bias attributable to label convention alone, per §9b). | Real footage, first order |

**Therefore: this fixture does not replace the marker session, and no result from it may be quoted as "our MPJPE".** It answers a narrower and currently unanswerable question — *what are sigma and visibility worth, in millimetres, against actual truth* — which is the number that decides whether Battle 4 (train our own detector) is the next battle.

---

## 2. The build, in increments — ordered by time-to-measurement

**The decisive finding across all four scouts: three of the four arms need no rendering at all.** `tests/test_commercial_multiview.py` already generates 2D observations by calling `camera.project(ground_truth)` at lines 51, 82, 172, 211, 245, 318, 501 and 577. The deliverable number — the (ii)→(iii) sigma delta — is reachable in days, with zero rendering and zero camera-replication risk.

### Increment 0 — the truth generator and the two controls (no render)

**Deliverable:** truth `.npz` + four `autoanim.body-observations` JSONL files + a scorer. **No changes to `src/`.**

Components, all present:

- **Motion.** Six owned SOMASKEL77 tracks from our own GEM-X runs on our own footage, verified present:
  ```
  .cache/autoanim_gnm/gem-x/outputs/autoanim_csg_dialogue/csg-dialogue-upper-body/soma_motion.npz   100 frames
  .cache/autoanim_gnm/gem-x/outputs/autoanim_dialogue/amy-cuddy-dialogue-body/soma_motion.npz        84
  .cache/autoanim_gnm/gem-x/outputs/autoanim_will_acting/will-stephen-acting-body/soma_motion.npz    96
  .cache/autoanim_gnm/gem-x/outputs/autoanim_squat/research-squat-640/soma_motion.npz                70
  .cache/autoanim_gnm/gem-x/outputs/autoanim_real/autoanim_fixture/soma_motion.npz                   67
  .cache/autoanim_gnm/gem-x/outputs/cpu_smoke/autoanim_fixture/soma_motion.npz                       67
  ```
  Each carries `local_rotations_xyzw (T,77,4)`, `root_translation_m (T,3)`, `rest_joint_positions_m (77,3)`, `rest_world_rotations_xyzw (77,4)`, `joint_positions_m (T,77,3)`, `contacts (T,6)`. Dialogue/acting distribution — the same distribution as the real fixture. ~480 frames total; clips are 67–100 frames so a 150-frame two-person scene needs concatenation plus two independent clips placed in one volume (the MammaSyn precedent §9a endorses). *Flag: `source_pts` is an int64 tick array not verified as uniform, so "16 s at 30 fps" is approximate.*
- **Forward kinematics.** `src/autoanim_gnm/soma_motion.py::soma_forward_kinematics` (line 231) then `project_soma_to_body_track` (line 782). Verified: this module imports only `numpy` plus in-repo modules — **no torch, no scipy** — so it runs in the project `.venv` and, if ever needed, inside Blender's Python.
- **Projection.** `CalibratedCamera.project` (line 183) on the rig from `load_camera_rig`.
- **Solver under test.** `reconstruct_multiview` (line 1389), `triangulate_point` (line 293), `solve_sequence_positions` (line 976) — unchanged.

**Two mandatory controls ship in this increment, before any arm reports a number:**

- **Zero-noise positive control.** GT 2D with no noise → full pipeline. Must recover truth well under 1 mm. Anything larger is a coordinate-convention or projection bug (rig world is Z-up metres; the pipeline persists `raw_triangulated_world_positions_z_up_m`), not an accuracy result.
- **Rest-pose negative control.** Replace the estimator output with a frozen mean pose. **Every gate must fail.** This is the standing rule applied to the fixture itself, and it is the control that would have caught the 98.3 mm rest-pose hand that scored 0.05 mm of jitter.

### Increment 1 — arm (ii), iid-baseline noise (no render)

Adds the noise injector and the scorer's fixed-denominator accounting. Produces the first true-3D-error number this project has ever had. Days.

### Increment 2 — schema `1.2` and solver plumbing (no render) — **the only forced `src/` change**

Arm (iii) cannot be run without it. `_person_array` (line 377) parses exactly `{"x","y","confidence"}`, and `confidence` is triple-loaded:

- **gate** — `minimum_confidence: float = 0.25` in both `triangulate_point` (line 298) and `solve_sequence_positions` (line 984);
- **weight** — `observed_weight = np.sqrt(np.clip(observations[frame_index, camera_index, joint_index, 2], 0.0, 1.0))` in `solve_sequence_positions`;
- **clipped at 1.0**, so any `1/σ²` above unity saturates.

There is no visibility channel anywhere. Overloading `confidence` would make the (ii)→(iii) delta measure *gate movement*, not sigma. **Precondition, hard:** budget `autoanim.body-observations/1.2` with explicit `sigma_px` and `visible` fields, plumb `1/σ` onto the reprojection residual in `triangulate_point` and `solve_sequence_positions`, and refuse to run (iii)/(iv) until it exists.

### Increment 3 — arm (iii), heteroscedastic sigma (no render)

**Delivers the number the whole fixture exists for.**

### Increment 4 — arm (iv), correlated occlusion + visibility (offscreen geometry, no photorealism)

Visibility is a ray/depth test against the posed mesh, not a render. `SOMA_neutral.npz` (`.cache/autoanim_gnm/gem-x/third_party/soma/assets/SOMA_neutral.npz`) is pure numpy: `mean (18056,3)`, `shapedirs (128,54168)`, `bind_pose_local/world (78,4,4)`, `t_pose_local/world (78,4,4)`, CSR skinning weights, `triangles (36108,3)`, UVs. Its `joint_names[1:]` is byte-identical to the 77 names in `src/autoanim_gnm/data/somaskel77-v1.json` (verified by the scout); the extra entry is a leading `Root`. LBS in numpy against this file is the cheapest path.

*Flag:* `SOMALayer` (`.../soma/soma.py:24`) defaults to `device="cuda"`, `mode="warp"`, calls `ensure_warp_initialized()` and runs `SkeletonTransfer` at construction. **Whether it runs on Mac CPU is untested.** Per the compute-split memory, treat any `SOMALayer` invocation as a Modal job; the numpy-LBS path avoids it entirely.

### Increment 5 — the paired in-domain test (one render pass + one detector run)

**Gate on arm (i).** Do not spend renderer effort before this passes. Protocol in §4.4.

### Increment 6 — arm (i), real SOMA-77 on renders

Last, and only behind Increment 5.

---

### 2.1 Decision recorded: which body drives the ground truth

The `mhr-blender` scout found two real paths and explicitly declined to resolve them. **This document resolves it: path (b), the SOMA rig.**

**(a) Pose MHR directly.** GT = `skel_state[:, :, :3]` from the traced `MHRDemo` (`.cache/autoanim_gnm/gem-x/third_party/soma/assets/MHR/mhr_model_lod1.pt`), 127 joints, `(tx,ty,tz, qx,qy,qz,qw, s)` in global space, centimetres, Y-up — verified at zero pose on LOD6: T-pose, `root` at y=92.4, wrists at x=±53.99, height 172.8. Native for hands. **Requires a hand-written MHR-127 → 19-canonical body mapping — precisely the definition-mismatch risk the fixture is supposed to eliminate.**

**(b) Pose the SOMA rig.** GT lands in somaskel77 natively, matching `soma_forward_kinematics`, matching the six owned motion tracks with **no retarget at all**, and flowing through `project_soma_to_body_track` into the 19-joint vocabulary the solver actually consumes (`joint_names` in the run report: `nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, root, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear`). Cost: loses MHR's 27-DoF hand parameterisation.

**Rationale:** the fixture's job is to measure the *estimator*, and the estimator solves 19 joints. Path (b) removes one hand-authored mapping from the truth chain. Path (a)'s advantage is hands, and hands are not in scope for any of the four arms.

**Design rule that survives either choice, and is not negotiable:** GT joints come from a traced/loaded model's own output — `skel_state` for MHR, `soma_forward_kinematics` for SOMA — never from a fresh Python re-implementation of forward kinematics. Definition mismatch is eliminated by construction, not by review.

### 2.2 Deferred, with a known defect attached: the hands path

If a hand arm is ever built on path (a), one convention divergence must be settled first. MHR's traced code builds its per-joint quaternion from `euler_xyz` as `R = Rz(z)·Ry(y)·Rx(x)` — scipy **extrinsic** `'xyz'`. `hand_fit._axis_rotation` (`src/autoanim_gnm/hand_fit.py:155`) composes `rotation = rotation * from_euler(axis, angle)` in `("x","y","z")` order, i.e. **intrinsic** `XYZ`. Measured at `(0.3, −0.7, 1.1)`:

```
MHR / scipy extrinsic 'xyz' : [ 0.2969, -0.2157,  0.5292,  0.7651]
hand_fit / intrinsic 'XYZ'  : [-0.0575, -0.3624,  0.4418,  0.8186]
```

Position-space MPJPE is unaffected — both are just FK producing points. But GT hand poses sampled through MHR's own 3-DoF joints would include poses **`hand_fit` cannot reach**, and the fixture would report a floor of "estimator error" that is a convention mismatch. Fix the composition order, or sample GT hand poses through `hand_fit.forward_kinematics` itself. **The scout did not verify whether the Euler order was deliberate — treat this as a finding to confirm with the author, not a settled bug.**

Independently confirmed as *correct*: `HAND_CHANNELS` (`hand_fit.py:52`) matches MHR's actual hand DoF exactly — `thumb0(ry,rz) thumb1(rx,ry,rz) thumb2(rz) thumb3(rz)` + four fingers `(rx,ry,rz)(rz)(rz)` = 27/hand.

---

## 3. Camera replication and its verification test

**Only arm (i) needs this.** Arms (ii)–(iv) use `CalibratedCamera.project` directly and carry zero camera-replication risk.

### 3.0 Decision recorded: which rig the fixture uses

Two scouts collided here, and the collision is real.

- The **camera-rig scout** verified the conversion empirically against `artifacts/soma77-full/camera-rig.json` (schema `autoanim.calibrated-camera-rig/1.0`, four cameras, 3840×2160, `fx` 2549.58–2560.36).
- The **licence scout** calls that rig a **blocker**: it is `_camera_rig_from_mamma_fixture` (`scripts/build_commercial_multiview_comparison.py:174`) converting `.cache/mamma/configs/examples/calib/iphones_outdoors.yaml`. `artifacts/commercial-multiview-soma77/run-report.json` labels itself `"test_fixture_license_scope": "MAMMA example footage/calibration: research comparison only"`, and `.cache/mamma/LICENSE` — verified on disk — is an MPI **non-commercial** licence covering "the MAMMA data, models, and software … including synthetic images and videos".

**Resolution.** The conversion mathematics and both verification tiers are rig-independent; the measured tolerances below stand as evidence that the *method* is exact. But the fixture's own rig is **procedurally generated from the §9b geometry class**, not converted from the YAML:

- four cameras, separations near 84.7° / 92.0° / 87.7° / 95.6° with two opposed pairs near 179.5° / 176.7°;
- baselines 6.3–9.7 m, depth 4.6–5.0 m to the volume centre;
- `fx ≈ fy ≈ 2550` at 3840×2160, deliberately with `fy != fx` on every camera and **at least one camera with `fy > fx`** (the real rig's D001 has this, and it is the case that breaks the naive pixel-aspect recipe — see §3.2);
- principal point offset ±25 px, non-zero on every camera;
- placements perturbed so the numbers are not MAMMA's numbers.

Copying exact calibration values is copying the data; an independently constructed rig of the same geometry class is not. **Flag for counsel either way** — this document is not an adjudication.

### 3.1 Rotation

The real rig's world is **Z-up, metres — Blender-native. No world-axis conversion is needed at all.** (Measured: camera Y points ≈ world −Z on all four cameras; `det(R) = 1` exactly.) The camera frames differ by a 180° rotation about X — OpenCV looks down +Z with +Y down, Blender down −Z with +Y up:

```python
R_blender = R_c2w @ np.diag([1.0, -1.0, -1.0])
camera.matrix_world = Matrix(np.block([[R_blender, C.reshape(3,1)], [np.zeros((1,3)), 1.0]]))
bpy.context.view_layer.update()
```

Quaternion shortcut if `rotation_quaternion` must be set (Blender is wxyz, as is the rig JSON): `(w,x,y,z)_c2w → (-x, w, z, -y)_blender`. Verified against the matrix form on all four cameras, `max |dR| = 3.3e-16`. **Prefer assigning `matrix_world`** — it sidesteps `rotation_mode` entirely.

### 3.2 Pixel aspect — do this first, it feeds `ycor`

Blender has one `lens`, so `fy != fx` goes through `render.pixel_aspect_*`, which must satisfy `pixel_aspect_x / pixel_aspect_y = fy / fx`. **Blender clamps both to a minimum of 1.0**, so the >1 factor must go on whichever axis needs it:

```python
if fy <= fx:
    pixel_aspect_x, pixel_aspect_y = 1.0, fx / fy
else:
    pixel_aspect_x, pixel_aspect_y = fy / fx, 1.0
ycor = pixel_aspect_y / pixel_aspect_x
```

### 3.3 Sensor, focal length, principal point

```python
camera.sensor_fit   = 'HORIZONTAL'      # explicitly; never leave it 'AUTO'
camera.sensor_width = sensor_mm         # arbitrary, cancels; 36.0 used below
camera.lens         = fx * sensor_mm / W
camera.shift_x      = (W / 2 - cx) / W
camera.shift_y      = (cy - H / 2) * ycor / W
```

Three things to internalise:

- **`sensor_fit='AUTO'` is a trap.** AUTO picks the fit axis from `pixel_aspect_x*W` vs `pixel_aspect_y*H`; a `pixel_aspect_x` of 1.00063 is exactly the value that flips that decision under a different aspect ratio.
- **Both shifts are normalised by `W`**, not by `H`, under horizontal fit (Blender's `viewfac = W`).
- **The sign is asymmetric**: `shift_x` is negated relative to `(cx − W/2)`; `shift_y` is not. Derivation: `dx = shift_x·viewfac`, `dy = shift_y·viewfac`, y-extent `±0.5·H·ycor`, giving `cx = W/2 − shift_x·W` and `cy = H/2 + shift_y·W/ycor`.
- **The principal point matters.** On the real rig it is 20–24 px off centre; ignoring the shifts is ~100 mm at the subject. Not a rounding concern.

Also set `render.resolution_percentage = 100` and leave `render.use_border` off — both silently rescale the effective grid.

### 3.4 Resolution invariance, and its one boundary

`lens`, `shift_x`, `shift_y` and the pixel aspects are all invariant to render resolution **at fixed aspect ratio** — `fx` and `cx` both scale with `W`, so the ratios do not move. One camera definition serves both 3840×2160 and 1280×720, which is exactly what `CalibratedCamera.scaled` (line 166) assumes. Verified numerically at both.

**The invariance holds only at 16:9.** `scaled()` applies independent x and y factors, so a render at, say, Battle 3's proposed 2056×1504 (1.367:1) produces anisotropic intrinsic scaling that a single Blender camera cannot represent. **Render the fixture at 16:9.**

Reference values from the real rig at `sensor_width = 36.0` (identical at 3840×2160 and 1280×720) — retained as a worked example of the conversion, not as the fixture's rig:

| cam | lens (mm) | shift_x | shift_y | pixel_aspect_x | pixel_aspect_y |
|---|---|---|---|---|---|
| A001 | 23.902332 | +0.00528453 | −0.00244718 | 1.0000000 | 1.0000998 |
| B001 | 23.972424 | +0.00389805 | +0.00062676 | 1.0000000 | 1.0001019 |
| C001 | 23.973248 | −0.00621074 | −0.00250823 | 1.0000000 | 1.0004463 |
| D001 | 24.003355 | +0.00427589 | +0.00368399 | 1.0006271 | 1.0000000 |

### 3.5 The verification test — two tiers, both already executed

**Pin the Blender version.** All of this was measured on **Blender 4.2.0**. The pixel-aspect clamp and `view_frame` semantics are version-dependent.

**Tier 1 — analytic.** No render. Per camera: build the Blender camera as above, then compare `CalibratedCamera.project` against `bpy_extras.object_utils.world_to_camera_view`, which reads `camera.view_frame(scene)` and therefore exercises Blender's real sensor-fit, shift and aspect logic. Convert with `u = ndc.x * W`, `v = (1 - ndc.y) * H`; `ndc.z` is depth in metres. Set scene resolution and pixel aspect *before* calling it.

Sample set, 227 points per camera: 200 uniform in the capture volume `x∈[−2,2], y∈[2,8], z∈[0,2]`, **plus 27 back-projected probes** placed 6 px inside each frame corner, each edge midpoint and the centre, at depths 2 / 5 / 9 m. The corner probes are the point of the exercise — centre-of-frame samples cannot expose a shift, aspect or fit error.

> **Gate:** max reprojection difference **≤ 0.01 px at 3840×2160**, **≤ 0.005 px at 1280×720**; max depth difference **≤ 1e-5 m**.
> **Measured:** 0.001759 px @ 3840, 0.000586 px @ 1280, 1.9e-6 m. That is mathutils float32 noise; the conversion is exact.

**Negative controls — mandatory, and what makes this proof rather than a passing test.** Both plausible mistakes were injected; both are caught by orders of magnitude:

| injected fault | max error @ 3840 | signature |
|---|---|---|
| `shift_y` sign flipped | **28.31 px** | constant offset ≈ `2·(cy − H/2)`, worst where the principal point is furthest off-centre |
| pixel aspect dropped (1.0 / 1.0) | **0.687 px** | pure slope in `dy` vs `(v − H/2)` equal to `\|1 − fy/fx\|`: measured −6.268e−4 where truth is 6.27e−4, +4.465e−4 where truth is 4.46e−4; `dx` untouched |

The pixel-aspect control is how the clamp was discovered: setting `pixel_aspect_y = fx/fy = 0.999373` silently clamps to 1.0 and leaves a 0.687 px systematic tilt. **A tolerance of 1 px would have passed it.**

**Tier 2 — render-level, resolving the half-pixel convention.** Tier 1 works in continuous coordinates and structurally cannot say where a pixel *index* sits. Cycles CPU, 8 samples, denoising off, `view_transform='Standard'`, black world, OPEN_EXR 32-bit, 1280×720; five emissive spheres (r = 0.03 m) back-projected to pixel targets (200,150), (1080,150), (200,570), (1080,570), (640,360) at 4 m; intensity-weighted centroid in a 51×51 window vs `project()`.

> **Gate:** ≤ 0.3 px (Blender's default 1.5 px Blackman-Harris filter is symmetric, so centroids survive it).
> **Measured:** centroid at array index `(c, r)` corresponds to continuous `(c + 0.5, r + 0.5)`; residuals **(−0.053, −0.017), (0.000, −0.023), (−0.011, −0.009), (+0.040, −0.006), (+0.005, −0.031)** px. Max 0.053 px.

**This closes the half-pixel question empirically.** The rig's `cx`/`cy` are corner-origin continuous coordinates — image centre at exactly `(W/2, H/2)`, pixel index `i` centred at `i + 0.5` — the same convention as `ndc.x * W`, and precisely what `scaled()`'s linear scaling of `cx` assumes. **No half-pixel correction anywhere in the fixture**, and detector outputs must be read the same way.

### 3.6 Where the check lives, and two construction-time guards

- Promote Tier 1 to `tests/` as a Blender-gated test. Have the fixture renderer emit the JSON report beside its output that `CLAUDE.md` mandates, carrying the rig-file SHA and both tier results, so the SHA chain covers camera fidelity and not just pipeline inputs.
- **Copy the `camera_bundle` pattern: check the convention where the camera is constructed, not only in a test.** `src/autoanim_gnm/camera_bundle.py::perspective_camera_from_calibration` (line 727) does the same `diag(1,-1,-1)` flip (line 740) *and* round-trips the convention at ingestion (lines 753–758) to catch a sign error before it silently mirrors every view. Nothing in the body lane does this; the fixture should.
- **`CalibratedCamera.project` has no behind-camera gate.** It divides by `depth` whatever the sign (contrast `camera_bundle.project_calibrated_points`, which raises at `depth <= 1e-6`). **GT-2D generation for arms (ii)–(iv) must gate `depth > 0` and in-frame itself.** Nothing downstream will.
- **Renormalise quaternions** before writing any synthetic rig JSON — `CalibratedCamera.__post_init__` rejects `abs(|q| − 1) > 1e-5`.
- **All four cameras must render at the same resolution.** `reconstruct_multiview` (line 1434) refuses a rig whose cameras ran at different frame sizes, and `pixel_scale` (line 1436) is a single scalar against `REFERENCE_DETECTOR_WIDTH_PX = 1280` (line 50). Feed the detector at 1280×720 to keep the pixel-denominated gates identical to the real runs, or re-derive every gate.

### 3.7 What does not exist yet, stated plainly

Grepping `shift_x|sensor_width|sensor_fit|angle_x` across `scripts/`, `src/`, `tests/`, `workers/`, `gnm/`, `native/` returns **zero hits**. Every Blender camera in this repo is hand-framed with a hard-coded `lens` (72.0 or 85.0 mm) and a `_look_at`. `scripts/inspect_video_acting_projection.py` is the closest precedent and only for the ndc→pixel step (lines 61–65); its camera is a review camera. `tests/test_articulation_projection.py` is facial constraint projection with no cameras; `tests/test_body_projection.py` is screen-space body constraints, not camera projection. The conversion above is new code.

---

## 4. The four arms

Shared conventions for all four:

- **Truth** = the posed SOMASKEL77 joints, projected to the 19-joint contract by `project_soma_to_body_track`, in Z-up metres.
- **Metrics on a fixed denominator** — all frames × all joints, unrecovered joints scored as failures, never dropped.
- **Bias and spread reported separately, per joint**, never pooled into one MPJPE.
- **≥ 5 seeds** on every noise arm; deltas quoted with spread.
- **Retained-observation count reported beside every millimetre figure**, plus `valid_joint_fraction`, `constraint_recovered_joint_fraction` and `interpolated_joint_fraction` separately (the real run reports 0.8912 / 0.00105 / 0.1077 respectively).

### 4.1 Arm (i) — real SOMA-77 detections on renders → true 3D error

**Isolates:** the detector's contribution to 3D error, on synthetic imagery. **Last to build.**

**Needs:**
- The camera replication of §3, at 16:9.
- A renderer. `soma/tools/vis_pyrender.py` is already headless (`_ensure_headless_pyrender`, `set_pyopengl_platform`) with `MeshRenderer`, `look_at`, `render_mesh` — what is missing is driving it from `CalibratedCamera.projection_matrix` instead of its own `compute_camera_pose`, plus lighting, material and background. Blender/Cycles is the alternative and the one §3's verification is written against.
- **The identical imaging chain.** `scripts/build_commercial_multiview_comparison.py::_extract_frames` runs `ffmpeg -vf "select=between(n,60,209),scale=1280:-2" -q:v {FRAME_JPEG_QUALITY}`, and the comment records that pinning JPEG quality lifted valid joints **82.8% → 88.2%** and cut temporal rejections **33 → 14** at width 1280. Render at 3840×2160, encode, extract through the same `scale` and `-q:v 2`. Anything that skips the downscale-and-mjpeg step is testing a different detector input.
- The detector, runnable locally: `workers/commercial_multiview/soma77_pose.py` against `.cache/autoanim_gnm/gem-x/inputs/onnx/vitpose.onnx` (present, ~3.4 GB, ONNX Runtime, no GPU host needed).

**Two decisions this arm must make explicitly and report both ways:**

- **Boxes.** SOMA-77 is top-down; it reads person boxes from an existing observation file (`soma77_pose.py:31–34`, `_square_box` at line 86), and `_detect_soma77` currently passes Apple Vision observations as `BOXES.jsonl`. Projected GT boxes remove the unaudited Apple Vision dependency (a licence win, §6) but hand the detector perfect framing the production path never gets — and increment 4's review already found a defect in exactly this construction (`_square_box` sizing against `max(w,h)` while the model consumes only the centre 192 of 256 columns). **Run both.**
- **Compression.** One run through the ffmpeg chain, one on lossless PNGs. The difference is the size of a proven first-order noise term, measured once.

**Produces:** per-joint **bias** (mean residual) and **spread** (dispersion) in millimetres against truth, plus the per-view GT-2D-versus-detection error — which is the first true 2D error number the project will have, and which retroactively calibrates arm (ii)'s noise scale.

**Quote only the spread when decomposing (i) − (ii).** The bias term bundles `SOMA77_TO_AUTOANIM` (`soma77_pose.py:61`), whose own comments concede the mismatch: `"left_hip": 67,  # LeftLeg`, and `"nose": 6,  # Head` with `nose has no SOMA counterpart; Head is the nearest and is not the same point`. Folding that constant into one MPJPE hides the dominant real-world error term.

### 4.2 Arm (ii) — ground-truth 2D + iid noise, fitted WITHOUT sigma

**Isolates:** the estimator's noise rejection, with a known input noise distribution. **First to build.**

**Needs:** the truth generator, `CalibratedCamera.project`, a noise injector, and the scorer. **No changes to `src/`.**

**Noise scale.** Start from the residuals we already have — 2.913 px median / 6.334 px p95 (`artifacts/commercial-multiview-soma77/run-report.json`, SOMA-77) and 4.589 / 10.877 px with `valid_joint_fraction` 0.8823 and 14 temporally rejected subject-frames (`artifacts/commercial-multiview-b2/run-report.json`, Apple Vision). **These are not the true 2D error** — they are measured against a track fitted to those same detections. Sweep around them; arm (i) replaces the guess with a measurement.

**Produces:** true 3D MPJPE as a function of injected 2D σ, on a fixed denominator, with retained counts.

### 4.3 Arm (iii) — the same noise, with the true sigma handed to the solver

**Isolates:** the value of per-observation uncertainty. **This is the deliverable.**

**The design trap sits inside this arm, not in a footnote.** If the noise is genuinely iid, **the true σ is a constant**, and inverse-variance weighting by a constant is ordinary least squares up to a global scale. The only thing (iii) would then change is the balance between the data block and the priors — `smooth_weight=2.0`, `length_weight=2.0`, `robust_scale_px=14.0` are absolute. The number the fixture exists to produce would measure an accidental reweighting, and could read ~0 mm ("σ is worth nothing" — the wrong strategic answer) or spuriously non-zero.

**Therefore, the arm is specified as follows and no other way:**

1. **Noise is heteroscedastic.** Draw per-observation σ to match the epipolar-disagreement deciles already measured in increment 5 — **2.3 / 4.2 / 7.5 / 14.9 / 30.6 px** (`docs/BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md:378`).
2. **The `minimum_confidence` mask is byte-identical between (ii) and (iii).** Only the weighting differs. Anything else and the delta measures gate movement.
3. **σ arrives through schema `1.2`'s `sigma_px`, entering as `1/σ` on the reprojection residual.** Never through `confidence`, which is clipped at 1.0 and also gates.
4. **Vacuity control, mandatory:** re-run (iii) with a *constant* σ equal to the population mean. **That run must land on (ii).** If it does not, the delta is measuring the prior balance, not sigma.

**Related prior art worth naming:** `src/autoanim_gnm/hand_fit.py::cross_view_weights` already builds an inverse-variance weight `1/(1 + (d/9px)²)` from epipolar disagreement — a σ *derived from geometry*, standing in for the σ SOMA-77 does not emit. Arm (iii) is the first time a solver on this pipeline sees a σ that is **true by construction**. It is also the test increment 6 could not run.

**Produces:** *(iii) − (ii)*, in millimetres, with 5-seed spread. **The number that decides Battle 4.**

### 4.4 Arm (iv) — correlated occlusion noise, with and without a visibility channel

**Isolates:** the value of a per-observation visibility bit under realistic, correlated failure.

**Needs:** posed mesh vertices, a per-camera ray/depth visibility test, geometric occluders, and schema `1.2`'s `visible` field plus its solver plumbing.

**The correlation structure decides the answer, so it must not be parameterised by hand.** Increment 5 already named the mechanism: two views confidently wrong the same way pass an epipolar test exactly as they pass the veto. One shared offset across all four views is unrealistically severe; independent per-view dropout is not correlated at all and will make the visibility channel look worthless. Real occlusion is correlated between *adjacent* views and independent across *opposed* ones — and this rig has two opposed pairs at 179.5° / 176.7°. **Generate occluders geometrically (a second body, a prop), read the correlation structure off the geometry, and report the measured cross-view error correlation matrix as a fixture output** so a reader can see what was actually injected.

**Produces:** *(iv)-with-visibility* − *(iv)-without*, in millimetres, plus the cross-view correlation matrix.

### 4.5 The paired in-domain test that gates arm (i)

Costs one render pass and one detector run. Run it before spending renderer effort.

1. Take an already-reconstructed motion — `artifacts/commercial-multiview-b2/subject-{00,01}.body-track.npz` (150 frames, `local_rotations_xyzw (150,55,4)`, `triangulated_world_positions_z_up_m (150,19,3)`), or better the SOMASKEL77 tracks — and render both subjects through the four cameras.
2. Push the renders through the same ffmpeg chain, then `workers/commercial_multiview/soma77_pose.py`.
3. Compare, **on the same poses**, against the distributions the real run already produced (`artifacts/soma77-full/work/{A,B,C,D}001-observations.jsonl`, 150 frames each, verified present): per-joint confidence, per-joint reprojection residual, `valid_joint_fraction`, temporal rejection count, epipolar-disagreement distribution. Also compare 2D error at the subject against the **16.2 mm measured on real footage** in `docs/BATTLE1_INCREMENT4_SOMA77_DETECTOR.md:26`.

Three caveats, each of which biases the answer:

- **The reference poses carry their own error.** This compares *detector behaviour statistics on similar poses*, not joint-for-joint correspondence.
- **The asymmetry runs one way.** SOMA-77 is itself synthetic-trained, so overlap does not prove the renders are realistic; **mismatch definitively proves they are not.**
- **A SOMA-rig-driven fixture removes the label-convention bias.** GT and detector share joint definitions exactly, so arm (i)'s error is a **lower bound** on real-footage error, not an estimate of it.

---

## 5. Pre-registered acceptance gates

Standing rule on this project: **no gate may be passable by a constant.** The fixture's version: **no arm may recover the truth because the truth was in its input.** Every gate below carries the degenerate solution it must reject. Bands are fixed **before** the runs report, as `docs/BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md` did — that pre-registration is why its "thrashing" verdict is credible.

| # | Gate | Threshold | Degenerate solution that must fail it |
|---|---|---|---|
| G1 | **Zero-noise positive control.** GT 2D, no noise, full pipeline, MPJPE vs truth | **< 1.0 mm** | Any coordinate-convention or projection bug. A Z-up/Y-up swap, an `xyzw`/`wxyz` reorder (the exact class of bug §9b's own correction records), or a mirrored `shift_x` all produce ≫ 1 mm. |
| G2 | **Rest-pose negative control.** Frozen mean pose substituted for the estimator | **Every other gate must FAIL** | A frozen constant. This is the control that would have caught the 98.3 mm rest-pose hand scoring 0.05 mm of jitter. If G2 passes anything, that gate is vacuous. |
| G3 | **Camera fidelity, Tier 1.** `project()` vs `world_to_camera_view`, 227 pts/cam incl. 27 corner/edge probes | **≤ 0.01 px @ 3840, ≤ 0.005 px @ 1280; depth ≤ 1e-5 m** | Centre-of-frame-only sampling. Measured: a flipped `shift_y` is 28.31 px and a dropped pixel aspect is 0.687 px — both invisible at frame centre, both caught here. **The negative controls are part of the gate, not commentary.** |
| G4 | **Camera fidelity, Tier 2.** Rendered emissive-sphere centroids vs `project()` | **≤ 0.3 px** (measured 0.053) | A half-pixel convention error. Tier 1 structurally cannot detect it. |
| G5 | **Independent camera construction.** Build the Blender camera by a different code path than `CalibratedCamera.projection_matrix`, then **triangulate a rendered rod of known length** | Rod length recovered to a pre-registered band | Self-validation. Reprojection RMS validates a camera against itself; this is the same protocol Battle 2's exit gate specifies, for the same reason. |
| G6 | **Arm (iii) vacuity control.** (iii) re-run with constant σ = population mean | **Must land on (ii)** within seed spread | A sigma delta that is really a prior-rebalance. If this fails, the headline number is not measuring sigma. |
| G7 | **Calibration sensitivity.** Perturb one camera by 1 mm / 0.05°, re-run | **Error must move measurably** | A fixture insensitive to calibration. If it does not move, the fixture is not measuring what we think it is. |
| G8 | **Fixed denominator.** Every mm figure reported over all frames × all joints, unrecovered scored as failures, with retained counts | Retained count reported, no pooling of `valid` / `constraint_recovered` / `interpolated` | Survivorship. `triangulate_point` has `inlier_threshold_px=14.0` and `minimum_confidence=0.25`; Battle 0's finding was that this discarded 41.8% of observations and made the *surviving* residual look flat. Sweep σ upward and MPJPE can go flat or *improve* as noise grows. |
| G9 | **Limb-length bias.** Two body shapes with deliberately different limb lengths (e.g. ±8% femur and humerus), per-limb length bias reported against truth | Pre-registered band, per limb | Canonical proportions. `estimate_limb_lengths_m` (line 932) fixes one length per subject per take **from the data**, then `solve_sequence_positions` constrains against it — so bone lengths are constants in the estimator (the reason the 0.00% reading was meaningless). And `positions_to_body_track` preserves canonical proportions, so a canonically-proportioned body makes the retarget gate vacuous too. |
| G10 | **Speed stratification.** Error reported stratified by joint speed, with at least one high-acceleration clip in the set | Error must vary with speed | Over-smooth motion. `solve_sequence_positions` carries a second-difference smoothness term (`smooth_weight=2.0`), and increment 5 showed the prior buys accuracy specifically because noise averages down across frames. Flat-in-speed error means the motion is too slow to test the prior. |
| G11 | **Association margin.** Minimum inter-subject 3D distance per frame + epipolar-affinity margin distribution reported alongside any switch count | Distributions published, not just the count | "Zero identity switches" is passable by a constant when bodies are widely spaced. Identical bodies also make the appearance cue useless. The margin distribution is what makes the count meaningful. |
| G12 | **Seeds.** Every noise arm ≥ 5 seeds; deltas quoted with spread | Spread reported | One seed. The deliverable is a difference of a few millimetres. |
| G13 | **Reprojection is NOT a gate.** | *Excluded by design* | In arms (ii)–(iv) reprojection is ~0 by construction — it **cannot fail**. `docs/FINGER_TRIANGULATION_GATE.md` already established three separate times that reprojection flatters bad geometry. Drop it from the fixture's gate set entirely. Note that `scripts/verify_commercial_multiview_artifact.py` enforces ≤6 px median / ≤12 px p95 / ≤25 px max (lines 111–113) and performs no projection itself; it verifies no camera and must not be mistaken for a fixture gate. |

---

## 6. Licence position

"Train-on-outputs" = may we render with it / detect with it, and use the result to train a detector we ship?

| Asset | Path / source | Licence | Commercial incl. train-on-outputs | Verified how | Flag |
|---|---|---|---|---|---|
| **Camera rig calibration** | `artifacts/soma77-full/camera-rig.json`, `artifacts/commercial-multiview-*/camera-rig.json`, built by `_camera_rig_from_mamma_fixture` (`scripts/build_commercial_multiview_comparison.py:174`) from `.cache/mamma/configs/examples/calib/iphones_outdoors.yaml` | MPI non-commercial (`.cache/mamma/LICENSE`) | **NO — bars both** | Licence text read on disk; `run-report.json` self-labels `"test_fixture_license_scope": "MAMMA example footage/calibration: research comparison only"` | **BLOCKER for the fixture's own rig.** Renders through those intrinsics/extrinsics are artefacts produced from MPI data; a detector trained on them is trained on MPI-derived data. **Remediation adopted in §3.0:** procedurally generate a rig of the same geometry class from §9b's measurements. Flag for counsel either way. |
| **Body model — MHR (upstream)** | `facebookresearch/MHR` | Apache-2.0, **weights included** | Yes | Research §8, primary sources | Clean at the source |
| — the copy we actually hold | `.cache/autoanim_gnm/gem-x/third_party/soma/assets/MHR/mhr_model_lod1.pt` (696 MB) | sha256 `352e271a…7377bc` = `mhr_model.pt` from HF `nvidia/GEM-X` per `workers/gem_x/provider-lock.json` | Governing terms **for this copy** unverified | `shasum -a 256`; matches the lock's NVIDIA entry byte-for-byte | **Provenance runs through NVIDIA's bundle, not Meta's release.** Trivially moot: pin a fresh copy from `facebookresearch/MHR`. The same SHA is already recorded in `src/autoanim_gnm/data/mhr-skeleton-v1.json`. |
| **SMPL / SMPL-X base bodies** | `.cache/.../soma/assets/SMPL/base_body.obj`, `.../SMPLX/base_body.obj`, `SMPL/smpl_anim.npy` | Research §8's named carve-out ("exclude pending legal review") | **NO** | Confirmed present on disk | **They sit in the same asset tree as MHR.** Any render script that walks `assets/` picks them up. Exclude by **explicit allow-list**, not by convention. |
| **SOMA body / correctives** | `SOMA_neutral.npz`, `correctives_model.pt`, same tree | SOMA-X code Apache-2.0 (`third_party/soma/LICENSE`, SPDX headers throughout); these bytes came via `nvidia/GEM-X` → NVIDIA OML | Yes (OML) | Provider lock + model card quoted in §9 docs | Same provenance caveat as MHR |
| **Detector — SOMA-77 / GEM-X** | `.cache/autoanim_gnm/gem-x/inputs/onnx/vitpose.onnx` | Code Apache-2.0; **weights NVIDIA Open Model License** | **Yes, with an attribution condition** | OML fetched (Last Modified 24 Oct 2025). Verbatim: *"NVIDIA claims no ownership rights in outputs"*; *"An output is not a Derivative Model"*; using a model *"or its outputs to create, train, fine tune, or otherwise improve an AI model"* triggers a **"Built on NVIDIA …"** notice obligation | Two open gaps: (1) **no archived OML text bound to our revision** — `docs/NVIDIA_BODY_DEPENDENCIES.md` §"Model-license snapshot gap" already demands this and it is still open; (2) the published notice string is Cosmos-flavoured, so **the exact attribution string for GEM-X is not established**. Do not invent one. |
| **Person boxes** | Currently Apple Vision, per `workers/commercial_multiview/soma77_pose.py:21–24` | Apple SLA | **Unaudited** | `docs/` grepped — no Apple Vision terms audit exists anywhere | Real gap in the asset gate. **In the fixture it is avoidable:** use projected GT boxes (§4.1) — but run production-style boxes too, and note that doing so re-imports the dependency. |
| **GEM-X bundled YOLOX** | Human-Art trained | non-commercial | **NO** | §9 docs | Already excluded by construction; keep it excluded |
| **Renderer — Blender / Cycles** | `/Applications/Blender.app` (4.2.0) | GPL; *"What you create with Blender is your sole property"* | Yes | Research §9 | Clean |
| **MPFB2 + MakeHuman assets** | `.cache/autoanim_gnm/body-provider/`, `scripts/blender_body_worker.py` | Assets **CC0 1.0**; extension code **AGPL** | Assets yes | Attestation file + `src/autoanim_gnm/body_provider.py` constants | AGPL §13 bites on *network* provision — keep any Modal render farm internal, do not expose it as a service |
| **HDRIs — Poly Haven** | not yet in `.cache` | CC0 | Yes | Research §9 | Clean, **but see §7.6** — NVIDIA trained GEM on HDRI Haven too |
| **Motions — the six owned SOMASKEL77 tracks** | `.cache/autoanim_gnm/gem-x/outputs/*/*/soma_motion.npz` | Ours, from our own footage | Yes | Files verified present | **Day-one source. Already in the target skeleton, no retarget.** |
| **Motions — CMU** | not on disk | "free for all uses"; **"may not resell… even in converted form"** | Renders yes; **files no** | Research §9a | The fixture's retained GT joint tracks *are* a converted form. **Fixture stays internal; never published as a benchmark.** |
| **Motions — ACCAD / 100STYLE** | not on disk | CC BY 3.0 / CC BY 4.0 | Yes | §9a | Attribution must land in the **fixture manifest**, not only in the plan |
| **Motions — Eyes JAPAN** | not on disk | CC BY 2.1 JP, page also says **BY-SA** | Yes if BY; **share-alike risk if BY-SA** | §9a, flagged ambiguous there | A referee fixture needs ~10 clips, not 7,300. **Prefer ASPset-510 (CC0) and ContactPose (MIT); skip the ambiguous source entirely** — zero cost, removes the question. |
| **Textures / skin** | CC0 MakeHuman assets on disk; nothing better owned | CC0 | Yes | Attestation | Clean but low realism. **§7.6 is where clean and valid pull in opposite directions.** MHR's OBJs carry UVs but **zero `mtllib`/`usemtl`**, and `soma/assets/images/` holds only documentation GIFs/PNGs — **no body textures anywhere**. |
| **Clothing / hair** | none owned | Daz AI program / Reallusion Enterprise, both negotiated | Unresolved | Research §9 | A purchase decision. The fixture ships without it, at a measurement cost. |

**Two rules that fall out of this table:**

1. **Do not treat licence-clean as solved for a bundle you did not download yourself.** MHR is the pattern: Apache-2.0 at Meta, NVIDIA bytes on our disk. SMPL/SMPL-X is the same pattern one level down — non-commercial assets physically co-resident with clean ones. §8's "licence for a dataset vs licences for its constituent parts" distinction applies to **directories**, not just datasets.
2. **The patent position is unchanged, and the fixture makes one exposure slightly worse.** US10395411B2 claim 1 reads on runtime template + shape blendshape + pose blendshape + skinning; the fixture *renders* through neural pose correctives as well as fitting through them, exercising the claim on both sides. Same FTO opinion the plan already lists as needing counsel. Not an engineering blocker.

**Worth stating out loud:** `artifacts/commercial-multiview-b2/run-report.json`'s licence-scope field means every accuracy number this project holds is not merely reference-free — it is measured on a fixture we cannot ship. **A procedurally-rigged synthetic fixture would be the project's first commercially clean fixture as well as its first with truth.**

---

## 7. How this fixture could silently lie

Thirteen mechanisms, each with the cheapest check that catches it.

**7.1 The deliverable delta is zero by construction if the noise is iid.** iid ⇒ true σ is a constant ⇒ inverse-variance weighting is OLS up to scale ⇒ (iii) only shifts the balance against `smooth_weight=2.0` / `length_weight=2.0` / `robust_scale_px=14.0`. Could read ~0 mm ("σ is worth nothing" — the wrong strategic answer) or spuriously non-zero.
**Check:** heteroscedastic σ from the increment-5 deciles (2.3 / 4.2 / 7.5 / 14.9 / 30.6 px); `minimum_confidence` mask byte-identical between (ii) and (iii); plus gate **G6** — constant-σ rerun must land on (ii). *(Specified into arm (iii), §4.3.)*

**7.2 Arms (ii)–(iv) cannot measure accuracy at all, only noise rejection.** GT 2D is produced by projecting the same joints you later compare against, through the same camera model `triangulate_point` inverts. Absent injected noise the residual is machine epsilon. No joint-definition error, no calibration error, no sync error, no distortion.
**Check:** state on the artefact that **only the deltas travel**; absolute MPJPE from these arms is never quoted as pipeline accuracy. Plus gate **G7**, the perturbed-rig sensitivity run.

**7.3 The gates are survivorship machines, and noise feeds them.** `inlier_threshold_px=14.0`, `minimum_confidence=0.25`. Battle 0's central finding: that threshold discarded 41.8% of observations and made the surviving residual look flat. Sweep σ up and MPJPE can go flat or improve.
**Check:** gate **G8** — fixed denominator, unrecovered scored as failures, retained counts beside every mm figure, the three fractions reported separately.

**7.4 Bone length and reprojection are the two known vacuous gates, and both are worse here.** `estimate_limb_lengths_m` fixes lengths from the data; `positions_to_body_track` preserves canonical proportions (the artefact's own limitations list says so). A canonically-proportioned body makes the retarget gate vacuous, and reprojection is ~0 by construction.
**Check:** gates **G9** (two body shapes at ±8% femur/humerus, per-limb bias) and **G13** (reprojection excluded from the gate set).

**7.5 Arms (iii)/(iv) need a schema that does not exist, and the obvious shortcut confounds them.** `_person_array` (line 377) parses `{x, y, confidence}` only; `confidence` gates *and* weights *and* clips at 1.0.
**Check:** schema `1.2` with `sigma_px` and `visible`, plus solver plumbing, as a **hard precondition**. Refuse to run (iii)/(iv) otherwise.

**7.6 Arm (i) may be measuring in-distribution performance and calling it detector noise.** SOMA-77 was trained on NVIDIA-internal Blender-style synthetic renders using RenderPeople characters and **HDRI Haven** environments. Our fixture would be Blender + Poly Haven HDRIs + a parametric body — close to the same recipe, and the licence-clean choices push it *closer*: CC0 MakeHuman skin on an unclothed body is exactly the sanitised render a synthetic-trained detector handles best. `docs/TEST_FIXTURES.md` already records this failure mode for the face lane — *"the low-poly synthetic render is not a legitimate proxy for a photographed human face."* A too-good arm (i) makes detector noise look small and sends the programme **away** from Battle 4, which is the one decision this fixture exists to inform.
**Check:** §4.5's paired test, against the 16.2 mm at the subject already measured on real footage. One detector run.

**7.7 Arm (i)'s "detector noise" silently bundles a constant.** `SOMA77_TO_AUTOANIM` maps `left_hip → 67 (LeftLeg)` and `nose → 6 (Head)`, the latter with the worker's own comment that it "is not the same point"; SOMA-77 has no ears at all. The resulting per-joint constant offset is the dominant real-world error term.
**Check:** report per-joint **mean residual (bias)** separately from **spread**, and quote (i) − (ii) only on the spread term.

**7.8 GT person boxes are a gift the production path never gets.**
**Check:** run arm (i) twice — GT boxes and production-style boxes — and report both.

**7.9 Compression is a decided-by-construction term.** Battle 0's actual win was `-q:v 2` at 1280, not resolution. Lossless PNGs into the detector remove a proven first-order noise term.
**Check:** one arm-(i) run each way. Measured once.

**7.10 The occlusion arm's correlation structure decides its answer.**
**Check:** geometric occluders, and publish the measured cross-view error correlation matrix as a fixture output. *(Specified into arm (iv), §4.4.)*

**7.11 Motion choice can flatter the temporal prior.** Retargeted, resampled, already-smooth mocap is the best case for a second-difference smoothness term. Note the fixture's day-one motion source is itself *estimator output* from GEM-X runs and is therefore already smoothed — this raises the risk rather than lowering it.
**Check:** gate **G10** — include a high-acceleration clip (`research-squat-640` is the closest owned candidate), report error stratified by joint speed.

**7.12 Association is trivially clean or trivially broken in two-person arms.** Identical bodies make the appearance cue useless; wide spacing makes "zero identity switches" passable by a constant.
**Check:** gate **G11** — publish minimum inter-subject 3D distance per frame and the epipolar-affinity margin distribution beside any switch count.

**7.13 The fixture itself must be falsifiable.** Five cheap runs, all mandatory before any number leaves the fixture: the zero-noise positive control (**G1**), the rest-pose negative control (**G2**), independent camera construction validated by a triangulated rod of known length (**G5** — the same protocol Battle 2's exit gate specifies, and for the same reason: reprojection RMS validates a camera against itself), ≥5 seeds (**G12**), and pre-registered bands fixed before the runs report.

---

## 8. What we still cannot claim afterwards — and why the marker session still happens

### 8.1 Claims the fixture supports

- *"Handing the solver a true per-observation σ is worth **X ± s mm** of true 3D error under heteroscedastic noise matching our measured epipolar deciles."* — arm (iii) − arm (ii).
- *"A visibility channel is worth **Y ± s mm** under geometrically-generated correlated occlusion, with the injected cross-view correlation matrix published."* — arm (iv) with/without.
- *"The estimator, given perfect 2D, recovers truth to **< 1 mm**."* — G1.
- *"On synthetic renders, SOMA-77 contributes **Z mm of spread** and **per-joint biases of b₁…b₁₉**."* — arm (i), spread and bias reported separately.

### 8.2 Claims it does not support

- **"Our MPJPE is N mm."** Absent by construction: calibration error, lens distortion, sync error, soft-tissue artefact, and joint-definition error — the last of which §9b identifies as producing 29–53 mm of systematic hip/knee error in sparse markerless systems from label convention alone.
- **"Arm (i) is our real detector error."** GT and detector share joint definitions, so arm (i) is a **lower bound**, not an estimate.
- **Any number from arms (ii)–(iv) as pipeline accuracy.** Only the deltas travel.

### 8.3 The one row that must never exist

The fixture's MPJPE is against **MHR/SOMA joint centres**. MAMMA's 13.5 mm is against **Vicon-derived SMPL-X joints**, and §9b's Ledger B establishes that a marker reference is itself ~20 mm from bone (>30 mm thigh soft-tissue artefact, 28.7 ± 4.0 mm acromion slide, 5–12 mm per axis of hip-centre regression, and a Vicon+MoSh++ pipeline that predicts held-out markers at 21.6 mm). **The two numbers are not on the same axis and must never appear in the same row or the same table.** State the metric and the reference on every claim, per the plan's own standing rule.

### 8.4 Why Battle 2 still happens

§9b's conclusion is that **20 mm is achievable at four cameras with no margin**, and the terms that consume that margin are exactly the ones this fixture cannot see: sync (≤5 ms budget against a median P95 body-joint speed of 1.76 m/s), calibration validated by triangulating known lengths rather than reprojection RMS, wardrobe and body scans, and a simultaneous marker reference. The synthetic fixture makes the estimator measurable and the detector *comparable*; only the shoot makes the system **validated**. It also produces the first calibration we can use commercially, which — per §6 — nothing we currently hold can do.

**The fixture's actual role in the sequence:** it removes the excuse for waiting. It delivers the sigma-and-visibility number in days, on a commercially clean rig, while Battle 2's lead time runs.

---

## 7a. A consistency check the fixture passed without being asked to

Two independent measurements land in the same place, and neither was tuned to the
other:

| | |
|---|---:|
| synthetic pipeline, true 3D error under noise calibrated from our own detector | **23.5 mm** |
| our real body track's per-frame **spread** against MAMMA, body joints, bias removed | **24.7 mm** |

The first is MPJPE against joints we posed ourselves. The second is
`BATTLE1_BODY_PROFILE_VS_MAMMA.md`, measured on real footage against a different
system entirely. The noise model was fitted to cross-view *epipolar disagreement*,
which is a quantity neither number is computed from.

They are not the same statistic — one is absolute error, the other is spread about
a systematic offset — so this is a consistency check, not a validation. But the two
routes to "how noisy is our reconstruction per frame" agree to within 5%, which is
better agreement than either deserves on its own, and it is the first evidence that
the synthetic noise model is not simply a number I chose.

**What it does not license.** It does not say our real error is 23.5 mm. The
fixture has no calibration error, no lens distortion, no sync error, no
soft-tissue artefact and no joint-definition error, and §1.3 lists all five as
first-order on real footage. It says the *estimator-plus-detector-noise* part of
the budget is around 23 mm, and that the other terms are additional.

## 8a. Correction to gate G6 — it cannot fail, and that is the problem

G6 was specified as the vacuity control on the headline number: re-run arm (iii)
with a **constant** sigma equal to the population mean, and require it to land on
arm (ii). It does — to 0.00 mm, on every seed, at every floor.

That is not a result. **It is an algebraic identity.** A constant confidence
multiplies every residual by the same `sqrt(c)`, which scales the least-squares
objective uniformly and cannot move its minimiser. Demonstrated on a 40x5 linear
problem: uniform 0.99 against constant 0.30 changes the solution by 0.0 to
numerical precision, while genuinely varying weights move it.

So G6 checks the plumbing — that nothing level-dependent leaks in through the
confidence *gate* — and nothing else. It is precisely the failure this project
adopted a standing rule against: **a gate no configuration can fail.** It was
written into the pre-registration, and it passed four times before the identity
was noticed.

**Replacement: G6b, the shuffled-sigma control.** Give each observation a sigma
drawn from the *same distribution* as the true one but assigned to a random
observation, so the weights have identical marginal statistics and carry no
information about which observation is actually bad. Arm (iii) must beat that, not
merely beat a constant. G6b can fail: if the informed and shuffled arms land
together, the delta is coming from the shape of the weight distribution rather
than from knowing which observations to distrust.

Every G6 "PASS" recorded before this correction should be read as a plumbing
check.

## 9a. Correction to flag 1 — the MHR parameter mapping is **still** unrecovered

**Checked before acting on it, and it does not reproduce.** Flag 1 below claims the
249 MHR parameter names can be scraped from the TorchScript pickle in order, and
recommends regenerating `mhr-skeleton-v1.json` with mapped limits and "correcting"
the comment at `src/autoanim_gnm/hand_fit.py:71`. Independent attempt:

- `mhr_model_lod6.pt` carries only 49 strings in `data.pkl` and no parameter names
  at all; `mhr_model_lod1.pt` carries 522, of which **104** are parameter-shaped.
- The three indices the flag cites do not land where it says. Against the raw
  string pool, index 7 is `l_foot`, 24 is `r_talocrural`, 79 is `l_lowarm`.
  Against the parameter-shaped subsequence, 7 is `root_rz`, 24 is
  `l_uparm_twist`, 79 is `l_middle1_rz`. Neither reproduces `spine_twist0`,
  `neck_twist`, `r_index1_rz`.
- And the count is wrong in a way that settles it: `parameter_limits_unmapped`
  carries 198 limits whose `parameter_index` reaches **203**, so the parameter
  space has at least 204 entries. A scrape yielding 104 parameter-shaped strings
  cannot be that list. `pickletools` emits each distinct string **once, at first
  appearance** — it is a deduplicated pool, not an ordered list, so index
  correspondence was never going to survive it.

The "validated by anatomical coherence" caveat was carrying the whole claim, and
coherence is not validation: any monotone subsequence of a joint-parameter
vocabulary reads plausibly.

**So the comment at `hand_fit.py:71` is correct and stands.** It says a wrong limit
is worse than none, because it silently constrains the solve to the wrong
manifold — and acting on flag 1 would have written exactly that wrong limit into
the solver, on the strength of a plausible-looking triple. Recovering the mapping
still needs `pymomentum` or a `get_parameter_names()` call on a machine that can
load the 696 MB module; until then `ANATOMICAL_LIMITS_RAD` stays.

**Flag 1 below is left as written, struck by this section, because the shape of the
error is the useful part.**

## 9. Open flags — carried forward unresolved

Recorded here rather than smoothed over. Each was raised by a scout who could not close it.

1. **MHR parameter names are a pickle-order scrape.** `parameter_names_list` is a plain `List[str]` in the TorchScript `data.pkl`; scraping the 249 names and indexing with `parameter_limits_unmapped.parameter_index` from `src/autoanim_gnm/data/mhr-skeleton-v1.json` gives an anatomically coherent result (`7 spine_twist0 −0.90 0.90`, `24 neck_twist −0.80 0.80`, `79 r_index1_rz −0.78 1.57`; layout 0–135 pose, 136–203 scale, 204–248 `blend_*`; all 198 limits land inside 0–203). **Validated only by anatomical coherence.** Cross-check against `model.get_parameter_names()` on a machine that can afford the 696 MB `torch.jit.load` (Modal) before treating it as final. Follow-up: regenerate `mhr-skeleton-v1.json` with mapped limits, and correct the `note` field and the comment at `src/autoanim_gnm/hand_fit.py:71`, both of which currently assert the mapping "could not be recovered … without Meta's pymomentum".
2. **`hand_fit`'s Euler order.** Extrinsic-vs-intrinsic divergence documented in §2.2. **Not verified as accidental** — confirm with the author before changing anything.
3. **`SOMALayer` on Mac CPU is untested.** Defaults to `device="cuda"`, `mode="warp"`, `ensure_warp_initialized()`, `SkeletonTransfer` at construction. Assume Modal.
4. **The rig world's up-axis is inferred, not documented.** Four camera centres at a common z ≈ 1.6 plus the `_z_up_m` array name are consistent, but this was not read off a calibration document.
5. **`vicon_radial_2` distortion is unverified.** `.cache/mamma/capture/calibration.py:34` declares the model `(pp_x, pp_y, rad_1, rad_2, rad_3)` but contains **no code applying it**. The formula could not be verified and no claim is made about magnitude. **Must be settled before Battle 2's own rig is ingested** — either the `autoanim.calibrated-camera-rig` schema grows distortion coefficients and the fixture renders through them, or footage is undistorted upstream. The AutoAnim body lane currently has **no distortion handling at all**; the only implementation in the repo is `_undistort_detection` in `src/autoanim_gnm/multiview_pipeline.py`, which belongs to the face lane's separate `camera_bundle` stack and is not wired to the body lane.
6. **Apple Vision terms are unaudited.** No audit exists anywhere in `docs/`.
7. **The NVIDIA OML snapshot gap is still open**, and the GEM-X attribution string is not established.
8. **`.cache/.../soma/assets/example_animation.npy`** is `(931, 94, 4, 4)` float32. **94 joints matches neither SOMA's 78 nor MHR's 127**, and it is absent from the repo's `ATTRIBUTIONS.MD`. Provenance and correspondence both unestablished. **Do not use it as a motion source.**
9. **Blender's numeric environment is a hard constraint.** Blender 4.2.0 ships Python 3.11.7 with **numpy 1.24.3 only — no torch, no scipy**. The project `.venv` (3.12.13) has scipy and trimesh but **no torch**; only the system `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` has torch 2.9.0. Consequences: MHR cannot be evaluated inside Blender (pose offline, cache to disk, render in a second process), and `autoanim_gnm.commercial_multiview` and `autoanim_gnm.hand_fit` both import scipy and therefore **cannot be imported inside Blender** — unlike `autoanim_gnm.body`, which `scripts/blender_body_worker.py` does import, and unlike `autoanim_gnm.soma_motion`, which is numpy-only (verified).
10. **`scripts/render_soma_skeleton_preview.py` is not Blender** — it is OpenCV drawing over video frames. Useful as a QA overlay reference (`SOMASKEL77_PARENTS` from `src/autoanim_gnm/soma_motion.py`), nothing more.
11. **A clean BVH → SOMASKEL77 converter does not exist and is not on this fixture's critical path.** `soma/pose_inversion.py::PoseInversion.fit` fits from **vertices**, and both shipped converters (`tools/convert_amass_to_soma.py`, `tools/smpl2soma.py`) route through a posed SMPL/SMPL-X mesh — the research-only path §9a explicitly rules out. A clean route would be BVH FK, a joint-name and rest-axis correspondence onto the 78-joint SOMA rig, then `soma/geometry/rig_utils.py::joint_local_to_world` and `remove_joint_orient_local`. Real work; deferred. None of CMU / Eyes JAPAN / ACCAD / 100STYLE / ASPset-510 / ContactPose is on disk — zero `.bvh` files anywhere in the repo.

---

## 10. Repo conventions this fixture must honour

- **Every output carries a JSON report beside it** with input SHAs and gate results (`CLAUDE.md`, `docs/VERIFICATION.md`). Check the SHA chain rather than assuming a build used current inputs. For the fixture the chain must cover **camera fidelity** (both §3.5 tiers) and the rig file, not just pipeline inputs.
- **`artifacts/` is gitignored.** Scripts must regenerate everything under it.
- **Worker pattern for anything headless in Blender**: `scripts/blender_body_worker.py` — `request.json → response.json`, `sys.path.insert` of `src/`, dedicated exit code, no network, SHA-verified assets, args parsed as `sys.argv[sys.argv.index("--")+1:]` (line 141). That is the template for a fixture worker; `scripts/export_macap_base_model_fbx.py` and `scripts/export_mamma_smplx_neutral_fbx.py` are the direct-exporter template.
- **Confirm a suspected defect with a second, independent measurement before acting.** Several "defects" in prior sessions were artefacts of the metric, not the rig — and §9b's own correction records two analysis-script bugs (a `wxyz`/`xyzw` reorder, and jitter measured on post-`np.interp`, post-Savitzky-Golay data) that produced confidently wrong conclusions. The fixture is a measurement instrument; it is subject to that rule more than anything else in the repo.
