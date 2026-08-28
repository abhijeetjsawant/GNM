# Commercial multiview body capture: MAMMA-quality path without MAMMA

## Decision

Do not ship MAMMA, its weights, its outputs, or the publicly downloaded SMPL-X
model under their default research licenses. MAMMA's license restricts its data,
models, software, and generated artifacts to non-commercial work and explicitly
forbids using them to train a commercial method. The public SMPL-X model also
defaults to non-commercial research; commercial licensing is available
separately. This is a product boundary, not merely an attribution task.

AutoAnim now has a clean-room calibrated multiview path that runs without any
MAMMA code, checkpoint, output, or SMPL-X model. On macOS it uses Apple's Vision
framework for 19-point 2D observations, then AutoAnim-owned geometry and
retargeting. The existing MAMMA example videos and calibration are used only as
a research comparison fixture. They must be replaced with owned/cleared capture
before production qualification.

This first implementation is a working sparse-body baseline, not a claim of
MAMMA visual parity. It reconstructs stable body motion from the same four-view,
five-second, two-performer clip and exports two animated AutoAnim-55 GLBs. It
does not yet estimate dense body shape or articulated fingers.

## What MAMMA does that made it look good

The checked-out MAMMA repository declares this five-stage graph in its README
and step documentation:

`ma_cap -> ma_masks -> ma_2d -> ma_3d -> ma_vis`

1. `ma_cap` synchronizes calibrated camera inputs.
2. `ma_masks` separates people with YOLO/SAM masks.
3. `ma_2d` runs the learned MammaNet dense body/hand landmark model.
4. `ma_3d` globally fits a shaped SMPL-X body across calibrated views and time.
5. `ma_vis` renders the fitted surface and camera overlays.

The quality difference is therefore not simply “four cameras versus one.” The
segmentation, dense landmarks, shaped parametric body prior, joint limits,
surface fitting, identity tracking, and temporal optimization all constrain the
same solution. Our earlier squat result worked because the full body was large,
unoccluded, and geometrically constrained. Dialogue acting frequently crops or
occludes wrists, fingers, hips, and legs, so a monocular or sparse per-frame
retargeter becomes underconstrained. More calibrated views materially help, but
they do not replace stronger observations and a global body prior.

## Implemented architecture

```text
Four synchronized videos + calibrated intrinsics/extrinsics
    |
    +-- Apple Vision 2D body observations (19 joints, every frame/view)
    |
    +-- permutation-based multi-person association
    |      camera-consensus reward
    |      previous 2D, skeleton, and 3D-root continuity
    |
    +-- confidence-weighted robust DLT triangulation
    |      camera-subset search + cheirality + reprojection rejection
    |
    +-- temporal root/body-volume rejection
    |      bounded acceleration + interpolation + Savitzky-Golay smoothing
    |
    +-- metric Z-up capture -> Y-up AutoAnim conversion
    |
    +-- sparse pose -> AutoAnim-55 IK + contact projection
    |
    +-- two BodyTrack JSON/NPZ files + animated GLBs + review viewer
```

Code ownership is deliberately split:

- `workers/commercial_multiview/apple_vision_pose.swift` performs only 2D
  detection and emits a versioned JSONL contract.
- `src/autoanim_gnm/commercial_multiview.py` owns camera geometry, association,
  triangulation, temporal repair, coordinate conversion, IK, and contacts.
- `scripts/build_commercial_multiview_comparison.py` owns fixture orchestration,
  provenance hashes, export, and review artifacts.
- `scripts/verify_commercial_multiview_artifact.py` is the fail-closed acceptance
  gate. It verifies metrics, shapes, finite values, normalized quaternions,
  contacts, GLBs, duration, and the declared absence of restricted dependencies.

GNM remains the face/head owner. This body phase never overwrites GNM jaw,
eyes, expression, oral anatomy, or tongue channels. The character-assembly layer
must compose GNM head motion with the body track through a reviewed neck seam.

## The critical mathematical finding

Minimizing reprojection residual alone is wrong for multi-person association.
An accidental pairing from two cameras can triangulate to a low-residual “ghost”
while the true four-camera correspondence has a slightly larger detector error.
The implemented objective rewards camera support before applying temporal 2D
and 3D continuity. A later bounded-acceleration and body-volume gate rejects
remaining implausible roots. The same-clip evaluation exposed this defect: the
earlier version produced metre-scale identity errors; the corrected version has
no root frame more than 300 mm from the MAMMA reference trajectory.

`association_objective_median` is intentionally unitless. It combines pixel
residuals, missing-camera penalties, and temporal terms. Raw accuracy metrics are
reported separately as reprojection errors in pixels.

## Same-clip result

The final research comparison processes frames `[60, 210)` at 30 Hz from all
four cameras: 150 reconstructed frames, two people, and approximately five
seconds of output.

Measured acceptance results:

| Gate | Result | Threshold |
|---|---:|---:|
| Direct finite 3D joint coverage | 82.77% | >= 80% |
| Interpolated joint fraction | 17.23% | <= 20% |
| Median reprojection error at 1280 detector width | 4.63 px | <= 6 px |
| P95 reprojection error | 11.05 px | <= 12 px |
| Maximum accepted reprojection error | 19.89 px | <= 25 px |
| Sparse joint -> fixed neutral-rig endpoint median | 163.98 mm | <= 180 mm (P0 only) |
| Sparse joint -> fixed neutral-rig endpoint P95 | 377.99 mm | <= 400 mm (P0 only) |
| Exported performers | 2 x AutoAnim-55 GLB | 2 required |

The endpoint gate is intentionally loose and is the clearest numeric evidence
that P0 is not dense-body parity. The fixed neutral body has generic torso, arm,
and leg proportions; direction/acting is retained, but joint endpoints can be
hundreds of millimetres from the reconstructed performer. A production result
requires the per-character shaped skeleton/body work in P2/P3. Reprojection
quality must never be presented as equivalent to final rig-fit quality.

For diagnostic comparison only, sparse joints were compared with the prior
MAMMA result. Subject 0 differed by 60.83 mm median / 184.43 mm P95 across the
available body joints; subject 1 by 79.83 mm median / 221.15 mm P95. Root median
differences were 54.99 mm and 40.39 mm. These are **method-to-method
differences**, not ground-truth accuracy. The largest remaining discrepancies
are shoulders, neck, and intermittently wrists, which matches the limitations of
19-point observations and occlusion.

## Production phases

### P0 — sparse commercial baseline (implemented)

- Apple Vision 19-joint detector on macOS.
- Four-camera calibrated reconstruction, multi-person association, temporal
  repair, AutoAnim-55 retarget, contacts, GLB export, and comparison viewer.
- Unit tests for robust outlier rejection, shuffled detector ordering, and
  finite quaternion-normalized BodyTrack compilation.
- Same-clip automated artifact gate.

Exit: current research fixture passes its numeric gate. It is suitable for
engineering continuation, not final capture quality certification.

### P1 — whole-body observations and hands

- Introduce a detector interface so Apple Vision and a GPU worker share the same
  versioned observation schema.
- Evaluate Apple's newer `DetectHumanBodyPoseRequest.detectsHands` path on the
  deployment OS, and an audited RTMW/RTMPose whole-body checkpoint on the RTX
  3090 or Modal. MMPose code is Apache 2.0, but every chosen checkpoint and its
  training datasets must receive an asset-level license record before use.
- Add palm, finger, toe, and face-adjacent confidence channels; retain GNM as
  face owner.
- Add close-hand multicamera fixtures and finger self-intersection/contact tests.

Exit: P95 wrists <= 80 mm against owned mocap ground truth; fingertip P95 <= 15
mm in close hand views; no axis inversions; no unlicensed asset in the manifest.

### P2 — global identity and temporal optimization

- Replace greedy frame association with a shot-level min-cost-flow/Viterbi solve.
- Add per-view appearance embeddings trained or licensed for commercial use.
- Run robust bundle adjustment over camera time offsets, roots, and joint tracks.
- Fit bone lengths once per character/shot and enforce joint limits and symmetry.
- Model occlusion explicitly rather than treating every low-confidence point as
  ordinary missing data.

Exit: zero identity switches on owned two-to-six-person stress captures; root
P95 <= 30 mm; limb P95 <= 50 mm; deterministic reruns.

### P3 — dense commercial body model

Two viable routes exist:

1. License SMPL-X commercially and isolate it behind the body-model interface.
   This is the shortest route to proven dense shape, hands, and skinning.
2. Build an owned parametric body: cleared scans, one canonical topology,
   blend-shape/PCA identity space, learned pose correctives, skinning weights,
   joint regressor, collision proxies, and differentiable multiview fitting.
   This offers maximum control but is a dataset and model program, not a small
   code phase.

Do not train from MAMMA output or its restricted training assets. A clean-room
model requires owned or explicitly commercial training data and documented
provenance.

Exit: owned scan benchmark within 5 mm mean surface error, stable cloth/body
silhouette under acting poses, no limb volume collapse, and commercial asset
audit passes.

### P4 — GNM character assembly

- Solve and persist per-character neck seam alignment, scale, rest transforms,
  material boundary, and LOD policy.
- Compose GNM facial performance and body motion on a shared timebase.
- Add gaze/neck arbitration so head intent from facial acting and torso intent
  from body capture do not double-rotate the neck.
- Validate dialogue shots containing head turns, shoulder contact, hands near
  face, tongue motion, blinks, and speech.

Exit: one five-to-ten-second owned multicamera dialogue shot exports as one
character with synchronized GNM face and body; no seam, timing, hand, palm,
tongue, or replay defect in reviewer sign-off.

### P5 — production qualification

- Capture owned ground truth with synchronized shutters, timecode, calibration
  board, and optical or inertial reference.
- Test camera dropout, occlusion, crossing identities, motion blur, cropped
  bodies, loose clothing, seated acting, props, and hands touching the face.
- Record per-shot confidence, manual correction handles, immutable provenance,
  and deterministic export hashes.
- Require legal review of every model, checkpoint, dataset, texture, and body
  asset; code license alone is insufficient.

Exit: all accuracy and reliability gates pass on held-out owned footage and a
solo animator can diagnose/correct failures without changing source code.

## External sources checked

- MAMMA license (non-commercial and no commercial training):
  https://github.com/cuevhv/mamma/blob/main/LICENSE
- GNM repository (GNM Head, Apache 2.0, commercial use):
  https://github.com/google/GNM
- Apple 2D body pose documentation (up to 19 body points):
  https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images
- Apple 3D body pose documentation (17 joints; most prominent person only):
  https://developer.apple.com/documentation/vision/identifying-3d-human-body-poses-in-images
- MMPose/RTMPose implementation and license:
  https://github.com/open-mmlab/mmpose
- SMPL-X default and commercial licensing information:
  https://smpl-x.is.tue.mpg.de/modellicense.html

This document is an engineering license boundary, not legal advice. Final
commercial clearance requires counsel and a per-asset bill of materials.
