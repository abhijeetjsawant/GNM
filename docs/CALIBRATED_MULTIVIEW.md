# Calibrated multiview capture and independent evaluation

Status: implemented and synthetic-tested; real-person production approval withheld.

## Why a camera sidecar is necessary

GNM vertices use model coordinates. Capture calibration uses a separate world
coordinate system. A supplied `world_to_camera` matrix therefore cannot replace
the original fitted GNM camera directly. Calibrated mode solves one shared
seven-degree-of-freedom similarity:

`gnm_to_camera[i] = world_to_camera[i] × gnm_to_world`

The shared transform contains scale, proper 3D rotation, and translation. Fit
views estimate identity, per-view neutral-expression nuisance, and that shared
transform. Held-out views do not enter any of those solves.

## Request contract

Send the ordered images and optional JSON sidecar to `POST /api/multiview`, or
use `autoanim-gnm multiview ... --calibration rig.json`. Upload order is the
canonical mapping; `index` and `filename` must agree with it exactly.

```json
{
  "schema_version": "autoanim.calibrated_multiview.v1",
  "camera_model": "opencv_radtan",
  "coordinate_convention": "opencv_world_to_camera_+x_right_+y_down_+z_forward",
  "meters_per_world_unit": 1.0,
  "calibration_rms_px": 0.18,
  "pose_error_degrees": 0.4,
  "scale_error_fraction": 0.003,
  "views": [
    {
      "index": 0,
      "filename": "front.png",
      "role": "front",
      "use": "fit",
      "image_size": [1080, 1920],
      "K": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
      "D": [0.01, -0.02, 0, 0, 0],
      "world_to_camera": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1.2],
        [0, 0, 0, 1]
      ],
      "visibility": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
  ]
}
```

The abbreviated example shows one view; a valid bundle requires at least three
`fit` views, at least one `held_out` view, and a nonzero baseline between fit
cameras. Every view requires exactly five OpenCV radial-tangential distortion
coefficients and 68 finite visibility weights in `[0,1]`, with at least 24
visible points.

## Strict ingress validation

The loader rejects the bundle before fitting when any of these invariants fail:

- UTF-8 JSON, finite numbers only, maximum 1 MiB, exact known fields;
- exact schema, camera model, coordinate convention, and ordered indices;
- exact decoded image-size and sanitized filename match;
- canonical finite `K`, zero skew, positive focal lengths, plausible principal point;
- exactly five finite `D` values;
- finite rigid `world_to_camera`, proper rotation, determinant `+1`, canonical last row;
- positive `meters_per_world_unit` and nonzero fit-camera baseline;
- at least three fit views and one held-out view;
- requester-declared calibration RMS at most 0.40 px, pose error at most 2 degrees, and
  scale error at most 1 percent.

Those three values are self-reported metadata, not measurements recomputed by
AutoAnim and not a cryptographic quality attestation. The canonical artifact
therefore names this result `declared_calibration_metadata_gate_passed`.

Images, the mapped 68 landmarks, and the dense 478 landmarks used for texture
masks are undistorted with OpenCV using the original `K` as the output matrix.

## Leakage boundary and gates

Only fit views may affect:

- shared identity coefficients;
- shared GNM-to-world registration;
- per-view nuisance expression;
- fit-view acceptance/rejection;
- texture projection, color harmonization, and source provenance.

Held-out views are projected once after fitting using identity-only neutral
geometry. They never appear in accepted/rejected arrays or texture source maps.
The calibrated result fails closed unless:

- fit aggregate NME is at most `0.025`;
- every accepted fit view NME is at most `0.040`;
- held-out aggregate NME is at most `0.035`;
- every held-out view NME is at most `0.050`;
- accepted fit yaw span is at least 120 degrees;
- identity coefficient saturation is at most 10 percent;
- at least 145/170 sparse-landmark identity directions remain observable.

The calibrated observability rank explicitly marginalizes the one shared
similarity and block-diagonal per-view expression nuisance, and applies the
final landmark-importance, detector-confidence, robust-residual, and view
weights. It does not reuse a rank conditional on already knowing registration.

## Artifacts and provenance

The exact uploaded sidecar is retained, hashed, included in the aggregate input
hash, and preserved in successful or failed job manifests. Successful jobs add:

- `capture-calibration.json`: canonical parsed bundle and source hash;
- `gnm-camera-registration.json`: scale/rotation/translation and reprojection error;
- `fit.npz`: original `K`, `D`, source extrinsics, `gnm_to_world`, effective
  GNM-to-camera matrices, fit/held-out/global indices, and all observations/predictions;
- `fit-report.json`: fit metrics, independently held-out metrics, registration,
  and similarity-marginalized observability.

`texture-maps.npz` keeps the bake-local source map for compatibility and adds
`texture_view_local_to_global` plus `source_view_global`. Color gains/biases use
the explicitly serialized local-to-global ordering; held-out indices cannot be
mistaken for texture sources when view roles are interleaved.

Artificial Euler cameras are not serialized in calibrated mode. Legacy mode is
unchanged and continues to identify its intrinsics as a dimension assumption.

## What has and has not been verified

Automated tests cover strict parsing, non-finite JSON, malformed intrinsics and
distortion, reflected/non-rigid extrinsics, image mismatch, shared-similarity
recovery in metre and millimetre world units, effective-camera pixel agreement,
nonzero image/68-point/478-point undistortion, circular-yaw branch cuts,
accepted-set convergence, locked-camera fitting, exact
sidecar retention, multipart size limits, and a five-fit/two-held-out leakage
test with interleaved roles. Corrupting held-out landmarks changes the verdict while identity
and registration remain bit-for-bit unchanged.

This is meaningful geometric evidence, but it is synthetic. The workspace does
not contain a consented, rights-cleared real person captured by a calibrated
multi-camera rig with an independent 3D scan or held-out likeness labels.
Consequently `production_validated` remains `false`. The next honest gate is a
local synchronized capture or licensed calibrated dataset subset, followed by
scan error, held-out reprojection, texture/lighting, and blinded likeness review.
