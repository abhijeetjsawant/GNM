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

## Audited production gap

The calibrated path is a sound sparse-geometry research baseline, not yet a
production likeness or appearance system. The important boundaries in the
current implementation are:

- identity is optimized from the 68-point GNM correspondence. Calibrated
  multiview exposes 170 landmark-supported identity directions; the remaining
  GNM identity directions are not made observable by adding more photographs;
- the dense 478 MediaPipe landmarks only form a convex-hull texture mask. They
  do not currently constrain the identity solve, facial parts, silhouette, or
  a high-frequency surface residual;
- held-out evaluation is independent of the fit, but it measures the same
  sparse detector correspondence rather than scan-to-surface error or dense
  likeness;
- texture baking is a deterministic CPU z-buffer and UV raster with incidence,
  projected-resolution, mask, and detector-confidence weights. It retains
  exhaustive observed/inpainted/generic provenance per GNM component;
- the 2026-07-19 linear-sRGB phase now interprets integer and float inputs as
  IEC 61966-2-1 sRGB, decodes before sampling/harmonization/blending/fill, and
  encodes the atlas exactly once. Provenance maps are unchanged and the result
  records the color-space contract. Inputs are still assumed sRGB: there is no
  embedded/profile-aware transform, chart calibration, illumination
  separation, semantic face/eye/mouth/occluder mask, or seam-energy measurement;
- multiview export is capped at 1024 square. The baker allocates float64
  `[view,height,width,3]` samples plus float64 weights, so merely raising the
  limit to 4K at 12 views would allocate about 6.4 GiB for those two arrays
  before raster buffers, images, and output maps;
- the automatic result is base color only. Normal, displacement, roughness,
  specular, and subsurface maps can be attached through the separately gated
  material-package workflow, but this pipeline does not recover them;
- the Three.js page is an orbit/wireframe/exposure preview. It does not lock to
  source cameras or display source/render pairs, residuals, confidence,
  provenance, seam heatmaps, held-out views, or unseen-light relighting.

These limits explain why a low sparse NME can coexist with a face that is not a
convincing likeness, and why a complete-looking RGB atlas can still contain
lighting seams and non-relightable highlights.

## Next phase: benchmark-first dense identity and measured appearance

The next implementation phase should preserve the current 68-point fit as the
initialization and trust boundary, then add evidence in five independently
testable work packages. It must not replace the production gate with a learned
model's confidence score.

1. **Source-backed calibration and fixtures.** Store raw ChArUco/checkerboard
   observations, solver flags, per-frame/per-camera residuals, covariance or
   repeatability, calibration software version, and hashes. Recompute the
   metrics instead of trusting the three declared summary fields. Require a
   minimum of five fit cameras and two untouched held-out cameras for a
   production fixture.
2. **Dense, part-aware GNM refinement.** Add dense facial correspondences,
   semantic facial-part masks, profile silhouette, and calibrated projection
   residuals. Optimize only directions supported by the evidence and retain a
   regularized GNM prior. Add an optional neutral residual surface for detail
   that the parametric identity basis cannot represent; keep animation on the
   low-frequency GNM layer.
3. **Color-managed, scalable texture bake.** The explicit sRGB-to-linear-sRGB
   working transform and single output encode are implemented and covered by
   adversarial black/white, round-trip, provenance, and calibrated-pipeline
   tests. Remaining work is to decode tagged input profiles into a documented
   interchange space, solve exposure/white-balance from chart and overlap
   evidence, replace the convex hull with semantic masks, and exclude hair,
   sclera, mouth opening, highlights, and transient occluders from skin
   harmonization. Use tiled/streamed accumulators and island dilation so 4K/8K
   does not scale as `views × atlas pixels` in memory.
4. **Measured PBR capture, not RGB inference claims.** Recover diffuse,
   specular, normal, and roughness only from a controlled polarized or otherwise
   calibrated illumination sequence. An uncontrolled photo may initialize
   appearance, but it must remain labeled baked RGB and non-relightable.
5. **Review workspace.** Add source-camera selection; synchronized
   source/render/difference views; fit-versus-held-out labels; landmark,
   silhouette, scan, confidence, provenance, and seam overlays; material-map
   isolation; a fixed unseen-light sweep; and an immutable reviewer decision.

The geometry work should be evaluated both with the GNM-only neutral surface
and with the optional residual enabled. Otherwise a detail layer can hide a
regression in the reusable parametric identity.

## Initial measurable acceptance gates

The following are initial engineering targets, not universal research
standards. A pilot capture must record the full distributions and may tighten
them; weakening a threshold requires a versioned decision and new blinded
review. Existing sparse gates above remain mandatory.

| Evidence | Initial pass condition |
| --- | --- |
| Calibration | Metrics recomputed from retained raw pattern detections; RMS `<=0.40 px`, per-view p95 `<=0.75 px`, repeated extrinsic rotation `<=0.5°`, metric scale error `<=0.5%`; at least 5 fit + 2 held-out cameras and `>=120°` fit yaw span. |
| Held-out sparse geometry | Existing aggregate 68-point NME `<=0.035` and every view `<=0.050`, with identity and registration bit-identical when held-out observations are corrupted. |
| Held-out dense geometry | On manually audited dense/part/silhouette labels, median reprojection `<2 px` and p95 `<5 px` per held-out camera; report each part and pose separately, never aggregate away a failed profile or mouth region. |
| Metric likeness | Against an independent neutral scan, after rigid metric alignment with no scale optimization: face-region point-to-surface median `<=1.0 mm`, p95 `<=2.5 mm`; surface-normal median `<=8°`, p95 `<=20°`. Also report GNM-only and GNM-plus-residual results. |
| Subset stability | Across at least three disjoint fit-camera subsets that pass coverage, canonical GNM-only face vertices differ by median `<=0.5 mm`, p95 `<=1.5 mm`; no coefficient may pass solely by saturating its bound. |
| Direct texture coverage | Skin observed fraction `>=0.90`, skin generic fraction `0`, skin inpainted fraction `<=0.05`, overlap fraction `>=0.30`; publish separate eye, teeth/gum, and tongue values rather than counting generic oral tiles as facial coverage. |
| Color and seams | Captured chart CIEDE2000 median `<=2`, p95 `<=4`; across valid overlapping skin observations and every UV-island seam, CIEDE2000 median `<=2`, p95 `<=5`, measured after the declared color transform in a fixed neutral render. |
| Held-out appearance | For controlled diffuse/polarized held-out cameras, masked linear-color PSNR `>=30 dB`, SSIM `>=0.95`, LPIPS `<=0.05`; report the same metrics without per-image exposure fitting so the evaluator cannot erase a color failure. |
| Relightable material | On lights absent from recovery, masked PSNR `>=30 dB`, SSIM `>=0.95`, LPIPS `<=0.05`; normal angular median `<=8°`, p95 `<=20°` against measured reference; repeat-capture roughness coefficient of variation `<=10%`. Fail the relightable claim if no independent reference exists. |
| Native detail | A pore/detail claim requires native maps `>=8192`, measured millimetres per texel, no upsampling, and a calibrated reference showing preserved contrast over approximately `0.1–0.5 mm` spatial detail. |
| Runtime and review | 4K bake peak RSS `<=2 GiB` and 8K peak RSS `<=4 GiB` on the documented reference machine; deterministic output hashes; zero failed source-camera associations; every held-out failure visible in the review workspace; blinded identity and look-development approval from at least two reviewers. |

PSNR, SSIM, and LPIPS are supporting image metrics, not substitutes for the
scan, provenance, color, and human-review gates. They are meaningful for
appearance only under controlled capture; scoring an uncontrolled photograph
against a lit render would primarily measure mismatched illumination.

## Data and licensing blockers

No fixture currently in this workspace can pass all of those gates. Public
datasets are useful for research regressions but do not establish permission
to ship a commercial likeness:

- [Meta Multiface](https://github.com/facebookresearch/multiface) provides
  dense calibrated multi-camera face capture and tracked meshes, but its
  [CC BY-NC 4.0 license](https://github.com/facebookresearch/multiface/blob/main/LICENSE)
  is noncommercial and the tracked geometry is not fully independent of the
  image capture being evaluated;
- the [NoW benchmark](https://now.is.tue.mpg.de/) supplies a standard
  scan-to-mesh evaluation for monocular neutral-face reconstruction, but its
  [license](https://now.is.tue.mpg.de/license.html) permits noncommercial
  scientific research and forbids redistribution and commercial use;
- [USC Digital Emily 2](https://vgl.ict.usc.edu/Research/DigitalEmily2/) is a
  useful polarization/microgeometry research reference, but its public package
  is not a drop-in AutoAnim calibrated-camera fixture and its usage terms must
  be approved before download or derived-artifact retention;
- the repository therefore still needs an in-house, explicitly consented and
  commercially authorized neutral subject with raw calibration frames,
  synchronized RGB/polarized views, a color chart, an independent metric scan,
  measured macro detail, held-out cameras/lights, and a signed retention and
  derivative-use policy.

Research supports the architecture but does not supply an implementation here.
[MICA](https://is.mpg.de/publications/mica-eccv2022) demonstrates why metric
identity needs stronger supervision than ordinary monocular reconstruction;
[DECA](https://deca.is.tue.mpg.de/) separates a parametric face from animatable
detail; part-guided fitting improves extreme-expression reconstruction in
[3DDFA-V3](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_3D_Face_Reconstruction_with_the_Geometric_Guidance_of_Facial_Part_CVPR_2024_paper.html).
For appearance, polarized smartphone capture recovers diffuse/specular response
and high-resolution normals in
[Azinović et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Azinovic_High-Res_Facial_Appearance_Capture_From_Polarized_Smartphone_Images_CVPR_2023_paper.html),
while a co-located phone-flash sequence is taken further in
[Han et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_High-Quality_Facial_Geometry_and_Appearance_Capture_at_Home_CVPR_2024_paper.pdf).
Those systems are evidence that the proposed capture tier is feasible, not code
or checkpoints included with AutoAnim.
