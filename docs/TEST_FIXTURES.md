# External test-fixture notice

The fixture downloader retains source attribution, pins immutable revisions,
and verifies SHA-256 before a real-input test runs. Downloaded media is not
part of this repository or an application asset.

## CREMA-D moving speech performance

- Dataset: **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- Creators/citation: H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur,
  A. Nenkova, and R. Verma, “CREMA-D: Crowd-sourced Emotional Multimodal
  Actors Dataset,” *IEEE Transactions on Affective Computing* 5(4), 2014.
- Official repository: <https://github.com/CheyneyComputerScience/CREMA-D>
- Pinned revision: `1658cd342dff90010aa843eaeebd53610a08b1dc`
- Clip: `VideoFlash/1001_DFA_ANG_XX.flv` (actor 1001, “Don't forget a
  jacket,” angry, unspecified intensity)
- SHA-256:
  `10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4`
- Size: 265,922 bytes
- Terms at the pinned revision: the database is offered under the Open Data
  Commons Open Database License (ODbL) 1.0, and individual contents under the
  Database Contents License (DbCL) 1.0. Review the official `LICENSE.txt` and
  satisfy attribution/share-alike obligations before any redistribution.

The clip is opt-in because an open database/content license is not a substitute
for a product-specific review of performer consent, publicity rights, biometric
processing, or the intended jurisdiction. AutoAnim uses it only as local test
and evaluation evidence; it is not training data and is never shipped.

Fetch and verify it with:

```sh
AUTOANIM_FETCH_CREMA_D=1 scripts/fetch_test_fixtures.sh
```

The real E2E test verifies decoding, tracking, dense geometry retargeting,
animated export, and source-video clock preservation. It does **not** prove
microexpression accuracy or production approval: CREMA-D has no per-frame FACS,
gaze, head-pose, lip-contact, or GNM ground-truth track.

## Existing fixtures

The same downloader also documents behavior in its output for the LibriSpeech,
scikit-image astronaut, public-domain official portrait, and opt-in RAVDESS
fixtures. See `docs/SPEC.md` and `docs/VERIFICATION.md` for their checksums,
terms, and current validation scope.

### LibriSpeech phone-alignment evidence

`tests/data/librispeech-5703-47212-0000-first-8s-mfa.TextGrid` is the first
8.0 seconds of the published Montreal Forced Aligner phone track for
LibriSpeech utterance `5703-47212-0000`. The final interval was clipped from
8.26 to 8.0 seconds to match the retained audio excerpt; no timings before the
cut were changed. The source row is pinned to
`changelinglab/librispeech-segment` revision
`63399486c618c9b35ba3c3089d12ff59c9ca50c1`. The checked-in TextGrid SHA-256
is `91a5925a00d0dc771f897cfda932714b1aeff6db3a250d163f77a304c7972365` and
the bound local WAV SHA-256 is
`f298d9abc89993008cd4711e1400ee84e5d4bcd01c55672eb514f33b65dc996b`.

The alignment was produced automatically with MFA, not independently reviewed
by a human, and contains no reviewed articulatory-apex tier. It therefore tests
real phone transport, normalization, scoring, retained-evidence reconstruction,
and motion inertia, but can never pass the production timing gate. The
[published alignment record](https://zenodo.org/records/2619474) and the
[pinned repackaged dataset](https://huggingface.co/datasets/changelinglab/librispeech-segment/tree/63399486c618c9b35ba3c3089d12ff59c9ca50c1)
are CC BY 4.0; retain attribution to Loren Lugosch and the cited MFA work.

## Calibrated synthetic GNM multiview positive control

The retained artifact at
`artifacts/verification/synthetic-calibrated-multiview-v1/` is a fully
synthetic, locally generated GNM-to-GNM positive control. It uses the
Apache-2.0 GNM model assets and contains no real-person images or biometric
data. `result.json` records the generation seed (`20260718`), input camera
poses, fit metrics, texture provenance, and export metrics.

The fixture renders one known GNM identity from five genuinely distinct
calibrated pinhole views: front, left and right three-quarter, and left and
right profile. Their true yaw angles are 0, +35.523, -35.523, +65.890, and
-65.890 degrees. Landmarks include 0.12-pixel synthetic noise and suppress
far-side eye and brow visibility in the profile views. All five views were
accepted, and the fitted yaw span was 131.994 degrees.

This artifact exercises the legacy synthetic core directly:
`MultiViewIdentityFitter` -> GNM atlas -> `bake_multiview_texture` -> component
provenance -> textured GLB export. It intentionally does not exercise upload,
image decoding, or MediaPipe detector ingress; the low-poly synthetic render is
not a legitimate proxy for a photographed human face. It also predates and does
not exercise the versioned camera-sidecar parser, shared GNM-to-world
registration, fit/held-out partition, or held-out gate.

### Calibrated results

- Identity fit: normalized mean error 0.00301523 and mean reprojection error
  0.140133 pixels; observable rank 170/170 and observability ratio 1.0 using
  `observable_subspace_constrained_least_squares_v1`.
- Bounds: coefficient limit +/-3.0, maximum absolute fitted coefficient
  1.084471, and zero saturated coefficients. Unobserved modes 170 through 252
  remain exactly neutral.
- Geometry: direct vertex RMS 2.402 mm and p95 3.786 mm; similarity-aligned RMS
  1.555 mm and p95 2.812 mm.
- Texture: 46,542 atlas texels, with 24,691 observed (53.051%), 11,582
  inpainted (24.885%), 10,269 generic (22.064%), and zero mirrored. Direct
  observation covers 61.894% of skin, 9.148% of the left eye, and 8.956% of
  the right eye. Teeth, gums, and tongue have no direct observations because
  the fixture uses a closed mouth; those components remain generic. Five view
  color fields were harmonized.
- Export: `fitted-textured.glb` is 1,052,404 bytes and contains 18,437 vertices,
  35,324 triangles, 616 seam duplicates, one draw call, and one embedded
  texture. Its SHA-256 is
  `70ab665dd96318d54c1c3bfc4a640ca24093fa386547b9642072485c0a3cd4ec`.
- Validation: the retained [Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator)
  report, `gltf-validator.json`, records 0 errors, 0 warnings, 0 infos, and 4
  `BUFFER_VIEW_TARGET_MISSING` hints with validator version 2.0.0-dev.3.10.
  Those hints concern optional `bufferView.target` declarations; they are not
  validator errors or warnings.
- Evidence bundle: `result.json`, `fit.npz`, `texture.png`,
  `texture-confidence.png`, `texture-provenance.png`, `texture-maps.npz`, the
  rendered views and masks, GLB vertex mapping, `fitted.obj`,
  `fitted-textured.glb`, and `gltf-validator.json`.
- Focused regression evidence: 14/14 multiview tests passed; 46 tests passed
  with 1 duplicate real-input test deselected across multiview fitting,
  multiview pipeline, texture baking, GNM texture, and glTF export suites.

These results are a deterministic synthetic proof of the original shared fitter,
GNM reconstruction, multiview texture core, and standards-valid export.
They are not evidence of real-person detector ingress, robustness to capture
conditions, biometric identity accuracy, likeness approval, or production
identity matching. A prior duplicate-photo detector experiment is not counted
as positive evidence: native MediaPipe creation aborted in the restricted
macOS test environment, and repeated copies of one image do not provide the
distinct viewpoints required by the calibrated fitter.

### Real calibrated face-fixture licensing and size audit

No small, explicitly commercial-compatible, neutral, calibrated real-person
face fixture was found. The remaining production-proof gap is therefore kept
open rather than being inferred from the synthetic positive control.

- [Meta Multiface](https://github.com/facebookresearch/multiface) provides
  calibrated multiview facial images, meshes, head poses, UV textures, audio,
  and camera parameters. The official mini configuration still selects one
  identity and two complete non-neutral expression archives, not individual
  camera-frame slices. The mini download is documented as 16.2 GB, and the
  official object index lists the neutral raw-image archive alone as
  3,108,300,800 bytes. Its
  [license](https://github.com/facebookresearch/multiface/blob/main/LICENSE) is
  CC BY-NC 4.0, so it cannot serve as a commercial product fixture; the license
  also does not itself grant all privacy or publicity rights. See the official
  [mini configuration](https://github.com/facebookresearch/multiface/blob/main/mini_download_config.json),
  [downloader](https://github.com/facebookresearch/multiface/blob/main/download_dataset.py),
  [camera documentation](https://github.com/facebookresearch/multiface/blob/main/documentation/CAMERA_VIEW.md),
  and [object index](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/6795937/index.html).
- [FaceScape's official license agreement](https://facescape.nju.edu.cn/static/License_Agreement.pdf)
  limits use to university researchers and faculty for non-commercial research,
  prohibits redistribution, and restricts publication of portraits except for
  an authorized subset. It is unsuitable for a shipped fixture.
- [Florence 3D Faces](https://www.micc.unifi.it/resources/datasets/florence-3d-faces/)
  supplies scans and videos under academic/non-commercial terms and requires a
  signed access agreement, as documented by its
  [official access page](https://www.micc.unifi.it/vim/3dfaces-dataset/index.html).
- [ETH3D](https://www.eth3d.net/) is calibrated but not face-specific and is
  released under CC BY-NC-SA 4.0, so it does not close the face or commercial
  licensing gap.
- [Google Nerfies](https://github.com/google/nerfies/blob/main/README.md)
  documents calibrated OpenCV cameras and image/camera JSON. Its official
  [0.1 release](https://github.com/google/nerfies/releases/tag/0.1) includes a
  1,863,540,655-byte validation-rig bundle, but the captures are dynamic scenes,
  not a neutral front/three-quarter/profile face benchmark. Although the code
  is [Apache-2.0](https://github.com/google/nerfies/blob/main/LICENSE), the
  repository does not explicitly establish participant likeness, publicity,
  or model-release rights for a production fixture.
- [Google HyperNeRF](https://github.com/google/hypernerf/blob/main/README.md)
  likewise provides calibrated dynamic-scene captures. Its official
  [v0.1 release](https://github.com/google/hypernerf/releases/tag/v0.1) contains
  bundles from roughly 336 MB to 1.8 GB, mostly object, hand, and person scenes
  rather than a neutral facial identity benchmark. Its
  [Apache-2.0 code license](https://github.com/google/hypernerf/blob/main/LICENSE)
  does not document participant likeness or model-release permissions.

### Commercial procurement and consented-capture route

The licensing audit also found two plausible procurement leads, but neither is
a ready-to-commit fixture. [Ten24 SP-6M](https://ten24.info/3d-scan-store/sp-6m/)
advertises paid opt-in commercial scans captured with more than 70 cameras,
including RAW imagery and expression sets. A purchase would still need a
written data-processing and model-training grant plus a small deliverable with
camera intrinsics, distortion, and camera-to-world extrinsics. Renderpeople's
[HumanDataset](https://renderpeople.com/humandataset/)
advertises hundreds of views and custom subsets, but its standard terms exclude
computer-vision and machine-learning use unless separately licensed; capture
consent and derived-biometric rights must therefore be established by contract.

The clean fallback is a local, explicitly consented nine-view capture. Use fit
views at yaw -70, -40, 0, +40, and +70 degrees, and reserve -55, -20, +20, and
+55 degrees for held-out evaluation. Capture at 12 MP or higher with at least
800 pixels across the face, a 70-100 mm full-frame-equivalent lens, locked
exposure/focus/white balance, two large soft sources, cross-polarization where
available, a ColorChecker, and a measured 100 mm scale reference. Two ChArUco
boards should remain visible for calibration. Repeat the sequence once to
measure capture stability.

Before the real fixture can count as production evidence, calibration RMS must
be <= 0.4 px, recovered camera pose error <= 2 degrees, and scale error <= 1%.
Fit only the five designated views; score the four held-out views independently
for reprojection, silhouette/landmark error, identity review, texture coverage,
seams, and color drift. A structured-light or photogrammetry reference with
<= 0.5 mm stated accuracy is preferred for metric geometry gates.

The current pipeline cannot ingest that evidence faithfully yet: it assumes
intrinsics during view analysis (`src/autoanim_gnm/multiview_pipeline.py`) and
does not accept versioned `K`, distortion, and `R|t` sidecars or run held-out
camera/reference evaluators. Those are implementation blockers, not missing
documentation, and must be closed before a real capture is reported as a
calibrated production benchmark.
