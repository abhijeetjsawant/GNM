# GNM Research: Architecture, Data Model, and Extension Surface

Research snapshot: 2026-07-18
GNM commit: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`
Upstream: <https://github.com/google/GNM>

## Executive verdict

GNM Head 3.0 is a high-quality parametric 3D head model and differentiable mesh generator. It is not a facial capture, audio analysis, animation authoring, or photorealistic rendering system. It gives downstream software an unusually rich endpoint: a fixed-topology 17,821-vertex head with identity, regional expression, neck/head/eye pose, teeth, tongue, eyeballs, UVs, vertex groups, and equivalent NumPy, JAX, PyTorch, and TensorFlow APIs.

That makes GNM a strong foundation for AI-driven facial animation if the missing translation layers are built around it:

- audio or text to timed mouth and emotion controls;
- image/video observations to regularized GNM coefficients;
- temporal smoothing, contact/collision checks, confidence reporting, rendering, and export;
- application workflows and evaluation data.

The released repository contains none of those perception layers. Google's root README calls perception and analysis technology roadmap work, and only GNM Head is currently catalogued. Any claim that GNM alone performs lip sync or photo-to-avatar reconstruction is false.

## What is actually released

The repository contains 61 tracked files and one model family:

| Item | Released state |
|---|---|
| Model | GNM Head |
| Version | 3.0 |
| Variant | `head` |
| Model asset | `gnm/shape/data/versions/v3_0/gnm_head.npz`, about 51 MB |
| Semantic decoders | expression and identity Keras H5 files, about 2.7 MB total |
| Sparse landmarks | one 68-point head definition |
| Backends | NumPy, JAX, PyTorch, TensorFlow |
| License | Apache-2.0 for GNM code and checked-in assets |
| Perception stack | not released |
| Learned albedo/texture | not released |
| Audio/animation model | not released |

The single model/variant constraint is explicit in [`gnm_specs.py`](../gnm/shape/data/versions/gnm_specs.py) and [`gnm_catalog.py`](../gnm/shape/data/versions/gnm_catalog.py). The shape-only schema is explicit in [`gnm_data_schema.py`](../gnm/shape/gnm_data_schema.py): template geometry, bases, joints, topology, UVs, regressors, skin weights, and vertex groups are present; a camera, appearance basis, lighting model, image encoder, or audio encoder is not.

## Formal model

GNM is a 3D morphable model with linear identity and expression bases followed by pose correctives and linear blend skinning.

For identity coefficients `beta`, expression coefficients `phi`, joint rotations `theta`, and translation `tau`, the bind-pose vertices are:

```text
V_bind = template + sum(beta_i * identity_basis_i)
                  + sum(phi_j * expression_basis_j)
```

Identity also changes the four joint locations:

```text
J_bind = template_joints + sum(beta_i * joint_identity_basis_i)
```

The implementation then adds rotation-dependent pose correctives, propagates the four joints through a kinematic chain, and applies artist-authored linear blend-skinning weights. The concrete execution order is visible in [`GNM.__call__`](../gnm/shape/gnm_xnp.py): validate/broadcast parameters, create bind-pose vertices, create identity-dependent joints, apply pose correctives, then skin into world space. The linear bind-pose equations are implemented in [`gnm_common.py`](../gnm/shape/gnm_common.py).

This order matters to extension code:

- identity and expression coefficients are additive only before posing;
- neck/head rotation changes geometry through both pose correctives and skinning;
- photo fitting should optimize an independent camera first, not misuse head pose to explain camera pose;
- audio mouth motion must use the expression basis because there is no jaw joint.

## Exact v3.0 data model

The checked-in NPZ contains 23 arrays:

| Field | Shape | Meaning |
|---|---:|---|
| `template_vertex_positions` | `[17821, 3]` | neutral dense head and internal anatomy |
| `vertex_identity_basis` | `[253, 17821, 3]` | identity displacements |
| `expression_basis` | `[383, 17821, 3]` | expression displacements |
| `template_joint_positions` | `[4, 3]` | neutral joints |
| `joint_identity_basis` | `[253, 4, 3]` | identity-dependent joint movement |
| `joint_names` | `[4]` | neck, head, left eye, right eye |
| `joint_parent_indices` | `[4]` | kinematic hierarchy |
| `skinning_weights` | `[4, 17821]` | per-joint vertex weights |
| `pose_correctives_regressor` | `[36, 53463]` | rotation corrective mapping |
| `joint_regressor` | `[4, 17821]` | vertices to joint locations |
| `triangles` | `[35324, 3]` | triangle topology |
| `quads` | `[17662, 4]` | quad topology |
| `triangle_uvs` | `[35324, 3, 2]` | triangle-corner UVs |
| `quad_uvs` | `[17662, 4, 2]` | quad-corner UVs |
| `mirror_indices` | `[17821]` | left/right vertex correspondence |
| `vertex_groups` | `[46, 17821]` | soft anatomical/semantic masks |

The 253 identity dimensions are region-concatenated:

- 170 head;
- 3 eyeball;
- 80 teeth.

The 383 expression dimensions in the actual asset are:

- 100 left periocular;
- 100 right periocular;
- 150 lower face;
- 32 tongue entries, consisting of `tongue_mean` plus 31 modes;
- 1 pupil dilation entry.

The bundled formal-definition PDF has a stale table that reports 31 tongue and 382 total expression dimensions. The actual NPZ, runtime, expression names, and current README all report 32/383. Runtime dimensions must be queried instead of copying the PDF table.

The face points along `+Z` and the neck points along `+Y`. This convention is stated in the formal definition and confirmed by the landmark coordinates.

## Parameters are statistical, not animator controls

The main identity names are `head_000` ... rather than semantic traits. Expression names are regional PCA components such as `left_eye_region_000` and `lower_face_region_000`. There is no `jawOpen`, `mouthFunnel`, ARKit, FACS action unit, phoneme, or viseme control surface.

The regional construction is still valuable. Speech controls can be restricted to the 150 lower-face and 32 tongue dimensions, while emotions and blinks can use the two 100-dimensional periocular blocks. Region masking is required because the semantic decoder can leak motion into unrelated regions.

## Semantic samplers

[`semantic_sampler.py`](../gnm/shape/semantic_sampler.py) wraps two conditional variational autoencoder decoders:

- identity: two binary gender categories and four broad demographic categories;
- expression: 20 labels including happy, surprise, disgust, stretch face, funneler, pucker, smile wide, lips roll in, winks, snarl, and tongue center.

These are generative samplers, not inverses. The identity sampler cannot infer a person's GNM coefficients from a photograph. The expression sampler creates plausible examples for a label, but does not turn audio or tracked action units into animation.

The H5 expression decoder is a small feed-forward network: `84 -> 64 -> 128 -> 256 -> 512 -> 383` with ReLU hidden layers. It can be evaluated exactly with NumPy and `h5py`; TensorFlow is not required at application runtime. Using zero latent noise produces deterministic class prototypes suitable for bootstrapping a viseme library. Local measurements confirmed distinct neutral, open, rounded, puckered, and smiling mouth geometry.

## Landmarks and fitting utilities

GNM releases one `HEAD_SPARSE_68` set. Each point is a barycentric combination of three mesh vertices, loaded by [`gnm_landmarks.py`](../gnm/shape/gnm_landmarks.py). This gives stable, differentiable 3D landmarks for fitting and evaluation.

Two cautions matter:

1. The checked-in left-jaw rows do not follow conventional iBUG-68 spatial order. A tested adapter is required before comparing them with a standard 68-point detector.
2. Sparse landmarks cannot observe every coefficient. Direct basis inspection shows zero 68-landmark motion for identity dimensions 170:253 (eyes and teeth) and expression dimensions 350:383 (tongue and pupil). A photo landmark fit must keep those values neutral.

The fitting helpers solve regularized least squares or project corresponding 3D vertices onto a linear basis. They are useful when the target already shares GNM topology, such as an artist-sculpted viseme mesh. They are not a single-image reconstruction pipeline: no image landmarks, camera, visibility, rasterizer, photometric term, or identity prior is supplied.

## Multi-framework implementation

GNM centralizes most behavior in a backend-agnostic `etils.enp` implementation and exposes thin NumPy, JAX, PyTorch, and TensorFlow subclasses. This is a strong integration choice:

- NumPy is sufficient for deterministic serving and export;
- PyTorch/JAX/TF make gradient-based research and training possible;
- identical topology and parameters allow offline fitting, web serving, and training to share files.

Two clean-room baselines loaded the real asset. A minimal NumPy/data/fitting environment passed:

```text
219 passed, 2 skipped, 28 parameterized subtests passed
```

The run covered the upstream NumPy, data loader, schema/base, landmarks, shared utilities, cross-backend abstraction, PCA projection, and regularized least-squares suites. It did not claim the optional TensorFlow/JAX/render suites passed in the initial minimal environment.

A separate fresh Python 3.13 install of `gnm/shape[all,dev]` succeeded and Google's official runner reported 278 passing tests in 37.609 seconds on this Apple Silicon host. The audit also exposed a test-discovery gap: the unittest runner does not descend into `fitting_utils/` and `visualization/` because those directories lack `__init__.py`. Direct additional runs passed 62 fitting/vertex-color tests and 21 camera/color tests, with three renderer-related skips. A green official runner therefore does not mean every checked-in test was collected.

Local NumPy measurements were about 8.69 ms for a warm single-frame model evaluation and about 261.5 ms for a 30-frame batch. These are hardware-specific development measurements, not product throughput guarantees.

## Rendering and export constraints

GNM includes topology, UVs, an edge-flow debug texture, camera helpers, and a `pyrender` path. It does not include a learned person-specific texture or production renderer. The current pyrender module forces OSMesa before import, while `pyrender` is not declared in `pyproject.toml`; this is fragile on macOS.

For an application, use:

- OpenCV overlays and CPU preview video for repeatable test artifacts;
- OBJ/NPZ/JSON export for geometry and coefficient data;
- a browser viewer or GLB generation for interactive inspection;
- a Linux/CUDA differentiable renderer only for later photometric inverse rendering.

The upstream package also has release-engineering defects that application setup must work around:

- `pyrender` is imported but absent from declared dependencies;
- the renderer forces OSMesa, which fails on a normal macOS installation without the OSMesa library;
- the built wheel is about 81 MB, includes tests and large README media, and omits the required `data/landmarks/head_sparse_68.txt` because `*.txt` is missing from package data;
- the root license was not present in the inspected wheel;
- core installation always pulls TensorFlow and notebook/OpenCV-era dependencies even for NumPy-only use;
- the README names a nonexistent `gnm_colab_viewer.py`; the actual file is `gnm_jupyter_viewer.py`;
- the project citation is still `coming soon`.

For this application, install GNM editable from the pinned repository, use the NumPy path, include the landmark source explicitly, and avoid the upstream pyrender backend on macOS.

## Capabilities

GNM is immediately useful for:

- dense, fixed-topology head generation;
- independent identity and facial expression control;
- gaze/eye rotation, neck/head pose, and global translation;
- visible teeth, gums, tongue, cornea, iris, sclera, and pupil dilation;
- batched animation generation;
- gradient-based coefficient optimization;
- corresponding-mesh projection into identity/expression space;
- UV-aware asset export and vertex-group-specific processing;
- synthetic data generation with exact ground truth.

## Limitations

- Only Head v3.0 is released; the broader ecosystem remains roadmap work.
- No audio, text, image, or video encoder is included.
- No jaw joint or animator-friendly control rig is included.
- No temporal model, coarticulation, blinks, collision, or lip/teeth/tongue contact model is included.
- No learned texture/albedo, hair, facial hair, wrinkles, material, illumination, or background model is included.
- A single view cannot identify absolute scale, depth, ears/rear skull, or hidden anatomy.
- The 68 landmarks cannot identify eye, teeth, tongue, pupil, or fine surface coefficients.
- Semantic expression classes omit native anger, sadness, fear, neutral, and phoneme targets.
- Expression decoder outputs are stochastic unless seeded and can mix regions.
- The model has documented demographic representation limits: binary gender training categories and four broad demographic groups.
- The current README has small structural/name discrepancies, and the formal PDF has a stale expression-count table.
- The official test runner omits subdirectory tests, and the wheel omits the landmark TXT required at runtime.
- The README describes rotations ambiguously as a `4x3 Rotation matrix`; the API consumes four axis-angle vectors in radians.
- GNM's Apache license is permissive, but alternative perception models often have research-only or non-commercial model licenses.

## What is feasible to build on top

Feasible now, using permissive/local components:

- offline audio-to-GNM cue animation with Rhubarb and calibrated semantic prototypes;
- transcript-guided, phone-accurate animation with a forced aligner;
- coarse single-photo visible-geometry fitting using a regularized 68-landmark objective;
- local web/CLI tooling for upload, fitting, preview, diagnostics, and export;
- synthetic labeled face, landmark, gaze, and expression datasets;
- expression retargeting from MediaPipe/ARKit-like controls through a calibrated adapter;
- multi-view or short-video identity fitting with shared identity coefficients;
- artist-authored viseme projection and animation libraries.

Not honest to promise from the released assets alone:

- a photorealistic or metrically accurate "3D clone" from one photo;
- production-quality emotional speech animation without paired training data or artist calibration;
- reliable identity from a heavily posed, occluded, low-resolution, or expressive single image;
- texture/hair reconstruction;
- clinically or biometrically valid anthropometric estimates.

## External research context

- [DECA: Detailed Expression Capture and Animation](https://arxiv.org/abs/2012.04012) demonstrates learned single-image FLAME shape/detail/albedo/expression/lighting estimation, but its released code/model has research-oriented licensing and an older environment.
- [EMOCA](https://emoca.is.tuebingen.mpg.de/) shows that landmark, photometric, and recognition losses alone do not preserve emotional expression well.
- [MICA](https://github.com/Zielon/MICA) is a stronger metric identity initializer, but released checkpoints depend on FLAME licensing that must be reviewed before commercial use.
- [WhisperX](https://arxiv.org/abs/2303.00747) improves word timing through voice activity detection and forced phoneme alignment.
- [VOCA](https://openaccess.thecvf.com/content_CVPR_2019/html/Cudeiro_Capture_Learning_and_Synthesis_of_3D_Speaking_Styles_CVPR_2019_paper.html), [FaceFormer](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.pdf), and [CodeTalker](https://openaccess.thecvf.com/content/CVPR2023/html/Xing_CodeTalker_Speech-Driven_3D_Facial_Animation_With_Discrete_Motion_Prior_CVPR_2023_paper.html) show that convincing speech animation is learned from paired audio and facial motion, not obtained from a static morphable model alone.
- [EmoTalk](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html) and [EmoVOCA](https://openaccess.thecvf.com/content/WACV2025/html/Nocentini_EmoVOCA_Speech-Driven_Emotional_3D_Talking_Heads_WACV_2025_paper.html) separate speech content from emotional motion and highlight the scarcity of emotional 3D training data.
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker) supplies 478 landmarks, 52 blendshapes, and a face transform, but its depth is not a metric personalized face reconstruction.
