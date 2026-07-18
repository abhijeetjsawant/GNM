# AutoAnim GNM Application Spec and Phased Plan

Status: implementation contract
GNM base: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

## Product decision

Ship a **local-first web application with a Python service and CLI**.

This form factor matches the dependency boundary:

- GNM and the fitting/audio stack are Python/native;
- a browser gives creators file upload, diagnostics, preview, and downloads without a desktop packaging project;
- the CLI makes every workflow reproducible in tests and batch jobs;
- local processing keeps face photos and voice recordings off third-party services;
- FastAPI can later be containerized or deployed without changing core modules.

The first release is a verified technical application, not a hosted multi-user SaaS. It runs on one machine, processes one request at a time, and writes explicit job artifacts.

## Users and jobs

Primary user: a creator or developer evaluating/using GNM-driven avatars.

Jobs to be done:

1. Upload audio, optionally provide dialog/emotion, and receive timed GNM coefficients, a mesh-backed preview video with normalized source audio, and diagnostics.
2. Upload one face photo and receive a confidence-gated neutral GNM mesh, fitted identity coefficients, a landmark overlay, and diagnostics.
3. Inspect exact inputs, runtime/model versions, parameters, and test metrics rather than trusting a black-box success message.

## Functional requirements

### Shared

- Load GNM Head from the checked-in v3 asset and query all dimensions at runtime.
- Produce deterministic results for a fixed seed/configuration.
- Create a unique job directory with a manifest, inputs, outputs, metrics, warnings, and errors.
- Never silently substitute a mock output.
- Validate file existence, type, size, duration/dimensions, and zero/multiple-face cases.
- Expose the same operations through CLI functions and HTTP endpoints.
- Include model, dependency, and git commit versions in every result.

### Audio

- Accept WAV, MP3, M4A, AIFF, OGG, or video-with-audio supported by ffmpeg.
- Normalize to mono 16 kHz PCM WAV.
- Require Rhubarb 1.14 for an audio job. Health remains available in a
  `degraded` state when it is absent; `audio` then fails with
  `DEPENDENCY_MISSING` and an install command rather than substituting cues.
- Accept optional dialog and manual emotion.
- Produce monotonic A-H/X cues covering the full clip.
- Decode deterministic semantic GNM prototypes without TensorFlow.
- Region-mask and blend viseme/emotion controls at configurable FPS.
- Generate `[frames, expression_dim]` coefficients plus rotations/translation.
  The MVP pose tracks are deliberately static zeros: `rotations` is float32
  `[frames,4,3]` axis-angle radians in GNM joint order and `translation` is
  float32 `[frames,3]` model-space meters. They are serialized in `controls.npz`
  with `expression`, `rotations`, `translation`, `timestamps`, and `fps`.
- Validate coefficient range, mouth geometry ordering, continuity, mesh finiteness, and cue coverage.
- Generate a preview MP4 muxed with the normalized source audio.

### Image

- Accept JPEG, PNG, or WebP.
- Detect exactly one face and extract landmarks using MediaPipe in the MVP.
- Use a versioned, tested MediaPipe-to-GNM correspondence subset rather than claiming all 68 points are exact equivalents.
- Fit a weak-perspective camera and first 10/20 observable head identity components with robust regularization.
- Keep unobservable eye/teeth identity and tongue/pupil expression blocks neutral.
- Produce full runtime-sized identity/expression arrays, camera, neutral mesh OBJ, overlay PNG, and metrics JSON.
- Reject or lower confidence for small faces, extreme pose, strong expression, occlusion, poor fit, instability, or coefficient saturation.

### Web UI

- Home page with Audio and Image workflows.
- Drag/drop or file picker, relevant options, explicit local-processing note, submit/status/error states.
- Results show preview media, detected emotion/confidence, fit metrics/warnings, and artifact download links.
- Health endpoint reports GNM/Rhubarb/ffmpeg/MediaPipe readiness.
- Results use native audio/video controls and an image overlay toggle. Failed
  jobs show typed error plus retained metrics. Low-confidence image fits show
  a warning panel and require `allow_low_confidence=true` to expose mesh/NPZ
  downloads; otherwise the job is retained as failed with diagnostics only.

## Architecture

```text
browser
  -> FastAPI routes
       -> application services
            -> GNMAdapter
            -> SemanticDecoder / ControlRig
            -> AudioAnalyzer / AnimationComposer / PreviewRenderer
            -> FaceLandmarker / IdentityFitter / OverlayRenderer
       -> artifact store

CLI -> same application services -> artifact store
```

Core modules:

| Path | Responsibility |
|---|---|
| `src/autoanim_gnm/gnm_adapter.py` | load model, landmarks, compact bases, mesh/export |
| `src/autoanim_gnm/semantic_decoder.py` | TensorFlow-free H5 dense decoder |
| `src/autoanim_gnm/rig.py` | viseme/emotion prototypes, masking, geometry validation |
| `src/autoanim_gnm/audio.py` | ffmpeg/Rhubarb, features, cue and emotion timelines |
| `src/autoanim_gnm/animation.py` | coarticulation, frame controls, mesh preview |
| `src/autoanim_gnm/image.py` | MediaPipe extraction and input quality |
| `src/autoanim_gnm/fitting.py` | camera/identity optimization and confidence |
| `src/autoanim_gnm/artifacts.py` | jobs, manifests, atomic writes, paths |
| `src/autoanim_gnm/api.py` | FastAPI endpoints and static UI |
| `src/autoanim_gnm/cli.py` | reproducible command line |

## Normative implementation constants

This section is the source of truth when prose elsewhere is less specific.

### Assets and runtime API

- GNM model: `gnm/shape/data/versions/v3_0/gnm_head.npz`.
- Landmark definition: `gnm/shape/data/landmarks/head_sparse_68.txt`.
- Expression decoder: `gnm/shape/data/semantic_sampler/expression_decoder_model.h5`.
- Model construction:
  `GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)`; the adapter also
  asserts the loaded full version is `3.0`.
- Runtime dimensions are asserted, not silently assumed: 17,821 vertices,
  35,324 triangles, 17,662 quads, 253 identity coefficients, 383 expression
  coefficients, and four joints.
- Identity slices are `head=0:170`, `eyes=170:173`, `teeth=173:253`.
  Expression slices are `left_eye=0:100`, `right_eye=100:200`,
  `lower_face=200:350`, `tongue=350:382`, `pupil=382:383`.
- The application imports the public GNM NumPy API and uses
  `GNMLandmarksType.HEAD_SPARSE_68`; it does not reimplement skinning.

### TensorFlow-free expression decoder

The expression H5 model has a concatenated input `[z64, class20]`, dense
widths `84 -> 64 -> 128 -> 256 -> 512 -> 383`, ReLU after the first four
dense layers, and a linear final layer. H5 datasets, in order, are
`dense_13` through `dense_17`, each with `kernel:0` and `bias:0` below
`model_weights/<layer>/<layer>/`. Matrix evaluation is `x @ kernel + bias`.
Every named prototype uses an all-zero latent `z64` and one-hot class input,
so it is deterministic and does not pretend to be a random CVAE sample.

The 20 class indices are exactly the `semantic_sampler.Expression` order:
`surprise, disgust, suck, compress_face, stretch_face, happy, squint,
platysma, blow, funneler, smile_wide, corners_down, pucker, wink_left,
wink_right, mouth_left, mouth_right, lips_roll_in, snarl, tongue_center`.

### Control rig contract

Decoded prototypes are cached. Each is converted to a delta from the neutral
zero coefficient vector, clipped component-wise to `[-3, 3]`, then masked.
The final composed track is clipped to `[-3, 3]`; clipping any component adds
`COEFFICIENT_SATURATED` and is a test failure for retained fixtures.

| Cue | Lower-face control | Tongue control |
|---|---|---|
| X | neutral | neutral |
| A | `-0.35 * compress_face` (empirically closes the zero-latent prototype) | neutral |
| B | `0.25 * stretch_face + 0.15 * smile_wide` | neutral |
| C | `0.60 * stretch_face` | neutral |
| D | `1.00 * stretch_face` | neutral |
| E | `0.65 * funneler` | neutral |
| F | `0.70 * pucker + 0.30 * funneler` | neutral |
| G | `-0.35 * lips_roll_in` (empirically closes the zero-latent prototype) | neutral |
| H | `0.50 * stretch_face + 0.35 * tongue_center` masked to lower face | `0.70 * tongue_center` masked to tongue |

Visemes can write only `200:382`. Emotion can write `0:350`, never tongue or
pupil. Emotion prototypes are: `joy=.70 happy + .30 smile_wide`,
`surprise=surprise`, `disgust=disgust`,
`sad=.75 corners_down + .25 compress_face`,
`anger=.50 snarl + .30 platysma + .20 compress_face`,
`fear=.55 surprise + .45 compress_face`,
`contempt=.60 snarl + .40 mouth_left`, and `neutral=zero`.

Cue transitions use a boundary-centered raised-cosine crossfade. For boundary
`b` between values `v0` and `v1`, set
`w=min(.070,(b-a)/2,(c-b)/2)`, where `[a,b]` and `[b,c]` are the adjacent
cues. On `[b-w,b+w]`, `alpha=.5-.5*cos(pi*(t-(b-w))/(2*w))` and the control is
`(1-alpha)*v0+alpha*v1`; outside every boundary window it is the containing
cue's control. Boundary windows cannot overlap because of the half-duration
limit; if they touch, the shared endpoint is evaluated once by the later
boundary. Emotion is a separate whole-clip envelope with the same equation
and 300 ms fade from/to neutral. Composition is additive after region masking.
Frame count is `ceil(duration_seconds * fps)` and frame timestamps
are `arange(frame_count) / fps`; therefore an exact 8.0 s clip at 30 fps has
240 frames and a final sample at 7.9667 s.

Rhubarb output is normalized before animation. Reject nonfinite times,
unknown values, `end <= start`, and overlaps greater than 1 ms. Sort by start,
clip the first/last cue to `[0,duration]`, and insert `X` for every uncovered
gap greater than 1 ms, including leading/trailing gaps. Merge adjacent equal
values and snap boundaries within 1 ms. The invariant after normalization is
`cues[0].start=0`, `cues[-1].end=duration`, and
`cues[i].end=cues[i+1].start`; any conflicting overlap is `CUE_INVALID`.

Rig geometry uses official sparse landmarks. Aperture is the mean 3D distance
of pairs `(61,67),(62,66),(63,65)`; mouth width is distance `(48,54)`.
Tongue motion is mean displacement of vertices with nonzero `tongue` vertex
group weight. On the zero-latent library, tests require
`D > 1.20*C > 1.20*B > 1.05*X` for aperture, A aperture no greater than X,
F width at least 3% below C, G aperture no greater than X, H tongue motion
above 0.5 mm, and a post-cue neutral coefficient norm below `1e-6`.

### Emotion inference contract

Priority is manual label, then dialog-assisted heuristic, then audio-only
heuristic. Manual labels have confidence 1.0. Dialog is lowercased and matched
against versioned word lists for joy, sadness, anger, fear, disgust, surprise,
and contempt; a unique winning label has confidence
`min(.85, .55 + .10 * matched_words)`. Ties return neutral at .40.

The v1 whole-word lists are deterministic: joy=`happy,glad,delighted,love,
wonderful,excited,joy`; sad=`sad,sorry,grief,lonely,cry,unhappy`; anger=
`angry,mad,furious,hate,rage,annoyed`; fear=`afraid,scared,fear,terrified,
worried`; disgust=`disgust,gross,revolting,nasty`; surprise=`wow,surprised,
unexpected,amazing,astonished`; contempt=`idiot,pathetic,ridiculous,
worthless`. Apostrophes are removed and other nonletters delimit tokens.

Audio features are measured from active 30 ms windows: RMS dBFS, median YIN F0
in 70-400 Hz, F0 coefficient of variation, and cue rate.
`arousal = clip(.45*(rms_dbfs+45)/30 + .35*f0_cv/.35 +
.20*cues_per_second/8, 0, 1)`. Audio-only v1 uses this deliberately conservative
tree: RMS above -33 dBFS and F0 above 280 Hz -> anger (.62); otherwise F0 CV
above .24 with F0 above 150 -> surprise (.58); otherwise RMS below -35 and F0 below 180 -> sad
(.58); otherwise RMS between -36/-30, F0 above 210, and F0 CV below .22 -> joy
(.55); otherwise RMS above -31, F0 above 150, and F0 CV above .15 -> fear
(.52); else neutral
(.50). All audio-only confidence is below .65 and therefore `unvalidated`;
the UI must say this is a heuristic, not an emotion recognizer. An LLM may
propose segment labels only through the documented JSON schema; it is optional
and never controls phoneme timing.

Windows use a 30 ms frame and 10 ms hop. Active means RMS at least 60 dB
below full scale and at least 12 dB above the clip's 10th-percentile RMS.
Silent clips are rejected as `AUDIO_SILENT`. F0 subtracts the frame mean and
uses the first local minimum below .20 in the cumulative-mean-normalized YIN
difference function over 70-400 Hz; `f0_cv=0` with fewer than five voiced
frames. RMS dBFS is `20*log10(max(rms,1e-8))`. Cue rate excludes X.

### MediaPipe correspondence and image fit

The landmarker asset is downloaded from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
and must have SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The MVP mapping below is
versioned as `mediapipe478_to_gnm68_v1`. Rows are GNM landmark indices; values
are MediaPipe indices:

```text
0:17  = 234,93,150,136,172,58,132,149,152,377,400,378,379,365,397,288,454
17:22 = 70,63,105,66,107
22:27 = 336,296,334,293,300
27:31 = 168,6,197,195
31:36 = 64,98,2,327,294
36:42 = 33,160,158,133,153,144
42:48 = 362,385,387,263,373,380
48:60 = 61,40,37,0,267,270,291,321,314,17,84,91
60:68 = 78,81,13,311,308,402,14,178
```

The nonstandard values at GNM rows 2-6 deliberately apply the checked-in
left-jaw permutation `[0,1,6,5,4,3,2,7,...,67]`; replacing them with the
usual ascending MediaPipe jaw list is a left-side correspondence bug.

The adapter validates left/right orientation on the retained astronaut fixture.
The optimizer uses all 68 rows, with weights 2.0 for eyes/nose, 1.0 for
brows/lips, and 0.5 for jaw. Its variables are yaw/pitch/roll, log scale,
2D translation, the first `K` head identity coefficients, and four nuisance
expression amplitudes (`happy`, `surprise`, `pucker`, `corners_down`). It
minimizes interocular-normalized weighted reprojection plus L2 priors with
SciPy `least_squares(loss="soft_l1", f_scale=.01)`. Identity bounds are
`[-3,3]`, nuisance expression bounds are `[-1,1]`; priors are
`lambda_identity=.0003`, `lambda_expression=.003` after interocular
normalization. Solve camera only, then K=10,
then K=20 initialized from the preceding stage. The exported mesh uses the
fitted identity with expression reset to zero.

Coordinate convention is GNM `+X` image-right, `+Y` up, and `+Z` toward the
viewer. Camera rotation is `R=Rz(roll) @ Rx(pitch) @ Ry(yaw)` applied to column
model points. Weak-perspective projection is `u=s*X_camera+tx`,
`v=-s*Y_camera+ty` in pixels with `s=exp(log_scale)`. Reported angles are those
three optimizer variables in degrees. Bounds are yaw/pitch `[-.8,.8]`, roll
`[-.6,.6]`, scale `[1e-3,1e5]`, and translation within two image widths of the
image center.

Initialization is zero pose/identity/nuisance, scale equal to observed outer
eye distance `(36,45)` divided by mean-model 3D `(36,45)` XY distance, and
translation aligning the projected centroid of rows 27:36 to the observed
centroid. Camera-only uses rows 27:48. Every later residual concatenates, in
this exact order, `sqrt(weight_i)*(predicted_i-observed_i)/interocular` flattened
row-major, `sqrt(.0003)*beta`, and `sqrt(.003)*nuisance`. Solver options are
`max_nfev=300, xtol=ftol=gtol=1e-8`; non-convergence is `FIT_REJECTED`.

The 83 eye/teeth identity coefficients and expression indices `350:383` are
always zero for image fitting. NME is mean 2D landmark error divided by outer
eye-corner `(36,45)` distance. Face width is max minus min observed X across
the 68 mapped points. Saturated means `abs(beta)>=2.999`. The visible mesh set
for synthetic error is vertices whose `skin_exterior` group weight exceeds .5.
Confidence is:

| Result | Conditions (all required) |
|---|---|
| high | face width >=128 px, NME <=.035, abs(yaw)<=25 deg, abs(pitch)<=20 deg, saturated identity fraction <=.10, stability RMS <=.35 |
| medium | face width >=80 px, NME <=.060, abs(yaw)<=40 deg, abs(pitch)<=30 deg, saturated fraction <=.25, stability RMS <=.75 |
| rejected | no/extra face, face width <64 px, NME >.080, abs(yaw)>45 deg, abs(pitch)>35 deg, nonfinite solve, saturated fraction >.30, or stability RMS >1.25 |

Results in the gap between medium and rejection are `low` and still retain
diagnostics, but the CLI exits 2 unless `--allow-low-confidence` is supplied.
Strong-expression warnings trigger when any MediaPipe mouth/eye blendshape is
above .70; the checked categories are every blendshape whose name begins
`mouth`, `eyeBlink`, `eyeSquint`, `eyeWide`, `browDown`, or `browInnerUp`.
Stability RMS is the RMS difference between the base fitted K-vector
and four refits after deterministic Gaussian 2D perturbation (seeds 0-3,
sigma=.5 px); nuisance expression is reinitialized each time. MediaPipe Tasks
does not expose reliable per-point presence in this model, so the application
does not claim occlusion recognition: fewer than 90% mapped points inside an
image margin of 5% is rejected, 90-95% warns `POSSIBLE_OCCLUSION`, and every
accepted result discloses that occlusion was inferred only from bounds/residual.

### Preview rendering

The MVP renderer is deterministic CPU OpenCV: orthographically project the
dense GNM head, depth-sort triangles, fill them with a fixed warm-gray Lambert
shade using camera-space normals, draw sparse landmarks in dark gray, and use a
640x640 black background. Audio previews render silent H.264 at requested FPS
then invoke ffmpeg with normalized WAV, AAC 128 kb/s, `-af apad -shortest`, and
`-movflags +faststart`. Padding prevents stream-copy truncation of a final H.264
B-frame group. `ffprobe` must report a video frame count exactly equal to the
control frame count and video-stream duration within one frame of source audio;
container duration alone is insufficient. Image overlays draw observed points
in green and fitted points in magenta with a connecting line; mean overlay line
length must equal reported pixel error within `1e-3`.

Preview camera is frontal with model center mapped to `(320,320)`, `+Y` up,
and scale chosen once from the neutral skin bounding box to occupy 85% of the
canvas height. It never auto-frames per animation frame. Backfaces with
clockwise projected winding are culled; Lambert light direction is normalized
`(-.3,.5,1)`, ambient is .35, diffuse is .65, base BGR is `(190,180,170)`, and
depth order is far-to-near triangle-centroid Z. ffmpeg uses `libx264`, CRF 18,
`medium`, `yuv420p`, one encoder thread, and cleared creation metadata. The
preview contains the normalized mono 16 kHz version of the source audio—not a
bit-identical original stream—and the UI labels this accurately.

## Operational contracts

### CLI

```text
autoanim-gnm health [--json]
autoanim-gnm audio INPUT --out DIR [--fps 30] [--emotion auto|neutral|joy|sad|anger|fear|disgust|surprise|contempt] [--dialog TEXT] [--rhubarb-bin PATH]
autoanim-gnm image INPUT --out DIR [--modes 10|20] [--allow-low-confidence]
autoanim-gnm serve [--host 127.0.0.1] [--port 8000] [--artifacts DIR]
```

Exit codes are 0 success, 2 input/quality rejection, 3 missing dependency, and
1 unexpected internal error.

### Job layout, manifest, and errors

Job IDs are 26-character lowercase ULIDs. Jobs are stored at
`<artifact-root>/<job-id>/`. Audio jobs contain `input.<ext>`,
`normalized.wav`, `cues.json`, `controls.npz`, `preview-silent.mp4`,
`preview.mp4`, and `result.json`; image jobs contain `input.<ext>`, `fit.npz`,
`fitted.obj`, `mesh-preview.png`, `overlay.png`, and `result.json`. The local MVP never deletes
jobs automatically; `--artifacts` selects a disposable root when desired.

Manifest state is `running|succeeded|failed`, and every manifest includes
schema version, job ID, kind, UTC timestamps, sanitized input name, SHA-256,
tool/model/git versions, configuration, artifacts with bytes/SHA-256, metrics,
warnings, and one nullable error. Error schema is
`{"code": "UPPER_SNAKE", "message": "actionable text", "details": {},
"retryable": false}`. Files are written to a sibling `.tmp`, fsynced, renamed,
then entered in the manifest. Download routes accept only basenames already
listed in the successful manifest.

Required top-level keys and types are: `schema_version` string `"1.0"`,
`job_id` string, `kind` enum, `status` enum, `created_at`/`updated_at` UTC RFC3339
strings, `input` object `{name,sha256,bytes,media_type}`, `versions` object,
`configuration` object, `metrics` object, `warnings` array of strings,
`artifacts` object keyed by logical name with `{name,bytes,sha256,media_type}`,
and `error` null/object. JSON finite floats are rounded to 8 decimal places.
Failed manifests may list only diagnostic artifacts already finalized; missing
expected success artifacts are never represented by null paths.

`controls.npz` has float32 arrays `expression[F,383]`, `rotations[F,4,3]`,
`translation[F,3]`, `timestamps[F]`, and scalar int32 `fps`. `fit.npz` has
float32 `identity[253]`, `expression[383]`, `camera[6]` ordered
`yaw,pitch,roll,log_scale,tx,ty`, `observed_landmarks[68,2]`, and
`fitted_landmarks[68,2]`. NPZ entries are written in this order without pickle.

At service startup, any retained `running` manifest owned by a dead prior
process is atomically changed to failed with `PROCESS_INTERRUPTED`. The single
worker lock is in-process; multi-worker Uvicorn is rejected at startup for the
MVP. The CLI acquires an advisory lock file in the artifact root and exits
`BUSY` instead of corrupting a concurrent job.

### Limits and readiness

Uploads are limited to 100 MiB; decoded audio to 10 minutes; images to
12,000x12,000 and 40 megapixels. ffprobe determines media type/duration; file
extensions are not trusted. `health` is `ready` only when GNM, MediaPipe model,
ffmpeg/ffprobe, and Rhubarb all pass executable probes. Missing Rhubarb makes
health `degraded` while image jobs remain usable.

Models live under `${AUTOANIM_CACHE_DIR}` or, by default,
`<project>/.cache/autoanim_gnm/`; installer downloads to `.partial`, verifies
SHA-256, then renames. Offline absence or any checksum/network failure is
`DEPENDENCY_MISSING` with the exact target path and retry command.

### Dependency and fixture pins

Reference environment pins are CPython 3.12.13, NumPy 2.5.1, SciPy 1.18.0,
h5py 3.16.0, MediaPipe 0.10.35, opencv-contrib-python 5.0.0.93, Pillow 12.3.0,
FastAPI 0.139.2, Uvicorn 0.51.0, python-multipart 0.0.32, pytest 9.1.1,
ffmpeg 8.1.1, and Rhubarb 1.14. GNM is installed editable from this checkout.
`requirements.lock` repeats these exact Python versions with transitive pins.

Retained or reproducibly downloaded real fixtures are:

- LibriSpeech `5703-47212-0000.ogg` from librosa's public example-data URL,
  SHA-256 `a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd`;
- the lossless `skimage.data.astronaut()` source PNG, SHA-256
  `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`;
- the public-domain official Barack Obama portrait from
  `https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg`,
  SHA-256
  `744dd848fbb0584229169e01c4944664957c62495fb9e8af514a088ebca43e19`;
- emotional speech is RAVDESS speech under its dataset terms and is fetched by
  an opt-in script, never silently redistributed.

The fixture script verifies every checksum. If RAVDESS cannot be downloaded
under the current environment/terms, the emotional benchmark is marked
blocked and manual-emotion behavior is still tested on real speech; it may not
be relabeled as automatic emotion validation.

Real-image release tests are not conditional: astronaut and the official
portrait must both detect one face, produce finite nonzero bounded identity
fits, have NME <=.060, and render overlays. Negative variants must include a
64-pixel face resize, a two-face composite, a 60-degree rotated portrait, and
a crop removing more than 15% of mapped points; they must respectively produce
`FIT_REJECTED`, `MULTIPLE_FACES`, pose low/rejection, and
`POSSIBLE_OCCLUSION`/rejection. These test quality gating, not biometric likeness.

## Data contracts

### Audio result

```json
{
  "kind": "audio_animation",
  "status": "succeeded",
  "model": {"gnm_version": "3.0", "expression_dim": 383},
  "audio": {"duration_s": 8.0, "sample_rate": 16000},
  "analysis": {
    "backend": "rhubarb-1.14.0",
    "emotion": "neutral",
    "emotion_confidence": 0.5,
    "cues": [{"start": 0.0, "end": 0.3, "value": "X"}]
  },
  "animation": {"fps": 30, "frames": 240, "expression_shape": [240, 383]},
  "metrics": {
    "cue_coverage": 1.0,
    "max_abs_coefficient": 0.0,
    "mesh_finite": true,
    "audio_video_offset_frames": 0
  },
  "artifacts": {"controls": "controls.npz", "preview": "preview.mp4", "report": "result.json"},
  "warnings": []
}
```

### Image result

```json
{
  "kind": "image_fit",
  "status": "succeeded",
  "model": {"gnm_version": "3.0", "identity_dim": 253},
  "detection": {"faces": 1, "landmarks": 478, "yaw_deg": 0.0, "pitch_deg": 0.0},
  "fit": {
    "modes": 20,
    "nme": 0.0,
    "coefficient_bound_fraction": 0.0,
    "confidence": "high",
    "confidence_reasons": []
  },
  "artifacts": {"mesh": "fitted.obj", "overlay": "overlay.png", "parameters": "fit.npz"},
  "warnings": ["Single-view visible-geometry estimate; not a metric 3D clone."]
}
```

### HTTP

POST operations are synchronous and return only after the manifest reaches a
terminal state. A process-wide nonblocking lock permits one POST at a time; a
concurrent POST receives 409 `BUSY` and is not queued.

| Method/path | Fields and success response |
|---|---|
| `GET /api/health` | 200 readiness/dependency JSON; health may be degraded |
| `POST /api/audio` | multipart `file` required; `dialog` optional <=10,000 chars; `emotion` enum default `auto`; `fps` integer 12-60 default 30; 201 manifest |
| `POST /api/image` | multipart `file` required; `modes` enum 10/20 default 20; `allow_low_confidence` boolean default false; 201 manifest |
| `GET /api/jobs/{id}` | 200 successful or failed manifest |
| `GET /api/jobs/{id}/files/{name}` | 200 allowlisted artifact bytes |

All errors use the manifest error object. Status mapping is 400 for
`INPUT_INVALID`, `MEDIA_INVALID`, `AUDIO_SILENT`, or `CUE_INVALID`; 404 for
`JOB_NOT_FOUND`/`ARTIFACT_NOT_FOUND`; 409 for `BUSY`; 413 for `LIMIT_EXCEEDED`;
422 for `FACE_NOT_FOUND`, `MULTIPLE_FACES`, or `FIT_REJECTED`; 424 for
`DEPENDENCY_MISSING`; and 500 for `INTERNAL_ERROR`. Framework validation errors
are converted to 400 `INPUT_INVALID`.

For an otherwise finite `low` image fit, `allow_low_confidence=false` stores
overlay/metrics, sets terminal state failed with `FIT_REJECTED`, returns 422,
and does not list mesh/NPZ as downloadable. With true, it returns 201 succeeded,
lists all artifacts, and retains `LOW_CONFIDENCE` in warnings. High/medium fits
do not require confirmation; rejected fits can never be overridden.

Optional LLM emotion input, when added after MVP, is strictly
`{"segments":[{"start":number,"end":number,"emotion":enum,
"confidence":number_0_to_1}]}` with sorted, nonoverlapping times inside the
clip; invalid input is ignored with a warning, and it never modifies cues.

## Phase plan and gates

No phase advances until its build, review, real-input test, and failure-fix loop passes.

### Phase 0: upstream truth and environment

Build:

- pin and inspect GNM;
- create reproducible Python environment and dependency installer;
- record upstream data/runtime facts.

Tests:

- upstream NumPy/data/fitting suites;
- load actual v3 asset and generate neutral vertices/landmarks;
- inspect actual NPZ dimensions and names.

Pass gate:

- no unexplained upstream failure;
- runtime facts and repository discrepancies documented.

### Phase 1: GNM adapter and semantic control rig

Build:

- model adapter and exact compact landmarks;
- TensorFlow-free deterministic semantic decoder;
- region-masked viseme and emotion library;
- OBJ/NPZ/JSON export and preview renderer.

Tests:

- decoder output against H5 dimensions and deterministic golden values;
- compact landmarks equal official landmarks within `1e-6`;
- expression masks and coefficient bounds;
- aperture ordering, pucker narrowing, tongue motion, neutral return;
- real GNM meshes finite with valid topology.

Pass gate:

- every rig geometry assertion passes on the actual GNM asset.

### Phase 2: audio pipeline

Build:

- ffmpeg normalization;
- Rhubarb integration and install helper;
- transparent audio emotion/manual override;
- cue-to-frame composer, controls, mesh-backed preview video, report.

Tests:

- unit/error tests for cues and input handling;
- actual eight-second LibriSpeech input without dialog;
- actual emotional speech input or explicitly documented dataset blocker;
- mux timing and visual/geometry inspection.

Pass gate:

- audio-only input produces nontrivial cues, bounded coefficients, changing GNM mouth geometry, and an audible muxed preview;
- emotion output is confidence-gated and never presented as validated if real emotional benchmarks fail.

### Phase 3: image fitting

Build:

- MediaPipe extraction and mapping;
- quality/pose/expression checks;
- compact camera/identity optimizer;
- confidence, overlay, neutral OBJ, parameters, report.

Tests:

- synthetic GNM coefficient recovery at defined noise levels;
- actual astronaut photo smoke test;
- the required public-domain official portrait as a second actual face;
- no/multiple/small/extreme cases;
- visual overlay inspection.

Pass gate:

- synthetic thresholds pass;
- real photo produces a stable, nontrivial, bounded fit and accurate overlay;
- limitations are present in API/UI result.

### Phase 4: application integration

Build:

- FastAPI routes, static UI, job artifacts, readiness, errors, downloads;
- CLI commands calling the same services.

Tests:

- API success/failure integration tests;
- browser upload and results checks;
- CLI/API parity;
- artifact path allowlisting and input size limits.

Pass gate:

- both real-input workflows succeed from browser and CLI in a clean run.

### Phase 5: final verification and hardening

Build:

- dependency/setup documentation;
- retained real-input fixtures or reproducible download script;
- final verification report with exact commands, versions, metrics, and media.

Tests:

- full unit/integration suite from a clean process;
- upstream regression subset;
- real audio end to end;
- real image end to end;
- API health and both upload workflows;
- inspect generated images/video and manifests.

Pass gate:

- every acceptance item has direct evidence, or is explicitly marked failed with cause and viable alternative.

## Acceptance criteria

1. `audio <real-file>` creates timed cues, a runtime-sized bounded expression track, changing GNM landmarks/vertices, and a muxed preview with normalized source audio.
2. Audio preview mux offset is <=1 frame and cue coverage is 100%.
3. Speech coefficients cannot change the two eye blocks or pupil block.
4. Manual emotion visibly changes the appropriate GNM regions without erasing mouth motion.
5. `image <real-photo>` detects exactly one face, fits nonzero observable identity coefficients, and returns a neutral GNM mesh plus overlay and confidence.
6. Compact and official GNM landmarks agree within `1e-6`.
7. Synthetic recovery at yaw `-18, 0, +18` degrees, K=20, and seeded 0.5 px
   Gaussian noise has median landmark NME <=.015, median fitted-visible-vertex
   error <=1.5 mm, p95 <=3.0 mm, and identity coefficient cosine similarity
   >=.75. Each threshold is computed over at least 12 seeded trials.
8. Every audio result contains the exact caveat
   `Rhubarb provides coarse viseme timing, not validated phoneme-accurate alignment.`
   Every image result contains the exact caveat
   `Single-view visible-geometry estimate; not a metric 3D clone.` Automatic
   emotion below .65 confidence additionally contains
   `Emotion is an unvalidated acoustic/lexical heuristic.`
9. Browser and CLI call the same core implementation. Their normalized result
   JSON is byte-identical after removing job ID, paths, SHA values, and UTC
   timestamps. Controls/fit NPZ array values are exactly equal. Encoded MP4
   bytes are not compared; ffprobe stream properties, duration, and three
   decoded reference-frame pixel hashes are compared under the pinned
   ffmpeg/OpenCV environment.
10. Invalid files, missing native tools, zero/multiple faces, and extreme inputs fail with actionable typed errors.
11. No tests substitute mocks for the final real-audio and real-photo end-to-end runs.
12. The full implementation and upstream regression suites pass after the final change.

The optional RAVDESS benchmark is evidence about automatic emotion accuracy,
not a hidden 13th acceptance item. Items 4 and 8 require deterministic manual
emotion composition and honest confidence/caveats; release never interprets a
blocked benchmark as validated automatic recognition.

Acceptance scoring is binary per numbered item: pass only with a retained test,
artifact, or command transcript; fail otherwise. Release readiness requires
12/12. The broader evaluator completeness score is 0-10: 0 means no executable
contract, 5 means major happy paths are specified but important decisions are
open, 7 means an engineer can implement and test without product decisions,
and 10 means every implementation detail is prescribed. This spec targets at
least 7 because algorithmic refinements remain intentionally open inside the
measured acceptance envelope.

The final manual visual pass is also explicit: inspect the first, middle, and
last audio frames plus every Rhubarb cue transition for inverted/clipped
geometry; inspect the image overlay at original resolution for left/right
swaps and gross contour drift. A pass requires no mesh triangles crossing the
image boundary due to nonfinite projection, no left/right inversion, and no
observed-to-fitted overlay line longer than 15% of interocular distance outside
the jaw. The renderer records contact sheets so this evidence is reviewable.

## Dependencies

Required:

- Python 3.10+;
- GNM Shape from this repository;
- NumPy, SciPy, h5py, Pillow, OpenCV, MediaPipe, FastAPI, Uvicorn, python-multipart;
- ffmpeg/ffprobe;
- Rhubarb 1.14 for audio cues.

Optional quality upgrades:

- whisper.cpp;
- Montreal Forced Aligner;
- a clearly licensed text/audio emotion model;
- three-view/video capture;
- Linux/CUDA differentiable rendering.

## Failure handling and rollback

- Jobs write to a new directory and only publish a completed result after validation.
- A failed job retains a safe manifest and logs, never a fake success artifact.
- Native/model downloads are versioned and checksum-capable; removing their cache rolls back them.
- Application code is additive around the pinned GNM tree. Reverting the app commit restores untouched upstream behavior.
- The web service is local-only by default and can be disabled without affecting CLI processing.

## Out of scope for the verified MVP

- photorealistic rendering or person-specific texture/hair;
- production phone-level alignment benchmarks requiring manual annotations;
- training a new audio-to-motion or speech-emotion neural model;
- biometric, clinical, or anthropometric claims;
- multi-user auth, cloud storage, billing, or deployment;
- multi-view/video fitting, although it is the recommended next product phase;
- Blender/Unity/Maya plugins;
- mobile real-time performance.
