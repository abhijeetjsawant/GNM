# Interactive 3D viewer research and implementation design

Research date: 2026-07-18. This document is a repository-grounded design for the
existing FastAPI application; it does not describe an unrelated rewrite.

## Decision

Keep the product as a local-first FastAPI web application and add a self-hosted
Three.js WebGL 2 viewer. Ship job geometry and animation as glTF 2.0 binary
(`.glb`), retain the current PNG/MP4 as the poster and universal fallback, and
make the browser's audio or video element the one playback clock.

Use these pinned frontend dependencies for the first implementation:

- the audited `three@0.183.2` runtime, including `GLTFLoader` and
  `OrbitControls`; current-pose export remains server-side;
- Vite as a development/build dependency, producing versioned static assets
  that FastAPI serves from the Python package;
- no CDN, remote environment map, analytics script, or JavaScript runtime on
  the end user's machine.

The npm registry reported Three.js `0.185.1` on 2026-07-18. The application
intentionally stays on its audited, checksum-pinned `0.183.2` bundle until a
separate compatibility pass validates an upgrade. Pin the exact package and
archive; upgrades are deliberate compatibility work, not a floating
dependency. The Three.js installation guide recommends npm plus a
build tool for applications with dependencies, while still producing ordinary
static files for production ([official installation guide](https://threejs.org/manual/en/installation.html)).

This form factor is the smallest dependable fit because the application already
owns uploads, job manifests, media previews, and an HTTP artifact boundary.
WebGL provides hardware-accelerated interaction on the same desktop and mobile
browsers without adding Electron, a native windowing layer, or a second API.
Three.js now requires WebGL 2; unsupported or lost GPU contexts fall back to the
existing preview rather than failing the job ([WebGLRenderer documentation](https://threejs.org/docs/pages/WebGLRenderer.html),
[MDN WebGL overview](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API)).

## What exists today

The original application had only CPU-rendered PNG/MP4 previews and a geometry-
only OBJ. The implemented application now has a real interactive asset path:

- seam-correct GLB export duplicates triangle-corner UV seams while retaining a
  source-vertex map and all six GNM anatomical components;
- static identity, multiview textured identity, learned-audio animation, and
  video-performance jobs all expose the same allowlisted viewer endpoint;
- animated GLBs use measured low-rank morph reconstruction with exact media
  time as the clock and a static fallback when reconstruction gates cannot pass;
- the paused `AnimationAction.time` is sampled directly, because
  `AnimationMixer.setTime()` cannot seek a completed clamped `LoopOnce` action
  backward correctly;
- the viewer provides orbit/zoom, bounded camera distance and polar angles,
  reset, exposure, surface, surface-plus-topology, and topology-only modes;
- source audio or browser-compatible source video is embedded beside the face
  and drives exact 3D time;
- the official Three.js `0.183.2` npm archive is SHA-256 pinned at bootstrap;
  `three.module.js`, its transitive `three.core.js`, `GLTFLoader`,
  `OrbitControls`, two utility imports, and the MIT license are served locally
  from a versioned allowlist;
- the viewer response has a restrictive CSP, no runtime CDN, live status,
  keyboard-focusable canvas, a visible WebGL failure message, and GPU/material/
  geometry disposal on page exit. `connect-src blob:` is deliberately narrow but
  required because Three 0.183.2's `ImageBitmapLoader` fetches embedded GLB image
  buffer views through object URLs; allowing blob images without blob fetches
  makes valid textured GLBs silently fall back to an untextured material;
- the application home page retains privacy-minimized recent-job summaries and
  direct links back to each viewable result.

The underlying GNM facts still govern the exporter: 17,821 source vertices,
35,324 triangles, six components, and triangle-corner UVs shaped
`[35324,3,2]`. `GNM.vertex_uvs` remains unsuitable across seams; the exporter
therefore uses `triangle_uvs` and deliberate seam duplication.

## Renderer comparison

| Choice | Strengths for this app | Blocking cost | License | Decision |
|---|---|---|---|---|
| Three.js | Direct access to meshes, morph weights, skeletons, points/lines, wireframe materials, cameras, render loop, and GPU lifecycle. `GLTFLoader`, `OrbitControls`, `AnimationMixer`, and `GLTFExporter` are official addons. | We must build the inspector, accessible DOM controls, timeline, and error recovery. | MIT ([official license](https://github.com/mrdoob/three.js/blob/dev/LICENSE)) | **Choose.** It is the least engine needed for custom GNM diagnostics. |
| Babylon.js | Full scene engine with glTF loading, morph targets, skins, animation, validation hooks, and a strong inspector/exporter. Its official loader defaults to loading morphs, node animation, and skins. | Larger and more opinionated engine surface than this single-purpose FastAPI UI; much of its scene/editor framework would be unused. | Apache-2.0 ([official repository](https://github.com/BabylonJS/Babylon.js), [loader options](https://doc.babylonjs.com/typedoc/classes/BABYLON.GLTFLoaderOptions)) | Good alternative if the product becomes a general 3D editor, not the minimal choice now. |
| `<model-viewer>` | Excellent declarative static/animated GLB display, camera controls, mobile gestures, posters, progress, AR, and useful screen-reader interaction prompts. | Its public abstraction centers on selecting/playing packaged animations and camera/material presentation. Custom per-frame diagnostics, landmark/line overlays, component inspection, wireframe/normals modes, and a shared waveform clock would require working around the component rather than with it. | Apache-2.0 ([official repository](https://github.com/google/model-viewer), [camera and animation API](https://modelviewer.dev/docs/index.html)) | Appropriate for a read-only embed/export preview, not the main workstation. |

All three licenses permit commercial use. Three.js has the smallest notice
surface for the selected bundle. Preserve its MIT text in
`THIRD_PARTY_NOTICES.md` and in source distributions. GNM itself is under the
repository's Apache-2.0 license. Avoid a third-party HDRI or matcap in the
default scene so an otherwise simple viewer does not acquire an ambiguous art
asset license.

## Delivery format comparison

| Format | What it can carry | Operational properties | Use here |
|---|---|---|---|
| Current OBJ | Geometry; OBJ can represent UVs/normals/material references, but this exporter does not. It has no standard skin or animation. | Text is large and slow to parse; multiple MTL/texture files complicate the artifact allowlist. Three.js also notes that OBJ notably does not contain animation ([animation manual](https://threejs.org/manual/en/animation-system.html)). | Keep as a legacy/data-science download only. |
| `.gltf` + `.bin` + images | Full glTF scene: geometry, PBR materials, textures, morph targets, skins, cameras, and animation. | Easy to inspect, but several relative files and requests must remain together. Useful for exporter debugging. | Optional debug export, not the production response. |
| `.glb` | The same glTF scene in a single little-endian binary container. | One allowlisted artifact, one request, no broken relative resources, no base64 overhead. Registered MIME is `model/gltf-binary`. | **Production default.** |

glTF is a royalty-free runtime delivery standard intended to minimize asset size
and runtime processing ([Khronos overview](https://www.khronos.org/gltf/)). Its
core supports PBR materials, textures, skins, morph targets, node transforms,
and animation of morph weights. GLB packages JSON, buffers, and images into one
binary blob ([glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)).
The specification uses meters, a right-handed coordinate system, +Y up, +Z
forward, and an asset facing +Z, matching GNM's documented head orientation.

## Scene and artifact architecture

### Static GLB

Each image-fit job gains `fitted.glb`; audio/video jobs gain `animation.glb`.
A versioned neutral `gnm-head-v3-neutral.glb` is built once into the static
application assets so neutral/fitted comparison does not duplicate the neutral
mesh in every job.

The GLB contains:

- a root named `AutoAnim_GNM_Head_v3` with `asset.extras.autoanim` containing
  exporter version, source GNM version, units, job kind, identity state, and
  SHA-256 provenance;
- one mesh node whose primitives carry one of six component tags corresponding
  to the GNM components. A component may use several material primitives (for
  example sclera/iris/pupil/cornea within an eye), but component visibility is
  still controlled as one logical group;
- indexed `POSITION`, smooth `NORMAL`, and `TEXCOORD_0` accessors;
- for animated jobs, `JOINTS_0`, `WEIGHTS_0`, a four-joint hierarchy, inverse
  bind matrices, position/normal morph targets, and an animation named
  `autoanim`;
- neutral studio PBR materials for skin, eyes, teeth/gums, and tongue. Textures
  are embedded only when the manifest says a valid UV texture exists;
- no camera or light in the export. The viewer supplies presentation, while the
  asset remains usable in other glTF clients.

Construct the render index by interning `(source_vertex_index, triangle_corner_uv,
component)` tuples. This preserves UV seams without degrading every triangle
to an unindexed triangle soup. Duplicate positions must also duplicate normals,
skin weights, and every morph delta. Recompute indexed smooth normals per
component after seam expansion; assert finite, near-unit output and preserved
triangle winding.

Do not map the uploaded portrait onto the head. A portrait is perspective image
space, not GNM UV space. `texture.mode` is one of `none`, `uv_debug`, or
`uploaded_uv`; person-specific texture estimation is a separate research
pipeline. The debug edge-flow texture appears only in the explicit “UV debug”
mode, never as a fitted appearance.

### Compact but faithful animation

Neither naïve option is shippable:

- the complete expression basis alone is about 78.1 MiB of raw float32 data;
- sending every deformed vertex is about 367 MiB per minute at 30 fps (3.67 GiB
  for the accepted ten-minute maximum), before protocol overhead.

Use a job-specific, standard glTF animation instead:

1. Bake the job's fixed identity into the base mesh and identity-adjusted bind
   joints.
2. Perform deterministic SVD on the actual expression control track
   `E[F,383]`. Convert the retained right-singular control directions through
   GNM's expression basis into `K_expression` vertex-delta morph targets.
3. GNM pose correctives are linear in the 36 values of
   `flatten(R_joint - I)`. Factor the actual pose-feature track separately and
   convert it through `pose_correctives_regressor` into `K_pose` pre-skin morph
   targets.
4. Export the exact four-joint GNM hierarchy, identity-specific joint positions,
   skinning weights, and inverse bind matrices. Convert per-frame axis-angle
   rotations to normalized XYZW quaternions and animate node rotations;
   animate the root joint's bind translation plus GNM global translation.
5. Write morph weights, joint rotations, and translation at the existing exact
   timestamps with `LINEAR` interpolation. Expression and pose correctives are
   applied before glTF skinning, matching GNM's evaluation order in
   [`gnm_xnp.py`](../gnm/shape/gnm_xnp.py) and
   [`gnm_common.py`](../gnm/shape/gnm_common.py).

Start with a total cap of 32 morph targets. At 24 position-only targets, the raw
morph basis is about 4.89 MiB and a ten-minute 30 fps weight track is about
1.65 MiB. Three.js r184 uploads morph targets through a data-array texture,
rather than consuming one vertex attribute per target
([official r184 renderer source](https://github.com/mrdoob/three.js/blob/r184/src/renderers/webgl/WebGLMorphtargets.js)).

Include per-target `NORMAL` deltas, calculated from the corresponding deformed
basis shape, for smooth facial lighting. Normal blending is an approximation
for combined nonlinear surface normals, while vertex reconstruction remains the
measured source of geometric truth. Screenshot tests must catch objectionable
lighting artifacts. If normal targets dominate size, a later custom derivative
normal shader may replace them only after cross-browser visual tests.

Rank is selected by reconstruction quality, not an arbitrary explained-variance
percentage. A browser-equivalent Python evaluator must compare reconstructed
vertices and 68 landmarks against `GNMAdapter.mesh/landmarks` over all frames
for clips up to 60 seconds and a deterministic set including endpoints, extrema,
high residuals, and regular intervals for longer clips. Initial gates:

- mesh p95 error <= 0.10 mm and maximum <= 0.50 mm;
- landmark p95 error <= 0.25 mm and maximum <= 1.00 mm;
- normalized quaternion error <= `1e-5` and duration error <= half a frame.

Record rank, scope, p95/max errors, and validator result in the manifest. If the
32-target cap cannot pass, do not silently show inaccurate motion. Mark the
interactive animation unavailable, add `VIEWER_RECONSTRUCTION_LIMIT`, and show
the verified MP4. The primary audio/video job may still succeed. Current
procedural and ARKit-retargeted tracks are expected to be low rank, but this is
an inference that the real-fixture phase must prove.

Use a dedicated deterministic `viewer_export.py` GLB writer backed by NumPy,
the Khronos schema, and conformance tests. Existing `trimesh` remains useful for
normal calculation and static round-trip inspection, but its documented scope
does not cover the skeleton/morph animation required here. Every produced GLB
must pass the official [Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator)
with zero errors.

### Playback clock

Do not start an independent Three.js clock. The audible/visible media is the
authority:

- audio job: an `<audio>` element reading the allowlisted normalized WAV;
- video job: a `<video>` element reading the browser-compatible proxy MP4;
- still image: no clock.

Activate and pause the `autoanim` action, then assign
`animationAction.time = clamp(media.currentTime, 0, duration)` and call
`mixer.update(0)` on each render. This action-local sampling is required because
`AnimationMixer.setTime()` resets action-local time; a clamped `LoopOnce` action
that has reached the end otherwise cannot seek backward correctly. Three.js's
animation APIs remain the underlying contract
([AnimationMixer](https://threejs.org/docs/pages/AnimationMixer.html)).
For audio, sample `audio.currentTime` in `requestAnimationFrame`; the existing
`timeupdate` event is unsuitable for smooth facial motion because browsers may
emit it anywhere from roughly 4–66 Hz depending on load
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/timeupdate_event)).
For video, prefer `requestVideoFrameCallback` and its `mediaTime`, falling back
to `requestAnimationFrame` and `video.currentTime`; the per-video-frame callback
is designed for frame processing and synchronization
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)).

On `play`, `pause`, `seeking`, `seeked`, `ratechange`, `ended`, and visibility
return, immediately snap the mixer and diagnostic playhead to media time. This
is important because animation frames are normally paused in background tabs
([MDN requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)).
Never accumulate deltas. Never autoplay audible media.

### Landmarks

Static jobs can store 68 positions directly. Animated landmarks should not add
68 XYZ values to every frame. Add a non-rendered `POINTS` primitive named
`AutoAnim_LandmarkSources` to the GLB containing only the GNM regressor's source
vertices, their base/morph attributes, and four skin weights. Put the compact
source-to-landmark indices and weights in that node's `extras`. The viewer reads
those public geometry attributes, copies the current morph weights, evaluates
only the small source set under the four joint matrices on CPU, and then applies
the regressor. This keeps landmark dots exact relative to the reconstructed
mesh, standard-container-only, and independent of UV seam duplication.

## Result and API contract

Preserve the current logical-artifact allowlist. The viewer block references
logical keys, never arbitrary filesystem names or client-supplied URLs:

```json
{
  "viewer": {
    "schema_version": "1.0",
    "status": "ready",
    "mode": "animation",
    "model_artifact": "viewer_model",
    "animation_clip": "autoanim",
    "clock_artifact": "normalized_audio",
    "timeline_artifact": "timeline",
    "poster_artifact": "preview",
    "duration_s": 8.0,
    "fps": 30,
    "coordinate_system": "+Y_up_+Z_forward_meters",
    "components": [
      "skin", "left_eye", "right_eye", "upper_teeth_and_gums",
      "lower_teeth_and_gums", "tongue"
    ],
    "texture": {"mode": "none"},
    "reconstruction": {
      "expression_rank": 12,
      "pose_rank": 6,
      "validation_scope": "all_frames",
      "mesh_p95_mm": 0.03,
      "mesh_max_mm": 0.21,
      "landmark_p95_mm": 0.05,
      "landmark_max_mm": 0.34
    }
  },
  "artifacts": {
    "viewer_model": {
      "name": "animation.glb",
      "media_type": "model/gltf-binary",
      "bytes": 5812345,
      "sha256": "..."
    },
    "normalized_audio": {
      "name": "normalized.wav",
      "media_type": "audio/wav",
      "bytes": 256044,
      "sha256": "..."
    }
  }
}
```

For an image job, `mode` is `static`, there is no clock or clip, and
`viewer_model` is `fitted.glb`. For a job that cannot meet the animation gate,
`status` is `static_only` or `unavailable`, `reason_code` is machine-readable,
and the poster/MP4 remains available.

Required server changes:

1. Mount package-owned, hashed frontend assets at `/static` using FastAPI's
   `StaticFiles` ([official FastAPI documentation](https://fastapi.tiangolo.com/tutorial/static-files/)).
   Replace the inline script/style with an external module and stylesheet so a
   strict content-security policy is possible.
2. Add `.glb -> model/gltf-binary`, `.gltf -> model/gltf+json`, `.bin ->
   application/octet-stream`, and `.ktx2 -> image/ktx2` to artifact MIME
   handling. Continue path-basename and terminal-manifest checks.
3. Expose `normalized.wav` as an audio artifact. For video jobs expose a
   normalized H.264/AAC MP4 (or a tested WebM alternative) as `viewer_media`.
4. Add the optional `viewer` object to the existing job result. No extra viewer
   endpoint is necessary; `GET /api/jobs/{id}` already supplies the contract
   and `/api/jobs/{id}/files/{name}` supplies immutable bytes.
5. Return `Cache-Control: private, max-age=31536000, immutable` for artifacts
   addressed by immutable job ID and checksum; keep job JSON and index HTML
   uncached. Retain ETag and `Last-Modified`.
6. Test byte ranges. Starlette `FileResponse` supports `Accept-Ranges: bytes`,
   `206`, and `416`, which matters for media seeking
   ([official Starlette response documentation](https://www.starlette.io/responses/)).
7. When the video-analysis pipeline exists, add `POST /api/video` with the same
   service/job lifecycle and terminal manifest as audio/image. Viewer code must
   depend only on the result contract, not on the tracking backend.

Do not change the current synchronous POST behavior merely to ship the viewer.
During processing, show an indeterminate state with the truthful stage name; do
not fabricate percentages. If real production timings establish that queued
work is needed, move all three pipelines together to `POST -> 202` plus job
polling as a separate service change rather than inventing a viewer-only queue.

Suggested content-security policy after externalizing assets:

```text
default-src 'self';
script-src 'self';
style-src 'self';
connect-src 'self' blob:;
img-src 'self' blob:;
media-src 'self' blob:;
worker-src 'self' blob:;
object-src 'none';
base-uri 'none';
frame-ancestors 'none'
```

User-supplied UV textures must be decoded with Pillow, constrained by encoded
bytes and pixel count (initially 4096 x 4096), converted to sRGB RGB/RGBA, and
re-encoded to strip metadata before embedding. The GLB itself is generated by
the server, not accepted as an untrusted upload.

## Product UI

### Desktop workspace

After a successful job, expand the result area into a focused workspace instead
of squeezing 3D into the current upload card:

```text
+--------------------------------------------------------------------------+
| Result: learned audio / fitted image    Ready      Export ▾   Fullscreen |
+------------------+------------------------------------+------------------+
| Source & mode    |                                    | Inspector        |
| Neutral / Fitted |             3D canvas              | Components       |
| Shaded / UV /    |                                    | Material         |
| Normals          |                                    | Lighting         |
|                  |                                    | Camera / QA      |
+------------------+------------------------------------+------------------+
| Play  00:02.43 / 00:08.00   waveform + cues + warnings      zoom        |
+--------------------------------------------------------------------------+
```

The canvas toolbar provides named, tooltip-backed controls:

- Play/Pause, time readout, front, left, right, and three-quarter presets;
- fit camera, reset camera, perspective/orthographic view, and fullscreen;
- studio, flat, normals, UV debug, and textured/albedo shading modes;
- wireframe, 68 landmarks, axes/grid, and background toggles;
- screenshot PNG and export menu.

The inspector contains collapsible sections:

- **Mesh:** neutral/fitted/current selector and visibility for skin, each eye,
  teeth/gums, and tongue;
- **Material:** base color/available texture, roughness, exposure, and UV debug;
- **Lighting:** fixed analytic studio preset, key intensity and direction, fill,
  and reset. No shadows by default;
- **Camera:** projection, FOV, numeric azimuth/elevation/distance, fit/reset;
- **Diagnostics:** backend, rank, reconstruction errors, dropped frames, sync
  offset, current cue, mouth aperture, energy, confidence, and warnings.

Use a perspective camera with target and near/far planes derived from the
loaded bounding sphere. Default to a front three-quarter studio view; one click
returns to exact front. `OrbitControls` supplies orbit, dolly, pan, and standard
one-/two-finger gestures while retaining +Y up
([official OrbitControls documentation](https://threejs.org/docs/pages/OrbitControls.html)).
Limit polar angle and distance so the user cannot lose the head or flip the
scene accidentally.

The export menu contains the server-authored animation/fitted GLB, MP4/PNG,
legacy OBJ where present, controls NPZ, timeline JSON, and cues. “Screenshot
PNG” is local canvas export. Add “Current pose GLB” only after a GLTFExporter
round-trip test proves the exported default pose is preserved; the official
exporter supports binary glTF, scenes, skins, morph targets, and animations
([GLTFExporter](https://threejs.org/docs/pages/GLTFExporter.html)).

### Timeline diagnostics

Keep media and 3D on the same playhead. Replace the current single canvas with:

- an accessible range input as the actual scrubber;
- a waveform/energy lane;
- mouth-aperture and speech-activity lanes;
- cue/viseme blocks with text or pattern, not color alone;
- emotion/accent lane and warning markers;
- for video, landmark confidence and head-pose lanes;
- textual current-time/current-cue/current-warning readout.

Canvas/SVG charts are visual aids and `aria-hidden`; the slider, labels, current
readout, warning list, and downloadable data are the semantic interface. Seeking
the slider updates the media element, which in turn updates the mixer. Timeline
zoom changes the visible range only, never the source timebase.

### Loading and recovery

Use explicit states rather than a permanent spinner:

| State | UI | Recovery |
|---|---|---|
| Processing job | Poster/skeleton and “Analyzing audio”, “Fitting identity”, or “Tracking video”; indeterminate because the API has no measured progress. | Cancel only when backend cancellation exists; otherwise allow navigation and show final result. |
| Fetching GLB/media | Byte progress when `Content-Length` is available, AbortController tied to job switch. | Retry viewer or open verified poster. |
| Parsing/compiling | “Preparing 3D”; await GLTFLoader and `renderer.compileAsync` before announcing ready. | Retry once, then poster fallback. |
| `WEBGL_UNAVAILABLE` | Static poster plus plain explanation. | Download GLB/MP4; try a supported browser/device. |
| `VIEWER_ASSET_INVALID` | Poster plus validator/reference ID. | Retry download; preserve all data exports. |
| `TEXTURE_FAILED` | Neutral material; mesh remains interactive. | Retry texture or choose studio shading. |
| `MOTION_TRACK_INVALID` / reconstruction limit | Static interactive mesh plus MP4. | Inspect quality report; download controls. |
| `CONTEXT_LOST` | Freeze poster and announce GPU reset. | Attempt one restore/reload, then poster fallback. |
| Media stall/error | Hold the last exact pose and show media error. | Retry media; do not run mesh ahead silently. |

Listen for `webglcontextlost`/`webglcontextrestored`; context loss is a standard
browser event and can be simulated in tests with `WEBGL_lose_context`
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/webglcontextlost_event)).

### Mobile and performance

- At widths below 720 px, use a full-width 1:1 or 4:3 canvas, horizontally
  scrollable primary toolbar, bottom-sheet inspector, and simplified timeline.
- Preserve vertical page scrolling; one-finger horizontal-dominant movement
  orbits, two fingers pan/pinch. Fullscreen landscape is the detailed mode.
- Minimum touch target is 44 CSS px. Do not hide a required function behind
  hover.
- Cap renderer pixel ratio at `min(devicePixelRatio, 2)` on desktop and `1.5`
  on constrained/mobile mode. Disable shadows and postprocessing; use analytic
  hemisphere/directional lights.
- Render on demand for static scenes. Run continuously only during media
  playback, active damping, resize, or camera manipulation. Pause when hidden
  and snap to media time on return.
- Abort fetches and dispose `AnimationMixer` actions, skeletons, controls,
  geometries, materials, textures, image bitmaps, and renderer state on job
  switch. Three.js does not automatically free GPU resources
  ([official cleanup guide](https://threejs.org/manual/en/how-to-dispose-of-objects.html)).
- Start without Draco, meshopt, KTX2, HDR environments, shadows, or
  postprocessing. Add compression only after size/latency measurement.
  `EXT_meshopt_compression` supports geometry, morph targets, and animation and
  is a later good fit; `KHR_texture_basisu` adds KTX2/Basis texture compression
  ([Khronos meshopt extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md),
  [Khronos BasisU extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)).

### Accessibility

- Give the canvas a stable accessible name such as “Fitted GNM head, front
  three-quarter view, mouth open at 2.43 seconds” and useful fallback content,
  including the poster. Canvas needs fallback text/sub-DOM to be accessible
  ([MDN canvas guidance](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Basic_usage)).
- Every canvas gesture has a DOM button/slider/keyboard equivalent with visible
  focus. `Space` toggles playback only when the workspace owns focus; arrows
  scrub by one frame, Shift+arrows by one second, Home/End seek bounds, `R`
  resets camera, and `0/1/2/3/4` select camera presets.
- Announce only state changes (“3D ready”, “Playback paused”, recoverable
  errors) through a polite live region; do not announce every animation frame.
- Provide a semantic diagnostics table and current-pose text alternative for
  users who cannot perceive the canvas.
- Meet WCAG AA contrast, never use hue alone for timeline meaning, and preserve
  200% text zoom without clipping controls.
- Honor `prefers-reduced-motion`: remove camera easing, pulsing loaders, and
  automatic view transitions, while keeping user-requested facial playback
  because it is the content. The media feature is broadly available and exists
  to reduce nonessential motion
  ([MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/%40media/prefers-reduced-motion)).

## Test strategy and hard gates

### Python/export tests

- UV seam expansion preserves each `(source vertex, triangle-corner UV)` pair,
  UVs remain in `[0,1]`, winding is preserved, indices stay in range, all six
  components remain, and normals are finite/unit.
- Static neutral and fitted GLB positions match the corresponding GNM/OBJ
  geometry within `1e-6` m after round trip. Bounds and units match.
- Identity-adjusted glTF skinning plus pose-corrective morphs matches
  `GNMAdapter.mesh` for zero, random bounded, and real-track controls at the
  stated reconstruction gates.
- SVD rank selection is deterministic, handles zero/constant/short tracks, and
  fails closed on NaN, nonmonotonic timestamps, excessive rank, or duration
  mismatch.
- GLB header, chunk lengths, 4-byte alignment, accessors, min/max, quaternion
  normalization, animation weight dimensions, joint tree, and embedded image
  MIME are verified. Two identical inputs produce identical hashes.
- Khronos glTF Validator reports zero errors for every generated neutral,
  fitted, audio, and video artifact.
- Job allowlisting denies undeclared/path-traversal names and accepts declared
  GLB/media. Assert content type, length, ETag, `Accept-Ranges`, valid `206`, and
  invalid `416` behavior.

### Browser/end-to-end tests

Use Playwright against the real FastAPI process and real generated artifacts:

- **Image:** submit the repository's actual portrait fixture, load fitted GLB,
  verify nonempty bounds/six components/68 landmarks, and capture front,
  three-quarter, wireframe, normals, and UV-debug screenshots. Verify the
  neutral/fitted geometry differs and poster fallback remains available.
- **Audio:** submit the actual eight-second human speech fixture to learned and
  fallback backends where available. Wait for `viewer-ready`, play, pause, seek
  to start/middle/end, change rate, and resume after visibility change. At
  checkpoints, compare Three.js morphed/skinned landmarks to the server's
  expected values and require media/mixer drift below one 30 fps frame.
- **Video:** submit an actual face-video fixture, verify proxy playback,
  `requestVideoFrameCallback` path, head/eye/mouth motion, seeking, and drift
  below one source frame. This gate cannot pass until the video analysis
  pipeline exists; do not replace it with a mocked track.
- Force GLB 404/corruption, texture decode failure, media failure, WebGL
  unavailability, and `WEBGL_lose_context`; assert accessible, useful fallback
  and no unhandled console error.
- At 390 x 844 and 844 x 390, assert no page-level horizontal overflow, all
  controls remain reachable, touch gestures do not trap vertical scroll, and
  device-pixel-ratio cap is applied.
- Keyboard-only traversal verifies focus order, accessible names, playback,
  scrubbing, camera presets, component toggles, export, and error recovery.
  Emulate reduced motion and high contrast.
- Switch among at least 20 results and assert WebGL context count remains one,
  `renderer.info.memory` returns to a stable band, fetches are aborted, and
  disposed scenes stop rendering.

Initial performance budgets on a warm local server:

- static neutral/fitted GLB <= 3 MiB uncompressed;
- typical <= 60 second animated GLB <= 16 MiB at the 32-target cap, including
  position and normal morph targets;
- first useful static render <= 1.5 s desktop / 3 s mobile emulation;
- playback p95 render frame <= 33 ms desktop / 50 ms mobile emulation;
- no long task > 200 ms after `viewer-ready`, and no growing GPU resource count
  during result switching.

Budgets are release gates, not promises about every device. If a real fixture
exceeds them, profile and either optimize, lower a non-geometric presentation
cost, or use the explicit fallback; never lower reconstruction correctness
without changing the published quality tier.

## Phased implementation plan

Every phase uses the same loop: build only that phase, review against this
contract (including edge cases and cleanup), run its focused tests plus the full
existing suite and real-input E2E, fix all failures, then repeat until green.
No phase advances on a waiver.

### Phase 0 — export/conformance spike

**Build:** implement UV-seam expansion, component materials, deterministic
static GLB serialization, neutral build asset, fitted image GLB, MIME support,
and validator integration. Add the optional viewer manifest in static mode.

**Dependencies:** current GNM adapter, triangle UVs, trimesh/Pillow, official
Khronos validator pinned for CI.

**Gate:** all static/export/API unit tests pass; neutral and real-photo fitted
GLBs validate with zero errors and round-trip to source geometry; no UI change
is needed to pass this phase.

### Phase 1 — static interactive viewer

**Build:** externalized Vite/Three.js bundle, FastAPI static mount, strict CSP,
GLTFLoader scene, camera/OrbitControls, analytic studio lighting, neutral/fitted
toggle, six component toggles, shaded/flat/normals/UV/wireframe/landmark modes,
resize/fullscreen, download menu, loading states, cleanup, and poster fallback.

**Dependencies:** Phase 0 artifacts; `three@0.184.0`; no animation work.

**Gate:** real portrait E2E and screenshot set pass in Chromium, WebKit, desktop,
and mobile viewports; keyboard/accessibility checks pass; WebGL unavailable,
corrupt asset, and context-loss paths recover; full Python suite remains green.

### Phase 2 — synchronized audio animation

**Build:** expression and pose-feature factorization, morph targets, exact GNM
skeleton/skin, animation accessors, normalized-audio artifact, media-master
AnimationMixer loop, waveform/cue diagnostics, seeking/rate/visibility behavior,
and reconstruction quality in the manifest.

**Dependencies:** Phase 1 viewer; existing final `controls.npz` contract. It is
backend-agnostic and therefore covers learned Audio2Face and procedural fallback.

**Gate:** real eight-second speech passes validator and all-frame mesh/landmark
quality; browser playback, pause, seek, end, loop, and rate tests stay within one
frame; learned and fallback paths both produce honest status; MP4 fallback is
tested; full suite passes.

### Phase 3 — video-driven motion

**Build:** `POST /api/video`, browser-compatible proxy media, final GNM control
track, viewer GLB using the same exporter, `viewer_media` clock contract, and
video-specific confidence/head-pose diagnostics. Viewer-specific code should be
limited to choosing video media time instead of audio time.

**Dependencies:** a production video-to-GNM tracking pipeline and a licensed,
redistributable real face-video test fixture. This repository does not have that
pipeline today.

**Gate:** actual video input produces a valid animated GLB, visibly and
numerically follows head/eye/mouth motion, seeks correctly, and remains within
one source frame through `requestVideoFrameCallback`; mocked tracks do not count.

### Phase 4 — textures, export, and production UX

**Build:** validated uploaded-UV texture path, embedded textures, material and
lighting inspector, current-pose export if its round trip passes, screenshot,
responsive bottom sheet, complete timeline lanes, diagnostics table, keyboard
map, reduced motion, localization-safe labels, and error-code catalog.

**Dependencies:** Phases 1–3 and an explicit UV texture input; portrait-to-UV
texture inference remains out of scope until separately built and evaluated.

**Gate:** real UV texture renders without seams on all components; export
round-trips preserve pose/material; mobile, accessibility, forced-error, job
switching, and memory tests pass; full audio/image/video E2E remains green.

### Phase 5 — measured optimization and release hardening

**Build only if measurement requires it:** meshopt/quantization, KTX2, worker
decode, generated thumbnails, long-clip rank tuning, async job progress, and
browser-specific workarounds. Keep uncompressed GLB as a conformance/debug
fixture and fallback.

**Gate:** compressed and uncompressed assets reconstruct within the same
geometry gates, validate, load in the chosen Three.js version, and improve
measured size or time materially without regressing startup, accessibility, or
offline behavior. Run the entire real-input matrix before release.

## Known limitations that the UI must state

- The viewer improves inspection and delivery; it does not improve the accuracy
  of audio retargeting, emotion inference, photo identity fitting, or future
  video tracking.
- A fitted image changes GNM identity geometry, not photorealistic appearance.
  No person texture exists unless a UV texture is separately supplied or a
  future appearance pipeline succeeds.
- The 68 landmarks are a sparse diagnostic, not proof of perceptual identity or
  lip-sync quality.
- More than 32 job-specific morph directions may exceed the interactive quality
  tier. The app exposes the reconstruction error and falls back instead of
  hiding that limitation.
- glTF places no format-level limit on morph targets, but the core specification
  only recommends that generic clients support at least eight morphed
  attributes. A valid 9–32-target animation is tested and supported in the
  pinned Three.js viewer; third-party DCC/viewer compatibility beyond eight
  targets is best effort and must be stated beside the GLB download.
- The current server serializes jobs under a process-local lock and returns only
  after completion. A viewer does not make that production-concurrent; queueing
  and worker isolation are separate service milestones.
- WebGL depends on browser and GPU health. PNG/MP4 and downloadable GLB remain
  first-class outputs, not an error-page afterthought.
