# AutoAnim: verified GNM facial-animation prototype

This checkout pins Google's GNM repository at
`3de70dfca5f3244620f44103c24b7cedc0dcb8b6` and adds a local Python/CLI/web
application for four evaluated workflows:

- real audio → Audio2Face-3D Claire on Apple Silicon → bounded 52-channel
  ARKit plus 16-channel tongue solve → semantic 383-D GNM retarget → dense
  mesh preview with audio, with a transparent Rhubarb procedural fallback;
- one real photo → MediaPipe landmarks → confidence-gated GNM visible-geometry
  fit → neutral OBJ/GLB, overlay, parameters, and explicit uncertainty;
- ordered front/three-quarter/profile photos → one shared perspective identity
  solve → provenance-aware UV texture bake → textured GLB and coverage audit;
- moving face video → exact-PTS MediaPipe performance capture → dense
  geometry-calibrated GNM expression, head, translation, and gaze animation.

The learned audio motion is materially smoother than the procedural baseline,
but its initial ARKit-to-GNM map is not artist-calibrated and is deliberately
reported as not production-approved. The fallback remains coarse viseme timing.
Automatic tone labels are an unvalidated acoustic/lexical heuristic. The photo
result is a visible-geometry estimate, not a metric 3D clone. Multiview improves
observability and texture coverage but cannot recover perfect hidden anatomy or
intrinsic albedo from ordinary RGB photos. Monocular video has no claim to
ground-truth microexpression capture.

Every GLB opens in the same media-synchronized Three.js viewer with surface,
texture, topology, exposure, orbit, zoom, and camera-reset controls. The exact
Three.js runtime is downloaded from the official npm package, checksum-pinned,
served locally under a restrictive CSP, and included in health readiness; the
viewer does not need a CDN after bootstrap.

## Quick start

Requirements: Python 3.12, ffmpeg/ffprobe, curl, unzip, tar, and `shasum`. The learned backend
also requires Apple Silicon macOS 15+, Swift 6+, and Xcode's Metal toolchain;
macOS/Linux can use the procedural fallback.

```bash
scripts/bootstrap.sh
scripts/bootstrap_a2f.sh
source .venv/bin/activate
export RHUBARB_BIN="$PWD/.cache/autoanim_gnm/rhubarb/rhubarb"
scripts/fetch_test_fixtures.sh

autoanim-gnm health --json
autoanim-gnm audio .cache/autoanim_gnm/fixtures/libri-human-speech-8s.wav --out artifacts/jobs --backend auto
autoanim-gnm image .cache/autoanim_gnm/fixtures/official-portrait.jpg --out artifacts/jobs
autoanim-gnm multiview front.jpg left-3q.jpg right-3q.jpg --roles front,left_3q,right_3q --out artifacts/jobs

# Calibrated audit mode: JSON maps every ordered image to measured K/D/R|t and
# reserves at least one camera as a leakage-proof held-out evaluation view.
autoanim-gnm multiview front.png left.png right.png profile.png \
  --calibration rig.json --out artifacts/jobs
autoanim-gnm video performance.mp4 --out artifacts/jobs

# Import controls claimed by a separately provisioned NVIDIA v3 worker.
# This is candidate-only: hashes validate content, not that v3/CUDA actually ran.
autoanim-gnm audio speech.wav --out artifacts/jobs --backend a2f-v3 \
  --v3-request worker-request.json --v3-response worker-response.json \
  --v3-model /pinned/v3/network.onnx \
  --v3-runtime worker-runtime-attestation.json \
  --v3-identity /pinned/v3/model_data_Claire.npz \
  --v3-schema worker-control-schema.json --v3-profile /pinned/v3

# Validate a complete commercial PBR facial-material package. The JSON spec
# carries map inventory, capture/provenance, rights, and evidence-backed claims.
# Its normal inventory entry must declare normal_encoding as unorm or
# signed_float; pixel-value inference is intentionally forbidden.
autoanim-gnm material /path/to/material-package --spec material-package.json

# Bind that validated package to an exact saved character revision. The first
# command emits a subject/revision/identity/UV attachment envelope; the second
# atomically publishes a new immutable PBR character revision. Complete package
# import is local/CLI-first because production atlases can be many gigabytes.
autoanim-gnm character --artifacts artifacts/jobs material-template CHARACTER_ID \
  --character-revision REVISION_ID --package-root /path/to/material-package \
  --spec material-package.json --attester "Lookdev supervisor" \
  --evidence-ref release://lookdev/shot-001 --evidence binding-evidence.pdf \
  --package-subject "Performer legal name" --same-subject-attested \
  --authored-for-attested \
  --displacement-midpoint 0.5 --displacement-scale-m 0.002 \
  --out material-attachment.json
autoanim-gnm character --artifacts artifacts/jobs import-material CHARACTER_ID \
  --character-revision REVISION_ID --package-root /path/to/material-package \
  --spec material-package.json --attachment material-attachment.json

# A successful audio/video take can be directed by a tool-disabled terminal LLM;
# the result includes editable beats plus a deterministic humanoid body/gaze track.
autoanim-gnm direct JOB_ID --artifacts artifacts/jobs --provider codex \
  --instructions "Restrained, reassuring; one small open-palm beat"
autoanim-gnm serve --host 127.0.0.1 --port 8000 --artifacts artifacts/jobs
```

Audio and video jobs now emit `oral-validation.json` and
`oral-glb-validation.json`. These audit every frame of GNM lip, tongue and
teeth geometry plus viewer reconstruction. They are structural reports, not
claims of phone accuracy, visibility, penetration-free geometry or artist
approval.

The global `--model-path`, `--rhubarb-bin`, `--a2f-runner`, and `--a2f-assets`
options may be placed before the subcommand when using non-default locations.
Use `--backend learned` to require the neural path or `--backend fallback` for
the deterministic compiler. `--backend a2f-v3` is an explicit, no-fallback
offline import of a sealed external-worker result; it requires the exact pinned
model/runtime/identity/schema/profile bindings shown above and never claims the
worker is authenticated or production-qualified. The web app is then available
at <http://127.0.0.1:8000>.

Run the normal suite after fetching fixtures:

```bash
pytest -q
```

The licensed RAVDESS emotional-speech fixture is opt-in and roughly 200 MiB:

```bash
AUTOANIM_FETCH_RAVDESS=1 scripts/fetch_test_fixtures.sh
```

The small CREMA-D moving-actor fixture is also opt-in. Review its
ODbL/DbCL attribution and performer/publicity caveats before fetching:

```bash
AUTOANIM_FETCH_CREMA_D=1 scripts/fetch_test_fixtures.sh
```

See [the external fixture notice](docs/TEST_FIXTURES.md) for the pinned source,
checksum, attribution, and validation scope.

Research and implementation evidence:

- [GNM architecture research](docs/RESEARCH.md)
- [pipeline feasibility](docs/FEASIBILITY.md)
- [application spec and phase gates](docs/SPEC.md)
- [production lipsync research, benchmark, and executed plan](docs/PRODUCTION_LIPSYNC_RESEARCH.md)
- [current audio production audit and remaining gates](docs/AUDIO_PRODUCTION_RESEARCH.md)
- [expanded multiview, texture, video, viewer research and phased plan](docs/EXPANDED_RESEARCH_AND_PLAN.md)
- [calibrated multiview camera contract and held-out evaluation](docs/CALIBRATED_MULTIVIEW.md)
- [interactive viewer design and validation contract](docs/VIEWER_RESEARCH.md)
- [requirement-by-requirement completion audit](docs/COMPLETION_AUDIT.md)
- [final verification and retained metrics](docs/VERIFICATION.md)
- [unified production character, acting, appearance, body, oral, security, and phased workflow](docs/PRODUCTION_WORKFLOW.md)

The original upstream project overview follows.

---

# GNM: Generative aNthropometric Model and Ecosystem

![GNM Teaser Image](assets/readme/gnm_logo.png)

Welcome to the **GNM Ecosystem** repository. GNM - pronounced as genome
(/ˈdʒiː.noʊm/) in reference to the human genome - strives to be the most
accurate and complete 3D parametric human model.

3D Morphable Models (3DMMs) are widely used across computer vision, computer
graphics, and generative AI for representing human geometry and appearance. GNM
introduces a state-of-the-art family of parametric statistical human models and
its associated perception stack.

Our roadmap includes releasing a comprehensive suite of statistical models
complemented by perception and analysis technology. To facilitate early
community research and open development, we are beginning our open-source
release with **GNM Head**, our high-fidelity statistical 3D model of the human
head.

The ecosystem is released under a permissive license suitable for both
non-commercial and commercial applications.


## GNM Ecosystem Packages

Here we list all the available GNM packages:

| Name | Description | Chips | Teaser |
| :--- | :--- | :--- | :---: |
| **[GNM Head](gnm/shape/README.md)** | Parametric 3D statistical human head and face geometry model providing fine-grained, disentangled control over identity, expressions, and head pose. The model contains controllable internal anatomy including eyeballs, teeth and tongue. Includes multi-framework backend support for **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, along with semantic parameter sampling. | [![CI Linux](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml)<br>[![CI macOS](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml)<br>[![CI Windows](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml)<br>[![Lint](https://github.com/google/gnm/actions/workflows/lint.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/lint.yml) | ![GNM Head Teaser](gnm/shape/assets/readme/teaser_heads_cropped.gif) ![GNM Head demo teaser](gnm/shape/assets/readme/gnm_head_demo.gif)

## Citation
If you use any part of the GNM Ecosystem in your work, please consider citing
the corresponding package. Relevant bibtex entries are listed below as well as
within the individual packages.

**GNM Head**

```bash
coming soon
```

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
