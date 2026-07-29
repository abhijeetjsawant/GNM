# GEM-X reproduction on Apple Silicon

Date: 2026-07-29  
Machine: MacBook Pro `Mac14,6`, Apple M2 Max (12 CPU cores), 32 GB RAM  
OS: macOS 26.2 (`25C56`), arm64

## Result

**Status: BLOCKED before inference. No real-video inference has completed, and
this report does not claim that GEM-X works on this machine.**

The official source checkout is pinned and the local fixture is verified. The
documented macOS setup was inspected and partially exercised. A long-running
submodule repair was stopped before completion, at which point further model
downloads were explicitly halted. The environment therefore has no ONNX Runtime,
no CoreML execution provider, no ONNX model files, and no inference output.

No AutoAnim production code was changed. The GEM-X checkout and all attempted
setup work live under the ignored project-local cache:

```text
/Users/abhi_macbook/Projects/apps/AutoAnim/.cache/autoanim_gnm/gem-x
```

## Source pin

| Item | Recorded value |
|---|---|
| Official source | `https://github.com/NVlabs/GEM-X.git` |
| GEM-X commit | `32992550dba114c62243fb55e361311972dce8f9` |
| Commit date | `2026-04-27T13:56:41-07:00` |
| Commit subject | `fix sam3db onnx auto-download and clean up demo docstring` |
| Branch at audit time | `main` tracking `origin/main` |
| Cache size after interrupted submodule operation | 2,984,352 KiB (`du -sk`) |
| Free disk after interruption | 52 GiB |

Pinned submodules and observed state:

| Submodule | Superproject pin | Observed state |
|---|---|---|
| `third_party/sam-3d-body` | `b5c765a0d89d789985e186d396315e7590887b94` | Git object present, worktree incomplete |
| `third_party/soma` | `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5` | Worktree present at `86632764684281dc98f31ab9c4aac36a4cdbc428`, not the pinned commit |
| `third_party/soma-retargeter` | `b12d9a3eeff6ea64d7029684e47d1e92b9a60c2c` | Not checked out |

The retargeter URL in `.gitmodules` is SSH-only:
`git@github.com:NVIDIA/soma-retargeter.git`. The first non-networked attempt
failed to resolve GitHub. The HTTPS rewrite attempt below ran for about 17
minutes and was user-interrupted before returning:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive
```

Because this checkout is not cleanly reproducible yet, it should not be used as
a production dependency pin.

## Documented macOS path and disk gate

The repository's supported path is:

```bash
git clone --recursive https://github.com/NVlabs/GEM-X.git
cd GEM-X
bash scripts/setup_mac.sh
source .venv/bin/activate
python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4
```

For this audit the intended setup command was:

```bash
cd /Users/abhi_macbook/Projects/apps/AutoAnim/.cache/autoanim_gnm/gem-x
bash scripts/setup_mac.sh --no-ramdisk
```

`--no-ramdisk` avoids the optional 2 GB RAM disk and does not alter the model
pipeline. The script:

1. initializes submodules;
2. creates a Python 3.12 environment;
3. installs PyTorch, SOMA, GEM, ONNX Runtime and conversion tools;
4. pulls SOMA Git LFS assets;
5. leaves the ONNX bundle to auto-download on first inference.

The script and `docs/INSTALL_MACOS.md` say approximately 4.6-5 GB. The current
official Hugging Face `nvidia/GEM-X/onnx` directory is **8.3 GB**, so that
estimate is stale. The eight files selected by
`gem.utils.hf_utils.download_all_onnx()` are:

| Remote file | Published size | Published SHA-256 when exposed by the official file page |
|---|---:|---|
| `onnx/gem_denoiser.onnx` | 1.51 MB | `20aab83c01bbd909a258ad0fa465458eda382c63dc80d56952b2ec952d6e192c` |
| `onnx/gem_denoiser.onnx.data` | 199 MB | not exposed during this audit |
| `onnx/gem_denoiser_no_imgfeat.onnx` | 188 MB | `65c576968bc9548b5c485522930b7985097b70a5e0154e5cd01241d6c806415f` |
| `onnx/gem_denoiser_no_imgfeat.onnx.data` | 1.16 GB | not exposed during this audit |
| `onnx/sam3db_backbone.onnx` | 879 kB | `e03fe953ff9762c3e96b67dcd78e52769e67798cd1f8bd82c4ef5dd7bde48b22` |
| `onnx/sam3db_backbone.onnx.data` | 3.36 GB | `479e7da52d115344a3fd3e7711dae58176f68c723486e80ba2c3efc046f4dbc6` |
| `onnx/vitpose.onnx` | 4.28 MB | `0982dbf4f1e8a48446a6fe35329711522b60210cce2a7499ca6ab93458c87f34` |
| `onnx/vitpose.onnx.data` | 3.39 GB | not exposed during this audit |

These are remote inventory values, not local-file verification. **No model or
checkpoint file was downloaded**, so there are no local model hashes to report.
After download, every local file must be SHA-256 checked and the three missing
official digests recorded before the assets are accepted.

At the gate, the checkout/environment occupied 0.62 GB. The interrupted
submodule operation grew the cache to 2.85 GiB. Adding the published 8.3 GB ONNX
bundle yields roughly 11.15 GB before transient download/cache duplication. That
is too close to a strict 12 GB cap to continue safely without first measuring
and relocating Hugging Face/Xet's temporary cache. The earlier estimate did not
account for the 2.2 GB submodule/object growth.

## Environment checks actually run

Prerequisites found:

```text
Python 3.12.13
uv 0.11.2
git-lfs 3.7.1
```

Lightweight imports in the existing `.venv`:

| Component | Result |
|---|---|
| Python | 3.12.13 |
| `torch` | imported, version 2.13.0 |
| `torchvision` | imported, version 0.28.0 |
| `numpy` | imported, version 2.5.1 |
| `gem` | imported, reports version 1.0.0 |
| `soma` | missing |
| `onnxruntime` | missing |
| `onnx` | missing |
| `opencv-python` / `cv2` | missing |
| `huggingface_hub` | missing |
| PyTorch MPS compiled | yes |
| PyTorch MPS available at runtime | **no** |

No ONNX provider can be reported because `onnxruntime` is not installed.
The repository intends to request `CoreMLExecutionProvider` first and fall back
to `CPUExecutionProvider`; that selection has not been tested here. The
unexpected `torch.backends.mps.is_available() == False` is a second environment
blocker that should be resolved before performance work, although the ONNX path
does not require PyTorch MPS for every stage.

## Rights-cleared local real-video fixture

The only distinct real-person video fixture found in the project is:

```text
/Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/production-next-video/01kxw83zz4rawnm0j3dyask1pz/input.flv
```

| Property | Value |
|---|---|
| SHA-256 | `10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4` |
| File size | 265,922 bytes |
| Duration | 2.235 seconds |
| Video | VP6F, 480×360, 30000/1001 fps |
| Audio | MP3, mono, 44.1 kHz |
| Framing | Real person, head-and-shoulders against green screen |

Twenty-four artifact paths contain this same byte-identical file; no duplicate
was created for GEM-X. It is a valid real input for an execution smoke test, but
it is a weak quality fixture for a full-body estimator because legs and most of
the torso are out of frame. A production evaluation still needs a consented,
full-body clip with hand and foot motion and known camera behavior.

## Exact attempts and blocker

Read-only checks:

```bash
git -C .cache/autoanim_gnm/gem-x rev-parse HEAD
git -C .cache/autoanim_gnm/gem-x submodule status
du -sk .cache/autoanim_gnm/gem-x
ffprobe -v error -show_entries \
  format=duration,size,bit_rate:stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels \
  -of json artifacts/production-next-video/01kxw83zz4rawnm0j3dyask1pz/input.flv
shasum -a 256 artifacts/production-next-video/01kxw83zz4rawnm0j3dyask1pz/input.flv
```

Submodule attempts:

```bash
git submodule update --init --recursive

git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive
```

The first failed at the SSH GitHub URL. The second did not complete before
interruption. Setup and model download were then stopped. The strict blocker is
therefore incomplete source dependencies plus an unsafe remaining disk margin
under the 12 GB gate—not an inference exception.

## Smallest safe continuation

First make the three submodules match the superproject pins and verify that
`git submodule status` has no `+`, `-`, or dirty marker. Then run the documented
setup without the RAM disk:

```bash
cd /Users/abhi_macbook/Projects/apps/AutoAnim/.cache/autoanim_gnm/gem-x
bash scripts/setup_mac.sh --no-ramdisk
```

Before downloading models, set a measured project-local Hugging Face cache and
ensure at least 13 GB additional free working space (8.3 GB final assets plus
download/Xet temporary headroom). Once setup and assets are complete, the
smallest real-video inference command is:

```bash
cd /Users/abhi_macbook/Projects/apps/AutoAnim/.cache/autoanim_gnm/gem-x
./.venv/bin/python scripts/demo/demo_soma_onnx.py \
  --video /Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/production-next-video/01kxw83zz4rawnm0j3dyask1pz/input.flv
```

Success must require exit code zero, non-empty motion output for all decoded
frames, finite SOMA parameters, a recorded ONNX provider for every session,
output file hashes/sizes, and measured wall time. Because the fixture is
head-and-shoulders only, a detector/no-full-body rejection is a legitimate
failed result and must not be relabeled as a passing inference.
