# NVIDIA body dependency ledger (N0)

Snapshot: `2026-07-29T17:50:36Z`

Scope: GEM-X, SOMA-X, SAM 3D Body, SOMA Retargeter, and candidate Kimodo
models for AutoAnim's body-motion pipeline

Evidence root:
`/Users/abhi_macbook/Projects/apps/AutoAnim/.cache/autoanim_gnm/gem-x`

## Release decision

**N0 status: BLOCKED for production; local GEM-X research preview executed.**

After this ledger's initial snapshot, the required SAM-3D-Body and SOMA-X
submodules were repaired to GEM-X's exact pins, the official macOS environment
was installed, and selected GEM-X/VitPose/SOMA model artifacts were downloaded
and locally hashed. A CPU-only `apple_silicon_preview` completed real-video
inference; exact hashes and results are in `docs/GEM_X_REPRODUCTION.md`.
SOMA Retargeter remains intentionally uninitialized because the executed path
does not request `--retarget`, and Kimodo remains absent.

Production use, packaging, redistribution, or deployment remains blocked
because:

1. several transitive asset terms are not yet mapped and dispositioned,
   including the custom SAM License, SOMA-shape's "proprietary" description,
   SMPL/SMPL-X-derived assets, and GarmentMeasurements/CAESAR lineage; and
2. the exact downloaded model bytes have hashes but do not yet have a complete
   accepted-license, commercial-use, hosting, redistribution, and derived-motion
   disposition;
3. the Apple path is a separately named CPU preview that skips SAM image
   features; the official Core ML path fails at runtime and no approved CUDA
   production worker has been qualified; and
4. Kimodo checkpoints, constraints runtime, and legal/security disposition are
   not present.

**No production approval may be issued until every unresolved model and
transitive term in this ledger has an owner, a written disposition, and
immutable evidence attached to the exact bytes being shipped.**

This document is an engineering evidence ledger, not legal advice or a license
grant.

## Evidence boundary

The detailed tables below preserve the bounded initial snapshot and must be
read as historical evidence. Subsequent execution repaired two required
submodules and downloaded only the selected model set documented in
`GEM_X_REPRODUCTION.md`; it did not accept a license or grant production use.

The status vocabulary is deliberately narrow:

| Status | Meaning |
|---|---|
| `VERIFIED_LOCAL` | The exact local bytes or Git object were inspected and identified by a content hash or commit OID. |
| `DECLARED_UPSTREAM` | An official publisher page makes the stated claim, but no matching local artifact was verified. |
| `NOT_PRESENT` | The dependency or artifact was not found in the bounded local search. |
| `UNRESOLVED` | Evidence exists, but ownership, license scope, compatibility, or reproducibility is not sufficiently dispositioned for production. |
| `BLOCKED` | The issue prevents N0 production approval. |

`VERIFIED_LOCAL` means provenance evidence was observed; it does **not** mean
the item is legally or technically approved.

## Local source ledger

### GEM-X superproject

| Field | Observed value |
|---|---|
| Official repository | [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) |
| Local origin | `https://github.com/NVlabs/GEM-X.git` |
| Local path | `.cache/autoanim_gnm/gem-x` |
| `git_commit_oid` | `32992550dba114c62243fb55e361311972dce8f9` |
| Tree OID | `9cbcabd8e352f15cecf2bb35972c4fb32d16d9a2` |
| Commit date | `2026-04-27T13:58:46-07:00` |
| Commit subject | `fix sam3db onnx auto-download and clean up demo docstring` |
| Tracking state | Local `main` equals the locally known `origin/main` |
| Fetch mode | Partial clone filter `blob:none` |
| Worktree state | Dirty through two submodules: `third_party/sam-3d-body` and `third_party/soma` |
| Status | `VERIFIED_LOCAL`, `BLOCKED` |

The local `origin/main` equality only proves equality to the last fetched
remote-tracking ref. It does not prove that the network repository has not
advanced since the last fetch.

### Exact submodule state

| Submodule | Official URL from `.gitmodules` | GEM-X gitlink pin | Observed local commit | Observed state | N0 status |
|---|---|---|---|---|---|
| `third_party/sam-3d-body` | `https://github.com/facebookresearch/sam-3d-body.git` | `b5c765a0d89d789985e186d396315e7590887b94` | `b5c765a0d89d789985e186d396315e7590887b94` | Commit matches, but 105 tracked paths are staged as deleted; the worktree occupies only 4 KiB. | `BLOCKED` |
| `third_party/soma` | `https://github.com/NVlabs/SOMA-X.git` | `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5` | `86632764684281dc98f31ab9c4aac36a4cdbc428` | Clean standalone checkout, but `+` in recursive submodule status means it does not match the GEM-X pin. | `BLOCKED` |
| `third_party/soma-retargeter` | `git@github.com:NVIDIA/soma-retargeter.git` | `b12d9a3eeff6ea64d7029684e47d1e92b9a60c2c` | none | `-` in recursive submodule status; directory is empty and uninitialized. The configured transport is SSH. | `BLOCKED` |

Observed standalone submodule metadata:

| Component | `git_commit_oid` | Tree OID | Commit date | Subject | Local `origin/main` |
|---|---|---|---|---|---|
| SAM 3D Body | `b5c765a0d89d789985e186d396315e7590887b94` | `7b5a2a23619659a679d64f096201c6504cbebe30` | `2026-02-19T10:16:00-08:00` | `add arXiv` | same OID |
| SOMA-X | `86632764684281dc98f31ab9c4aac36a4cdbc428` | `7f266658ae365df3748ba7d949c9eeff5f3e59ee` | `2026-06-05T19:56:16Z` | `Update public SOMA-X mirror` | same OID |

The currently present SOMA-X bytes therefore cannot be called the dependency
selected by GEM-X commit `32992550...`. A production manifest must contain
one explicit, tested combination rather than silently accepting the local
submodule drift.

### Components not cloned

| Component | Official source | Local result | Status |
|---|---|---|---|
| Kimodo | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | No local Git checkout found in the bounded workspace/cache search. | `NOT_PRESENT`, `BLOCKED` for a Kimodo-backed feature |
| Separate SOMA-X clone | [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) | No second checkout found; the only observed checkout is the GEM-X submodule above. | `NOT_PRESENT` |

## Local file and disk ledger

### Source and environment footprint

| Path | Observed allocated size | Note |
|---|---:|---|
| GEM-X tree including nested data | 2.8 GiB | Research cache, not a release artifact |
| GEM-X Git object storage | 1.1 GiB | Includes submodule object and LFS cache data |
| GEM-X `.venv` | 611 MiB | Incomplete environment |
| SOMA-X submodule | 1.1 GiB | Includes materialized LFS assets |
| SAM 3D Body worktree | 4 KiB | Effectively missing |
| SOMA Retargeter worktree | 0 B | Uninitialized |

The data volume had 926 GiB total, 824 GiB used, 52 GiB available, and 95%
reported capacity at the snapshot. This is a storage-pressure observation, not
a GPU-memory measurement. Model download and packaging must account for final
artifacts, Git/LFS duplication, Hugging Face/Xet cache, and temporary files.

### Content-addressed local control files

| File | Bytes | SHA-256 | Interpretation |
|---|---:|---|---|
| GEM-X `LICENSE` | 11,387 | `d1a7d615ab8eff4de143b1456f46dabf232f54daf0fcf9a70442bb6f637a9e95` | Apache License 2.0 text |
| GEM-X `ATTRIBUTIONS.md` | 13,656 | `cd2981c52f484c6f20506a01de356ca05cc016a741c8801785e88bae3763a4ee` | Bundled third-party notices, including the custom SAM License |
| GEM-X `requirements.txt` | 436 | `a9e4b4705516409a2acc74020e674b7a82cf36202907c12f6cf3411b2ba5902d` | CUDA-oriented pinned top-level requirements |
| GEM-X `scripts/setup_mac.sh` | 9,596 | `5f88b0368cc5084249fa1d688339f23cd74bf7d413d60690c0e5fcb3f8eddbde` | macOS bootstrap logic |
| GEM-X `.gitmodules` | 356 | `ecadd6280afc22db161e9e8198904d5869eb1962da18277cdeaed309b3b83803` | Submodule sources |
| SOMA-X `LICENSE` | 11,387 | `d1a7d615ab8eff4de143b1456f46dabf232f54daf0fcf9a70442bb6f637a9e95` | Apache License 2.0 text |
| SOMA-X `ATTRIBUTIONS.MD` | 22,239 | `cb4fe4ecc22c6f18f7085a14f0bad13ce5edf7a04e1427f1223196f09103d59e` | Anny and MHR attribution/license text |
| SOMA-X `README.md` | 16,947 | `adfaee5f3b863c29443d0f82256ebf8966c64c9ea98ed50160fd2f2938bd5bce` | Calls SOMA-shape a "proprietary PCA-based model" and says the codebase is Apache-2.0 |
| SOMA-X `docs/model_card.md` | 9,959 | `87df45fa9328e1a4329bf0bf76d8f3df510f5ed15be202d327ff8a95b4e15427` | Says SOMA is ready for commercial use and released under Apache-2.0 |

### Materialized SOMA-X model assets

These are the substantial local parameter/model artifacts. Their SHA-256
digests equal the corresponding Git LFS object IDs observed in the SOMA-X
checkout.

| Asset | Bytes | SHA-256 / LFS OID | Evidence status | License disposition |
|---|---:|---|---|---|
| `assets/SOMA_neutral.npz` | 27,084,937 | `0bfde60903533f2a4abc989579d0d161fb0dc8d5301ecd17fd4a9168be543083` | `VERIFIED_LOCAL` | `UNRESOLVED`: exact asset-to-license mapping required |
| `assets/correctives_model.pt` | 70,549,157 | `411d061a651051a7a4ec941543495d35eac21e5a7a69c8ce042b89cd12772a43` | `VERIFIED_LOCAL` | `UNRESOLVED`: trained/derived asset lineage must be recorded |
| `assets/MHR/mhr_model_lod1.pt` | 696,110,248 | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` | `VERIFIED_LOCAL` | `UNRESOLVED`: official MHR is Apache-2.0, but the exact shipped asset mapping still needs sign-off |
| `assets/MHR/mhr_model_lod6.pt` | 2,080,751 | `28839b7ee0eb38eb3943cf26cff61a306de78f86227a0f17bb044252c7bc06ac` | `VERIFIED_LOCAL` | same as above |
| `assets/SOMA_template_rig.usda` | 344,934,193 | `79f02176534ebc0671c76bf09203bbee8e1621f8b6d34cc63aec69b73460bba2` | `VERIFIED_LOCAL` | `UNRESOLVED`: exact asset-to-license mapping required |

Other materialized SOMA-X LFS content includes Anny, GarmentMeasurements, MHR,
SMPL, and SMPL-X wrap/base meshes plus animations and documentation media. The
presence of an LFS object proves local bytes exist; it does not resolve the
rights of upstream source assets or derived data.

## Model and checkpoint ledger

### Runtime models

| Model or bundle | Official location | Officially declared terms | Local byte evidence | N0 decision |
|---|---|---|---|---|
| GEM-X `gem_soma.ckpt` | [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X) | Model card labels the model `nvidia-open-model-license`; code is Apache-2.0. | `NOT_PRESENT`; `inputs/` is absent and no `.ckpt` was found. | `BLOCKED` |
| GEM-X ONNX bundle | [nvidia/GEM-X files](https://huggingface.co/nvidia/GEM-X/tree/main/onnx) | Same model repository and model-license declaration. | `NOT_PRESENT`; no `.onnx` or external ONNX data file was found. | `BLOCKED` |
| SAM 3D Body checkpoint/MHR bundle | [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) | SAM Materials are governed by Meta's custom SAM License. Access may require acceptance. | `NOT_PRESENT`; the submodule code worktree is also incomplete. | `BLOCKED` |
| Kimodo-SOMA-RP-v1.1 | [nvidia/Kimodo-SOMA-RP-v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1) | Publisher says "ready for commercial use" under the NVIDIA Open Model License; 282M parameters, 30-joint output, 30 fps, maximum 10 seconds. | `NOT_PRESENT`; no repo, revision, files, hashes, or license snapshot. | Candidate only; `BLOCKED` |
| Kimodo-SOMA-SEED-v1.1 | [nvidia/Kimodo-SOMA-SEED-v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1.1) | Publisher says "ready for commercial use" under the NVIDIA Open Model License; trained on the public BONES-SEED dataset. | `NOT_PRESENT`; no repo, revision, files, hashes, or license snapshot. | Candidate only; `BLOCKED` |
| Kimodo-SMPLX-RP-v1 | [official Kimodo model table](https://github.com/nv-tlabs/kimodo#kimodo-models) | NVIDIA R&D Model License, not the NVIDIA Open Model License. | `NOT_PRESENT` | **Excluded from every production configuration.** |

The Kimodo model cards describe Linux/Windows and NVIDIA Ampere, Blackwell, and
Lovelace as supported targets. The repository says full-GPU generation needs
about 17 GB VRAM, or under 3 GB when the text encoder is moved to CPU. Those are
publisher claims, not measurements on this Mac.

Kimodo's use of a 30-joint motion skeleton is not a contradiction with the
repository's `somaskel77` interchange: the production integration must preserve
the exact adapter and file schema selected by the application. No implicit
joint insertion or deletion is licensed by this ledger.

### Model-license snapshot gap

The current [NVIDIA Open Model License
Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
is marked "Last Modified: October 24, 2025." It declares commercial use and
derivative-model rights subject to its conditions, including:

- compliance with NVIDIA's [Trustworthy AI
  terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/);
- termination conditions involving litigation and guardrail circumvention;
- distribution of the agreement and the prescribed NVIDIA attribution notice;
- separate terms for separately licensed components; and
- the possibility of future updates for legal or regulatory requirements.

No local copy of that agreement was found tied to a GEM-X or Kimodo model
revision. Before download or use, the release process must archive the exact
license text accepted by the responsible legal entity, hash it, record the
acceptance time and actor, and bind it to the immutable model revision and file
digests.

## License and provenance matrix

### Direct code and model terms

| Dependency | Verified or official source | Term observed | Production disposition |
|---|---|---|---|
| GEM-X code | Local `LICENSE`; [official repository](https://github.com/NVlabs/GEM-X) | Apache-2.0 | Code term identified; NOTICE/attribution obligations still must be packaged. |
| GEM-X model | [official model card](https://huggingface.co/nvidia/GEM-X) | NVIDIA Open Model License | `UNRESOLVED`: no local model/license bytes, immutable revision, or acceptance record. |
| SOMA-X code | Local `LICENSE`; [official repository](https://github.com/NVlabs/SOMA-X) | Apache-2.0 | Code term identified; asset-level scope remains unresolved. |
| Kimodo code | [official repository](https://github.com/nv-tlabs/kimodo) | Apache-2.0 | `DECLARED_UPSTREAM`; no local source pin or dependency inventory. |
| Kimodo-SOMA RP/SEED models | Official model cards linked above | NVIDIA Open Model License | `UNRESOLVED`: candidate terms only until exact bytes/revision/license are recorded and accepted. |
| Kimodo-SMPLX model | [NVIDIA Internal Scientific R&D Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/) | Internal, scientific R&D in a non-production environment; production and distribution are prohibited. | `BLOCKED` and excluded from production. |
| SOMA Retargeter code | [official repository](https://github.com/NVIDIA/soma-retargeter) | Apache-2.0 | `DECLARED_UPSTREAM`; local submodule is uninitialized and transitive terms are unreviewed. |

### Transitive components and assets

| Component or lineage | Evidence | Observed term or claim | Required disposition |
|---|---|---|---|
| SAM 3D Body | Local GEM-X `ATTRIBUTIONS.md`; [official SAM license](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) | Custom SAM License, last updated 2025-11-19. It is not Apache-2.0. It includes redistribution, attribution for publications, law/privacy/trade-control, no-reverse-engineering, termination, and other conditions. | Legal review and written use/distribution decision; archive exact accepted license. The incomplete local checkout must not be shipped. |
| MHR | Local SOMA-X `ATTRIBUTIONS.MD`; [official MHR repository](https://github.com/facebookresearch/MHR) | Apache-2.0 upstream. | Map each bundled MHR asset/corrective to source, revision, and license; preserve notices. |
| Anny | Local SOMA-X `ATTRIBUTIONS.MD`; [official Anny repository](https://github.com/naver/anny) | Apache-2.0 is reproduced locally for Anny. | Map bundled wrap/base assets to an immutable upstream revision and preserve notice. |
| SOMA-shape | Local SOMA-X `README.md` and `docs/model_card.md`; [official SOMA-X repository](https://github.com/NVlabs/SOMA-X) | README calls it a "proprietary PCA-based model." The model card says SOMA is "ready for commercial use" and released under Apache-2.0, but the local files do not unambiguously map those statements to each LFS asset. | Obtain an authoritative asset-level mapping for `SOMA_neutral.npz` and related shape data. Retain the published commercial-use/Apache evidence, but do not infer that a root code license automatically settles every model/data asset. |
| SMPL-Body-derived interoperability assets | Local SOMA-X README acknowledgement and materialized `assets/SMPL/*` LFS files; [official SMPL-Body license](https://smpl.is.tue.mpg.de/bodylicense.html) | SMPL-Body is described as CC BY 4.0, with attribution requirements; the broader SMPL model has separate terms. | Identify which local objects are SMPL-Body outputs, derived data, or independently authored; satisfy credit and patent review; exclude separately licensed SMPL model files unless approved. |
| SMPL-X | SOMA-X README; [official SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html) | Default download license is for non-commercial research; commercial licensing is separate. SOMA-X says SMPL/SMPL-X model files are not redistributed and must be downloaded separately. | Production configuration must not download or package SMPL-X model files without a commercial agreement. Verify the lineage of bundled wrap/base assets independently. |
| GarmentMeasurements and CAESAR-derived shape data | SOMA-X README; [official GarmentMeasurements repository](https://github.com/mbotsch/GarmentMeasurements) | Upstream repository is GPL-3.0; SOMA-X says its PCA identity model was trained on the CAESAR dataset. | Legal review of code use, converted PCA data, dataset rights, redistribution, and whether this identity backend can be included. Keep disabled until cleared. |
| guided-diffusion | Local GEM-X `ATTRIBUTIONS.md`; [official repository](https://github.com/openai/guided-diffusion) | MIT | Preserve applicable notice; confirm whether code is actually included in the selected build. |
| PyTorch3D / ACTOR | Local GEM-X `ATTRIBUTIONS.md`; [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [ACTOR](https://github.com/Mathux/ACTOR) | BSD-3-Clause | Preserve applicable notices; confirm included files and versions. |
| YOLOX | Local GEM-X `ATTRIBUTIONS.md`; [official repository](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | Preserve applicable notice; pin actual included version. |
| ByteTrack | Local GEM-X `ATTRIBUTIONS.md`; [official repository](https://github.com/ifzhang/ByteTrack) | MIT | Preserve applicable notice; pin actual included version. |

Provider claims about training-data ownership or permissive licensing are
recorded as `DECLARED_UPSTREAM`; this audit did not independently verify
training-data contracts. Runtime consumers do not automatically acquire the
training datasets, and those datasets must not be copied into the product
unless separately approved.

## Runtime and reproducibility ledger

### Observed host

| Field | Observation |
|---|---|
| Host | MacBook Pro `Mac14,6` |
| SoC | Apple M2 Max, 12 CPU cores, 38 GPU cores |
| Memory | 32 GB unified memory |
| OS | macOS 26.2, build `25C56`, arm64 |
| Metal | Supported by the host |
| Python | 3.12.13, Clang 22.1.1 |
| `uv` | 0.11.2 |
| Git LFS | 3.7.1, Darwin arm64 |

### Observed Python environment

| Check | Result |
|---|---|
| PyTorch | 2.13.0 |
| Torchvision | 0.28.0 |
| NumPy | 2.5.1 |
| `torch.cuda.is_available()` | `False`; PyTorch reports no CUDA version |
| `torch.backends.mps.is_built()` | `True` |
| `torch.backends.mps.is_available()` | `False` |
| `gem` import | Present from the local source tree; reports `1.0.0` |
| `soma` import | Missing |
| `onnxruntime` import | Missing |
| `onnx` import | Missing |
| OpenCV import | Missing |
| Hugging Face Hub import | Missing |
| `pip` module | Missing from the virtual environment |
| Installed distributions | 13; normalized listing SHA-256 `55bd040acf5f491124a4ef5f979e8a1bc8fcf584c85797ff7bbcac98efe34736` |

The 13 distributions are Jinja2 3.1.6, MarkupSafe 3.0.3, filelock 3.32.0,
fsspec 2026.7.0, mpmath 1.3.0, networkx 3.6.1, NumPy 2.5.1, Pillow 12.3.0,
setuptools 83.0.0, SymPy 1.14.0, PyTorch 2.13.0, Torchvision 0.28.0, and
typing_extensions 4.16.0.

This environment is not a realization of the checked-in
`requirements.txt`, which requests CUDA 12.6 builds of PyTorch 2.10.0 and
Torchvision 0.25.0 plus NumPy 1.23.5 and many other packages. The repository
also has loose requirements in `setup.cfg`, while the macOS bootstrap installs
unversioned packages. No complete lockfile was found for GEM-X. Therefore the
environment is not reproducible and its currently importable packages do not
demonstrate a working GEM-X runtime.

No production inference, real-input execution, performance benchmark, or output
artifact was produced from this dependency snapshot.

## Unresolved issues register

| ID | Blocking issue | Required owner | Required evidence |
|---|---|---|---|
| `N0-SRC-001` | SOMA-X local commit differs from GEM-X gitlink. | Body integration engineer | Clean recursive source manifest with exact superproject and submodule OIDs; no `+`, `-`, or dirty marker. |
| `N0-SRC-002` | SAM worktree has 105 tracked deletions. | Body integration engineer | Clean checkout at the selected pin plus file manifest and tests. |
| `N0-SRC-003` | SOMA Retargeter is uninitialized and uses an SSH URL in `.gitmodules`. | Body integration engineer | Either initialize at the pin with an auditable source URL or formally exclude it from the product. |
| `N0-MDL-001` | GEM-X checkpoint/ONNX bytes are absent. | ML platform engineer | Immutable model repository revision, full filename/size/SHA-256 manifest, model card, license hash, and acceptance record. |
| `N0-MDL-002` | Kimodo candidate source/model bytes are absent. | ML platform engineer | Exact source commit, model revision and file hashes, validated install, output-schema test, and license acceptance record. |
| `N0-LIC-001` | SAM custom terms are not dispositioned. | Legal/compliance owner | Written decision covering use, packaging, redistribution, no-reverse-engineering, privacy, trade controls, and termination. |
| `N0-LIC-002` | SOMA model and bundled asset license mapping is incomplete. | Legal/compliance + asset owner | Per-file or per-directory provenance/SBOM for SOMA, SOMA-shape, correctives, MHR, Anny, SMPL/SMPL-X-derived data, and GarmentMeasurements. |
| `N0-LIC-003` | NVIDIA model-license bytes are not archived against exact model revisions. | Legal/compliance owner | Hashed license/Trustworthy-AI snapshot, accepting entity/actor/time, selected model IDs/revisions, and redistribution plan. |
| `N0-LIC-004` | Kimodo-SMPLX uses the R&D license. | Product owner | Permanent production denylist entry and configuration test proving it cannot be selected. |
| `N0-ENV-001` | Local environment is incomplete, unlocked, and has no usable PyTorch accelerator. | Runtime engineer | Locked environment, SBOM, vulnerability/license scan, import/provider smoke tests, and target-hardware benchmark. |
| `N0-CAP-001` | Remaining disk is 52 GiB at 95% capacity; download/cache duplication is unbudgeted. | Release engineer | Measured peak disk plan with cache location, required headroom, cleanup/recovery procedure, and package size budget. |

## N0 exit gate

N0 passes only when all boxes below are evidenced. A verbal assertion or a
successful import is insufficient.

### Source reproducibility

- [ ] Select and record one GEM-X `git_commit_oid`.
- [ ] Produce a clean recursive submodule manifest at the selected pins.
- [ ] Record the source archive/tree hashes used by CI and release packaging.
- [ ] Pin Kimodo source if Kimodo is enabled; otherwise prove it is excluded.
- [ ] Replace mutable branch references in manifests with immutable commit OIDs.

### Model reproducibility

- [ ] Choose the production GEM-X representation: checkpoint, ONNX bundle, or
  an explicitly versioned conversion of one of them.
- [ ] Record the immutable Hugging Face/NGC revision, every model filename,
  byte length, SHA-256, and download origin.
- [ ] Do the same for exactly one production Kimodo-SOMA model if acting
  generation is enabled.
- [ ] Verify hashes before loading and fail closed on a missing or changed file.
- [ ] Store provider-raw output manifests so conversions can be reproduced.

### License and provenance

- [ ] Archive and hash every applicable code, model, data, and asset license.
- [ ] Record the accepting legal entity, authorized actor, time, and model
  revision for click-through or custom model terms.
- [ ] Complete the transitive asset mapping described by `N0-LIC-002`.
- [ ] Produce release `LICENSES/`, `NOTICE`, attribution, source-offer, and
  model-notice artifacts as applicable.
- [ ] Confirm the production distribution model—local app, optional download,
  hosted inference, or bundled weights—against every term.
- [ ] Add a hard denylist for research-only assets, including
  Kimodo-SMPLX-RP-v1 and unlicensed SMPL/SMPL-X model files.
- [ ] Obtain written legal/compliance disposition for all `UNRESOLVED` rows.

### Runtime and supply chain

- [ ] Create a platform-specific lock with hashes for the target macOS runtime
  and, if used, the NVIDIA Linux worker runtime.
- [ ] Generate an SBOM containing Python, native, Git submodule, LFS, model, and
  separately downloaded assets.
- [ ] Run license and vulnerability scans and disposition every unknown,
  copyleft, custom, or restricted item.
- [ ] Verify the actual execution provider for every model session; do not infer
  acceleration from host Metal support.
- [ ] Run clean-machine install, offline re-install from the release cache,
  real-video inference, Kimodo generation, schema validation, and uninstall
  tests.
- [ ] Measure peak disk, memory/VRAM, latency, and package size on each supported
  target.

### Approval record

- [ ] Engineering signs the immutable source/model/runtime manifest.
- [ ] Security signs the SBOM, vulnerability disposition, model-loading policy,
  and artifact-integrity controls.
- [ ] Legal/compliance signs the license and provenance register.
- [ ] Product signs the enabled model/backends and excluded configurations.
- [ ] Release engineering attaches all evidence to a versioned N0 approval
  record.

Until every item passes, the release state remains **BLOCKED — research only**.

## Official references

- [GEM-X source](https://github.com/NVlabs/GEM-X)
- [GEM-X model repository and model card](https://huggingface.co/nvidia/GEM-X)
- [SOMA-X source](https://github.com/NVlabs/SOMA-X)
- [SOMA-X documentation](https://nvlabs.github.io/SOMA-X/stable/)
- [SOMA-X model card](https://nvlabs.github.io/SOMA-X/stable/model_card.html)
- [SAM 3D Body source](https://github.com/facebookresearch/sam-3d-body)
- [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)
- [SOMA Retargeter source](https://github.com/NVIDIA/soma-retargeter)
- [Kimodo source and official model table](https://github.com/nv-tlabs/kimodo#kimodo-models)
- [Kimodo documentation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/)
- [Kimodo-SOMA-RP-v1.1 model card](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
- [Kimodo-SOMA-SEED-v1.1 model card](https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1.1)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [NVIDIA Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)
- [NVIDIA Internal Scientific R&D Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/)
- [MHR source and Apache-2.0 declaration](https://github.com/facebookresearch/MHR)
- [Anny source](https://github.com/naver/anny)
- [SMPL-Body license](https://smpl.is.tue.mpg.de/bodylicense.html)
- [SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html)
- [GarmentMeasurements source and GPL-3.0 declaration](https://github.com/mbotsch/GarmentMeasurements)
