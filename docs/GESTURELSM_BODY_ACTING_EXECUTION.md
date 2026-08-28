# Learned speech-driven body acting: execution plan and status

Status: local integration, SpeakType alignment, Modal four-seed inference,
corrected retarget, and GNM composition complete. All four corrected candidates
pass the automatic body-motion gates, but remain review-only because qualitative
animator approval and character attachment/look-development are outstanding.

This phase replaces the visibly rigid four-pose acting preview with a learned
holistic speech-motion candidate lane. It follows the Sol Pro recommendation
to falsify GestureLSM Shortcut-ReFlow before investing in a larger production
stack. Google GNM remains the sole owner of facial identity, expressions,
eye-local motion, jaw, lips, teeth, tongue, and oral contacts.

## Product boundary

```text
audio + supplied transcript + retained word alignment
        -> sealed hash-bound request
        -> isolated GestureLSM worker on RTX 3090, four seeds
        -> immutable native SMPL-X55 response
        -> trusted importer and independent FK validation
        -> direct rest-relative SMPL-X55 -> AutoAnim55 retarget
        -> derived foot contacts and bounded root/contact cleanup
        -> ranked but never auto-approved body candidates
        -> GNM face/body ownership-safe composition
        -> connected 55-joint GLB review artifact
```

An optional LLM may later emit a bounded semantic acting plan—intent, energy,
gesture families, spatial limits, stance, and emphasis regions. It must never
emit joint rotations, translations, IK targets, or per-frame keyframes.
GestureLSM has no native free-form direction conditioning, so the first test
does not pretend to condition it with a prompt. Direction-aware candidate
selection and bounded cleanup are the next adapter layer. SynTalker is the
planned direction-conditioned research lane if this baseline passes.

## Hardware split

- RTX 3090: pinned GestureLSM/SMPL-X inference and candidate generation.
- RTX 3070: reproducibility check after the 3090 run passes.
- Apple Silicon Mac: request preparation, validation, retarget, projection,
  GNM composition, review, editing, and export.

The 3090 is the first worker because 24 GB VRAM provides safer margin for the
denoiser, RVQ decoders, SMPL-X evaluation, and upstream rendering than the
3070's 8 GB.

## Implemented milestones

### M1 — Pinned worker and sealed boundary: complete locally

- GestureLSM repository pinned to
  `f52ac2f53dd1b99beb74bbcdbdf1a98118a36a70`.
- Hugging Face model revision pinned to
  `a59d981c1529f558a87a3a4c07b8b437cfa13ee8`.
- Shortcut-ReFlow denoiser, upper-body, hand, lower-body decoders, and upstream
  config are path-, size-, and SHA-256-pinned.
- Supplied transcript and TextGrid bypass upstream Whisper and MFA. The tested
  local lane uses SpeakType's FluidAudio Parakeet v3 word timestamps and does
  not require a ChatGPT subscription or OpenAI API key.
- Torch and upstream provider code remain outside the trusted app process.
- The importer rejects unknown/duplicate JSON members, symlinks, compressed or
  object NPZ arrays, path traversal, mismatched hashes, shapes, dtypes,
  hierarchy, pins, request identity, timebase, and independent-FK errors.

### M2 — Direct retarget and candidate contract: complete locally

- Exact SMPL-X55 hierarchy and semantic mapping are versioned.
- Jaw and both eye joints are stripped because GNM owns them.
- Rest-relative global rotation deltas are transferred into AutoAnim55; this
  avoids direct local-axis copying and preserves all 30 target finger joints.
- SMPL-X anatomical-left `+X` is explicitly reflected into AutoAnim
  anatomical-left `-X` for both root translation and world-rotation deltas.
  A regression test proves a native downward shoulder pose cannot become a
  raised target arm.
- Target-neutral input maps to target neutral, quaternion norms/sign continuity
  are checked, and FK error must remain within 2 mm.
- Raw and projected tracks are retained separately.
- The virtual target `Root` is intentionally unmapped; SMPL-X pelvis rotation
  belongs on target `Hips`. Mapping both target joints to pelvis consumed the
  motion on `Root` and left `Hips` falsely static. The semantic contract and a
  regression test now enforce `Hips -> pelvis` with no duplicate `Root` source.
- Candidate ranking reports body/finger activity and duplicates, but never
  silently approves a candidate.

### M3 — GNM fusion and review export: complete locally

- `build_audio_acting_shot.py` accepts a sealed request plus one or more
  validated GestureLSM responses.
- The body track is composed on the same exact 48 kHz timeline as the retained
  GNM face arrays.
- GestureLSM's 208 learned samples are shortest-arc quaternion-resampled onto
  the authoritative 211-sample GNM clock; foot contacts are discarded and
  re-derived after resampling.
- The connected character GLB has all 55 body/finger joints.
- Learned shots remain blocked from publishing until an animator approves a
  specific candidate and body identity/attachment calibration is complete.
- The deterministic body compiler remains as the unchanged A condition.

### M4 — Real four-seed CUDA A/B: automatic gates pass; human review pending

SpeakType/Parakeet v3 transcribed the exact seven-second source locally as:

> Will you keep asking for the rules that fence out the newcomers while you
> scale behind them on tens of billions?

The result contained 21 words, word timings from 0.160 to 6.960 seconds, mean
confidence 0.997, and a 7.011-second source duration. The converter emits a
contiguous 24-interval TextGrid with explicit silence/PAD gaps, lower-cased
lexical labels, and an unmodified readable transcript. The sealed request uses
seeds 7, 11, 17, and 23.

Completed on Modal workspace `ragnarok-space`:

- L40S CUDA probe: 46,068 MiB VRAM, driver 580.95.05;
- exact source/model bootstrap into a persistent volume;
- face-VQ support pin: `ce9240dcf08b1629855708c00d0aef116dccfdd35a0f6e2eb1d4aba430ae4ac9`;
- neutral SMPL-X support pin: `bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74`;
- sealed request upload, GPU execution, and response download entrypoint.
- released 334-tensor generator checkpoint audited and loaded strictly;
- the released demo's 256/768 encoder mismatch, `audio`/`audio_onset` mismatch,
  broken DataParallel fallback, and optional MoviePy preview dependency fixed by
  hash-pinned compatibility patches;
- successful L40S audio-to-motion run, trusted response import, independent FK
  validation, direct retarget, projection, and candidate report generation;
- direct Python requirements frozen from the successful Python 3.10.13 image
  and SHA-256-bound in `provider-lock.json`.

The first Modal L40S run `ap-P32NuXVRGXn8aCbOL8M1I6` produced four distinct,
strictly loaded candidates in 72.18 seconds. It exposed a semantic hierarchy
error: both target `Root` and `Hips` consumed the source pelvis. That made Hips
look constant and double-counted travel through the hierarchy. Because the
mapping is part of the signed response contract, the responses were not reused.
A corrected four-seed run, `ap-1tDrVpzkvdvyO7MZq8cQHn`, regenerated all four
signed responses in 72.74 seconds against the corrected contract.

All corrected candidates have 52 nonconstant joints, all 30 finger joints
active, finite arrays, exact end ticks, and pelvis rotation spans from 1.097 to
1.550 rad. Their unbounded root travel was 0.279–0.344 m. A declared
restrained-standing projection now scales horizontal root drift to a 0.23 m
envelope before contact projection, while preserving vertical translation and
every joint rotation. Final travel is 0.230–0.233 m, ground penetration is zero,
and maximum contact/root correction is 10.95–19.31 mm. All four therefore pass
the current automatic body-motion gates; none is auto-approved.

The first visual render initially showed raised arms. Native SMPL-X evidence
put both hands 0.2–0.7 m below the head, proving the model output was not the
cause. The retarget omitted the SMPL-X `+X`-left to AutoAnim `-X`-left basis
reflection; after correction, target hands remain 0.15–0.71 m below the head
and the acting silhouette is structurally plausible. The corrected seed 7
review still contains an overly deep forward crouch and generic gestures. This
is a learned-performance/selection problem rather than a remaining axis or
skinning failure. Seed 7 is rendered for review but is not automatically or
manually approved.

## RTX execution

On the RTX 3090 host:

```bash
python scripts/bootstrap_gesturelsm.py \
  --install-root /opt/autoanim/providers \
  --install

PYTHONPATH=/path/to/AutoAnim/src:/path/to/AutoAnim \
conda run -n autoanim-gesturelsm python workers/gesturelsm/infer.py \
  --request /tmp/shot-001-request/request.json \
  --provider-root /opt/autoanim/providers/GestureLSM \
  --checkpoint /opt/autoanim/providers/GestureLSM/ckpt/shortcut_reflow.bin \
  --config /opt/autoanim/providers/GestureLSM/configs/shortcut_reflow_test.yaml \
  --smplx-model-directory /opt/autoanim/providers/GestureLSM/datasets/hub/smplx_models \
  --output-directory /tmp/shot-001-response
```

For an RTX installation, freeze `conda list --explicit`, hash every upstream
support artifact, and store the lock with the run report. Modal's successful
direct Python dependency set is already pinned in `requirements-modal.txt`;
the CUDA base image and all provider/model/support artifacts are separately
recorded by `provider-lock.json`.

## Acceptance gates

At least three of four candidates must satisfy all hard gates:

- body/audio end tick mismatch: 0;
- GNM-owned array changes: 0 bytes;
- nonconstant wrists, chest, pelvis, and at least 20/30 finger joints;
- planted-foot drift p95 <= 20 mm and maximum <= 40 mm;
- ground penetration p99 <= 10 mm;
- zero joint-limit violations after projection;
- no per-frame root correction greater than 50 mm;
- no duplicate numerical candidates;
- restrained-standing root travel <= 0.25 m.

Two animators then review deterministic A against blinded B candidates using
the same face, character, audio, camera, and lighting. Continue only if the
selected B is preferred in at least 75% of pairwise judgments, averages at
least 3.5/5 for naturalness, semantic appropriateness, and face/body timing,
and needs no more than 15 minutes of correction for seven seconds.

## Current verification evidence

- 143 focused alignment, provider, security, retarget, resampling, projection,
  composition, body, SOMA, Blender-worker, and GestureLSM tests pass.
- Python compilation passes for `src`, `scripts`, and the GestureLSM worker.
- Both compatibility patches apply cleanly to the exact pinned checkout.
- Modal L40S run `ap-2niSzHj5aogbihMOowGCzy` generated and downloaded a
  208-frame, full-window candidate for the seven-second request; all 334
  generator tensors and all 27 face-VQ tensors loaded strictly. The original
  demo's dropped tail (128 frames for this input) is covered by overlap/padding
  and the emitted end tick is exact.
- The trusted importer accepted the candidate with exact independent rigid FK;
  all arrays were finite, 52 joints were nonconstant, and all 30 finger joints
  were active.
- Full-timeline reproducibility run `ap-hNN6mYiSsZhxgV9x8Sj23Y` also passed.
  GPU numeric drift was bounded to 0.000490 rad (rotations), 0.164 mm
  (positions), and 0.029 mm (root translation); ticks were exact.
- A local contract-level end-to-end run produced a connected 211-frame,
  seven-second, 55-joint GLB using a real SMPL-X motion example from the
  GestureLSM repository. This proves the data path, not learned quality or
  audio synchronization.
- The real Parakeet/GestureLSM/GNM run produced a connected 211-sample GLB and
  a verified 7.011-second H.264/AAC review file at
  `artifacts/audio-acting-shot/acting-shot.mp4`. Its SHA-256 is
  `45425dd0d4b8ca0de587a7cb8df2b5527185df120d1e7fbe767f4e16ddf65a1d`.
- The repository-wide suite is currently blocked by a reproducible native
  MediaPipe abort in the pre-existing video timing test during face-landmarker
  construction. It is outside this body lane and is not being treated as a
  passing assertion.

The phase is not production-validated. The corrected real four-seed run passes
the automatic numerical gates, but blinded animator review has not happened and
the current seed 7 performance is not strong enough for approval. The next
work is semantic candidate planning/ranking, pose-quality rejection (including
excessive crouch), blinded review, and reproduction on the second RTX machine.
