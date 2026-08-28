# Battle 1, increment 4 — the detector that was already on disk

Status: landed 2026-08-28. The largest single accuracy improvement of the lane
so far, and a partial correction to Battle 0's framing.

## What happened

The GPU survey concluded that no commercially-clean whole-body 2D pose
checkpoint exists. It flagged one unverified exception. The exception holds, and
it was sitting in `.cache` the whole time.

NVIDIA **GEM-X** bundles `Dinov3_ViTPose_huge_metrosim_256x192` — a ViT-Huge
emitting **77 SOMA keypoints**: body, feet, face, and 15 articulated finger
joints per hand. Confirmed by inspecting the ONNX graph directly:
`heatmaps [1, 77, 64, 48]`.

## The result

Identical frames, identical calibration, identical reconstruction.

| detector | valid | interp | reprojection | 2D error at subject | bone instability | temporal rejections |
|---|---:|---:|---:|---:|---:|---:|
| Apple Vision (19) | 88.2% | 6.0% | 4.59 px | 25.6 mm | 9.4% | 14 |
| MediaPipe (33) | 32.6% | 65.1% | 3.05 px | 17.0 mm | 50.8% | 133 |
| SOMA-77, clipped box *(defective — see below)* | 88.9% | 10.8% | 2.82 px | 15.7 mm | 5.5% | 0 |
| **GEM-X SOMA-77, corrected** | **89.1%** | 10.8% | 2.91 px | **16.2 mm** | **3.5%** | **0** |

Against Apple Vision: **2D error at the subject −37%**, **bone-length instability
−63%**, **temporal rejections 14 → 0**, valid joints slightly up.

Note what the box fix did and did not do. Reprojection got marginally *worse*
(2.82 → 2.91 px) while bone instability improved sharply (5.5% → 3.5%). That is
the expected shape: the clipped version's flattering reprojection was partly the
same survivorship pattern seen twice before in this lane — border-pinned joints
are consistent with each other while being wrong. Bone length is a physical
invariant and cannot be gamed that way, which is why it is the more trustworthy
of the two numbers here.

**2D error at the subject falls 25.6 mm → 16.2 mm, a 37% reduction** — on the
exact quantity Battle 0 identified as Apple Vision's model-limited ceiling.

> **Basis, stated so it is not over-quoted.** These millimetres are derived from
> the *reprojection residual*, which is both fitted and gate-censored, exactly as
> flagged throughout `BATTLE0_DETECTOR_WIDTH_FINDINGS.md`. It is a relative
> comparison between detectors on identical footage, not a ground-truth-verified
> accuracy. No accuracy claim in this lane is ground-truth-verified until Battle
> 2 delivers an owned reference capture.

**Bone-length instability falls 9.4% → 3.5%** — a 63% gain, and still short of
the 2% gate, which moved to Battle 4 in increment 2 precisely because no
detector then available could approach it. 3.5% narrows that gap a long way; it
does not close it.

 That was the hypothesis: this
model predicts *skeletal joint centres*, where Apple Vision and MediaPipe predict
*surface* landmarks. Surface landmarks slide relative to bone as a subject turns,
which is the mechanism increment 2 identified behind the instability. Interior
joint centres do not slide. The measurement supports the mechanism.

**Temporal rejections fall 14 → 0.**

### The one number it appears to lose on is not real

Its interpolated fraction is 10.8% against Apple Vision's 6.0%. That is entirely
structural: SOMA-77 has no ear landmarks, and the two ears are 2 of our 19 slots
= 10.5%. Per-joint:

> over the 17 joints SOMA-77 actually emits: **99.3% direct triangulation**

against Apple Vision's 88.2% over 19. It is not losing coverage; it is declining
to invent two joints it does not model. The reconstruction already falls back to
the neck for missing head landmarks, and GNM owns the head regardless.

Note this is **not** the survivorship pattern that made MediaPipe's reprojection
look good — there, coverage collapsed to a third. Here coverage is *higher* and
error is *lower* at the same time.

## Licence position

Verified against primary sources before use.

- **Code** Apache-2.0. **Weights** NVIDIA Open Model License — *not* Apache, as
  GEM-X's own `docs/MODEL_OVERVIEW.md` incorrectly claims. The README and
  HuggingFace card both say: *"Use of the source code is governed by the Apache
  License, Version 2.0. Use of the associated model is governed by the NVIDIA
  Open Model License Agreement."*
- The OML is genuinely permissive: *"Models are commercially usable"*, *"NVIDIA
  claims no ownership rights in outputs"*. Conditions: attribution passthrough on
  redistribution, Trustworthy AI compliance, termination on guardrail
  circumvention or litigation. The Trustworthy AI clause that touches us bars
  *illegal* biometric processing without consent — a compliance obligation, not a
  field-of-use bar on mocap.
- **Training data:** *"GEM is trained exclusively on internally generated
  synthetic video data"* — Bones RigPlay-1 (NVIDIA-owned), RenderPeople
  (NVIDIA-licensed), 3,500 internal synthetic characters. *"All training data and
  the SOMA parametric body model are owned by NVIDIA or released under permissive
  licenses, making GEM ready for commercial use."*

**Two encumbrances avoided by construction.** GEM-X's bundled person detector is
YOLOX trained on **Human-Art**, authorised "for non-commercial purposes" only —
so our worker takes boxes from the Apple Vision detections we already have,
keeping that checkpoint out of the path. And the `--no-imgfeat` ONNX path never
touches Meta's SAM materials.

**One open question, worth an email to NVIDIA.** The backbone is architecturally
Meta's DINOv3 ViT-H+/16 and nothing public states whether it was initialised from
Meta's pretrained weights. If it was, DINOv3's agreement flows down alongside the
OML — commercial use still permitted, redistribution must carry it, and
"NVIDIA-owned only" would describe the fine-tuning data rather than the
pretraining corpus.

## A defect the adversarial review found in this worker

The 7.1 px cross-detector agreement validated the transform on **one person in
one frame**, which was not tight enough. A 33-agent review, worktree-isolated,
found a real error in my box construction.

The model consumes only the centre 192 of the 256 crop columns. My `_square_box`
sized a square on `max(width, height)`, which gives the subject a horizontal
field of view of only `0.75 x 1.25 x max(w, h)` — narrower than a wide pose.
GEM-X's reference `get_bbx_xys` fits the hull to the 192:256 ratio *before*
enlarging, so its horizontal FOV is always at least 1.2x the hull width.

Measured on the fixture before the fix:

| | |
|---|---:|
| person-frames losing at least one joint out of the input | **353 / 1179 (30%)** |
| joints outside the model input | 1067 / 18156 (5.9%) |
| clipped joints pinning to the heatmap border (within 4 px) | **38.2%**, vs 1.8% in-view — a 21x enrichment |
| clipped joints that still passed the 0.25 confidence gate | 410 |

A joint outside the input cannot receive a heatmap peak, so the argmax saturates
at a border cell. Confidence does partially signal it — median 0.454 for clipped
joints against 0.930 in-view, and the gate discarded ~31% of them — so it is not
wholly silent, but 410 wrong joints still reached triangulation.

Fixed by fitting the hull to the visible window before enlarging:
`side = max(height, width * CROP / MODEL_WIDTH) * BOX_PADDING`. After the fix,
**0 of 1179 person-frames and 0 of 18156 joints are clipped.**

**The headline numbers above were measured with this defect present** and are
re-baselined below. The review also notes the fix is not a uniform win: growing a
wide hull's box can pull a neighbouring performer into the crop in a two-person
scene, so the improvement should be measured rather than assumed.

### One divergence that is not a bug

`_crop` maps the box across 255 columns while `_decode` assumes 192/256 of the
box side — a 256/255 inconsistency. It is inherited from the reference, which
does exactly the same thing, so our worker reproduces GEM-X bitwise. Left alone
deliberately: matching the reference is worth more than being 0.4% more
self-consistent than it.

## Correctness of the reimplementation

The worker is written against the ONNX graph and published preprocessing rather
than importing GEM-X, so it depends only on numpy, cv2 and onnxruntime. The
transform is validated against an independent detector: on the same person in
the same frame, its keypoints agree with Apple Vision's to a **median of 7.1 px**.
A wrong affine or decode would be off by hundreds.

One bug caught in passing: an early draft pushed heatmap peaks through a sigmoid.
The peaks already lie in [0.895, 0.992], so that flattened every joint to ~0.72
and destroyed what discrimination the confidence carried.

## What this corrects

Battle 0 concluded Apple Vision is model-limited and no rig change reaches past
it. That stands. But the framing around it — that the detector ceiling was a
property of the *category* — was too broad. A better model existed, was
commercially licensed, and was already on disk. **Measuring the alternative
should have come before concluding the ceiling was general.**

## Integration, and the full acceptance gate

The build script now takes `--detector {apple_vision,soma77}`. The soma77 path
runs Apple Vision first for person boxes, then SOMA-77 for keypoints;
`run-report.json` records which detector produced the artifact, read from the
observations rather than the CLI argument.

Both artifacts pass `verify_commercial_multiview_artifact.py`. Gate by gate:

| gate | Apple Vision | SOMA-77 | |
|---|---:|---:|---|
| valid joint fraction | 0.8823 | **0.8912** | ↑ |
| interpolated fraction | 0.0602 | 0.1077 | ↑ — the two ears SOMA-77 does not model |
| median reprojection | 4.589 px | **2.913 px** | **−37%** |
| p95 reprojection | 10.877 px | **6.334 px** | **−42%** |
| max reprojection | 19.663 px | 18.228 px | ↓ |
| temporal rejections | 14 | **0** | |
| **retarget endpoint median** | 157.0 mm | **120.0 mm** | **−24%** |
| **retarget endpoint p95** | 359.7 mm | **249.0 mm** | **−31%** |
| status | pass | **pass** | |

The retarget endpoint pair is the most product-relevant row: it measures how far
the reconstructed joints sit from the fitted rig, which is what an animator
actually sees. `COMMERCIAL_MULTIVIEW_BODY.md` set those at ≤180 mm median and
≤400 mm p95 and called them "intentionally loose, P0 only". At 120 / 249 mm they
are now comfortably inside, on the same fixture.

## Actions that need you, not me

| action | why | when |
|---|---|---|
| **Email NVIDIA** re: whether `vitpose.pth`'s DINOv3-architecture backbone was initialised from Meta's pretrained weights | Blocks nothing today. Gates our redistribution posture: if yes, DINOv3's agreement flows down alongside the NVIDIA OML | Before any redistribution |
| **Add the NVIDIA Open Model License attribution notice** to the ship checklist | The OML requires attribution passthrough on redistribution, alongside the Apache and CC notices already listed | Ship checklist |

## Follow-ups this opens

- **The pipeline is now two-stage**: Apple Vision for person boxes, SOMA-77 for
  keypoints. A clean dedicated person detector is still wanted; this is a working
  arrangement, not an intended design.
- **SOMA-77 emits 15 articulated finger joints per hand and we discard them.**
  That is precisely the missing input for the N5.1 hand blocker, which the
  pipeline map names as the forward path's release blocker. It deserves its own
  increment.
- **This is a CPU-bound ViT-Huge**: ~25 minutes for 600 frames × 2 people here.
  On a GPU it is seconds. That is the concrete case for the lifted on-device
  constraint.
- Battle 4's target moves. The question is no longer "beat 25.6 mm" but "beat
  15.7 mm with dense landmarks".
