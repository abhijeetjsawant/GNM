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

| detector | valid | + recovered | interp | reprojection | 2D error at subject | bone instability | temporal rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| Apple Vision (19) | 88.2% | 94.0% | 6.0% | 4.59 px | 25.6 mm | 9.4% | 14 |
| MediaPipe (33) | 32.6% | 34.9% | 65.1% | 3.05 px | — | 50.8% | 133 |
| **GEM-X SOMA-77** | **88.9%** | 89.2% | 10.8% | **2.82 px** | **15.7 mm** | **5.5%** | **0** |

**2D error at the subject falls 25.6 mm → 15.7 mm, a 39% reduction** — on the
exact quantity Battle 0 identified as Apple Vision's model-limited ceiling.

**Bone-length instability falls 9.4% → 5.5%.** That was the hypothesis: this
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
