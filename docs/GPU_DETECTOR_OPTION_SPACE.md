# What lifting the on-device constraint actually unlocks

Status: decision briefing, 2026-08-28. Prompted by the clarification that the
pipeline may run on a bigger GPU when quality requires it, rather than having to
run on the MacBook.

Source: a 72-agent survey with adversarial verification of every licence and
capability claim. 44 findings survived; 27 came back **restricted**, 12
**unverified**, 8 **clean**. The synthesis agent hit an account limit, so this
briefing is written from the verified findings directly.

## 1. The answer

**No commercially-clean whole-body 2D pose checkpoint exists in the open
ecosystem.** The menu was licence-gated, not compute-gated, so removing the
MacBook constraint unlocks **exactly zero new shippable checkpoints**.

The root taint is one dataset. COCO-WholeBody, verbatim: *"COCO-WholeBody
dataset is **ONLY** for research and non-commercial use."* Its 133-keypoint
skeleton *is* the whole-body format, so every open 133-kp checkpoint inherits it
by construction. RTMW's "Cocktail14" is contaminated four ways over
(COCO-WholeBody, Halpe, InterHand, UBody) and MMPose ships those weights with
**no weights licence at all** — silence is not a grant. ViTPose's whole-body
variants trace to the same sets. Sapiens v1 is CC-BY-NC. OpenPose is
academic-only. Ultralytics is AGPL-3.0.

**A trap worth naming:** HuggingFace licence tags launder nothing.
`usyd-community/vitpose-plus-base` carries 3.2M downloads under a permissive tag
over encumbered training data.

## 2. But three genuinely clean things did turn up

| asset | licence | why it matters |
|---|---|---|
| **Meta SAM 3D Body** | SAM License (19 Nov 2025) — **no** non-commercial clause, no MAU cap, no field-of-use restriction | Ships **commercially usable checkpoints**, and predicts into **MHR** — the Apache-2.0 body model this plan already chose. Verify independently before relying on it |
| **SAM 2** | Apache-2.0, **code and checkpoints** | The mask stage is clean. MAMMA's ablation put masks at 31.96 → 18.33 px on two-person 2D error |
| **Unity PeopleSansPeople** | Apache-2.0 | A privacy-preserving synthetic human data generator — relevant to Battle 3 |

Two more worth chasing, both currently **unverified**:

- **We may already own a clean whole-body 2D detector.** GEM-X reportedly bundles
  a ViTPose predicting 77 SOMA keypoints (body + hands) under NVIDIA's terms.
  If that holds, it is a materially better interim detector than Apple Vision's
  19 sparse joints. **Verify before building on it.**
- **SDPose-Body** (Sept 2025): MIT, *"trained exclusively on COCO-2017 train2017,
  no extra data"* — the narrowest data story found. Body-only, but COCO's own
  image-provenance caveat still applies.

**Sapiens2** is the one near-miss: its grant permits commercial use, but its
acceptable-use policy forbids surveillance applications, it disclaims all
warranty of non-infringement and pushes provenance risk onto the licensee, and
its training data is undisclosed. Not a foundation to build a product on.

## 3. What the GPU actually changes — with numbers

Less than it might seem, because the expensive half of the plan was **already**
written for Modal.

- **Inference is nearly free.** Our 150-frame, 4-camera, two-person shot costs
  **185–448 s of single-GPU time, i.e. $0.10–$0.49** at MAMMA's published rates.
- **Training a MAMMA-class ViT-Base dense-landmark model to 300k iterations is
  ~$600–720** on Modal. That is not the cost centre.
- **The cost centre is the synthetic corpus** — MammaSyn is 955k images from 2.8k
  rendered sequences — and its cost is published nowhere.
- **Dense landmarks cost ~80× sparse-keypoint inference.** That is exactly what
  the accuracy is bought with, and it is precisely what an on-device budget
  could never have afforded. This is the real unlock.
- Backbone capacity has sharply diminishing returns: **7.3× the parameters buys
  +3.3 AP**. Bigger is not the answer; *denser* is.
- **Input resolution to the pose network is worth more than a backbone step**
  (ViTPose-B, 224² → higher). That is the lever a per-person crop pulls — and
  note this is about the *crop*, not the source frame width Battle 0 ruled out.
- Top-down beats bottom-up by **5–6.5 AP** on crowded multi-person.
- More cameras is *not* where GPU money should go — MAMMA saturates around 12.

## 4. Plan changes

**Delete:** Battle 1's "MediaPipe as the interim detector". Measured worse than
Apple Vision on our own fixture (`BATTLE1_INCREMENT3_DETECTOR_COMPARISON.md`).
Two plan documents still recommended it; corrected.

**Restate:** "Battle 2 is the critical path for measurement" is too strong. It is
the critical path for the **final gate**. Reference-free progress on the detector
can start before we own a capture.

**Stop:** treating a per-person ROI crop as an Apple Vision improvement. Apple
Vision is model-limited; a crop is a lever on *our own* detector, in Battle 4.

**Start earlier:** a toy-scale end-to-end vertical slice of Battle 3 → Battle 4
on Modal — render a small synthetic corpus, train a small dense-landmark model,
measure the scaling trend. At ~$700 for a full training run, the information is
cheap relative to committing to the corpus blind.

## 5. Two honesty items this surfaced

- **Every accuracy measurement in Battles 0 and 1 was made on footage we have
  concluded we may not use commercially.** The MAMMA example fixture is the
  development fixture. This is known and recorded, but it bears restating: none
  of those numbers can appear in a commercial claim, and Battle 2 is what
  replaces them.
- **A restricted checkpoint is on disk**:
  `.cache/autoanim_gnm/gem-x/inputs/hub/checkpoints/yolox_x_8xb8-300e_humanart-a39d44ed.onnx`
  (396 MB), a person detector trained on Human-Art, in the GEM-X lane's runtime
  path. `.cache/` is gitignored so it has never reached a commit, but if the
  GEM-X lane ever ships, that detector ships with it. Needs a licence pass.
- Relatedly: **RTMDet has never had one.** Battle 1 names it for person boxes
  specifically to avoid AGPL YOLO, but the standing asset rule was never applied
  to it.

## 6. One more encumbrance in the MAMMA recipe

Reproducing MAMMA verbatim inherits encumbered weights at a point the earlier
research missed: *"For MammaNet and CameraHMR, we initialize the transformer
with ViTPose-B weights."* Our detector must initialise from something else, or
from scratch. Worth knowing before Battle 4 starts, not during.
