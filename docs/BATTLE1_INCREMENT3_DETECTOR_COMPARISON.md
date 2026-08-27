# Battle 1, increment 3 — second detector, measured

Status: negative result, 2026-08-27. Plan: `docs/OWNED_BODY_CAPTURE_PLAN.md`.

## Two blockers cleared, one plan item closed

### The MediaPipe viability blocker was wrong

`docs/MAMMA_MULTIVIEW_EXECUTION.md` records MediaPipe 0.10.35 aborting in native
code on this host, and the plan carried the detector swap as blocked on that.

That abort is the **legacy** `mediapipe.python.solutions` graph — which is not
even importable in the installed build (`ModuleNotFoundError`). The **Tasks**
runtime is present, and its native graph was probed on this machine against a
`.task` asset already in `.cache`, so nothing was downloaded to find out:

```
TASKS GRAPH RAN OK — CPU via XNNPACK, no abort
```

### The model asset cleared the licence gate

Verified against primary sources before download: all three Pose Landmarker
bundles are **Apache-2.0**, stated in the model card Google links from the
official docs. Training data is Google's own — 30k consented images plus 85k
fitness images, **no third-party dataset named**. The shipped `.task` is a plain
zip containing two TFLite CNNs and **no GHUM model**; GHUM was used offline to
generate synthetic Z ground truth, so the gated GHUM licence does not attach to
the artifact.

Downloaded from pinned versioned URLs; sizes and MD5s match the independently
fetched values exactly (`full` 9,398,198 B / `83879689…`, `heavy` 30,664,242 B /
`453dec4d…`).

## A configuration error I caught before reporting it

The first run produced `No camera observes every requested subject` — MediaPipe
finding 0 or 1 person in most frames. Reported as-is that would have been a
false negative. It was **my configuration**: MediaPipe's 0.5 defaults are tuned
for a close-up single subject.

| detection/presence/tracking floor | frames with 2 people | mean people/frame |
|---|---:|---:|
| 0.5 (default) | 1 / 40 | 0.57 |
| 0.2 | 32 / 40 | 1.77 |
| **0.1** | **33 / 40** | **1.82** |
| 0.05 | 35 / 40 | 1.88 |

The worker now defaults to 0.1 and exposes the floor. The reconstruction side
already gates on confidence and rejects outliers, so it is better to let weak
detections through and let geometry judge them.

## The comparison

Identical frames, identical calibration, identical reconstruction. MediaPipe
`heavy` at a 0.1 floor.

| detector | valid | + recovered | interpolated | median reprojection | bone instability | temporal rejections |
|---|---:|---:|---:|---:|---:|---:|
| **Apple Vision** | **88.2%** | **94.0%** | 6.0% | 4.59 px | **9.4%** | **14** |
| MediaPipe heavy | 32.6% | 34.9% | 65.1% | 3.05 px | 50.8% | 133 |

**Apple Vision wins decisively.** MediaPipe resolves barely a third of the
joints, its limb lengths swing by half their own length, and it produces nearly
ten times the temporal rejections.

Its lower reprojection error is **survivorship**, the same artifact Battle 0
found: when only a third of joints survive the gates, the survivors are the easy
ones. Reprojection error over a differently-censored population is not
comparable — that lesson is now twice-learned in this lane.

Root cause: MediaPipe's multi-person mode is weak on a wide stage. Even at a 0.1
floor, camera D001 sees two people in only 38 of 150 frames.

## Caveat that limits what this can claim

MediaPipe has **no neck and no pelvis landmark**. Both are derived as
shoulder/hip midpoints. Apple Vision predicts both directly. That is a
definitional difference of exactly the class Battle 0 identified as the dominant
error term, so part of the gap is mapping, not model. It does not change the
verdict — a 55-point coverage gap is not a mapping artifact — but it means the
result should be read as "MediaPipe as we can map it" rather than as a clean
measurement of the underlying model.

## What this closes

Battle 1's "MediaPipe as the interim detector" item is **closed, negative**.
Apple Vision stays as the interim detector.

The deeper reading: both are small on-device models, and both are model-limited
in the same way Battle 0 described. Swapping one lightweight detector for another
was never going to reach the target. That is the third independent measurement
pointing at Battle 4.

## The constraint that just disappeared

The pipeline does **not** have to run on the MacBook — a bigger GPU is available
when quality requires it, and all NVIDIA inference in this project already runs
on Modal.

Every detector considered in Battle 1 — Apple Vision, MediaPipe — was chosen
under the on-device assumption. That assumption is gone, which changes the option
space rather than the conclusion: the models we could reach on-device are not
good enough, and now we are not restricted to them.

## Artifacts

- `workers/commercial_multiview/mediapipe_pose.py` — the second detector
- `autoanim.body-observations/1.1` — detector-neutral observation contract;
  1.0 still loads and reads as `apple_vision`
- `artifacts/mediapipe-detector/` — MediaPipe observations over the same frames
