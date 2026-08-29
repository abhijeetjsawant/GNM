#!/usr/bin/env python3
"""Fourth 2D/3D body estimator: Meta's SAM 3D Body, predicting into MHR.

Unlike the other three workers this one does not emit 2D landmarks. It emits a
**parametric MHR body** — the same 127-joint rig `src/autoanim_gnm/data/mhr-skeleton-v1.json`
already describes, verified to match joint for joint. That matters because the
largest systematic term this lane has measured is SOMA-77's landmark convention
disagreeing with the reference (neck 39.9 px, ankles 27.8, hips 20-25, no ears at
all). A model that predicts *into our own body model* cannot disagree about
conventions; there are none to cross.

It is monocular and single-image, at 72.5 mm PVE on EMDB — an order of magnitude
worse than our four-camera geometry, so it is **not** a replacement for the
pipeline. It is a candidate for the stage the substitution harness measured as
dominant, and its per-view outputs are meant to be fused across the rig.

Two things it gives that our current stack cannot:

* `pred_global_rots` — per-joint world **rotations**. Our reconstruction estimates
  positions only, which is what blocked the local-frame calibration in the arm (i)
  work and what a skeletal animation needs anyway;
* `hand_pose_params` — hands from a learned prior, against the 27-DoF constrained
  chain increments 5 and 6 built by hand.

It also accepts our calibrated intrinsics directly, which retires the MoGe
field-of-view stage the reference ComfyUI workflow needs: guessing the camera is
their problem, not ours.

    python workers/commercial_multiview/sam3d_body_pose.py RIG.json BOXES.jsonl FRAME...

Requires the venv at `.cache/autoanim_gnm/sam-3d-body/venv` and the checkpoint
beside it. See `docs/BATTLE1_COMPONENT_SWAP.md`.

**Shims, and why they are here rather than in Meta's tree.** The upstream code
hardcodes CUDA in several places (`sam_3d_body_estimator.py:160`,
`sam3d_body.py:1034`, `:1246`) and there is no CPU or MPS path. The SAM License
forbids reverse engineering and a vendored checkout should stay pristine, so every
redirect lives here, in our code, and is reversible by deleting this file.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCHEMA_VERSION = "autoanim.body-mhr/1.0"
ESTIMATOR = "meta_sam_3d_body_dinov3"
PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / ".cache/autoanim_gnm/sam-3d-body/model.ckpt"
MHR_MODEL = PROJECT / ".cache/autoanim_gnm/gem-x/third_party/soma/assets/MHR/mhr_model_lod1.pt"
SOURCE = PROJECT / ".cache/autoanim_gnm/gem-x/third_party/sam-3d-body"
# Arrays worth keeping. The mesh is 18,439 vertices per person per frame and is not
# needed to score anything, so it is dropped unless asked for.
KEEP = ("pred_keypoints_3d", "pred_keypoints_2d", "pred_joint_coords", "pred_global_rots",
        "mhr_model_params", "body_pose_params", "hand_pose_params", "shape_params",
        "scale_params", "expr_params", "global_rot", "pred_cam_t", "focal_length",
        "bbox", "lhand_bbox", "rhand_bbox")


def install_cpu_shims(device: str = "cpu"):
    """Redirect the upstream's hardcoded CUDA onto a device that exists here."""

    import torch

    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)

    def to(x: Any, target: Any) -> Any:
        target = device if target == "cuda" else target
        if isinstance(x, dict):
            return {k: to(v, target) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to(v, target) for v in x)
        if torch.is_tensor(x):
            if target == "numpy":
                return x.detach().cpu().numpy()
            if x.dtype == torch.float64 and str(target).startswith("mps"):
                x = x.float()
            return x.to(target)
        return x

    import sam_3d_body.utils.dist as dist

    dist.recursive_to = to
    # Every importer bound the name at import time, so rebind each of them.
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("sam_3d_body") and hasattr(
            module, "recursive_to"
        ):
            module.recursive_to = to
    return to


def box_from_joints(joints: dict[str, Any], padding: float = 0.15) -> np.ndarray | None:
    points = [
        (float(j["x"]), float(j["y"]))
        for j in joints.values()
        if np.isfinite(j.get("x", np.nan)) and float(j.get("confidence", 0.0)) >= 0.2
    ]
    if len(points) < 4:
        return None
    array = np.asarray(points, dtype=np.float64)
    low, high = array.min(axis=0), array.max(axis=0)
    pad = padding * (high - low)
    return np.asarray([low[0] - pad[0], low[1] - pad[1], high[0] + pad[0], high[1] + pad[1]])


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(PROJECT / "src"))
    import torch

    install_cpu_shims("cpu")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from autoanim_gnm import commercial_multiview as cm

    rig = cm.load_camera_rig(Path(sys.argv[1]))
    boxes_by_frame: dict[int, list[dict[str, Any]]] = {}
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            boxes_by_frame[int(record["frame_index"])] = record.get("people", [])
    frames = [Path(v) for v in sys.argv[3:]]

    model, cfg = load_sam_3d_body(str(CHECKPOINT), device="cpu", mhr_path=str(MHR_MODEL))
    estimator = SAM3DBodyEstimator(model, cfg)

    import cv2

    for path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        index = int(path.stem)
        # The camera the frame came from is the one whose name the parent directory
        # carries, which is how every other worker in this lane is invoked.
        camera = next((c for c in rig if c.name == path.parent.name), rig[0])
        intrinsics = np.asarray(camera.scaled(width, height).intrinsics, dtype=np.float32)
        people: list[dict[str, Any]] = []
        for person in boxes_by_frame.get(index, []):
            box = box_from_joints(person.get("joints", {}))
            if box is None:
                continue
            out = estimator.process_one_image(
                str(path), bboxes=box.reshape(1, 4),
                cam_int=torch.from_numpy(intrinsics)[None],
            )
            if not out:
                continue
            record = {k: np.asarray(out[0][k]).tolist() for k in KEEP if k in out[0]}
            record["index"] = len(people)
            people.append(record)
        print(json.dumps({
            "schema_version": SCHEMA_VERSION, "estimator": ESTIMATOR,
            "frame_index": index, "width": width, "height": height,
            "camera": camera.name, "image_path": str(path), "people": people,
        }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
