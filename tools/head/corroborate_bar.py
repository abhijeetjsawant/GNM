#!/usr/bin/env python3
"""Is MAMMA's ~29 deg of head-on-torso motion performer motion, or fit drift?

`HEAD_ORIENTATION_MEASURED.md` §1 makes MAMMA's head-relative-to-thorax spread the
reason a head gate is not vacuous, and flags that no second instrument had confirmed it.
This is that instrument, and it is independent on both axes that matter:

  * a **different detector** -- Apple Vision's ears against MAMMA's own dense 2D;
  * a **different estimator** -- independent per-point triangulation against a
    parametric SMPL-X fit with a temporal prior.

Both sides compute the same physical quantity: the **signed yaw of the head's lateral
axis relative to the shoulder axis, about world up**. MAMMA's head lateral axis is its
eye axis, which is rigidly attached to the head joint by construction; Apple Vision's is
its ear axis, which `ear_null_template.py` shows is not a face template.

Neither side is truth. Agreement means two unrelated instruments see the same head
turning; disagreement means at least one is drifting, and the gate's premise weakens.

Blind to: a common-mode error. If both instruments inherit the same bias from the shared
camera geometry, they agree and are both wrong. Nothing here is an accuracy claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_map import mamma_index_for  # noqa: E402

MA3D = Path("artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground")
CACHE = Path("artifacts/head-lane/three-detector-cache.npz")
L_EYE, R_EYE, L_SHOULDER, R_SHOULDER, PELVIS, HEAD = 23, 24, 16, 17, 0, 15


def signed_yaw(lateral: np.ndarray, reference: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Signed angle from `reference` to `lateral`, measured about `up`, in degrees."""
    def flatten(v):
        v = v - np.outer(v @ up, up)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    a, b = flatten(reference), flatten(lateral)
    cross = np.cross(a, b) @ up
    return np.degrees(np.arctan2(cross, np.einsum("ni,ni->n", a, b)))


def main() -> None:
    blob = np.load(CACHE)
    av = blob["apple_vision"]
    # MAMMA's subject indices are NOT ours -- see subject_map.py. Pairing them by
    # index returns |r| = 0.21 / 0.03, i.e. nothing, and that was the first result
    # this script produced.
    mapping = mamma_index_for(av)
    report: dict = {}

    for subject in range(2):
        joints = np.load(MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        # Up, per system, derived rather than assumed: the mean pelvis->head direction.
        m_up = (joints[:, HEAD] - joints[:, PELVIS]).mean(axis=0)
        m_up /= np.linalg.norm(m_up)
        mamma = signed_yaw(joints[:, L_EYE] - joints[:, R_EYE],
                           joints[:, L_SHOULDER] - joints[:, R_SHOULDER], m_up)

        world = av[subject]
        ear = world[:, JOINT_INDEX["left_ear"]] - world[:, JOINT_INDEX["right_ear"]]
        shoulder = world[:, JOINT_INDEX["left_shoulder"]] - world[:, JOINT_INDEX["right_shoulder"]]
        ok = np.isfinite(ear).all(axis=1) & np.isfinite(shoulder).all(axis=1)
        vision = np.full(len(ear), np.nan)
        vision[ok] = signed_yaw(ear[ok], shoulder[ok], np.array([0.0, 0.0, 1.0]))

        both = ok & np.isfinite(mamma)
        a, b = mamma[both], vision[both]
        # Sign convention differs between the two world frames; report both and take
        # the magnitude, since the physical claim is co-variation, not handedness.
        r = float(np.corrcoef(a, b)[0, 1])
        report[f"subject_{subject:02d}"] = {
            "mamma_body_id": mapping[subject],
            "frames_compared": int(both.sum()),
            "mamma_yaw_deg": {"sd": float(a.std()), "range": float(a.max() - a.min())},
            "apple_vision_yaw_deg": {"sd": float(b.std()), "range": float(b.max() - b.min())},
            "pearson_r": r,
            "pearson_r_abs": abs(r),
            # A shuffled control: the same two series, one of them permuted. If the
            # correlation survives shuffling it was never about time.
            "shuffled_control_r_abs_p95": float(np.percentile(
                [abs(np.corrcoef(a, np.random.default_rng(seed).permutation(b))[0, 1])
                 for seed in range(200)], 95)),
        }

    Path("artifacts/head-lane/corroborate-bar.json").write_text(json.dumps(report, indent=2))
    for subject, row in report.items():
        print(f"{subject}: {row['frames_compared']:3d} frames  "
              f"|r| = {row['pearson_r_abs']:.3f}  (shuffled control p95 "
              f"{row['shuffled_control_r_abs_p95']:.3f})   "
              f"MAMMA yaw sd {row['mamma_yaw_deg']['sd']:5.1f}deg, "
              f"AV yaw sd {row['apple_vision_yaw_deg']['sd']:5.1f}deg")


if __name__ == "__main__":
    main()
