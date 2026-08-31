#!/usr/bin/env python3
"""The head fit, with Apple Vision's ears added to the objective.

§7.4's pre-registered recommendation, and §6a's first named next step. The reasoning is
already measured and is not a guess:

  * §4 -- Apple Vision's ears are the widest and best-conditioned head baseline on this
    footage (158-190 mm, 6.4x and 5.5x tighter in length than SOMA-77's skull axis), and
    they survive a null template that predicts them from nose, eyes and neck.
  * §1 -- but their *per-frame* yaw scatters at 21-41 deg, so they cannot drive a
    rotation channel directly.
  * §2b -- and their 2D are epipolar-consistent at 0.29-0.76x their own body control.

Excellent 2D, wide baseline, noisy per frame: that is precisely an observation for a
multi-frame fit to average, which is where §7.4 said they belong. SOMA-77's five head
points span the skull but are all within ~145 mm of its centre; the ears add the widest
lateral pair available, which is the axis yaw is least constrained about.

**Cross-detector subject matching is derived, never assumed** -- the same lesson
`subject_map.py` banked after MAMMA's indices turned out to be crossed. Apple Vision's
subjects are matched to ours by 3D root agreement, and the margin is asserted.

Blind to: convention. An Apple Vision ear and a SOMA-77 joint centre are different
points under different definitions, but the fit does not need them to share one -- each
landmark gets its own free position in the rigid template, so a convention offset is
absorbed as template geometry rather than error. What it *does* assume is that both
detectors' points are rigid with respect to the skull, which is true of ears and of
skull joints alike.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, load_camera_rig, load_observation_jsonl,
)
from associate import CAMERAS, OUT, RIG  # noqa: E402
from solve_head import (  # noqa: E402
    NAMES as SOMA_NAMES, gather, held_out_px, initialise, log_so3, rodrigues, solve,
)
from triangulate_soma import triangulate  # noqa: E402

EARS = ("right_ear", "left_ear")
WEIGHTS = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
TOLERANCE = 0.10
MINIMUM_MARGIN = 5.0


def apple_vision_ears() -> np.ndarray:
    """[our_subject, frame, camera, ear, 3] of (x, y, confidence), subjects matched to ours."""
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(
        f"artifacts/commercial-multiview-soma77/work/{name}-observations.jsonl") for name in CAMERAS]
    cameras = [rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
               for name in CAMERAS]
    calls: list[np.ndarray] = []
    real = cm.associate_frame_graph

    def recording(*args, **kwargs):
        associated, cost = real(*args, **kwargs)
        calls.append(np.array(associated, copy=True))
        return associated, cost

    _, _, _, raw = cm.reconstruct_multiview(cameras, observations, subject_count=2,
                                            sample_rate_hz=30, associator=recording)
    associated = np.stack(calls)  # [frame, subject, camera, joint, 3]

    # --- match Apple Vision's subjects to ours, by 3D root agreement ------------
    ours = triangulate([0])[0][:, :, 0]  # SOMA Hips, [subject, frame, 3]
    distances = np.zeros((2, 2))
    for mine in range(2):
        for theirs in range(2):
            root = raw[theirs][:, JOINT_INDEX["root"]]
            both = np.isfinite(root).all(axis=1) & np.isfinite(ours[mine]).all(axis=1)
            distances[mine, theirs] = float(
                np.median(np.linalg.norm(root[both] - ours[mine][both], axis=1)) * 1000.0)
    straight, crossed = distances[0, 0] + distances[1, 1], distances[0, 1] + distances[1, 0]
    mapping = {0: 0, 1: 1} if straight < crossed else {0: 1, 1: 0}
    margin = max(straight, crossed) / max(min(straight, crossed), 1e-9)
    if margin < MINIMUM_MARGIN:
        raise SystemExit(f"Apple Vision subject match ambiguous ({margin:.2f}x)\n{distances}")
    print(f"apple vision subject match: {mapping}  margin {margin:.1f}x  "
          f"(matched median {min(straight, crossed) / 2:.1f} mm)")

    frames = associated.shape[0]
    out = np.full((2, frames, len(CAMERAS), len(EARS), 3), np.nan)
    for mine, theirs in mapping.items():
        for slot, name in enumerate(EARS):
            out[mine, :, :, slot] = associated[:, theirs, :, JOINT_INDEX[name]]
    return out


def main() -> None:
    ears = apple_vision_ears()
    report: dict = {"rule": "largest weight within +10% in-frame reprojection of weight 0",
                    "landmarks": list(SOMA_NAMES) + list(EARS)}
    output: dict[str, np.ndarray] = {}
    for subject in range(2):
        soma, cameras = gather(subject)
        observations = np.concatenate([soma, ears[subject]], axis=2)
        seen = np.isfinite(observations[..., :2]).all(axis=3) & (observations[..., 2] >= 0.25)
        print(f"subject {subject}: ear observations {int(seen[:, :, 5:].sum())} of "
              f"{seen[:, :, 5:].size}, soma {int(seen[:, :, :5].sum())} of {seen[:, :, :5].size}")

        template0, rotations0, translations0 = initialise(subject)
        # Seed each ear 80 mm out along the template's own eye axis -- a plausible
        # starting point that the fit is entirely free to move, since every template
        # position is a free parameter. Nothing here constrains the answer.
        ear_seed = np.zeros((len(EARS), 3))
        lateral = template0[SOMA_NAMES.index("LeftEye")] - template0[SOMA_NAMES.index("RightEye")]
        lateral = lateral / max(np.linalg.norm(lateral), 1e-9)
        ear_seed[0] = -0.080 * lateral
        ear_seed[1] = +0.080 * lateral
        template = np.vstack([template0, ear_seed])

        curve: dict[float, float] = {}
        fits: dict[float, dict] = {}
        every = np.ones(len(CAMERAS), dtype=bool)
        for weight in WEIGHTS:
            fit = solve(observations, cameras, template, rotations0, translations0, weight, every)
            fits[weight] = fit
            curve[weight] = float(np.nanmean(
                [held_out_px(fit, observations, cameras, c) for c in range(len(CAMERAS))]))
            print(f"  subject {subject}  weight {weight:8.1f} -> in-frame {curve[weight]:.3f} px")
        ceiling = curve[0.0] * (1.0 + TOLERANCE)
        best = max(w for w in WEIGHTS if curve[w] <= ceiling)
        fit = fits[best]
        output[f"subject_{subject:02d}_head_world"] = rodrigues(fit["rotations"])
        output[f"subject_{subject:02d}_head_position_m"] = fit["translations"]
        output[f"subject_{subject:02d}_template_m"] = fit["template"]
        report[f"subject_{subject:02d}"] = {
            "in_frame_px_by_weight": {str(k): v for k, v in curve.items()},
            "chosen_weight": best, "chosen_in_frame_px": curve[best], "ceiling_px": ceiling,
            "ear_separation_mm": float(np.linalg.norm(
                fit["template"][5] - fit["template"][6]) * 1000.0),
            "template_extent_mm": {
                name: float(np.linalg.norm(fit["template"][i]) * 1000.0)
                for i, name in enumerate(list(SOMA_NAMES) + list(EARS))},
        }
        print(f"subject {subject}: chose {best} at {curve[best]:.3f} px, "
              f"fitted ear separation {report[f'subject_{subject:02d}']['ear_separation_mm']:.1f} mm")
    np.savez(OUT / "head-solve.npz", **output)
    (OUT / "head-solve-ears.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
