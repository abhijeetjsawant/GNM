#!/usr/bin/env python3
"""Does the delivered foot ORIENTATION carry any foot information?

`LeftFoot` and `RightFoot` are not constants in the shipped track -- they turn, median
18-32 deg of local rotation. That looks like a solved foot and a naive "does it move?"
gate would pass it. But the pipeline's 19 landmark targets end at the **ankle**: there is
no toe target, so nothing observed constrains the foot's rotation about the ankle. Whatever
those channels carry, it is not measured foot orientation.

This tests it against the one independent measurement now available: the **triangulated
`Foot->ToeBase` direction** from SOMA-77's retained landmarks, which `tools/feet/toe_gate.py`
shows is stable to ~2x the leg controls in length.

  * If the delivered foot axis tracks the measured one, the channel carries real
    information and the toes would only refine it.
  * If it does not, the delivered foot orientation is **motion without information** --
    which is worse than a constant, because a constant is visibly degenerate and this is
    not.

Mean-removed, because the rig's rest convention and SOMA-77's skeletal convention need not
share a zero; what is being asked is whether the two axes turn TOGETHER, not whether they
point the same way. A constant offset is a convention, a varying one is an error.

Blind to: accuracy of either axis. This measures agreement between two of our own
estimates, so it can show the delivered channel is uninformed; it cannot show the toe
direction is right.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names, skeleton_for_track_dict  # noqa: E402
from triangulate_soma import triangulate  # noqa: E402

SOMA = {"LeftFoot": 69, "LeftToeBase": 70, "RightFoot": 74, "RightToeBase": 75}
ORDER = list(SOMA)
SLOT = {n: i for i, n in enumerate(ORDER)}
TRACKS = Path("artifacts/commercial-multiview-soma77")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main() -> None:
    positions, _, used = triangulate([SOMA[n] for n in ORDER])
    report = {}
    for subject in range(2):
        track = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")
        q = track["local_rotations_xyzw"]
        root = track["root_translation_m"]
        track_doc = json.loads(
            (TRACKS / f"subject-{subject:02d}.body-track.json").read_text()
        )
        names = track_doc["joint_names"]
        skeleton = skeleton_for_track_dict(track_doc)   # D3: the track's own rest
        world = forward_kinematics_positions(root, q, skeleton=skeleton)
        pos = positions[subject]
        block = {}
        for side, foot_j, toe_j in (("L", "LeftFoot", "LeftToes"), ("R", "RightFoot", "RightToes")):
            rig = unit(world[:, names.index(toe_j)] - world[:, names.index(foot_j)])
            soma_f, soma_t = f"{'Left' if side=='L' else 'Right'}Foot", f"{'Left' if side=='L' else 'Right'}ToeBase"
            meas = pos[:, SLOT[soma_t]] - pos[:, SLOT[soma_f]]
            ok = used[:, subject] & np.isfinite(meas).all(axis=1) & np.isfinite(rig).all(axis=1)
            m = unit(meas[ok])
            r = rig[ok]
            raw = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", m, r), -1.0, 1.0)))
            # remove the constant convention offset: rotate the rig axis by the mean
            # rotation that best aligns the two, then re-score. Kabsch on directions.
            U, _, Vt = np.linalg.svd(r.T @ m)
            d = np.sign(np.linalg.det(U @ Vt))
            R = U @ np.diag([1.0, 1.0, d]) @ Vt
            aligned = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", m, r @ R), -1.0, 1.0)))
            block[side] = {
                "frames": int(ok.sum()),
                "raw_angle_median_deg": float(np.median(raw)),
                "after_removing_constant_offset": {
                    "median_deg": float(np.median(aligned)),
                    "p95_deg": float(np.percentile(aligned, 95)),
                },
                # the control: what a CONSTANT rig axis would score against the same
                # measured axis, on the same frames. If the delivered channel cannot beat
                # this, its motion carries nothing.
                "constant_axis_control": None,
            }
            const = np.broadcast_to(unit(r.mean(axis=0)), r.shape)
            U, _, Vt = np.linalg.svd(const.T @ m)
            d = np.sign(np.linalg.det(U @ Vt))
            Rc = U @ np.diag([1.0, 1.0, d]) @ Vt
            ca = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", m, const @ Rc), -1.0, 1.0)))
            block[side]["constant_axis_control"] = {
                "median_deg": float(np.median(ca)), "p95_deg": float(np.percentile(ca, 95))}
        report[f"subject_{subject:02d}"] = block

    Path("artifacts/feet-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/feet-lane/delivered-foot.json").write_text(json.dumps(report, indent=2))
    print(f"{'':30s} {'median':>9s} {'p95':>9s}   {'CONSTANT ctrl':>20s}")
    for subject, block in report.items():
        for side, b in block.items():
            a = b["after_removing_constant_offset"]; c = b["constant_axis_control"]
            verdict = "carries information" if a["median_deg"] < c["median_deg"] - 1.0 else \
                      "NO BETTER THAN A CONSTANT"
            print(f"{subject} foot {side} ({b['frames']:3d} fr){'':6s} {a['median_deg']:8.2f}° "
                  f"{a['p95_deg']:8.2f}°   {c['median_deg']:8.2f}° / {c['p95_deg']:6.2f}°   {verdict}")


if __name__ == "__main__":
    main()
