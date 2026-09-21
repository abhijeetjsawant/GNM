"""D4 B3 (REPORTED): every mapped MHR joint against our own captured landmarks.

This is `tools/compare/delivered_vs_capture.py`'s ROLE on the MHR joint set. That instrument
itself cannot run on this delivery, and the failure is measured, not asserted: it reaches
`d3_skeleton_gate.glb_joint_positions`, which assumes every skin joint carries a rotation
channel, and momentum's export gives 45 rotation channels to 127 joints -- `KeyError: 1`
(`artifacts/compare/d4-body/logs/22-b3-dvc-attempt.log`). Beyond that it is built on the
AutoAnim-55 rig: `PAIRS`, `rest_translations_m`, `BodyTrack`, `skeleton_for_track`. None of that
exists on an MHR track. So its *definitions* are reused here and its *code* is not:

  * the delivered file is read back **from its own bytes** (`d4_glb_closure.glb_joint_positions`,
    which reuses `d3_skeleton_gate`'s glTF primitives), never from the code path that wrote it;
  * the reference is `triangulated_world_positions_z_up_m` from the SAME delivery directory --
    the landmarks that delivery was fitted to, so only the delivery can move a figure;
  * the pairing is the delivered track's own declared `landmark_to_joint`, read out of the
    delivered JSON, not a resemblance of names.

BLIND TO, carried over verbatim because it is the same blindness: (a) orientation -- a joint on
its landmark can still be rotated about it; (b) the mesh -- this scores the SKELETON the file
carries, and the skin bound to it can balloon without moving a number (that is B1's job); (c) the
landmarks themselves, our own triangulation, so a common-mode detector error is inside the
reference; (d) whether a change is an improvement in the world.

AND ONE MORE, specific to this step: the locator offsets are PINNED at zero, so the fit is asked
to put MHR's joint exactly on our landmark. Where MHR's joint convention and SOMA-77's landmark
convention genuinely differ, that difference appears here as error and cannot be told apart from
a bad fit. FITTER_PLAN section 7 is the record of that under-fit; lane H's marker session owns it.

    .venv/bin/python tools/compare/d4_b3_placement.py --delivery DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
# allow_pickle below: this repository's own build output under the gitignored artifacts/ tree.
from d4_glb_closure import glb_joint_positions, to_capture  # noqa: E402

GROUPS = {"hips": ("left_hip", "right_hip"), "knees": ("left_knee", "right_knee"),
          "ankles": ("left_ankle", "right_ankle"), "neck": ("neck",),
          "shoulders": ("left_shoulder", "right_shoulder"),
          "elbows": ("left_elbow", "right_elbow"), "wrists": ("left_wrist", "right_wrist"),
          "root": ("root",), "head": ("nose",), "eyes": ("left_eye", "right_eye")}

# The performer's own measured segments, from the captured landmarks, against the same segment
# on the delivered skeleton. FITTER_PLAN section 7's table: expect ~20 mm mean absolute error,
# because pinned offsets refuse to model the convention gap they are pinned across.
SEGMENTS = (("left_upper_arm", "left_shoulder", "left_elbow"),
            ("right_upper_arm", "right_shoulder", "right_elbow"),
            ("left_lower_arm", "left_elbow", "left_wrist"),
            ("right_lower_arm", "right_elbow", "right_wrist"),
            ("left_upper_leg", "left_hip", "left_knee"),
            ("right_upper_leg", "right_hip", "right_knee"),
            ("left_lower_leg", "left_knee", "left_ankle"),
            ("right_lower_leg", "right_knee", "right_ankle"),
            ("shoulder_width", "left_shoulder", "right_shoulder"),
            ("hip_width", "left_hip", "right_hip"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    delivery = arguments.delivery.resolve()
    report = {"delivery": str(delivery), "reference": "triangulated_world_positions_z_up_m from "
                                                      "the same delivery", "subjects": {}}
    for subject in (0, 1):
        track = np.load(delivery / f"subject-{subject:02d}.body-track.npz", allow_pickle=True)
        declared = json.loads((delivery / f"subject-{subject:02d}.body-track.json")
                              .read_text(encoding="utf-8"))["landmark_to_joint"]
        names_capture = [str(n) for n in track["consumed_joint_names"]]
        capture = np.asarray(track["triangulated_world_positions_z_up_m"], np.float64)
        joint_names, positions = glb_joint_positions(delivery / f"subject-{subject:02d}.glb")
        world = to_capture(positions)
        frames = min(world.shape[0], capture.shape[0])

        per_landmark, per_group = {}, {}
        for landmark, joint in declared.items():
            reference = capture[:frames, names_capture.index(landmark)]
            delivered = world[:frames, joint_names.index(joint)]
            distance = np.linalg.norm(delivered - reference, axis=1) * 1000.0
            distance = distance[np.isfinite(distance)]
            per_landmark[landmark] = {"joint": joint,
                                      "median_mm": round(float(np.median(distance)), 3),
                                      "p95_mm": round(float(np.percentile(distance, 95)), 3),
                                      "frames": int(distance.size)}
        for group, members in GROUPS.items():
            values = [per_landmark[m]["median_mm"] for m in members if m in per_landmark]
            if values:
                per_group[group] = round(float(np.median(values)), 3)

        segments = {}
        for name, first, second in SEGMENTS:
            a, b = declared[first], declared[second]
            delivered = np.linalg.norm(world[:frames, joint_names.index(a)]
                                       - world[:frames, joint_names.index(b)], axis=1) * 1000.0
            measured = np.linalg.norm(capture[:frames, names_capture.index(first)]
                                      - capture[:frames, names_capture.index(second)],
                                      axis=1) * 1000.0
            keep = np.isfinite(measured) & np.isfinite(delivered)
            segments[name] = {
                "delivered_mm_median": round(float(np.median(delivered[keep])), 2),
                # the delivered bone is rigid; the spread says so rather than assuming it
                "delivered_mm_spread_p95_minus_p05": round(
                    float(np.percentile(delivered[keep], 95)
                          - np.percentile(delivered[keep], 5)), 2),
                "captured_mm_median": round(float(np.median(measured[keep])), 2),
                "abs_error_mm": round(float(abs(np.median(delivered[keep])
                                                - np.median(measured[keep]))), 2)}
        report["subjects"][f"subject_{subject:02d}"] = {
            "per_landmark": per_landmark, "group_median_mm": per_group,
            "all_landmarks_median_mm": round(
                float(np.median([v["median_mm"] for v in per_landmark.values()])), 3),
            "segments": segments,
            "segment_mean_abs_error_mm": round(
                float(np.mean([s["abs_error_mm"] for s in segments.values()])), 2)}
        row = report["subjects"][f"subject_{subject:02d}"]
        print(f"subject {subject:02d}: all-landmark median {row['all_landmarks_median_mm']} mm; "
              f"groups {row['group_median_mm']}; segment mean |error| "
              f"{row['segment_mean_abs_error_mm']} mm")
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
