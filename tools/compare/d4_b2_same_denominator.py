"""D4 B2: same denominator, on the CONSUMED input.

Astra's finding 3: comparing saved arrays passes on the wrong denominator, because the pre-card
fitter read the RAW array while the rig was built from the smoothed one -- and the two files
happened to agree. So this compares the array the MHR adapter was ACTUALLY HANDED (written by the
build itself, across the process boundary, into `converter-inputs/`) against the array at the rig
converter's input, and then re-derives momentum's markers from that array through the declared
mapping and the declared unit conversion and checks them against what momentum received.

Four things are checked, and a failure in any one is a FAIL:
  1. the consumed array is byte-identical to the rig converter's input on the same build, AND to
     the rig delivery's own `triangulated_world_positions_z_up_m` from a separate `--body rig`
     build -- same dtype, same shape, same bytes, so the NaN pattern is included;
  2. joint names, order, and the per-frame validity mask are identical;
  3. subject and frame order: the tracks' `ticks` are identical;
  4. every marker value and occlusion flag momentum received is re-derived from that array
     through `MAP` and `(x, z, -y) * 100` and matches exactly.

And it RECORDS what the MHR route does not consume. Equal body arrays do not mean equal
information: the rig additionally consumes the solved head landmarks, the toes and `Spine1`, and
the MHR map uses 17 of the 19 body landmarks (SOMA-77 emits no ears).

    .venv/bin/python tools/compare/d4_b2_same_denominator.py --delivery DIR --rig-build DIR \
        --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# allow_pickle below: every file read here is this repository's OWN build output under the
# gitignored artifacts/ tree (npz with object arrays; CLAUDE.md). Nothing third-party is loaded.

ROOT = Path(__file__).resolve().parents[2]

# The mapping is NOT imported from the fitter: it is read back out of the DELIVERED track's own
# `landmark_to_joint` declaration, and the unit conversion is written out here from the card's
# words, so the check is an independent re-derivation rather than the adapter agreeing with
# itself. (It also keeps this instrument on `.venv`, which has no pymomentum.)
def to_mhr_cm(p_zup_m: np.ndarray) -> np.ndarray:
    """Capture Z-up metres -> MHR Y-up centimetres: `(x, z, -y) * 100`."""
    return np.stack([p_zup_m[..., 0], p_zup_m[..., 2], -p_zup_m[..., 1]], axis=-1) * 100.0


# What the rig route consumes beyond the 19-joint body array, and the MHR route does not.
OMITTED_FEEDS = {
    "head_landmarks_by_camera": "the five skull-rigid SOMA-77 landmarks the head solve uses "
                                "(Head, HeadEnd, Jaw, LeftEye, RightEye)",
    "toe_landmarks_by_camera": "LeftToeBase / RightToeBase, the ball of each foot (D-lane feet)",
    "spine_landmarks_by_camera": "SOMA-77 Spine1, which D7 gives Hips a frame of its own from",
}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--rig-build", type=Path, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    delivery = arguments.delivery.resolve()
    inputs = delivery / "converter-inputs"
    rig_side = np.load(inputs / "rig-converter-input.npz", allow_pickle=True)
    report = {"delivery": str(delivery), "rig_build": str(arguments.rig_build.resolve()),
              "subjects": {}, "omitted_feeds": OMITTED_FEEDS}
    for subject in (0, 1):
        consumed = np.load(inputs / f"subject-{subject:02d}-consumed.npz", allow_pickle=True)
        rig_track = np.load(arguments.rig_build / f"subject-{subject:02d}.body-track.npz",
                            allow_pickle=True)
        markers = np.load(delivery / f"subject-{subject:02d}.markers.npz", allow_pickle=True)
        track = np.load(delivery / f"subject-{subject:02d}.body-track.npz", allow_pickle=True)
        declared = json.loads((delivery / f"subject-{subject:02d}.body-track.json")
                              .read_text(encoding="utf-8"))["landmark_to_joint"]
        key = str(markers["source_array_key"])
        handed = np.asarray(consumed[key])
        rig_input = np.asarray(rig_side[f"subject_{subject:02d}_triangulated_world_positions_z_up_m"])
        rig_delivered = np.asarray(rig_track["triangulated_world_positions_z_up_m"])
        names_consumed = [str(n) for n in consumed["joint_names"]]
        names_markers = [str(n) for n in markers["source_joint_names"]]
        names_track = [str(n) for n in track["consumed_joint_names"]]

        # 4. the markers momentum received, re-derived from that array through the declared route
        array_cm = to_mhr_cm(np.asarray(handed, np.float64))
        landmark_names = list(declared)
        expected_positions = np.zeros((handed.shape[0], len(landmark_names), 3))
        expected_occluded = np.zeros((handed.shape[0], len(landmark_names)), bool)
        for column, landmark in enumerate(landmark_names):
            row = names_consumed.index(landmark)
            values = array_cm[:, row]
            finite = np.isfinite(values).all(axis=1)
            expected_positions[finite, column] = values[finite]
            expected_occluded[:, column] = ~finite
        got_positions = np.asarray(markers["marker_positions_mhr_cm"])
        got_occluded = np.asarray(markers["marker_occluded"])

        checks = {
            "consumed_array_key": key,
            "1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build":
                _bytes_equal(handed, rig_input),
            "1b_handed_array_is_byte_identical_to_the_rig_BUILD_s_delivered_array":
                _bytes_equal(handed, rig_delivered),
            "1c_the_array_is_the_SMOOTHED_repaired_one":
                key == "triangulated_world_positions_z_up_m",
            "2a_joint_names_and_order_identical_everywhere":
                names_consumed == names_markers == names_track
                == [str(n) for n in rig_side["joint_names"]],
            "2b_validity_mask_identical":
                bool(np.array_equal(np.isfinite(handed), np.isfinite(rig_delivered))),
            "3_frame_order_identical":
                bool(np.array_equal(np.asarray(consumed["ticks"]),
                                    np.asarray(rig_track["ticks"]))),
            "4a_marker_values_match_the_declared_mapping_and_conversion":
                bool(np.array_equal(expected_positions, got_positions)),
            "4b_occlusion_flags_match": bool(np.array_equal(expected_occluded, got_occluded)),
            "marker_names_are_the_declared_map":
                [str(n) for n in markers["marker_names"]] == landmark_names,
            "landmarks_consumed": len(landmark_names),
            "landmarks_available": len(names_consumed),
            "landmarks_not_consumed": [n for n in names_consumed if n not in declared],
            "declared_landmark_to_joint": declared,
            "occluded_marker_fraction": round(float(got_occluded.mean()), 6),
            "max_abs_marker_difference_cm": float(np.abs(expected_positions - got_positions).max()),
        }
        checks["verdict"] = "PASS" if all(
            value for name, value in checks.items()
            if name[0].isdigit() or name == "marker_names_are_the_declared_map") else "FAIL"
        report["subjects"][f"subject_{subject:02d}"] = checks
        print(f"subject {subject:02d}: {checks['verdict']}  "
              f"(key {key}, {checks['landmarks_consumed']} of {checks['landmarks_available']} "
              f"landmarks, marker delta {checks['max_abs_marker_difference_cm']} cm)")
    report["verdict"] = ("PASS" if all(s["verdict"] == "PASS" for s in report["subjects"].values())
                         else "FAIL")
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("B2:", report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
