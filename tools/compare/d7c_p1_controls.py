#!/usr/bin/env python3
"""D7c: P1's two controls, and WHY ONE OF THEM CANNOT BE BUILT.

The card asks for two builds, each failing a named contract:

  (1) a build that OVERWRITES the projection's foot locals after the projection -- must fail
      P1, may pass P2 (a foot's own rotation does not move its origin);
  (2) a build that CLEARS the nonempty snapshot mask -- must fail P1 on the mask, may pass P2
      (the geometry stays planted).

CONTROL (1) IS REFUSED BY THE SHIPPING PATH ITSELF, and that is a stronger result than P1
catching it. `d7c_pelvis_rest_delivery.py --mode control-overwrite-locals` first ASSERTS that
the projection really did change a foot local (it will not run as a no-op), then restores the
pre-projection rotations -- and `BodyTrack.__post_init__` runs `validate_body_track`, which
raises before a single file is written:

    BodyValidationError: left foot contact moved 0.00884243 m (limit 0.00001000 m)

So the mutation cannot reach a delivered artifact at all. That is the same thing D9b found of
its lock-without-correction degenerate: refused by the shipping path, which is stronger than a
gate catching it after the fact. It does, however, leave P1's DETECTION of that mutation
unexercised by a built artifact -- so this file exercises it directly on the delivered bytes,
which is the honest way to close the clause:

  * take the delivery's own post-projection snapshot and its delivered track;
  * apply each mutation to the DELIVERED arrays offline -- the same mutation the build would
    have made, no more;
  * run P1's own comparison on the result and require it to FAIL, naming the channel.

Nothing here is a build and nothing here ships. It measures the INSTRUMENT: that P1 detects
the two failure modes the card names.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_p1_controls.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/swap-harness", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(f"PYTHONPATH trap: {autoanim_gnm.__file__}")

import d3_skeleton_gate as d3  # noqa: E402
from d7c_projection_preservation import PROTECTED_JOINTS, snapshot  # noqa: E402

DELIVERY = ROOT / "artifacts/compare/d7c-pelvis-rest/delivery"
FOOT_JOINTS = ("LeftFoot", "LeftToes", "RightFoot", "RightToes")
OUT = ROOT / "artifacts/compare/d7c-pelvis-rest/p1-controls.json"


def compare(delivered: dict, frozen: dict, names: list[str]) -> list[str]:
    """P1's own comparison, on arrays rather than on a track. Same channels, same order."""
    failing = []
    if not np.array_equal(delivered["root"], frozen["root_translation_m"]):
        failing.append("root_translation_m")
    if not np.array_equal(delivered["contacts"], frozen["foot_contacts"]):
        failing.append("foot_contacts")
    for joint in PROTECTED_JOINTS:
        slot = names.index(joint)
        if not np.array_equal(delivered["rotations"][:, slot],
                              frozen["local_rotations_xyzw"][:, slot]):
            failing.append(f"local::{joint}")
    return failing


def main() -> int:
    report = {
        "title": "D7c: P1's two controls",
        "why_control_1_is_not_a_build": (
            "`BodyTrack.__post_init__` runs `validate_body_track`, which raised "
            "`left foot contact moved 0.00884243 m (limit 0.00001000 m)` before a single "
            "file was written. The mutation cannot reach a delivered artifact at all -- the "
            "SHIPPING PATH refuses it, which is stronger than P1 catching it afterwards. "
            "The build's own no-op assertion had already passed, so the projection really "
            "did change a foot local and the control was not degenerate."),
        "what_is_measured_here": (
            "the INSTRUMENT: that P1 detects the two failure modes the card names, applied "
            "to the DELIVERED bytes offline. Nothing here is a build and nothing ships."),
        "controls": {},
    }
    ok = True
    for subject in (0, 1):
        frozen = snapshot(DELIVERY, subject)
        track = d3.load_track(DELIVERY, subject)
        names = list(track.joint_names)
        base = {"root": np.asarray(track.root_translation_m),
                "rotations": np.asarray(track.local_rotations_xyzw),
                "contacts": np.asarray(track.foot_contacts)}
        clean = compare(base, frozen, names)

        overwritten = {k: np.array(v, copy=True) for k, v in base.items()}
        for joint in FOOT_JOINTS:
            overwritten["rotations"][:, names.index(joint)] = frozen[
                "pre_local_rotations_xyzw"][:, names.index(joint)]
        cleared = {k: np.array(v, copy=True) for k, v in base.items()}
        cleared["contacts"][:] = False

        rows = {
            "the_unmutated_delivery": {"failing_channels": clean,
                                       "P1": "PASS" if not clean else "FAIL"},
            "control_1_foot_locals_overwritten": {},
            "control_2_contact_mask_cleared": {},
        }
        for key, mutated in (("control_1_foot_locals_overwritten", overwritten),
                             ("control_2_contact_mask_cleared", cleared)):
            failing = compare(mutated, frozen, names)
            rows[key] = {"failing_channels": failing,
                         "P1": "FAIL" if failing else "PASS",
                         "detected_as_required": bool(failing)}
            ok &= bool(failing)
        rows["control_1_foot_locals_overwritten"]["samples_changed"] = int((~np.all(
            overwritten["rotations"] == base["rotations"], axis=-1)).sum())
        rows["control_2_contact_mask_cleared"]["mask_was_nonempty"] = [
            int(v) for v in base["contacts"].sum(0)]
        report["controls"][f"subject_{subject:02d}"] = rows
        print(f"subject {subject}: clean {rows['the_unmutated_delivery']['P1']}; "
              f"control 1 {rows['control_1_foot_locals_overwritten']['P1']} "
              f"{rows['control_1_foot_locals_overwritten']['failing_channels']}; "
              f"control 2 {rows['control_2_contact_mask_cleared']['P1']} "
              f"{rows['control_2_contact_mask_cleared']['failing_channels']}")
    report["both_controls_detected_on_both_performers"] = ok
    report["verdict"] = "PASS" if ok else "FAIL"
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nverdict {report['verdict']}; wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
