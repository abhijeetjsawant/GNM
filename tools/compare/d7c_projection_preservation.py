#!/usr/bin/env python3
"""D7c's P: does the delivered file still carry what the projection decided?

THREE CONTRACTS, KEPT APART, because each catches something the others cannot. The card
splits them for that reason and this file never merges their verdicts.

  P1 CHANNEL PRESERVATION. A watcher captures the candidate's track by value IMMEDIATELY
     after its SINGLE call to `project_generated_foot_contacts` -- the function's own return,
     not a re-derivation. The DELIVERED body-track, first AUTHENTICATED against the GLB's
     `body_track_sha256` so it is the track the shipped file was written from, must be
     bit-identical to that snapshot on the root translation, the `foot_contacts` mask, and
     the locals of `Root`, `Hips`, both upper and lower legs, both feet and both toes --
     exactly the ancestry D9b's post-projection re-solve is required to preserve. Comparison
     is at the dtype the dataclass stores (float32 for the root and the rotations, bool for
     the mask), so a cast can never be mistaken for a change or hide one.

  P2 ANCHOR LOCK. Every accepted contact run, taken from the FROZEN SNAPSHOT MASK and never
     inferred from GLB velocity, must hold Foot AND Toes at the run's first KEYED sample
     within `CONTACT_TOLERANCE_M` when forward kinematics is run on THE GLB'S OWN ARRAYS.
     KEYED SAMPLES ONLY: the samplers are LINEAR (`body_export.py:615`) and between-key
     playback already reads 0.67 / 1.31 mm at interval midpoints on the SHIPPED GLBs, which
     is B6's report and not this clause.

  P3 TRAVEL REPORT. Planted-foot travel for both builds on the FROZEN UNION of both builds'
     run intervals, each interval keeping its own side and boundaries, so a contact the
     candidate LOSES stays in the comparison instead of vanishing from the denominator. The
     candidate is NOT required to be planted on a baseline-only run: contact selection may
     change, and the card says so.

WHY P1 IS THE ONE THAT CATCHES THE CONTROLS, and why the anchor check alone guarantees
nothing. A build that overwrites the projection's foot LOCALS after the projection fails P1
and MAY PASS P2 -- a foot's own rotation does not move its origin. A build that clears the
nonempty snapshot mask fails P1 on the mask and MAY PASS P2 -- the geometry stays planted,
there is simply nothing left to check. Both controls are built by
`d7c_pelvis_rest_delivery.py --mode control-*`, which refuses to run as a no-op, and both
must FAIL P1 here.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_projection_preservation.py \\
        --baseline artifacts/commercial-multiview-soma77 \\
        --candidate artifacts/compare/d7c-pelvis-rest/delivery \\
        --out artifacts/compare/d7c-pelvis-rest/projection-preservation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

from autoanim_gnm import body_export as be  # noqa: E402
from autoanim_gnm.body import CONTACT_TOLERANCE_M  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402

# The exact ancestry the card names. Nothing is added to it and nothing is dropped.
PROTECTED_JOINTS = (
    "Root", "Hips",
    "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "LeftToes", "RightToes",
)
FOOT_SIDES = (("LeftFoot", "LeftToes"), ("RightFoot", "RightToes"))


def runs_of(flags: np.ndarray) -> list[tuple[int, int]]:
    on = np.flatnonzero(flags)
    if not on.size:
        return []
    splits = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
    return [(int(part[0]), int(part[-1])) for part in splits]


def snapshot(directory: Path, subject: int) -> dict:
    """The post-projection track the watcher saved, at the dtype the dataclass stores."""
    path = directory / f"projection-snapshots/projection-snapshot-{subject:02d}.npz"
    if not path.exists():
        raise SystemExit(f"{path} is missing: rebuild with the watcher installed")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def delivered(directory: Path, subject: int) -> dict:
    """The delivered track, AUTHENTICATED against the GLB's own `body_track_sha256`."""
    track = d3.load_track(directory, subject)
    document, _ = d3.read_glb(directory / f"subject-{subject:02d}.glb")
    stamped = document["asset"]["extras"]["body_track_sha256"]
    computed = be._body_track_sha256(track)
    return {"track": track, "glb_body_track_sha256": stamped,
            "recomputed_sha256": computed, "authenticated": bool(stamped == computed)}


def p1(directory: Path, subject: int) -> dict:
    """Channel preservation: the delivered track against the frozen snapshot, bit for bit."""
    frozen = snapshot(directory, subject)
    row = delivered(directory, subject)
    track = row["track"]
    names = list(track.joint_names)
    channels: dict = {}
    root_ok = bool(np.array_equal(np.asarray(track.root_translation_m),
                                  frozen["root_translation_m"]))
    contacts_ok = bool(np.array_equal(np.asarray(track.foot_contacts),
                                      frozen["foot_contacts"]))
    channels["root_translation_m"] = {
        "bit_identical": root_ok,
        "dtype": str(np.asarray(track.root_translation_m).dtype)}
    channels["foot_contacts"] = {
        "bit_identical": contacts_ok,
        "snapshot_contacts": [int(v) for v in frozen["foot_contacts"].sum(0)],
        "delivered_contacts": [int(v) for v in np.asarray(track.foot_contacts).sum(0)]}
    local = np.asarray(track.local_rotations_xyzw)
    for joint in PROTECTED_JOINTS:
        slot = names.index(joint)
        same = bool(np.array_equal(local[:, slot],
                                   frozen["local_rotations_xyzw"][:, slot]))
        moved = int((~np.all(local[:, slot]
                             == frozen["local_rotations_xyzw"][:, slot], axis=-1)).sum())
        channels[f"local::{joint}"] = {"bit_identical": same, "frames_that_differ": moved}
    passes = (row["authenticated"]
              and all(entry["bit_identical"] for entry in channels.values()))
    return {
        "contract": ("the delivered body-track is bit-identical to the post-projection "
                     "snapshot on the root translation, the contact mask and the locals of "
                     "the ten protected joints"),
        "authentication": {k: v for k, v in row.items() if k != "track"},
        "channels": channels,
        "failing_channels": [k for k, v in channels.items() if not v["bit_identical"]],
        "verdict": "PASS" if passes else "FAIL",
    }


def p2(directory: Path, subject: int) -> dict:
    """Anchor lock, from the GLB's own arrays, on KEYED samples only."""
    frozen = snapshot(directory, subject)
    mask = np.asarray(frozen["foot_contacts"])
    names, positions, _rest = d3.glb_joint_positions(
        directory / f"subject-{subject:02d}.glb")
    index = {name: slot for slot, name in enumerate(names)}
    rows: list[dict] = []
    worst = 0.0
    for side, (foot, toes) in enumerate(FOOT_SIDES):
        for start, end in runs_of(mask[:, side]):
            entry = {"side": foot, "run": [start, end], "frames": end - start + 1}
            for joint in (foot, toes):
                anchor = positions[start, index[joint]]
                travel = np.linalg.norm(positions[start:end + 1, index[joint]] - anchor,
                                        axis=1)
                entry[f"{joint}_max_m"] = float(travel.max())
                worst = max(worst, float(travel.max()))
            entry["holds"] = bool(max(entry[f"{foot}_max_m"], entry[f"{toes}_max_m"])
                                  <= CONTACT_TOLERANCE_M)
            rows.append(entry)
    return {
        "contract": ("every accepted contact run, taken from the FROZEN SNAPSHOT MASK, holds "
                     "Foot AND Toes at the run's first KEYED sample within "
                     f"CONTACT_TOLERANCE_M = {CONTACT_TOLERANCE_M} m, forward-kinematicked "
                     "on the GLB's own arrays"),
        "mask_source": "the post-projection snapshot, never GLB velocity",
        "keyed_samples_only": ("the samplers are LINEAR; between-key playback already reads "
                              "0.67 / 1.31 mm at interval midpoints on the SHIPPED GLBs and "
                              "is B6's report, not this clause"),
        "runs": rows,
        "worst_travel_m": worst,
        "band_m": CONTACT_TOLERANCE_M,
        "verdict": "PASS" if all(row["holds"] for row in rows) else "FAIL",
    }


def p3(baseline: Path, candidate: Path, subject: int) -> dict:
    """Travel on the FROZEN UNION of both builds' runs. A REPORT; never a verdict."""
    out: dict = {
        "contract": ("planted-foot travel for both builds on the frozen UNION of both "
                     "builds' run intervals, each interval keeping its own side and "
                     "boundaries, so a contact the candidate LOSES stays in the comparison"),
        "the_candidate_is_not_required_to_be_planted_on_a_baseline_only_run": (
            "contact selection may change under a new pelvis frame; the card says so and "
            "replaces D9b's identical-contacts clause with this report"),
        "intervals": [],
    }
    masks = {}
    geometry = {}
    for label, directory in (("baseline", baseline), ("candidate", candidate)):
        with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
            masks[label] = np.asarray(archive["foot_contacts"])
        names, positions, _ = d3.glb_joint_positions(
            directory / f"subject-{subject:02d}.glb")
        geometry[label] = (positions, {n: i for i, n in enumerate(names)})
    for side, (foot, toes) in enumerate(FOOT_SIDES):
        union = {(label, start, end)
                 for label in masks for start, end in runs_of(masks[label][:, side])}
        for label, start, end in sorted(union, key=lambda row: (row[1], row[0])):
            entry = {"side": foot, "run": [start, end], "declared_by": label}
            for arm in ("baseline", "candidate"):
                positions, index = geometry[arm]
                for joint in (foot, toes):
                    anchor = positions[start, index[joint]]
                    travel = np.linalg.norm(
                        positions[start:end + 1, index[joint]] - anchor, axis=1)
                    entry[f"{arm}_{joint}_max_mm"] = round(float(1e3 * travel.max()), 5)
                entry[f"{arm}_declares_this_run"] = bool(
                    masks[arm][start:end + 1, side].all())
            out["intervals"].append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-p1", choices=("PASS", "FAIL"), default="PASS",
                        help="a CONTROL is built to FAIL P1; say so and the exit code "
                             "reflects whether it did what it was built to do")
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    candidate = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    report: dict = {
        "title": "D7c P -- projection preservation, three contracts kept apart",
        "baseline": str(baseline), "candidate": str(candidate), "label": args.label,
        "expected_p1": args.expect_p1,
        "subjects": {},
    }
    for subject in (0, 1):
        row = {"P1_channel_preservation": p1(candidate, subject),
               "P2_anchor_lock": p2(candidate, subject),
               "P3_travel_report": p3(baseline, candidate, subject)}
        report["subjects"][f"subject_{subject:02d}"] = row
        print(f"subject {subject}: P1 {row['P1_channel_preservation']['verdict']} "
              f"(failing: {row['P1_channel_preservation']['failing_channels']}) "
              f"P2 {row['P2_anchor_lock']['verdict']} worst "
              f"{row['P2_anchor_lock']['worst_travel_m']:.3e} m, "
              f"{len(row['P2_anchor_lock']['runs'])} runs")
    p1_verdicts = {s: row["P1_channel_preservation"]["verdict"]
                   for s, row in report["subjects"].items()}
    p2_verdicts = {s: row["P2_anchor_lock"]["verdict"]
                   for s, row in report["subjects"].items()}
    report["P1_verdicts"] = p1_verdicts
    report["P2_verdicts"] = p2_verdicts
    report["P1_as_expected"] = all(v == args.expect_p1 for v in p1_verdicts.values())
    report["verdict"] = ("PASS" if report["P1_as_expected"]
                         and (args.expect_p1 == "FAIL"
                              or all(v == "PASS" for v in p2_verdicts.values()))
                         else "FAIL")
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nP1 {p1_verdicts} (expected {args.expect_p1}) | P2 {p2_verdicts}")
    print(f"verdict {report['verdict']}; wrote {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
