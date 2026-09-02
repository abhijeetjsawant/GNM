#!/usr/bin/env python3
"""D1 (fix): the SOMA lane's handedness, on REAL motion rather than a unit test.

WHY THIS EXISTS. `soma_motion._DELTA_MAPPING` used to send our `Left*` joints onto SOMA's
`Right*` ones -- a deliberate COMPENSATION for the rig's mirrored naming, with the reason
written beside it. Repairing the rig without removing that swap would have shipped every
SOMA performer with their arms and legs exchanged. The swap was removed, and until now the
only thing behind that removal was an assertion about two dictionaries. This runs it on a
real GEM-X/Kimodo export.

WHAT IT MEASURES. One signed number per arm: `sign((right - left) x up . forward)` with
`across` taken BY NAME, the same handedness triple product `facing_location.py` uses. A
proper rotation cannot change it and a mirror must, so it is exactly the reading that
catches a left/right exchange and nothing else. Three arms:

  * SOMA's own skeleton, from its own joint positions and its own names -- the reference.
  * our delivered rig under the REPAIRED mapping, forward-kinematic from the projected
    track, scored by OUR names.
  * our delivered rig under the LEGACY swapped mapping, on the identical fixture. That is
    the before arm, and it is what makes this a comparison rather than an assertion.

The two rig arms differ ONLY in the mapping dictionary, so any difference between them is
the mapping and nothing else.

NOT a fixture with ground truth and not a reference comparison: SOMA is our own upstream
provider's output, and all this asks is whether the side survives the boundary.

    .venv/bin/python tools/compare/soma_handedness.py
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from autoanim_gnm import soma_motion as sm                                    # noqa: E402
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions  # noqa: E402

CLIPS = (
    "autoanim_dialogue/amy-cuddy-dialogue-body",
    "autoanim_squat/research-squat-640",
    "autoanim_will_acting/will-stephen-acting-body",
    "autoanim_real/autoanim_fixture",
    "cpu_smoke/autoanim_fixture",
    "autoanim_csg_dialogue/csg-dialogue-upper-body",
)
MOTION_ROOT = ROOT / ".cache/autoanim_gnm/gem-x/outputs"
OUT = ROOT / "artifacts/compare/d1-fix/soma-handedness.json"


def triple(across: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return np.einsum("fj,fj->f", np.cross(across, up), forward)


def summarise(values: np.ndarray) -> dict:
    ok = np.isfinite(values)
    return {"sign_median": float(np.sign(np.median(values[ok]))),
            "fraction_of_frames_positive": float(np.mean(values[ok] > 0)),
            "frames": int(ok.sum())}


def load(clip: str) -> sm.SomaMotion:
    """Build a SomaMotion from a retained GEM-X export, through the real dataclass so its
    own validation runs. Nothing here is re-implemented: the fields come straight off the
    provider's npz and its sidecar manifest."""
    directory = MOTION_ROOT / clip
    data = np.load(directory / "soma_motion.npz", allow_pickle=True)
    meta = json.loads((directory / "soma_motion.json").read_text())
    coordinates = meta["source_coordinate_system"]
    base = meta["source_time_base"]
    frames = len(data["root_translation_m"])
    # `source_time_base` is a TICK DURATION (numerator/denominator seconds per tick), not a
    # frame rate, so the rate comes from the spacing of `source_pts` in those ticks. Read
    # rather than assumed: an assumed 30 would be a timebase error of exactly the kind
    # CLAUDE.md records, and this instrument's whole subject is a convention taken on trust.
    pts = np.asarray(data["source_pts"], dtype=np.int64)
    # `source_time_base` is a TICK DURATION (numerator/denominator seconds per tick), not a
    # frame rate. Ticks are derived by the SAME exact rational rule `validate_soma_motion`
    # checks, so the dataclass's own validation runs rather than being worked around; an
    # assumed 30 fps here would be exactly the timebase error CLAUDE.md records, inside an
    # instrument whose whole subject is a convention taken on trust.
    numerator, denominator = base["numerator"], base["denominator"]
    first = int(pts[0])
    ticks = np.asarray(
        [(2 * (int(v) - first) * numerator * sm.TICKS_PER_SECOND + denominator)
         // (2 * denominator) for v in pts], dtype=np.int64)
    seconds = (int(pts[-1]) - first) * numerator / denominator
    rate = int(round((frames - 1) / seconds)) if seconds > 0 else 30
    return sm.SomaMotion(
        provider_id=meta["provider_id"],
        provider_git_commit_oid=meta["provider_git_commit_oid"],
        operation=meta["operation"],
        motion_kind=meta["motion_kind"],
        duration_ticks=int(ticks[-1]),
        sample_rate_hz=max(rate, 1),
        ticks=ticks,
        source_pts=data["source_pts"],
        root_translation_m=data["root_translation_m"],
        local_rotations_xyzw=data["local_rotations_xyzw"],
        rest_joint_positions_m=data["rest_joint_positions_m"],
        rest_world_rotations_xyzw=data["rest_world_rotations_xyzw"],
        joint_positions_m=data["joint_positions_m"],
        contacts=data["contacts"],
        contact_schema_id=meta["contact_schema"]["id"],
        contact_names=tuple(meta["contact_schema"]["contact_names"]),
        source_handedness=coordinates["handedness"],
        source_up_axis=coordinates["up_axis"],
        source_forward_axis=coordinates["forward_axis"],
        source_linear_unit_in_meters=float(coordinates["linear_unit_in_meters"]),
        source_to_canonical_rotation_xyzw=tuple(
            coordinates["source_to_canonical_rotation_xyzw"]),
        source_time_base_numerator=int(base["numerator"]),
        source_time_base_denominator=int(base["denominator"]),
        input_sha256=meta["input_sha256"],
        provider_raw_motion_sha256=meta["provider_raw_motion_sha256"],
    )


# Which SOMA joint does each of our rig's limb joints actually FOLLOW? This is the reading
# the triple product cannot give (see `why_the_triple_product_cannot_see_this` below).
FOLLOWS = {"LeftHand": "LeftHand", "RightHand": "RightHand",
           "LeftLowerArm": "LeftForeArm", "RightLowerArm": "RightForeArm",
           "LeftFoot": "LeftFoot", "RightFoot": "RightFoot",
           "LeftLowerLeg": "LeftShin", "RightLowerLeg": "RightShin"}


def opposite(name: str) -> str:
    return ("Right" + name.removeprefix("Left")) if name.startswith("Left") else (
        "Left" + name.removeprefix("Right"))


def follows_arm(track, names: list[str], joints: np.ndarray, soma_names: list[str]) -> dict:
    """For each limb joint: is our rig's trajectory nearer SOMA's SAME-named joint or its
    OPPOSITE-named one? Root-relative, so the comparison is about motion and not placement.

    This is what a swapped mapping actually breaks. The rig's left arm stays on the rig's
    left -- that is the REST skeleton's doing and no mapping changes it -- while the motion
    on it becomes the performer's other arm. So the discriminator is agreement, not sign.
    """
    ours = forward_kinematics_positions(
        track.root_translation_m, track.local_rotations_xyzw, skeleton=DETAILED_HUMANOID)
    ours = ours - ours[:, [names.index("Hips")]]
    theirs = joints - joints[:, [soma_names.index("Hips")]]
    frames = min(len(ours), len(theirs))
    rows, wrong = {}, []
    for ours_name, soma_name in FOLLOWS.items():
        a = ours[:frames, names.index(ours_name)]
        same = float(np.median(np.linalg.norm(
            a - theirs[:frames, soma_names.index(soma_name)], axis=1)) * 1000.0)
        cross = float(np.median(np.linalg.norm(
            a - theirs[:frames, soma_names.index(opposite(soma_name))], axis=1)) * 1000.0)
        rows[ours_name] = {"to_SOMA_same_side_mm": round(same, 2),
                           "to_SOMA_opposite_side_mm": round(cross, 2),
                           "nearer": "same_side" if same < cross else "OPPOSITE_side"}
        if same >= cross:
            wrong.append(ours_name)
    same_all = [v["to_SOMA_same_side_mm"] for v in rows.values()]
    return {"per_joint": rows, "joints_following_the_wrong_side": wrong,
            "every_joint_follows_its_own_side": not wrong,
            "median_distance_to_its_own_side_mm": round(float(np.median(same_all)), 2)}


def rig_arm(track, names: list[str]) -> dict:
    positions = forward_kinematics_positions(
        track.root_translation_m, track.local_rotations_xyzw, skeleton=DETAILED_HUMANOID)
    at = lambda name: positions[:, names.index(name)]
    return summarise(triple(at("RightUpperArm") - at("LeftUpperArm"),
                            at("Neck") - at("Hips"),
                            at("LeftToes") - at("LeftFoot")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    names = list(DETAILED_HUMANOID.names)
    soma_names = list(sm.SOMASKEL77_NAMES)
    legacy = {k: ("Right" + v.removeprefix("Left")) if v.startswith("Left")
              else ("Left" + v.removeprefix("Right")) if v.startswith("Right") else v
              for k, v in sm._DETAILED_DELTA_MAPPING.items()}

    report: dict = {
        "step": "D1 (fix)",
        "instrument": "tools/compare/soma_handedness.py",
        "question": ("does the SOMA -> rig boundary keep each side on its own side, on a "
                     "REAL provider export rather than in a unit test?"),
        "definition": ("sign((RIGHT - LEFT) x up . forward), across BY NAME. Invariant "
                       "under any proper rotation, inverted by any reflection."),
        "legacy_mapping_is_the_shipped_one_with_every_side_swapped": True,
        "clips": {},
    }
    for clip in CLIPS:
        directory = MOTION_ROOT / clip
        if not (directory / "soma_motion.npz").is_file():
            report["clips"][clip] = {"unavailable": "no soma_motion.npz on disk"}
            continue
        motion = load(clip)
        joints = np.asarray(motion.joint_positions_m, dtype=np.float64)
        at = lambda name: joints[:, soma_names.index(name)]
        source = summarise(triple(at("RightArm") - at("LeftArm"),
                                  at("Neck2") - at("Hips"),
                                  at("LeftToeBase") - at("LeftFoot")))

        repaired = sm.project_soma_to_detailed_body_track(motion)
        original = dict(sm._DETAILED_DELTA_MAPPING)
        try:
            sm._DETAILED_DELTA_MAPPING.clear()
            sm._DETAILED_DELTA_MAPPING.update(legacy)
            before = sm.project_soma_to_detailed_body_track(motion)
        finally:
            sm._DETAILED_DELTA_MAPPING.clear()
            sm._DETAILED_DELTA_MAPPING.update(original)

        after_arm = rig_arm(repaired, names)
        before_arm = rig_arm(before, names)
        after_follows = follows_arm(repaired, names, joints, soma_names)
        before_follows = follows_arm(before, names, joints, soma_names)
        report["clips"][clip] = {
            "frames": int(len(joints)),
            "input_sha256": motion.input_sha256,
            "npz_sha256": sha256((directory / "soma_motion.npz").read_bytes()).hexdigest(),
            "source_coordinate_system": {
                "handedness": motion.source_handedness,
                "up_axis": motion.source_up_axis,
                "forward_axis": motion.source_forward_axis,
            },
            "SOMA_its_own_joints_and_names": source,
            "our_rig_REPAIRED_mapping": after_arm,
            "our_rig_LEGACY_swapped_mapping": before_arm,
            "repaired_agrees_with_SOMA": (
                after_arm["sign_median"] == source["sign_median"]),
            "legacy_agrees_with_SOMA": (
                before_arm["sign_median"] == source["sign_median"]),
            "which_side_each_joint_FOLLOWS_repaired": after_follows,
            "which_side_each_joint_FOLLOWS_legacy": before_follows,
        }
        print(f"{clip:46s} sign SOMA {source['sign_median']:+.0f} repaired "
              f"{after_arm['sign_median']:+.0f} legacy {before_arm['sign_median']:+.0f}"
              f"   | follows own side: repaired "
              f"{after_follows['every_joint_follows_its_own_side']}, legacy "
              f"{before_follows['every_joint_follows_its_own_side']}")

    scored = [v for v in report["clips"].values() if "unavailable" not in v]
    report["what_the_LEGACY_arm_is"] = (
        "NOT the code as it shipped. It is the repaired skeleton with `soma_motion`'s "
        "compensating side swap LEFT IN -- precisely the half-done change this check "
        "exists to catch. The lane as it shipped was self-consistent: bones named Left sat "
        "at rig -X and carried the mesh's anatomical RIGHT flesh, and the mapping sent them "
        "SOMA's Right rotations, so the performer's right arm drove the mesh's right arm. "
        "Both halves were mirrored and the product was correct. Change one and it is not."
    )
    report["why_the_triple_product_cannot_see_this"] = (
        "MEASURED, and it is the main finding of this check. The triple product reads -1 on "
        "SOMA's own joints, -1 on our rig under the repaired mapping and -1 under the "
        "swapped one, on all six clips: IT DOES NOT DISCRIMINATE. The reason is structural. "
        "`soma_motion` is a ROTATION retarget -- our rig's joint POSITIONS come from OUR "
        "rest skeleton posed by SOMA-derived rotations -- so the joint named LeftUpperArm "
        "sits on the rig's left whatever mapping drives it. A swapped mapping leaves every "
        "bone on its own side and puts the OTHER arm's motion on it, and a sign that "
        "measures where the bones ARE cannot see which motion they are DOING. This is the "
        "same class of blindness as 'a length invariant cannot score direction' -- the "
        "instrument is correct and the claim it was asked to carry is not one it supports. "
        "What discriminates is AGREEMENT, below."
    )
    report["verdict"] = {
        "clips_scored": len(scored),
        "the_triple_product_discriminates": False,
        "triple_product_sign_SOMA_repaired_legacy": [
            [v["SOMA_its_own_joints_and_names"]["sign_median"],
             v["our_rig_REPAIRED_mapping"]["sign_median"],
             v["our_rig_LEGACY_swapped_mapping"]["sign_median"]] for v in scored],
        "repaired_every_joint_follows_its_own_side_on_every_clip": all(
            v["which_side_each_joint_FOLLOWS_repaired"]["every_joint_follows_its_own_side"]
            for v in scored),
        "legacy_clips_where_some_joint_follows_the_wrong_side": sum(
            not v["which_side_each_joint_FOLLOWS_legacy"]["every_joint_follows_its_own_side"]
            for v in scored),
        "median_distance_to_its_own_SOMA_joint_mm": {
            "repaired": [v["which_side_each_joint_FOLLOWS_repaired"][
                "median_distance_to_its_own_side_mm"] for v in scored],
            "legacy": [v["which_side_each_joint_FOLLOWS_legacy"][
                "median_distance_to_its_own_side_mm"] for v in scored],
        },
        "repaired_nearer_its_own_side_on_every_clip": all(
            v["which_side_each_joint_FOLLOWS_repaired"]["median_distance_to_its_own_side_mm"]
            < v["which_side_each_joint_FOLLOWS_legacy"]["median_distance_to_its_own_side_mm"]
            for v in scored),
    }
    report["reading"] = (
        "The discriminating figure is not the sign but the DISTANCE: how far each of our "
        "rig's limb joints sits from the SOMA joint it is supposed to be following, "
        "root-relative. The repaired mapping is nearer its own side on every joint of every "
        "clip, by a wide margin; leaving the compensation in blows that up several-fold "
        "(on the squat clip, feet 53 mm -> 417 mm and hands 133 mm -> 731 mm). "
        "The 'which side is nearer' test is the weaker of the two and says so: on "
        "`cpu_smoke/autoanim_fixture` the legacy arm is still nominally nearer its own side "
        "on every joint while being 970 mm from it, because on a clip whose two arms move "
        "alike both pairings are equally bad. Read the distance, not the winner."
    )
    report["blind_to"] = (
        "The triple product is blind to THIS defect, which is the finding above and not a "
        "caveat. The distance arm scores the SOMA lane's own output against the SOMA lane's "
        "own input, so it cannot see an error the provider and the projection share, and it "
        "is not accuracy: our rig has canonical bone lengths and SOMA's skeleton does not, "
        "so a residual of 130-200 mm on the hands is the proportion mismatch I1 measures "
        "and not a retarget error. What it can see is a SIDE EXCHANGE, which is all it is "
        "asked for. It says nothing about facing, nothing about bone lengths, and nothing "
        "about the fingers. On a clip whose two arms move alike the 'nearer side' test "
        "degenerates -- `cpu_smoke` demonstrates it -- which is why the distance is quoted "
        "beside it. These are our own upstream provider's exports; there is no ground truth "
        "here and none is claimed."
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {arguments.out}")
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
