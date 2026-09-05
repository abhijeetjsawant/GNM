#!/usr/bin/env python3
"""Every mapped joint of a DELIVERED GLB against the pipeline's own captured landmarks.

THE FIGURE THAT SAW THE D7 DEFECT, and the reason it exists. Every joint instrument in
this lane before it re-solved the track through the converter and scored the result:
`tools/swap-harness/retarget_cost.py` does exactly that, and it has **no spine landmark**,
so it re-derives `Hips` from the trunk line whatever the delivery carries. It reads D3's
torso figure (14.46 / 26.21 mm) on the D7 delivery unchanged. It is not wrong; it is
answering a different question -- *what does the converter do to these landmarks?* -- and
it is structurally BLIND to a change that only the delivered file carries.

This reads the delivered file **from its own bytes** (D3's lesson,
`docs/reviews/performer-skeleton-2026-09-03.md`): `d3_skeleton_gate.glb_joint_positions`
parses the GLB, walks its own `children` hierarchy and its own animation channels exactly
as a glTF viewer would, and nothing about AutoAnim is assumed. The result is converted
back into the capture's Z-up metres and scored against
`triangulated_world_positions_z_up_m` from the SAME delivery directory's
`subject-XX.body-track.npz`. **Nothing is re-detected and nothing is re-triangulated**:
the landmarks are the ones that delivery was solved onto, so the only thing that can move
a figure is the delivery.

WHICH RIG JOINT SCORES WHICH LANDMARK. The converter aims each bone so that its CHILD's
origin lands on the landmark: `LeftShoulder` is turned until `LeftUpperArm`'s origin
reaches `left_shoulder`, `LeftUpperLeg` until `LeftLowerLeg`'s origin reaches
`left_knee`, and so on. So the joint that carries a landmark's claim is the child, and
`PAIRS` below is that correspondence and not a resemblance of names. `Hips` is scored
twice -- the rig's `Hips` JOINT against the hip-landmark midpoint, and the LEG-ROOT
midpoint against it, because D2b's root formula puts the leg roots there and not `Hips`
(`_leg_root_offset`). `Head` is scored against `nose`, which in the SOMA-77 adapter is
index 6, the **`Head` skeletal joint** and not a surface nose (CLAUDE.md): it is a
joint-vs-joint pairing, reported, never banded.

BLIND TO. (a) Orientation: a joint whose origin is on its landmark can still be rotated
about it, and every finger, toe, eye and the whole head-on-neck rotation is invisible
here. (b) The mesh: this scores the SKELETON the file carries, and the skin bound to it
can balloon or tear without moving a number -- that is what the silhouette is for.
(c) The landmarks themselves: they are our own triangulation, so a common-mode detector
error is inside the reference and cannot appear as an error. (d) It cannot see whether a
change is an improvement in the world -- only whether the delivered skeleton sits closer
to the points the delivery claims to have been solved onto.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/delivered_vs_capture.py \
        --delivery D3=artifacts/compare/delivered-before-d7-2026-09-05 \
        --delivery D7=artifacts/commercial-multiview-soma77 \
        --out artifacts/compare/d7b-trunk/delivered-vs-capture.json

THE FLOORS. Two are reported, each read from EACH delivery's own file and never carried
across arms. `trunk_length_floor` is D7b's: the residual a straight rigid trunk cannot
remove. `arm_aim_floor` is D9's, and it is the same idea for the operation D9 performs --
the elbow put on the ray from the delivered `UpperArm` origin at the rest upper-arm length,
then the wrist chained from that placed elbow. A placement figure quoted without its floor
says nothing about whether an aim can do better, which is the only question a step like
this asks.

Any number of `--delivery LABEL=PATH` arms; every pairwise difference of two arms is
block-bootstrapped on IDENTICAL draws (block 15, 2000 draws, fixed seed), paired frame by
frame, so the same denominator holds across arms by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import d3_skeleton_gate as d3  # noqa: E402
from autoanim_gnm.body import BodyTrack  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402

# The moving-block bootstrap this lane uses everywhere. Per-frame agreement has lag-1
# autocorrelation 0.99, so ordinary resampling is invalid (CLAUDE.md).
BLOCK, DRAWS, SEED = 15, 2000, 20260905

# rig joint -> capture landmark. The rig joint is the CHILD whose origin the converter
# aims onto the landmark; see the module docstring.
PAIRS: tuple[tuple[str, str], ...] = (
    ("Neck", "neck"),
    ("LeftUpperArm", "left_shoulder"),
    ("RightUpperArm", "right_shoulder"),
    ("LeftLowerArm", "left_elbow"),
    ("RightLowerArm", "right_elbow"),
    ("LeftHand", "left_wrist"),
    ("RightHand", "right_wrist"),
    ("LeftUpperLeg", "left_hip"),
    ("RightUpperLeg", "right_hip"),
    ("LeftLowerLeg", "left_knee"),
    ("RightLowerLeg", "right_knee"),
    ("LeftFoot", "left_ankle"),
    ("RightFoot", "right_ankle"),
    ("Head", "nose"),
)
# Grouped for the report; the gate reads these names.
GROUPS = {
    "trunk": ("Neck",),
    "hips": ("LeftUpperLeg", "RightUpperLeg", "hips_joint_vs_hip_midpoint",
             "leg_root_midpoint_vs_hip_midpoint"),
    "knees_ankles": ("LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot"),
    "arms": ("LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm",
             "LeftHand", "RightHand"),
    "head_reported_only": ("Head",),
}
REFERENCE = (
    "the delivery's OWN triangulated landmarks "
    "(`triangulated_world_positions_z_up_m` in its `subject-XX.body-track.npz`), in "
    "absolute capture Z-up metres. Not MAMMA, not a re-detection, not a re-solve."
)


# --------------------------------------------------------------------------- the readers
REFERENCE_ARRAY = {"smoothed": "triangulated_world_positions_z_up_m",
                   "raw": "raw_triangulated_world_positions_z_up_m"}
REFERENCE_TEXT = {
    "smoothed": REFERENCE,
    "raw": (
        "the delivery's own RAW triangulation "
        "(`raw_triangulated_world_positions_z_up_m`), NaNs intact -- the pre-fill, "
        "pre-solve, pre-smoothing points, scored only on the frames where the landmark "
        "actually triangulated. Added for D8, whose repair moves the SMOOTHED array by "
        "construction: scoring a D8 delivery against the smoothed landmarks would score "
        "it against points D8 itself repaired, and the raw array is the one reference "
        "that is bit-identical across the D7b and D8 builds."),
}


def capture_positions(directory: Path, subject: int,
                      reference: str = "smoothed") -> np.ndarray:
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        return np.asarray(archive[REFERENCE_ARRAY[reference]], np.float64)


def delivered_positions(directory: Path, subject: int) -> tuple[dict[str, int], np.ndarray]:
    """`[frame, joint, 3]` in CAPTURE Z-up metres, forward-kinematicked from the GLB's bytes.

    glTF here is the rig's Y-up world; `positions_to_body_track` built it with
    ``rig = (cx, cz, -cy)``, so the inverse is ``capture = (rx, -rz, ry)``.
    """

    names, positions, _rest = d3.glb_joint_positions(
        directory / f"subject-{subject:02d}.glb")
    capture = np.stack(
        [positions[..., 0], -positions[..., 2], positions[..., 1]], axis=-1)
    return {name: index for index, name in enumerate(names)}, capture


def track_rest(directory: Path, subject: int) -> dict[str, np.ndarray]:
    track = BodyTrack.from_dict(json.loads(
        (directory / f"subject-{subject:02d}.body-track.json").read_text()))
    return {name: np.asarray(rest, np.float64)
            for name, rest in zip(track.joint_names, track.rest_translations_m)}


# ------------------------------------------------------------------------- the statistics
def tilt_degrees(capture: np.ndarray) -> np.ndarray:
    """Trunk tilt from vertical, per frame, from the CAPTURED landmarks alone.

    Camera-independent and identical for every arm, because every arm is scored against
    the same triangulated positions -- which the caller asserts byte-identical.
    """

    hip_mid = 0.5 * (capture[:, JOINT_INDEX["left_hip"]]
                     + capture[:, JOINT_INDEX["right_hip"]])
    trunk = capture[:, JOINT_INDEX["neck"]] - hip_mid
    cosine = trunk[:, 2] / np.linalg.norm(trunk, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def tercile_masks(tilt: np.ndarray) -> dict[str, np.ndarray]:
    low, high = np.percentile(tilt, 33.3), np.percentile(tilt, 66.7)
    return {
        "whole": np.ones(len(tilt), dtype=bool),
        "upright": tilt <= low,
        "middle": (tilt > low) & (tilt < high),
        "bent": tilt >= high,
    }


def block_starts(frames: int) -> np.ndarray:
    """The draws. Generated ONCE per frame count and shared by every arm and every joint.

    Same denominator: two arms differing only in the delivery are resampled on the very
    same block indices, so the paired difference's interval carries no draw noise of its
    own.
    """

    rng = np.random.default_rng(SEED)
    return rng.integers(0, max(frames - BLOCK, 1),
                        size=(DRAWS, max(frames // BLOCK, 1)))


def bootstrap_median(values: np.ndarray, starts: np.ndarray) -> list[float]:
    """Block bootstrap of the median. NaN-aware, and identical on a NaN-free array.

    `--reference raw` leaves a NaN wherever the landmark did not triangulate, so a draw
    can contain them; a draw with nothing finite in it contributes nothing rather than a
    NaN that would poison the percentile.
    """

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not np.isfinite(values).any():
        return [float("nan"), float("nan")]
    draws = []
    for row in starts:
        block = np.concatenate([values[s:s + BLOCK] for s in row])[:n]
        if np.isfinite(block).any():
            draws.append(float(np.nanmedian(block)))
    if not draws:
        return [float("nan"), float("nan")]
    draws = np.asarray(draws)
    return [round(float(np.percentile(draws, 2.5)), 4),
            round(float(np.percentile(draws, 97.5)), 4)]


def _round(value) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 3)


def summarise(error_mm: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    finite = np.isfinite(error_mm)
    if not finite.any():
        return {"frames_scored": 0}
    out = {
        "median_mm": _round(np.nanmedian(error_mm)),
        "p95_mm": _round(np.nanpercentile(error_mm, 95)),
        "max_mm": _round(np.nanmax(error_mm)),
    }
    if not finite.all():
        # `--reference raw` only. Stated per joint, because a landmark that triangulated
        # on 140 of 150 frames and one that triangulated on all 150 do not share a
        # denominator and the report must not let them look as though they do.
        out["frames_scored"] = int(finite.sum())
        out["frames_total"] = int(finite.size)
    for name in ("upright", "bent"):
        subset = error_mm[masks[name]]
        out[f"{name}_tercile_median_mm"] = (
            _round(np.nanmedian(subset)) if np.isfinite(subset).any() else None)
    return out


# ------------------------------------------------------------------------------ the floor
def trunk_length_floor(directory: Path, subject: int) -> tuple[np.ndarray, float]:
    """`| L_rest - ||neck_landmark - Spine_origin|| |` per frame, in mm, and `L_rest` in mm.

    The residual a straight rigid trunk chain CANNOT remove, from THIS delivery's own
    Spine origin. `Spine`, `Chest` and `UpperChest` share one world rotation, and their
    rests are collinear +Y with zero X and Z on both performers, so the `Neck` origin is
    exactly `Spine_origin + R . (rest_Chest + rest_UpperChest + rest_Neck)` and the best
    any aim can do is put the neck ON the ray -- leaving the LENGTH difference. That share
    belongs to a flexible spine (D5), not to an aim.

    Note it is read from EACH delivery's own Spine origin. D7's floor is not D7b's: a
    different pelvis frame puts the Spine origin somewhere else, and carrying one arm's
    floor onto another would be scoring against a number instead of a construction.
    """

    rest = track_rest(directory, subject)
    length = float(sum(np.linalg.norm(rest[name])
                       for name in ("Chest", "UpperChest", "Neck")))
    index, world = delivered_positions(directory, subject)
    capture = capture_positions(directory, subject)
    chord = np.linalg.norm(
        capture[:, JOINT_INDEX["neck"]] - world[:, index["Spine"]], axis=1)
    return 1000.0 * np.abs(length - chord), 1000.0 * length


def distributed_flexion_recovery(directory: Path, subject: int) -> dict:
    """How much of the floor a flexion SPREAD over Spine/Chest/UpperChest would recover.

    ANALYTIC, and it builds nothing. The chord shortens because the trunk is one straight
    segment of fixed length `L`; a chain of the same three segments hinged at Chest and
    UpperChest spans a shorter chord for the same total length. Take the pelvis-to-thorax
    angle `phi` per frame -- the angle between the pelvis's own +Y (the `Hips` world axis
    the delivery carries) and the aim `neck - Spine_origin` -- split it EQUALLY over the
    two interior hinges, and the polyline's chord is the recoverable span. The equal split
    is an assumption and is stated as one: real lumbar and thoracic flexion is not equal,
    and only spine ratios (D5) can say what it is.

    Reported for D5. It is not a band and nothing here is fitted to it.
    """

    rest = track_rest(directory, subject)
    lengths = np.asarray([np.linalg.norm(rest[name])
                          for name in ("Chest", "UpperChest", "Neck")])
    total = float(lengths.sum())
    index, world = delivered_positions(directory, subject)
    capture = capture_positions(directory, subject)
    spine_origin = world[:, index["Spine"]]
    aim = capture[:, JOINT_INDEX["neck"]] - spine_origin
    chord = np.linalg.norm(aim, axis=1)
    # The pelvis's own up axis, in capture metres: `Hips` -> `Spine` is the rig's +Y
    # turned by the Hips world rotation, and it is read from the delivered file.
    pelvis_up = world[:, index["Spine"]] - world[:, index["Hips"]]
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    phi = np.arccos(np.clip(np.sum(unit(pelvis_up) * unit(aim), axis=1), -1.0, 1.0))
    # Equal split over the two interior hinges: in the plane of the bend the three
    # segments sit at -phi/2, 0 and +phi/2 about the mean direction, so the polyline's
    # chord is the vector sum below and it is shorter than `total` by construction.
    span = np.zeros((len(phi), 2))
    for k in range(3):
        angle = (k - 1.0) * phi / 2.0
        span[:, 0] += lengths[k] * np.cos(angle)
        span[:, 1] += lengths[k] * np.sin(angle)
    curved_chord = np.linalg.norm(span, axis=1)
    floor = np.abs(total - chord)
    curved = np.abs(curved_chord - chord)
    masks = tercile_masks(tilt_degrees(capture))
    return {
        "assumption": ("the pelvis-to-thorax angle phi is split EQUALLY over the two "
                       "interior hinges (Chest and UpperChest). Real lumbar and thoracic "
                       "flexion is not equal; only spine ratios (D5) can say what it is."),
        "handed_to": "D5 -- spine ratios and a flexible chain",
        "segment_lengths_mm": [round(1000.0 * float(v), 2) for v in lengths],
        "phi_median_deg": round(float(np.degrees(np.median(phi))), 3),
        "phi_bent_tercile_median_deg": round(
            float(np.degrees(np.median(phi[masks["bent"]]))), 3),
        "straight_chain_residual_mm": {
            "median": round(float(np.median(1000.0 * floor)), 3),
            "bent_tercile_median": round(
                float(np.median(1000.0 * floor[masks["bent"]])), 3)},
        "equal_split_curved_chain_residual_mm": {
            "median": round(float(np.median(1000.0 * curved)), 3),
            "bent_tercile_median": round(
                float(np.median(1000.0 * curved[masks["bent"]])), 3)},
        "recovered_mm": {
            "median": round(float(np.median(1000.0 * (floor - curved))), 3),
            "bent_tercile_median": round(
                float(np.median(1000.0 * (floor - curved)[masks["bent"]])), 3)},
    }


# -------------------------------------------------------------------------- the arm floor
# The take's own frame ids run 60..209 and the array index is `id - 60`
# (`tools/compare/d8_occlusion_silhouette.py` carries the same two constants, from the
# observations themselves). Frame id 118 is the length-limited frame the D9 card names.
FIRST_FRAME_ID = 60
WINDOW_FIRST_ID, WINDOW_LAST_ID = 85, 125
ARM_SIDES: tuple[tuple[str, str, str, str, str], ...] = (
    ("left", "LeftUpperArm", "LeftLowerArm", "LeftHand", "left"),
    ("right", "RightUpperArm", "RightLowerArm", "RightHand", "right"),
)


def arm_aim_floor_per_frame(directory: Path, subject: int) -> dict[str, np.ndarray]:
    """The residual an AIM cannot remove on the arms, per frame, in mm, for THIS delivery.

    The companion of :func:`trunk_length_floor`, for the operation D9 actually performs,
    and it is a CHAINED floor because the operation is chained.

    `UpperArm`'s origin is read from the delivered file itself -- the real one, wherever
    the trunk chord and the clavicle's fixed length happen to have put it -- and D9 does
    not move it (that miss is D5's: the trunk chord and a missing shoulder translation).
    From that origin the best any aim can do is put the elbow ON THE RAY to the captured
    elbow at the rest upper-arm length, so the elbow residual is
    ``| ||captured_elbow - origin|| - L_upper |``: a LENGTH error alone, exactly the shape
    of the trunk floor. The wrist is then chained from that PLACED elbow, not from the
    captured one -- because that is where the delivered elbow will be -- so its floor is
    ``| ||captured_wrist - placed_elbow|| - L_lower |`` and it inherits the elbow's
    length error as a displacement of its own origin.

    Read from EACH delivery's own file, never carried across arms: a different `UpperArm`
    origin is a different floor, and quoting one build's floor against another would be
    scoring against a number instead of a construction (`trunk_length_floor`'s own note).

    BLIND TO the same things the instrument is: this is a floor on PLACEMENT, and a bone
    that reaches its landmark can still be rolled about its own axis.
    """

    rest = track_rest(directory, subject)
    index, world = delivered_positions(directory, subject)
    capture = capture_positions(directory, subject)
    out: dict[str, np.ndarray] = {}
    for side, upper, lower, hand, landmark in ARM_SIDES:
        length_upper = float(np.linalg.norm(rest[lower]))
        length_lower = float(np.linalg.norm(rest[hand]))
        origin = world[:, index[upper]]
        elbow = capture[:, JOINT_INDEX[f"{landmark}_elbow"]]
        wrist = capture[:, JOINT_INDEX[f"{landmark}_wrist"]]
        shoulder = capture[:, JOINT_INDEX[f"{landmark}_shoulder"]]
        reach = elbow - origin
        span = np.linalg.norm(reach, axis=1)
        placed = origin + (length_upper / span)[:, None] * reach
        reach_wrist = wrist - placed
        span_wrist = np.linalg.norm(reach_wrist, axis=1)
        out[f"{side}_elbow"] = 1000.0 * np.abs(span - length_upper)
        out[f"{side}_wrist"] = 1000.0 * np.abs(span_wrist - length_lower)
        out[f"{side}_captured_upper_arm_mm"] = 1000.0 * np.linalg.norm(
            elbow - shoulder, axis=1)
        out[f"{side}_rest_upper_arm_mm"] = np.full(len(span), 1000.0 * length_upper)
        out[f"{side}_rest_lower_arm_mm"] = np.full(len(span), 1000.0 * length_lower)
    return out


def arm_aim_floor(directory: Path, subject: int, masks: dict[str, np.ndarray],
                  delivered: dict[str, np.ndarray] | None = None) -> dict:
    """The per-side floor summarised, with the delivered arm's EXCESS over it beside it.

    `delivered` is this delivery's own per-joint error arrays (`measure`'s rows). The
    excess is the figure a 3 mm band cannot see on its own: after D9 the delivered elbow
    residual should EQUAL the floor to storage precision, because it is length-only by
    construction, and a gap larger than that means the change did more than aiming.
    """

    per_frame = arm_aim_floor_per_frame(directory, subject)
    frames = len(per_frame["left_elbow"])
    window = np.zeros(frames, dtype=bool)
    window[max(WINDOW_FIRST_ID - FIRST_FRAME_ID, 0):
           min(WINDOW_LAST_ID - FIRST_FRAME_ID + 1, frames)] = True
    out: dict = {
        "definition": (
            "elbow: the residual left when the elbow is put ON THE RAY from the "
            "delivered UpperArm origin to the captured elbow, at the rest upper-arm "
            "length -- a LENGTH error alone. wrist: chained from that PLACED elbow to "
            "the captured wrist at the rest lower-arm length."),
        "read_from": "this delivery's own GLB origin and its own captured landmarks",
        "frame_ids": {"first_array_index_is_frame_id": FIRST_FRAME_ID,
                      "window": [WINDOW_FIRST_ID, WINDOW_LAST_ID]},
        "sides": {},
    }
    for side, upper, lower, hand, _landmark in ARM_SIDES:
        row: dict = {
            "rest_upper_arm_mm": round(float(per_frame[f"{side}_rest_upper_arm_mm"][0]), 2),
            "rest_lower_arm_mm": round(float(per_frame[f"{side}_rest_lower_arm_mm"][0]), 2),
        }
        for part, joint in (("elbow", lower), ("wrist", hand)):
            values = per_frame[f"{side}_{part}"]
            cell = {
                "median_mm": _round(np.median(values)),
                "p95_mm": _round(np.percentile(values, 95)),
                "max_mm": _round(np.max(values)),
                "window_median_mm": _round(np.median(values[window])),
                "bent_tercile_median_mm": _round(np.median(values[masks["bent"]])),
                "upright_tercile_median_mm": _round(np.median(values[masks["upright"]])),
            }
            if delivered is not None:
                excess = delivered[joint] - values
                cell["delivered_minus_floor_mm"] = {
                    "median": _round(np.median(excess)),
                    "p95": _round(np.percentile(excess, 95)),
                    "max_abs": _round(np.max(np.abs(excess))),
                }
            row[part] = cell
        # The length-limited frame the D9 card names. Reported ALWAYS, not only when it
        # is bad: a floor that is 36 mm on one frame is not an aim failing, it is a
        # captured upper arm longer than the performer's own bone (a D8 leftover on the
        # yellow), and the card pre-registered it as such.
        row["frame_id_118"] = {
            "array_index": 118 - FIRST_FRAME_ID,
            "elbow_floor_mm": _round(per_frame[f"{side}_elbow"][118 - FIRST_FRAME_ID]),
            "wrist_floor_mm": _round(per_frame[f"{side}_wrist"][118 - FIRST_FRAME_ID]),
            "captured_upper_arm_mm": _round(
                per_frame[f"{side}_captured_upper_arm_mm"][118 - FIRST_FRAME_ID]),
            "why": ("length-limited, not aim-limited: compare `captured_upper_arm_mm` "
                    "with `rest_upper_arm_mm` above. Where the captured segment is "
                    "LONGER than the performer's own rest bone no aim can reach it (D9's "
                    "card names performer 1's left arm here: 333 against a rest of 277, "
                    "a D8 leftover on the yellow); where it is shorter the bone "
                    "overshoots by the same construction"),
        }
        out["sides"][side] = row
    return out


# ------------------------------------------------------------------------------ the sweep
def measure(directory: Path, reference: str = "smoothed") -> dict[int, dict[str, np.ndarray]]:
    """Per-frame error in mm for every mapped joint, per subject. The raw arrays.

    Under `--reference raw` a frame where the landmark did not triangulate carries NaN
    rather than a number, and every statistic below is NaN-aware. The tilt terciles are
    still computed from the SMOOTHED array: they are a frame CLASSIFIER, not a score, and
    the classifier must be the same for every arm and every joint or the terciles stop
    meaning one thing. It is byte-identical across the arms this instrument compares,
    which the caller asserts.
    """

    out: dict[int, dict[str, np.ndarray]] = {}
    for subject in (0, 1):
        capture = capture_positions(directory, subject, reference)
        index, world = delivered_positions(directory, subject)
        rows: dict[str, np.ndarray] = {}
        for rig, landmark in PAIRS:
            rows[rig] = 1000.0 * np.linalg.norm(
                world[:, index[rig]] - capture[:, JOINT_INDEX[landmark]], axis=1)
        hip_mid = 0.5 * (capture[:, JOINT_INDEX["left_hip"]]
                         + capture[:, JOINT_INDEX["right_hip"]])
        rows["hips_joint_vs_hip_midpoint"] = 1000.0 * np.linalg.norm(
            world[:, index["Hips"]] - hip_mid, axis=1)
        rows["leg_root_midpoint_vs_hip_midpoint"] = 1000.0 * np.linalg.norm(
            0.5 * (world[:, index["LeftUpperLeg"]] + world[:, index["RightUpperLeg"]])
            - hip_mid, axis=1)
        rows["_tilt_deg"] = tilt_degrees(capture_positions(directory, subject, "smoothed"))
        out[subject] = rows
    return out


def report(deliveries: dict[str, Path], reference_mode: str = "smoothed") -> dict:
    labels = list(deliveries)
    arms = {label: measure(path, reference_mode) for label, path in deliveries.items()}
    # SAME DENOMINATOR. Every arm must be scored against the same landmarks, or the
    # difference between two arms is not the delivery.
    #
    # WHICH ARRAY CARRIES THAT PROPERTY DEPENDS ON THE STEP. Up to D7b the change lived
    # inside the converter and could not move a landmark, so the SMOOTHED array was
    # identical across arms and was the reference. D8 repairs the capture between the raw
    # triangulation and the smoothed array, so the smoothed array MOVES BY CONSTRUCTION
    # and can no longer be the shared denominator; the RAW array is captured before the
    # repair and takes its place. `--reference raw` is that mode, and it also stops the
    # instrument scoring a D8 delivery against points D8 itself repaired.
    shared = {}
    for subject in (0, 1):
        first = capture_positions(deliveries[labels[0]], subject, reference_mode)
        shared[f"subject_{subject:02d}"] = {
            label: bool(np.array_equal(first,
                                       capture_positions(path, subject, reference_mode),
                                       equal_nan=True))
            for label, path in deliveries.items()}
    payload: dict = {
        "title": "delivered joints, from each GLB's own bytes, against the captured landmarks",
        "reference_mode": reference_mode,
        "reference": REFERENCE_TEXT[reference_mode],
        "blind_to": [
            "orientation -- a joint on its landmark can still be rotated about it",
            "the MESH bound to the skeleton (that is the silhouette's job)",
            "common-mode detector error, which is inside the reference",
            "whether a change is right in the world; only whether the delivered skeleton "
            "sits closer to the points that delivery was solved onto",
        ],
        "retarget_cost_is_blind_to_this_class": (
            "tools/swap-harness/retarget_cost.py RE-SOLVES the track through the converter "
            "with no spine landmark, so it reads D3's torso figure on the D7 delivery "
            "unchanged. It is kept as it is and labelled."),
        "bootstrap": {"block": BLOCK, "draws": DRAWS, "seed": SEED,
                      "paired": "the per-frame DIFFERENCE array is resampled, on draws "
                                "shared by every arm and every joint"},
        "deliveries": {label: str(path) for label, path in deliveries.items()},
        "triangulated_landmarks_byte_identical_across_arms": shared,
        "subjects": {},
    }
    all_shared = all(all(row.values()) for row in shared.values())
    payload["same_denominator"] = all_shared

    for subject in (0, 1):
        frames = len(arms[labels[0]][subject]["_tilt_deg"])
        masks = tercile_masks(arms[labels[0]][subject]["_tilt_deg"])
        starts = block_starts(frames)
        joints = [name for name in arms[labels[0]][subject] if not name.startswith("_")]
        per_joint: dict = {}
        for joint in joints:
            row: dict = {label: summarise(arms[label][subject][joint], masks)
                         for label in labels}
            differences: dict = {}
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    delta = arms[b][subject][joint] - arms[a][subject][joint]
                    differences[f"{b}_minus_{a}"] = {
                        # NaN-aware: under `--reference raw` a frame where the landmark
                        # did not triangulate carries NaN in BOTH arms and drops out of
                        # the pairing. Identical to np.median on the NaN-free default.
                        "median_mm": _round(np.nanmedian(delta))
                        if np.isfinite(delta).any() else None,
                        "frames_paired": int(np.isfinite(delta).sum()),
                        "ci95_mm": bootstrap_median(delta, starts),
                        "bent_tercile_median_mm": (
                            _round(np.nanmedian(delta[masks["bent"]]))
                            if np.isfinite(delta[masks["bent"]]).any() else None),
                        "bent_tercile_ci95_mm": bootstrap_median(
                            delta[masks["bent"]],
                            block_starts(int(masks["bent"].sum()))),
                    }
            row["paired_differences"] = differences
            per_joint[joint] = row
        floors = {}
        for label, path in deliveries.items():
            residual, length = trunk_length_floor(path, subject)
            floors[label] = {
                "L_rest_mm": round(length, 2),
                "median_mm": round(float(np.median(residual)), 3),
                "bent_tercile_median_mm": round(
                    float(np.median(residual[masks["bent"]])), 3),
                "upright_tercile_median_mm": round(
                    float(np.median(residual[masks["upright"]])), 3),
                "p95_mm": round(float(np.percentile(residual, 95)), 3),
            }
        payload["subjects"][f"subject_{subject:02d}"] = {
            "frames": frames,
            "tilt_deg": {
                "median": round(float(np.median(arms[labels[0]][subject]["_tilt_deg"])), 2),
                "tercile_edges": [
                    round(float(np.percentile(arms[labels[0]][subject]["_tilt_deg"], 33.3)), 2),
                    round(float(np.percentile(arms[labels[0]][subject]["_tilt_deg"], 66.7)), 2)],
                "bent_frames": int(masks["bent"].sum()),
                "upright_frames": int(masks["upright"].sum())},
            "joints": per_joint,
            "groups": GROUPS,
            "trunk_length_floor": floors,
            "arm_aim_floor": {
                label: arm_aim_floor(path, subject, masks, arms[label][subject])
                for label, path in deliveries.items()},
            "distributed_flexion_would_recover": {
                label: distributed_flexion_recovery(path, subject)
                for label, path in deliveries.items()},
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", action="append", required=True,
                        metavar="LABEL=PATH", help="repeatable; order is preserved")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", choices=sorted(REFERENCE_ARRAY), default="smoothed",
                        help="which of the delivery's own landmark arrays to score "
                             "against. `smoothed` is the historical default and its "
                             "output is byte-identical to before this option existed. "
                             "`raw` is D8's: the pre-fill, pre-solve, pre-smoothing "
                             "points, NaNs intact, which is the one array a D8 build "
                             "shares bit for bit with a D7b build.")
    args = parser.parse_args()
    deliveries: dict[str, Path] = {}
    for entry in args.delivery:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--delivery wants LABEL=PATH, got {entry!r}")
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if not resolved.exists():
            raise SystemExit(f"{resolved} does not exist")
        deliveries[label] = resolved
    payload = report(deliveries, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"reference: {payload['reference_mode']}")
    for name, subject in payload["subjects"].items():
        print(f"\n{name}  frames {subject['frames']}  tilt median "
              f"{subject['tilt_deg']['median']} deg")
        header = "  ".join(f"{label:>10}" for label in deliveries)
        print(f"  {'joint':<34}{header}")
        for joint, row in subject["joints"].items():
            cells = "  ".join(
                f"{(row[label].get('median_mm') if row[label].get('median_mm') is not None else float('nan')):>10.1f}"
                for label in deliveries)
            print(f"  {joint:<34}{cells}")
        print(f"  {'-- bent tercile --':<34}")
        for joint, row in subject["joints"].items():
            cells = "  ".join(
                f"{(row[label].get('bent_tercile_median_mm') if row[label].get('bent_tercile_median_mm') is not None else float('nan')):>10.1f}"
                for label in deliveries)
            print(f"  {joint:<34}{cells}")
        print(f"  trunk length floor: {json.dumps(subject['trunk_length_floor'])}")
        for label, floor in subject["arm_aim_floor"].items():
            for side, row in floor["sides"].items():
                print(f"  arm aim floor [{label}] {side:<5} "
                      f"elbow {row['elbow']['median_mm']:>6.2f} "
                      f"(p95 {row['elbow']['p95_mm']:>6.2f}, window "
                      f"{row['elbow']['window_median_mm']:>6.2f})  "
                      f"wrist {row['wrist']['median_mm']:>6.2f} "
                      f"(p95 {row['wrist']['p95_mm']:>6.2f}, window "
                      f"{row['wrist']['window_median_mm']:>6.2f})  "
                      f"frame 118 {row['frame_id_118']['elbow_floor_mm']:>6.2f} / "
                      f"{row['frame_id_118']['wrist_floor_mm']:>6.2f}")
    print(f"\nsame denominator: {payload['same_denominator']}")
    print(f"wrote {args.out}")
    return 0 if payload["same_denominator"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
