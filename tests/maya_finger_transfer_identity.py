#!/usr/bin/env python
"""Run the source-derived HumanIK finger identity case under mayapy."""

from __future__ import annotations

import json
import math
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from retarget_macap_to_maya_hik import (  # noqa: E402
    HIK_MAPPING,
    SOURCE_NAMESPACE,
    _characterize,
    _configure_retarget,
    _deform_joints,
    _finger_joints,
    _finger_pairs,
    _import_source,
    _import_target,
    _local_rest,
    _restore_pose,
    _rest_pose,
    _set_source,
    _start_maya,
    _transfer_fingers,
    _world_rest,
)


SOURCE = os.path.join(
    ROOT, "artifacts", "macap-base-model", "macap-Base-model-5s-id-00.fbx"
)
TARGET = os.path.join(ROOT, "artifacts", "macap-hik", "macap-HIK-Skeleton.ma")


def _abduction(cmds, bases, tips):
    directions = []
    for base, tip in zip(bases, tips):
        a = cmds.xform(base, q=True, ws=True, t=True)
        b = cmds.xform(tip, q=True, ws=True, t=True)
        vector = [b[index] - a[index] for index in range(3)]
        length = math.sqrt(sum(value * value for value in vector))
        directions.append([value / length for value in vector])
    angles = []
    for first, second in zip(directions[:-1], directions[1:]):
        dot = sum(first[index] * second[index] for index in range(3))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
    return sum(angles) / len(angles)


def _rotation_error_degrees(first, second):
    from maya.api.OpenMaya import MTransformationMatrix

    first_q = MTransformationMatrix(first).rotation(asQuaternion=True)
    second_q = MTransformationMatrix(second).rotation(asQuaternion=True)
    dot = abs(
        first_q.x * second_q.x
        + first_q.y * second_q.y
        + first_q.z * second_q.z
        + first_q.w * second_q.w
    )
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def main():
    cmds, mel = _start_maya()
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc")

    source_joints = _import_source(cmds, SOURCE)
    keys = cmds.keyframe(source_joints, q=True, timeChange=True) or []
    start, end = int(round(min(keys))), int(round(max(keys)))

    muted = _rest_pose(cmds, source_joints)
    source_fingers = [
        cmds.ls("%s:%s" % (SOURCE_NAMESPACE, name), long=True)[0]
        for name in _finger_joints()
    ]
    source_rest = _world_rest(cmds, source_fingers)
    source_local_rest = _local_rest(cmds, source_fingers)
    source_character, _ = _characterize(cmds, mel, source_joints)
    _restore_pose(cmds, muted)

    target_character = _import_target(cmds, mel, TARGET)
    pairs = _finger_pairs(cmds, mel, target_character)
    target_rest = _world_rest(cmds, [target for _, target in pairs])
    target_local_rest = _local_rest(cmds, [target for _, target in pairs])
    rest_basis_offsets = []
    for source, target in pairs:
        rest_basis_offsets.append(
            _rotation_error_degrees(
                source_local_rest[source][0], target_local_rest[target][0]
            )
        )
    deform = _deform_joints(cmds, mel, target_character)
    _set_source(mel, target_character, source_character)
    _configure_retarget(cmds, target_character)
    cmds.bakeResults(
        deform,
        simulation=True,
        time=(start, end),
        sampleBy=1,
        disableImplicitControl=True,
        preserveOutsideKeys=False,
        sparseAnimCurveBake=False,
        removeBakedAttributeFromLayer=False,
        bakeOnOverrideLayer=False,
        minimizeRotation=True,
        controlPoints=False,
        shape=False,
    )

    diagnostics = {}
    _transfer_fingers(
        cmds,
        pairs,
        start,
        end,
        rest_tables=(source_rest, target_rest),
        diagnostics=diagnostics,
    )

    digits = ("index", "middle", "ring", "pinky")
    source_bases = [
        cmds.ls("%s:left_%s1" % (SOURCE_NAMESPACE, digit), long=True)[0]
        for digit in digits
    ]
    source_tips = [
        cmds.ls("%s:left_%s2" % (SOURCE_NAMESPACE, digit), long=True)[0]
        for digit in digits
    ]
    pair_by_name = {
        source.split(":")[-1]: target for source, target in pairs
    }
    target_bases = [pair_by_name["left_%s1" % digit] for digit in digits]
    target_tips = [pair_by_name["left_%s2" % digit] for digit in digits]
    abduction_errors = []
    for frame in range(start, end + 1):
        cmds.currentTime(frame)
        abduction_errors.append(
            abs(
                _abduction(cmds, source_bases, source_tips)
                - _abduction(cmds, target_bases, target_tips)
            )
        )

    parent_mismatches = []
    for source, target in pairs:
        source_parent = cmds.listRelatives(source, parent=True, fullPath=True)[0]
        target_parent = cmds.listRelatives(target, parent=True, fullPath=True)[0]
        source_parent_name = source_parent.split(":")[-1]
        target_parent_name = target_parent.split("|")[-1].split(":")[-1]
        expected_suffix = HIK_MAPPING[source_parent_name]
        if not target_parent_name.endswith("_" + expected_suffix):
            parent_mismatches.append([source, target])

    summary = {
        "frame_range": [start, end],
        "pair_count": len(pairs),
        "parent_mapping_mismatches": len(parent_mismatches),
        "source_rotate_orders": sorted(
            {int(cmds.getAttr("%s.rotateOrder" % source)) for source, _ in pairs}
        ),
        "target_rotate_orders": sorted(
            {int(cmds.getAttr("%s.rotateOrder" % target)) for _, target in pairs}
        ),
        "mean_rest_basis_offset_deg": sum(rest_basis_offsets) / len(rest_basis_offsets),
        "max_rest_basis_offset_deg": max(rest_basis_offsets),
        "stage_mean_residual_deg": {
            stage: sum(values) / len(values) for stage, values in diagnostics.items()
        },
        "stage_max_residual_deg": {
            stage: max(values) for stage, values in diagnostics.items()
        },
        "mean_finger_abduction_error_deg": sum(abduction_errors) / len(abduction_errors),
        "worst_finger_abduction_error_deg": max(abduction_errors),
    }
    print("FINGER_IDENTITY " + json.dumps(summary, sort_keys=True))
    if os.environ.get("ASSERT_FINGER_IDENTITY") == "1":
        if (
            summary["parent_mapping_mismatches"]
            or summary["mean_finger_abduction_error_deg"] > 1.0e-3
            or max(summary["stage_max_residual_deg"].values()) > 1.0e-5
        ):
            raise AssertionError(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
