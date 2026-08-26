#!/usr/bin/env python
"""Build a HumanIK skeleton that carries the macap base model's exact anatomy.

HumanIK's skeleton generator ships a generic 1.73 m figure. Every slot's rest
position is a plain attribute on ``HIKSkeletonGeneratorNode`` though, so the
generator can be driven from the macap base skeleton instead. The result is a
HumanIK-native rig - canonical slot names, auto-characterizes anywhere - whose
bone lengths, joint placement, and finger spacing are the base model's own, so
retargeting macap takes onto it is close to an identity operation.

Run through mayapy:

    /Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy \
        scripts/build_macap_hik_skeleton.py BASE.fbx OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_macap_to_maya_hik import (  # noqa: E402
    FINGER_DIGITS,
    HIK_MAPPING,
    SOURCE_NAMESPACE,
    _import_into_namespace,
    _sha256,
    _start_maya,
    _top_ancestor,
)


CHARACTER_NAME = "macap"
# The generator lays a skeleton out in centimetres, Y up, standing on y = 0.
UNIT_SCALE = 1.0
GROUND_TOLERANCE_CM = 1e-6
PLACEMENT_TOLERANCE_CM = 0.01
# Structure switches chosen so the generated slot set matches the macap rig:
# three spine joints, one neck, one clavicle per side, three bones per digit.
GENERATOR_STRUCTURE = {
    "SpineCount": 3,
    "NeckCount": 1,
    "ShoulderCount": 1,
    "FingerJointCount": 3,
    "ToeJointCount": 1,
    "WantIndexFinger": True,
    "WantMiddleFinger": True,
    "WantRingFinger": True,
    "WantPinkyFinger": True,
    "WantExtraFinger": False,
    "WantFingerBase": False,
    "WantInHandJoint": False,
    "WantFootThumb": False,
    "WantInFootJoint": False,
    "WantHipsTranslation": False,
    "WantUpperArmRollBone": False,
    "WantLowerArmRollBone": False,
    "WantUpperLegRollBone": False,
    "WantLowerLegRollBone": False,
}


def _arguments():
    values = sys.argv[1:]
    if len(values) != 3:
        raise SystemExit("Expected BASE.fbx OUTPUT.fbx REPORT.json")
    base, output, report = (os.path.abspath(value) for value in values)
    if not os.path.isfile(base):
        raise SystemExit("Base model is missing: %s" % base)
    if not output.lower().endswith(".fbx"):
        raise SystemExit("Output must use the .fbx extension: %s" % output)
    for path in (output, report):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    return base, output, report


def _upright(position):
    """Map the imported frame to Y up.

    The macap exporter leaves the Z-up to Y-up conversion as a -90 degree X
    rotation on the exported roots, so Maya reads the model lying along -Z. This
    is done arithmetically rather than by rotating the scene: the mesh is bound
    to the skeleton, so rotating its transform as well would double-transform it.
    """

    return [position[0], -position[2], position[1]]


def _base_rest(cmds, path):
    """Import the base model and read its rest skeleton, standing on y = 0."""

    _import_into_namespace(cmds, path, SOURCE_NAMESPACE, file_type="FBX")
    joints = cmds.ls("%s:*" % SOURCE_NAMESPACE, type="joint", long=True) or []
    if not joints:
        raise RuntimeError("Base model FBX contains no joints")
    assemblies = cmds.ls("%s:*" % SOURCE_NAMESPACE, assemblies=True, long=True) or []
    tilt = sorted({round(cmds.getAttr("%s.rotateX" % node), 3) for node in assemblies})
    if not tilt or not all(abs(value + 90.0) < 1.0 for value in tilt):
        raise RuntimeError("Unexpected root tilt on the base model: %r" % tilt)

    meshes = cmds.ls("%s:*" % SOURCE_NAMESPACE, type="mesh", noIntermediate=True, long=True)
    if not meshes:
        raise RuntimeError("Base model FBX contains no mesh to stand on the ground")
    sole = None
    for mesh in meshes:
        box = cmds.exactWorldBoundingBox(mesh)
        for x in (box[0], box[3]):
            for y in (box[1], box[4]):
                for z in (box[2], box[5]):
                    height = _upright([x, y, z])[1]
                    sole = height if sole is None else min(sole, height)

    rest = {}
    for joint in joints:
        name = joint.split("|")[-1].split(":")[-1]
        position = _upright(cmds.xform(joint, q=True, ws=True, t=True))
        rest[name] = [position[0], position[1] - sole, position[2]]
    return rest, sole


def _finger_tip(rest, side, digit):
    """Extrapolate a tip for HumanIK's fourth finger bone, which macap lacks."""

    distal = rest["%s_%s3" % (side, digit)]
    middle = rest["%s_%s2" % (side, digit)]
    direction = [distal[i] - middle[i] for i in range(3)]
    length = math.sqrt(sum(value * value for value in direction))
    if length < 1e-6:
        raise RuntimeError("Degenerate %s %s bone on the base model" % (side, digit))
    return [distal[i] + direction[i] for i in range(3)]


def _slot_positions(rest):
    """Every HumanIK slot the generated skeleton needs, in generator space."""

    positions = {}
    for macap_name, slot in HIK_MAPPING.items():
        if macap_name not in rest:
            raise RuntimeError("Base model has no joint %s" % macap_name)
        positions[slot] = rest[macap_name]
    for side, prefix in (("left", "Left"), ("right", "Right")):
        for digit, hik_digit in zip(
            FINGER_DIGITS, ("Thumb", "Index", "Middle", "Ring", "Pinky")
        ):
            positions["%sHand%s4" % (prefix, hik_digit)] = _finger_tip(rest, side, digit)
    return positions


def _create_generator(cmds, mel):
    mel.eval('hikCreateCharacter("%s")' % CHARACTER_NAME)
    character = mel.eval("hikGetCurrentCharacter()")
    if not character:
        raise RuntimeError("HumanIK did not create a character")
    generator = cmds.createNode("HIKSkeletonGeneratorNode")
    cmds.connectAttr("%s.CharacterNode" % generator, "%s.SkeletonGenerator" % character)
    mel.eval('hikReadDefaultCharPoseFileOntoSkeletonGeneratorNode("%s")' % generator)
    mel.eval('hikSetSkeletonGeneratorDefaults("%s")' % generator)
    return character, generator


def _apply(cmds, generator, positions):
    for attribute, value in GENERATOR_STRUCTURE.items():
        plug = "%s.%s" % (generator, attribute)
        if cmds.objExists(plug):
            cmds.setAttr(plug, value)
    applied, skipped = {}, []
    for slot, position in sorted(positions.items()):
        plug = "%s.%sT" % (generator, slot)
        if not cmds.objExists(plug):
            skipped.append(slot)
            continue
        for axis, value in zip("xyz", position):
            cmds.setAttr("%s%s" % (plug, axis), float(value) * UNIT_SCALE)
        applied[slot] = [round(float(value), 6) for value in position]
    return applied, skipped


def main():
    base_path, output_path, report_path = _arguments()
    cmds, mel = _start_maya()
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc")

    rest, sole = _base_rest(cmds, base_path)
    positions = _slot_positions(rest)
    character, generator = _create_generator(cmds, mel)
    applied, skipped = _apply(cmds, generator, positions)
    mel.eval('hikSetCurrentCharacter("%s")' % character)
    mel.eval("hikUpdateCurrentSkeleton()")

    # The generator distributes the intermediate spine joints evenly between the
    # first spine and the neck instead of honouring their supplied positions, so
    # the built skeleton is nudged onto the requested rest afterwards. Working
    # root-to-leaf means each correction carries its subtree, and only positions
    # move: HumanIK's own bone orientations are left alone.
    prefix = "%s_" % character
    built = [
        joint for joint in cmds.ls(type="joint", long=True) or []
        if joint.split("|")[-1].startswith(prefix)
    ]
    for joint in sorted(built, key=lambda name: name.count("|")):
        slot = joint.split("|")[-1][len(prefix):]
        if slot in applied:
            cmds.xform(joint, worldSpace=True, translation=applied[slot])
    mel.eval('hikCharacterLock("%s", 1, 1)' % character)

    # hikCreateCharacter uniquifies the name, and the generator prefixes every
    # joint with whatever it settled on.
    prefix = "%s_" % character
    generated = {
        joint.split("|")[-1]: cmds.xform(joint, q=True, ws=True, t=True)
        for joint in cmds.ls(type="joint", long=True) or []
        if joint.split("|")[-1].startswith(prefix)
    }
    if not generated:
        raise RuntimeError("HumanIK produced no skeleton")

    errors, worst, missing = {}, 0.0, []
    for slot, wanted in sorted(applied.items()):
        name = "%s%s" % (prefix, slot)
        if name not in generated:
            missing.append(slot)
            continue
        actual = generated[name]
        distance = math.sqrt(sum((actual[i] - wanted[i]) ** 2 for i in range(3)))
        errors[slot] = round(distance, 6)
        worst = max(worst, distance)

    # Delete the imported base model so only the HumanIK skeleton is exported.
    cmds.delete(cmds.ls("%s:*" % SOURCE_NAMESPACE))
    roots = sorted({
        _top_ancestor(cmds, joint)
        for joint in cmds.ls(type="joint", long=True) or []
        if joint.split("|")[-1].startswith(prefix)
    })
    cmds.select(roots, replace=True, hierarchy=True)
    mel.eval("FBXExportBakeComplexAnimation -v false")
    mel.eval("FBXExportInputConnections -v false")
    mel.eval("FBXExportSkins -v false")
    mel.eval("FBXExportShapes -v false")
    mel.eval("FBXExportConstraints -v false")
    mel.eval("FBXExportUpAxis y")
    mel.eval('FBXExport -f "%s" -s' % output_path.replace("\\", "/"))
    if not os.path.isfile(output_path):
        raise RuntimeError("Skeleton export produced no FBX")

    # FBX cannot carry an HIKCharacterNode, so a Maya scene is written too: that
    # is what the retarget script can consume directly as a characterized target.
    scene_path = os.path.splitext(output_path)[0] + ".ma"
    cmds.file(rename=scene_path)
    cmds.file(save=True, type="mayaAscii", force=True)
    if not os.path.isfile(scene_path):
        raise RuntimeError("Skeleton export produced no Maya scene")

    failures = []
    if skipped:
        failures.append("generator has no slot for: %s" % ", ".join(sorted(skipped)))
    if missing:
        failures.append("skeleton is missing: %s" % ", ".join(sorted(missing)))
    if worst > PLACEMENT_TOLERANCE_CM:
        failures.append("worst joint placement error is %.6f cm" % worst)

    report = {
        "schema_version": "autoanim.macap-hik-skeleton/1.0",
        "status": "passed" if not failures else "failed",
        "maya_version": cmds.about(version=True),
        "character_name": character,
        "base_model": base_path,
        "output_fbx": output_path,
        "output_scene": scene_path,
        "ground_offset_cm": sole,
        "slots_driven": len(applied),
        "joints_generated": len(generated),
        "worst_placement_error_cm": worst,
        "placement_errors_cm": errors,
        "structure": GENERATOR_STRUCTURE,
        "base_sha256": _sha256(base_path),
        "output_sha256": _sha256(output_path),
        "output_scene_sha256": _sha256(scene_path),
        "failures": failures,
    }
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "slots_driven", "joints_generated",
        "worst_placement_error_cm", "failures")}))
    if failures:
        raise RuntimeError("macap HumanIK skeleton build failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
