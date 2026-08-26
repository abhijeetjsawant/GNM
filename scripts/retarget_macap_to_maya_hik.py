#!/usr/bin/env python
"""Retarget a macap animated FBX onto one of Maya's shipped HumanIK characters.

Maya ships fully HumanIK-rigged example characters under
``Examples/Animation/Rigs``. Each already carries an ``HIKCharacterNode`` and a
solver, so the work here is to characterize the macap SMPL-X skeleton as a
HumanIK source, point the shipped character at it, bake the retargeted motion
onto the target's deform skeleton, and export that alone.

Run through mayapy:

    /Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy \
        scripts/retarget_macap_to_maya_hik.py \
        SOURCE.fbx TARGET.ma OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys


# macap SMPL-X joint -> HumanIK slot. Every HumanIK required slot is covered,
# plus the optional spine, shoulder, toe, and finger slots the rig can fill.
HIK_MAPPING = {
    "pelvis": "Hips",
    "spine1": "Spine",
    "spine2": "Spine1",
    "spine3": "Spine2",
    "neck": "Neck",
    "head": "Head",
}
for _side, _hik in (("left", "Left"), ("right", "Right")):
    HIK_MAPPING[f"{_side}_hip"] = f"{_hik}UpLeg"
    HIK_MAPPING[f"{_side}_knee"] = f"{_hik}Leg"
    HIK_MAPPING[f"{_side}_ankle"] = f"{_hik}Foot"
    HIK_MAPPING[f"{_side}_foot"] = f"{_hik}ToeBase"
    HIK_MAPPING[f"{_side}_collar"] = f"{_hik}Shoulder"
    HIK_MAPPING[f"{_side}_shoulder"] = f"{_hik}Arm"
    HIK_MAPPING[f"{_side}_elbow"] = f"{_hik}ForeArm"
    HIK_MAPPING[f"{_side}_wrist"] = f"{_hik}Hand"
    for _digit, _hik_digit in (
        ("thumb", "Thumb"),
        ("index", "Index"),
        ("middle", "Middle"),
        ("ring", "Ring"),
        ("pinky", "Pinky"),
    ):
        for _bone in (1, 2, 3):
            HIK_MAPPING[f"{_side}_{_digit}{_bone}"] = f"{_hik}Hand{_hik_digit}{_bone}"

FINGER_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SOURCE_NAMESPACE = "macap"
TARGET_NAMESPACE = "hik"
SOURCE_CHARACTER = "macapSource"
# Opening the shipped .ma scenes reports this on machines without the stereo
# plug-in. It does not affect the rig, so it is the one tolerated load error.
TOLERATED_LOAD_ERRORS = ("stereoCamera",)
# Pass this instead of a scene path to retarget onto HumanIK's own generated
# skeleton: canonical slot names, four bones per digit, and no mesh, so nothing
# in the delivered file depends on Autodesk's example characters.
GENERATED_TARGET = "hik-skeleton"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments():
    values = sys.argv[1:]
    if len(values) != 4:
        raise SystemExit("Expected SOURCE.fbx TARGET.ma OUTPUT.fbx REPORT.json")
    source, output, report = (os.path.abspath(values[i]) for i in (0, 2, 3))
    target = values[1] if values[1] == GENERATED_TARGET else os.path.abspath(values[1])
    paths = (source,) if target == GENERATED_TARGET else (source, target)
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit("Required source file is missing: %s" % path)
    if not output.lower().endswith(".fbx"):
        raise SystemExit("Output must use the .fbx extension: %s" % output)
    for path in (output, report):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    return source, target, output, report


def _start_maya():
    import maya.standalone

    maya.standalone.initialize("python")
    import maya.cmds as cmds
    import maya.mel as mel

    for plugin in ("mayaHIK", "mayaCharacterization", "fbxmaya", "quatNodes", "matrixNodes"):
        try:
            cmds.loadPlugin(plugin, quiet=True)
        except RuntimeError:
            pass
    for script in (
        "hikGlobalUtils.mel",
        "hikCharacterControlsUI.mel",
        "hikCharacterControlsUtils.mel",
        "hikDefinitionOperations.mel",
        "hikInputSourceUtils.mel",
    ):
        mel.eval('source "%s"' % script)
    return cmds, mel


def _import_into_namespace(cmds, path, namespace, file_type=None):
    """The FBX plug-in ignores file(namespace=...), so set the namespace first.

    Both skeletons carry joints called ``head``, ``neck`` and ``jaw``; without
    separate namespaces the mapping would silently address the wrong rig.
    """

    if not cmds.namespace(exists=namespace):
        cmds.namespace(add=namespace)
    cmds.namespace(set=":%s" % namespace)
    try:
        options = {"i": True, "ignoreVersion": True, "prompt": False}
        if file_type:
            options["type"] = file_type
            options["options"] = "fbx"
        else:
            options["loadReferenceDepth"] = "none"
        cmds.file(path, **options)
    finally:
        cmds.namespace(set=":")


def _import_source(cmds, path):
    """Import the macap take and stand it up in Maya's Y-up world.

    Blender writes the Z-up to Y-up conversion as a -90 degree X rotation on the
    exported root rather than into the data, so Maya reads the character lying
    along -Z. Parenting under one group and rotating it back is a rigid fix that
    leaves every joint transform and the skinning untouched.
    """

    _import_into_namespace(cmds, path, SOURCE_NAMESPACE, file_type="FBX")
    joints = cmds.ls("%s:*" % SOURCE_NAMESPACE, type="joint", long=True) or []
    if not joints:
        raise RuntimeError("Imported macap FBX contains no joints")
    assemblies = cmds.ls("%s:*" % SOURCE_NAMESPACE, assemblies=True, long=True) or []
    tilt = sorted({round(cmds.getAttr("%s.rotateX" % node), 3) for node in assemblies})
    if tilt and all(abs(value + 90.0) < 1.0 for value in tilt):
        group = cmds.group(assemblies, name="%s_upright" % SOURCE_NAMESPACE, world=True)
        cmds.setAttr("%s.rotateX" % group, 90.0)
    elif tilt and any(abs(value) > 1.0 for value in tilt):
        raise RuntimeError("Unexpected root tilt on the macap import: %r" % tilt)
    return cmds.ls("%s:*" % SOURCE_NAMESPACE, type="joint", long=True)


def _rest_pose(cmds, joints):
    """Mute the take and restore the bind pose so HumanIK characterizes at rest.

    Zeroing the rotate channels is not the bind pose here: the exporter writes
    each bone's orientation into ``rotate`` rather than ``jointOrient``, so
    zeroing collapses the whole skeleton into one vertical chain with the legs
    pointing up. HumanIK would then characterize that as the reference stance
    and retarget limbs inverted. The FBX's own bindPose is restored instead,
    locally so the uprighting group still applies.
    """

    muted = []
    for joint in joints:
        for attribute in (
            "rotateX", "rotateY", "rotateZ",
            "translateX", "translateY", "translateZ",
        ):
            plug = "%s.%s" % (joint, attribute)
            if cmds.connectionInfo(plug, isDestination=True):
                cmds.mute(plug)
                muted.append(plug)
    poses = cmds.ls("%s:*" % SOURCE_NAMESPACE, type="dagPose") or []
    if not poses:
        raise RuntimeError("Imported macap FBX carries no bind pose to characterize")
    for pose in poses:
        cmds.dagPose(pose, restore=True)
    return muted


def _restore_pose(cmds, muted):
    for plug in muted:
        cmds.mute(plug, disable=True, force=True)


def _characterize(cmds, mel, joints):
    mel.eval('hikCreateCharacter("%s")' % SOURCE_CHARACTER)
    character = SOURCE_CHARACTER
    mapped, missing = {}, []
    by_name = {joint.split("|")[-1].split(":")[-1]: joint for joint in joints}
    for smplx_name, hik_name in sorted(HIK_MAPPING.items()):
        joint = by_name.get(smplx_name)
        if joint is None:
            missing.append(smplx_name)
            continue
        slot = int(mel.eval('hikGetNodeIdFromName("%s")' % hik_name))
        mel.eval('setCharacterObject("%s","%s",%d,0)' % (joint, character, slot))
        mapped[smplx_name] = hik_name
    if missing:
        raise RuntimeError("macap rig is missing mapped joints: %s" % ", ".join(missing))
    mel.eval("hikUpdateDefinitionUI()")
    # Locking with validation saves the reference stance. Without it HumanIK has
    # no characterized source and the target solves against nothing.
    mel.eval('hikCharacterLock("%s", 1, 1)' % character)
    if not int(mel.eval('hikIsDefinitionLocked("%s")' % character)):
        raise RuntimeError("HumanIK refused to lock the macap character definition")
    return character, mapped


def _finger_joints():
    """Source finger joints in root-to-tip order, which the transfer relies on."""

    order = []
    for side in ("left", "right"):
        for digit in FINGER_DIGITS:
            for bone in (1, 2, 3):
                order.append("%s_%s%d" % (side, digit, bone))
    return order


def _world_matrix(cmds, node):
    from maya.api.OpenMaya import MMatrix

    return MMatrix(cmds.xform(node, q=True, ws=True, matrix=True))


def _rotation_matrix(cmds, node):
    """A node's world orientation with translation and scale removed.

    The macap import carries a 100x unit scale and its own world offset while a
    generated HumanIK skeleton is authored in centimetres at the origin, so raw
    world matrices are not comparable between the two rigs. Only orientation is
    transferred, so the rest is stripped here rather than being carried through
    the composition and discarded at the end.
    """

    from maya.api.OpenMaya import MMatrix

    values = list(_world_matrix(cmds, node))
    rows = (values[0:3], values[4:7], values[8:11])
    flattened = []
    for row in rows:
        length = (row[0] ** 2 + row[1] ** 2 + row[2] ** 2) ** 0.5
        if length < 1e-12:
            raise RuntimeError("Degenerate orientation on %s" % node)
        flattened.extend([value / length for value in row] + [0.0])
    flattened.extend([0.0, 0.0, 0.0, 1.0])
    return MMatrix(flattened)


def _local_rest(cmds, joints):
    """Parent-relative rest matrix per joint, keyed by joint name."""

    rest = {}
    for joint in joints:
        parents = cmds.listRelatives(joint, parent=True, fullPath=True) or []
        if not parents:
            raise RuntimeError("Finger joint %s has no parent" % joint)
        rest[joint] = (
            _rotation_matrix(cmds, joint) * _rotation_matrix(cmds, parents[0]).inverse(),
            parents[0],
        )
    return rest


def _world_rest(cmds, joints):
    """World-space rest orientation per joint, keyed by joint name."""

    return {joint: _rotation_matrix(cmds, joint) for joint in joints}


def _finger_pairs(cmds, mel, target_character):
    pairs = []
    for name in _finger_joints():
        source = "%s:%s" % (SOURCE_NAMESPACE, name)
        slot = int(mel.eval('hikGetNodeIdFromName("%s")' % HIK_MAPPING[name]))
        target = mel.eval('hikGetSkNode("%s",%d)' % (target_character, slot))
        if not target or not cmds.objExists(target):
            raise RuntimeError("Target rig has no joint for %s" % HIK_MAPPING[name])
        pairs.append((cmds.ls(source, long=True)[0], cmds.ls(target, long=True)[0]))
    return pairs


def _transfer_fingers(cmds, pairs, start, end, rest_tables=None, diagnostics=None):
    """Copy finger rotations rig-to-rig instead of leaving them to HumanIK.

    HumanIK transfers finger curl faithfully but loses most of the adduction, so
    adjacent fingers close on one another. Both rigs carry the same three bones
    per digit, so each joint's world-rest rotation delta is applied to the
    target and resolved back through the target parent.

    The rigs can use different joint axes even when their rest positions match.
    Measure the source motion against its world-space rest frame, apply that
    delta to the target's world-space rest frame, then convert the result through
    the target parent. Maya matrices use row-vector composition, so local-to-world
    is ``local * parent_world``.
    """

    from maya.api.OpenMaya import MMatrix, MTransformationMatrix

    def record(stage, actual, expected):
        if diagnostics is None:
            return
        actual_q = MTransformationMatrix(MMatrix(actual)).rotation(asQuaternion=True)
        expected_q = MTransformationMatrix(MMatrix(expected)).rotation(asQuaternion=True)
        dot = abs(
            actual_q.x * expected_q.x
            + actual_q.y * expected_q.y
            + actual_q.z * expected_q.z
            + actual_q.w * expected_q.w
        )
        diagnostics.setdefault(stage, []).append(
            math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
        )

    if rest_tables is not None:
        source_rest, target_rest = rest_tables
    else:
        source_rest = _world_rest(cmds, [pair[0] for pair in pairs])
        target_rest = _world_rest(cmds, [pair[1] for pair in pairs])

    for frame in range(start, end + 1):
        cmds.currentTime(frame)
        for source, target in pairs:
            source_world_rest = source_rest[source]
            target_world_rest = target_rest[target]
            rest_basis = source_world_rest.inverse() * target_world_rest
            record("rest_pose", source_world_rest * rest_basis, target_world_rest)
            source_world = _rotation_matrix(cmds, source)
            delta = source_world_rest.inverse() * source_world
            record("parent_space_extraction", source_world_rest * delta, source_world)
            desired_world = target_world_rest * delta
            record("rest_delta_composition", desired_world, target_world_rest * delta)
            target_parent = cmds.listRelatives(target, parent=True, fullPath=True)[0]
            target_parent_world = _rotation_matrix(cmds, target_parent)
            target_local = desired_world * target_parent_world.inverse()
            desired = target_local * target_parent_world
            record("parent_to_world_composition", desired, desired_world)
            euler = MTransformationMatrix(MMatrix(desired)).rotation(asQuaternion=False)
            record("quaternion_to_euler", euler.asMatrix(), desired)
            cmds.xform(
                target,
                worldSpace=True,
                rotation=[math.degrees(value) for value in (euler.x, euler.y, euler.z)],
            )
            record("world_xform_assignment", _rotation_matrix(cmds, target), desired)
            cmds.setKeyframe(
                target, attribute=("rotateX", "rotateY", "rotateZ"), time=frame
            )
    return [pair[1] for pair in pairs]


def _import_target(cmds, mel, path):
    if path == GENERATED_TARGET:
        if not cmds.namespace(exists=TARGET_NAMESPACE):
            cmds.namespace(add=TARGET_NAMESPACE)
        # No namespace here: the generator prefixes joint names with the
        # character name, which would otherwise pick the namespace up twice.
        existing = set(cmds.ls(type="HIKCharacterNode") or [])
        if True:
            mel.eval("hikCreateSkeleton()")
            created = [
                node for node in cmds.ls(type="HIKCharacterNode") or []
                if node not in existing
            ]
            if len(created) != 1:
                raise RuntimeError("hikCreateSkeleton produced %r" % created)
            # The source character is still current at this point, so the
            # generator would otherwise rebuild the wrong skeleton.
            mel.eval('hikSetCurrentCharacter("%s")' % created[0])
            mel.eval("hikUpdateCurrentSkeleton()")
    else:
        _import_into_namespace(cmds, path, TARGET_NAMESPACE)
    if path == GENERATED_TARGET:
        characters = [
            node for node in cmds.ls(type="HIKCharacterNode") or [] if node not in existing
        ]
    else:
        characters = [
            node
            for node in cmds.ls(type="HIKCharacterNode")
            if node.startswith("%s:" % TARGET_NAMESPACE)
        ]
    if len(characters) != 1:
        raise RuntimeError("Expected one HumanIK character in the target, found %r" % characters)
    return characters[0]


def _top_ancestor(cmds, node):
    while True:
        parents = cmds.listRelatives(node, parent=True, fullPath=True) or []
        if not parents:
            return node
        node = parents[0]


def _deform_joints(cmds, mel, target_character):
    """Every joint under the characterized skeleton root, control rig excluded.

    Walking from the Hips slot works for both an imported rig and HumanIK's own
    generated skeleton, which carries no namespace to filter on.
    """

    root = _top_ancestor(cmds, _slot_node(cmds, mel, target_character, "Hips"))
    joints = [root] if cmds.nodeType(root) == "joint" else []
    joints += cmds.listRelatives(root, allDescendents=True, fullPath=True, type="joint") or []
    keep = [j for j in joints if "_Ctrl_" not in j.split("|")[-1]]
    if not keep:
        raise RuntimeError("Target rig exposes no deform joints")
    return keep


# HumanIK ships these compensations on so a retarget looks plausible on a
# differently proportioned body. They re-solve the hips from the target's own
# mass centre and leg length, which walks the root off the performer's path.
# Turning them off transfers the take instead of reinterpreting it.
RETARGET_PROPERTIES = {
    "ScaleCompensationMode": 0,
    "MassCenterCompensationMode": 0,
    "HipsHeightCompensationMode": 0,
    "AnkleHeightCompensationMode": 0,
    "AnkleProximityCompensationMode": 0,
    "ReachActorLeftAnkle": 0.0,
    "ReachActorRightAnkle": 0.0,
    "ReachActorLeftAnkleRotationRotation": 0.0,
    "ReachActorRightAnkleRotation": 0.0,
}


def _slot_node(cmds, mel, character, slot_name):
    slot = int(mel.eval('hikGetNodeIdFromName("%s")' % slot_name))
    node = mel.eval('hikGetSkNode("%s",%d)' % (character, slot))
    if not node or not cmds.objExists(node):
        raise RuntimeError("Target character has no %s" % slot_name)
    return cmds.ls(node, long=True)[0]


def _configure_retarget(cmds, target_character):
    """Disable HumanIK's proportion compensations after the source is set.

    ``hikSetCharacterInput`` restores these to their shipped values, so this has
    to run after the source is attached or the writes are silently discarded.
    """

    properties = cmds.listConnections(target_character, type="HIKProperty2State") or []
    if not properties:
        return None, {}
    node = properties[0]
    applied = {}
    for attribute, value in RETARGET_PROPERTIES.items():
        plug = "%s.%s" % (node, attribute)
        if not cmds.objExists(plug):
            raise RuntimeError("Target HumanIK properties lack %s" % attribute)
        before = cmds.getAttr(plug)
        cmds.setAttr(plug, value)
        if cmds.getAttr(plug) != value:
            raise RuntimeError("HumanIK discarded the retarget setting %s" % attribute)
        applied[attribute] = [before, value]
    return node, applied


def _set_source(mel, target_character, source_character):
    mel.eval('hikSetCurrentCharacter("%s")' % target_character)
    # A generated skeleton arrives with its definition unlocked, and HumanIK
    # silently refuses a source until it is locked.
    if not int(mel.eval('hikIsDefinitionLocked("%s")' % target_character)):
        mel.eval('hikCharacterLock("%s", 1, 1)' % target_character)
    mel.eval("hikUpdateCharacterList()")
    mel.eval("hikUpdateSourceList()")
    mel.eval('hikSetCharacterInput("%s","%s")' % (target_character, source_character))
    input_type = int(mel.eval('hikGetInputType("%s")' % target_character))
    if input_type != 3:
        raise RuntimeError("HumanIK did not accept the macap character as source")


def main():
    source_path, target_path, output_path, report_path = _arguments()
    cmds, mel = _start_maya()
    cmds.file(new=True, force=True)
    # The macap takes are 30 Hz. Matching the scene rate first keeps imported
    # keys on whole frames instead of fractional ones.
    cmds.currentUnit(time="ntsc")

    # HumanIK's skeleton generator attaches to whatever character is current, so
    # a generated target has to exist before the source character is created.
    generated_target = None
    if target_path == GENERATED_TARGET:
        generated_target = _import_target(cmds, mel, target_path)

    source_joints = _import_source(cmds, source_path)
    start = int(round(cmds.playbackOptions(q=True, minTime=True)))
    end = int(round(cmds.playbackOptions(q=True, maxTime=True)))
    keys = cmds.keyframe(source_joints, q=True, timeChange=True) or []
    if keys:
        start, end = int(round(min(keys))), int(round(max(keys)))
    if end <= start:
        raise RuntimeError("Imported macap FBX carries no usable frame range")

    muted = _rest_pose(cmds, source_joints)
    source_finger_rest = _world_rest(
        cmds,
        [cmds.ls("%s:%s" % (SOURCE_NAMESPACE, name), long=True)[0] for name in _finger_joints()],
    )
    source_character, mapped = _characterize(cmds, mel, source_joints)
    _restore_pose(cmds, muted)

    target_character = generated_target or _import_target(cmds, mel, target_path)
    finger_pairs = _finger_pairs(cmds, mel, target_character)
    target_finger_rest = _world_rest(cmds, [pair[1] for pair in finger_pairs])
    deform = _deform_joints(cmds, mel, target_character)
    _set_source(mel, target_character, source_character)
    property_node, retarget_properties = _configure_retarget(cmds, target_character)

    cmds.playbackOptions(minTime=start, maxTime=end, animationStartTime=start, animationEndTime=end)
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

    finger_targets = _transfer_fingers(
        cmds, finger_pairs, start, end,
        rest_tables=(source_finger_rest, target_finger_rest),
    )

    source_root = cmds.ls("%s:pelvis" % SOURCE_NAMESPACE, long=True)[0]
    source_head = cmds.ls("%s:head" % SOURCE_NAMESPACE, long=True)[0]
    target_hips = _slot_node(cmds, mel, target_character, "Hips")
    target_head = _slot_node(cmds, mel, target_character, "Head")

    def _upness(lower, upper):
        a = cmds.xform(lower, q=True, ws=True, t=True)
        b = cmds.xform(upper, q=True, ws=True, t=True)
        delta = [b[i] - a[i] for i in range(3)]
        length = sum(value * value for value in delta) ** 0.5
        return (delta[1] / length) if length > 1e-9 else 0.0

    def _spread(joints, hand, span):
        """Mean gap between adjacent fingertips, normalised by hand length."""

        points = [cmds.xform(joint, q=True, ws=True, t=True) for joint in joints]
        anchor = cmds.xform(hand, q=True, ws=True, t=True)
        reference = cmds.xform(span, q=True, ws=True, t=True)
        scale = sum((reference[i] - anchor[i]) ** 2 for i in range(3)) ** 0.5
        if scale < 1e-6:
            return 0.0
        gaps = [
            sum((points[i + 1][j] - points[i][j]) ** 2 for j in range(3)) ** 0.5
            for i in range(len(points) - 1)
        ]
        return (sum(gaps) / len(gaps)) / scale

    def _abduction(bases, tips):
        """Mean angle between adjacent digits' proximal bones, in degrees.

        Positional spread depends on each rig's finger lengths; the angle between
        neighbouring proximal bones does not, so it is the fair comparison for a
        rotation transfer.
        """

        directions = []
        for base, tip in zip(bases, tips):
            a = cmds.xform(base, q=True, ws=True, t=True)
            b = cmds.xform(tip, q=True, ws=True, t=True)
            vector = [b[i] - a[i] for i in range(3)]
            length = sum(value * value for value in vector) ** 0.5
            if length < 1e-9:
                return 0.0
            directions.append([value / length for value in vector])
        angles = []
        for i in range(len(directions) - 1):
            dot = sum(directions[i][j] * directions[i + 1][j] for j in range(3))
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
        return sum(angles) / len(angles)

    source_bases = [cmds.ls("%s:left_%s1" % (SOURCE_NAMESPACE, d), long=True)[0]
                    for d in ("index", "middle", "ring", "pinky")]
    source_tips = [cmds.ls("%s:left_%s2" % (SOURCE_NAMESPACE, d), long=True)[0]
                   for d in ("index", "middle", "ring", "pinky")]
    target_bases, target_tips = [], []
    for digit in ("Index", "Middle", "Ring", "Pinky"):
        for bone, bucket in ((1, target_bases), (2, target_tips)):
            slot = int(mel.eval('hikGetNodeIdFromName("LeftHand%s%d")' % (digit, bone)))
            bucket.append(mel.eval('hikGetSkNode("%s",%d)' % (target_character, slot)))

    source_spread_joints = [
        cmds.ls("%s:left_%s3" % (SOURCE_NAMESPACE, digit), long=True)[0]
        for digit in ("index", "middle", "ring", "pinky")
    ]
    target_spread_joints = []
    for digit in ("Index", "Middle", "Ring", "Pinky"):
        slot = int(mel.eval('hikGetNodeIdFromName("LeftHand%s3")' % digit))
        target_spread_joints.append(mel.eval('hikGetSkNode("%s",%d)' % (target_character, slot)))
    source_hand = cmds.ls("%s:left_wrist" % SOURCE_NAMESPACE, long=True)[0]
    target_hand = mel.eval(
        'hikGetSkNode("%s",%d)' % (target_character, int(mel.eval('hikGetNodeIdFromName("LeftHand")')))
    )

    samples = {"frames": [], "source_upness": [], "target_upness": [],
               "source_root": [], "target_root": [],
               "source_finger_spread": [], "target_finger_spread": [],
               "source_finger_abduction_deg": [], "target_finger_abduction_deg": []}
    for frame in range(start, end + 1):
        cmds.currentTime(frame)
        samples["frames"].append(frame)
        samples["source_upness"].append(_upness(source_root, source_head))
        samples["target_upness"].append(_upness(target_hips, target_head))
        samples["source_root"].append(cmds.xform(source_root, q=True, ws=True, t=True))
        samples["target_root"].append(cmds.xform(target_hips, q=True, ws=True, t=True))
        samples["source_finger_spread"].append(
            _spread(source_spread_joints, source_hand, source_spread_joints[1])
        )
        samples["target_finger_spread"].append(
            _spread(target_spread_joints, target_hand, target_spread_joints[1])
        )
        samples["source_finger_abduction_deg"].append(_abduction(source_bases, source_tips))
        samples["target_finger_abduction_deg"].append(_abduction(target_bases, target_tips))

    def _correlate(first, second):
        count = len(first)
        mean_a = sum(first) / count
        mean_b = sum(second) / count
        cov = sum((first[i] - mean_a) * (second[i] - mean_b) for i in range(count))
        var_a = sum((value - mean_a) ** 2 for value in first) ** 0.5
        var_b = sum((value - mean_b) ** 2 for value in second) ** 0.5
        return cov / (var_a * var_b) if var_a > 1e-9 and var_b > 1e-9 else 0.0

    # Torso up-ness is the pose check the root trajectory cannot make: HumanIK
    # rescales translation by proportion, so positions may correlate even when
    # the retargeted character is oriented wrongly.
    upness_correlation = _correlate(samples["source_upness"], samples["target_upness"])
    upness_error = max(
        abs(samples["source_upness"][i] - samples["target_upness"][i])
        for i in range(len(samples["frames"]))
    )
    root_correlation = [
        _correlate([p[axis] for p in samples["source_root"]],
                   [p[axis] for p in samples["target_root"]])
        for axis in range(3)
    ]
    failures = []
    if upness_correlation < 0.9:
        failures.append("torso orientation correlates at only %.3f" % upness_correlation)
    if upness_error > 0.35:
        failures.append("worst per-frame torso orientation error is %.3f" % upness_error)
    if min(root_correlation) < 0.8:
        failures.append("root trajectory correlates at only %r" % root_correlation)
    spread_ratios = [
        samples["target_finger_spread"][i] / samples["source_finger_spread"][i]
        for i in range(len(samples["frames"]))
        if samples["source_finger_spread"][i] > 1e-6
    ]
    worst_spread_ratio = min(spread_ratios) if spread_ratios else 0.0
    mean_spread_ratio = sum(spread_ratios) / len(spread_ratios) if spread_ratios else 0.0
    abduction_error = [
        abs(samples["target_finger_abduction_deg"][i] - samples["source_finger_abduction_deg"][i])
        for i in range(len(samples["frames"]))
    ]
    worst_abduction_error = max(abduction_error)
    mean_abduction_error = sum(abduction_error) / len(abduction_error)
    # The residual is the two rigs' differing neutral finger splay, which a
    # rest-relative transfer preserves by design; the gate catches a broken
    # transfer, not that difference.
    if mean_abduction_error > 15.0:
        failures.append(
            "finger abduction differs from the source by %.2f degrees on average"
            % mean_abduction_error
        )
    # Spread is reported, not gated: it is normalised by each rig's own hand
    # length, so a target with proportionally longer, closer-set fingers scores
    # low even when the transfer is exact. Abduction error above is the fidelity
    # gate; spread describes the target's anatomy.

    # Strip the source rig, both HumanIK character definitions, and the control
    # rig so the delivered file carries only the baked target character.
    cmds.delete(cmds.ls("%s:*" % SOURCE_NAMESPACE))
    for node in cmds.ls(type="HIKCharacterNode") + cmds.ls(type="HIKSolverNode") + cmds.ls(
        type="HIKState2SK"
    ) + cmds.ls(type="HIKRetargeterNode"):
        if cmds.objExists(node):
            cmds.delete(node)
    control_rig = [j for j in cmds.ls(type="joint", long=True) if "_Ctrl_" in j.split("|")[-1]]
    roots = sorted({j.split("|")[1] for j in control_rig if j.count("|") > 1})
    for node in roots:
        if cmds.objExists(node):
            cmds.delete(node)

    cmds.select(clear=True)
    remaining = sorted({_top_ancestor(cmds, joint) for joint in deform})
    for mesh in cmds.ls(type="mesh", noIntermediate=True, long=True) or []:
        remaining.append(_top_ancestor(cmds, mesh))
    remaining = sorted(set(remaining))
    cmds.select(remaining, replace=True, hierarchy=True)
    mel.eval('FBXExportBakeComplexAnimation -v true')
    mel.eval('FBXExportBakeComplexStart -v %d' % start)
    mel.eval('FBXExportBakeComplexEnd -v %d' % end)
    mel.eval('FBXExportBakeComplexStep -v 1')
    mel.eval('FBXExportInputConnections -v false')
    mel.eval('FBXExportSkins -v true')
    mel.eval('FBXExportShapes -v true')
    mel.eval('FBXExportEmbeddedTextures -v false')
    mel.eval('FBXExportConstraints -v false')
    mel.eval('FBXExportUpAxis y')
    mel.eval('FBXExport -f "%s" -s' % output_path.replace("\\", "/"))
    if not os.path.isfile(output_path):
        raise RuntimeError("HumanIK retarget produced no FBX")

    report = {
        "schema_version": "autoanim.macap-hik-retarget/1.0",
        "status": "passed" if not failures else "failed",
        "maya_version": cmds.about(version=True),
        "source_fbx": source_path,
        "target_scene": target_path,
        "target_character": target_character,
        "output_fbx": output_path,
        "frame_range": [start, end],
        "mapped_joints": len(mapped),
        "mapping": mapped,
        "deform_joints_baked": len(deform),
        "retarget_properties": retarget_properties,
        "source_sha256": _sha256(source_path),
        "output_sha256": _sha256(output_path),
        "upness_correlation": upness_correlation,
        "worst_upness_error": upness_error,
        "root_trajectory_correlation": root_correlation,
        "finger_transfer": "world-rest delta with explicit target parent conversion",
        "finger_joints_transferred": len(finger_targets),
        "worst_finger_spread_ratio": worst_spread_ratio,
        "mean_finger_spread_ratio": mean_spread_ratio,
        "worst_finger_abduction_error_deg": worst_abduction_error,
        "mean_finger_abduction_error_deg": mean_abduction_error,
        "samples": samples,
        "failures": failures,
    }
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "frame_range", "mapped_joints", "deform_joints_baked",
        "upness_correlation", "worst_upness_error",
        "mean_finger_spread_ratio", "worst_finger_abduction_error_deg",
        "mean_finger_abduction_error_deg", "failures")}))
    if failures:
        raise RuntimeError("HumanIK retarget verification failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
