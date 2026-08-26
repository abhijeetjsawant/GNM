#!/usr/bin/env python
"""Skin the macap base mesh onto the HumanIK skeleton built from the same rig.

The HumanIK skeleton carries the base model's joints to within 78 nm, so the
base mesh's existing skin weights transfer by influence rather than being solved
again: every macap joint has a HumanIK slot, and only the face joints - which
HumanIK has no slots for - fold into Head. The result is a HumanIK-native
character with a visible body, the base model's proportions, and the finger
spacing the shipped Autodesk scans do not have.

Run through mayapy:

    /Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy \
        scripts/bind_macap_mesh_to_hik.py SKELETON.ma BASE.fbx OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_macap_to_maya_hik import (  # noqa: E402
    HIK_MAPPING,
    SOURCE_NAMESPACE,
    TARGET_NAMESPACE,
    _import_into_namespace,
    _sha256,
    _slot_node,
    _start_maya,
    _top_ancestor,
)


MESH_NAME = "macap_HIK_Body"
MAX_INFLUENCES = 8
# HumanIK has no jaw or eye slots. Those weights are folded into Head rather
# than dropped, so the head surface keeps its full influence mass.
FACE_JOINTS = ("jaw", "left_eye_smplhf", "right_eye_smplhf")
FACE_FALLBACK = "Head"
WEIGHT_SUM_TOLERANCE = 1e-4


def _arguments():
    values = sys.argv[1:]
    if len(values) != 4:
        raise SystemExit("Expected SKELETON.ma BASE.fbx OUTPUT.fbx REPORT.json")
    skeleton, base, output, report = (os.path.abspath(value) for value in values)
    for path in (skeleton, base):
        if not os.path.isfile(path):
            raise SystemExit("Required input is missing: %s" % path)
    if not output.lower().endswith(".fbx"):
        raise SystemExit("Output must use the .fbx extension: %s" % output)
    for path in (output, report):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    return skeleton, base, output, report


def _skin_cluster(cmds, mesh):
    history = cmds.listHistory(mesh, pruneDagObjects=True) or []
    clusters = cmds.ls(history, type="skinCluster") or []
    if len(clusters) != 1:
        raise RuntimeError("Expected one skinCluster on the base mesh, found %r" % clusters)
    return clusters[0]


def _read_weights(cmds, mesh, cluster):
    """Read the base skin as a dense [vertex][influence] table via the API."""

    from maya.api.OpenMaya import MDagPath, MFnSingleIndexedComponent, MFn, MSelectionList
    from maya.api.OpenMayaAnim import MFnSkinCluster

    selection = MSelectionList()
    selection.add(mesh)
    dag = selection.getDagPath(0)
    dag.extendToShape()
    component = MFnSingleIndexedComponent().create(MFn.kMeshVertComponent)
    MFnSingleIndexedComponent(component).setCompleteData(cmds.polyEvaluate(mesh, vertex=True))

    selection = MSelectionList()
    selection.add(cluster)
    skin = MFnSkinCluster(selection.getDependNode(0))
    weights, influence_count = skin.getWeights(dag, component)
    influences = [path.partialPathName() for path in skin.influenceObjects()]
    if influence_count != len(influences):
        raise RuntimeError("skinCluster influence count does not match its influence list")
    return list(weights), influences


def _upright_duplicate(cmds, mesh, ground_offset):
    """A static copy of the base mesh, stood upright and put on the ground.

    The copy is made without input connections so it carries no skinCluster:
    transforming a still-bound mesh would move it once through its own transform
    and again through the skeleton.
    """

    duplicate = cmds.duplicate(mesh, name=MESH_NAME, inputConnections=False)[0]
    duplicate = cmds.parent(duplicate, world=True)[0] if cmds.listRelatives(
        duplicate, parent=True
    ) else duplicate
    # The FBX import drives the mesh transform, and duplicating carries those
    # connections across; break them so the copy is a free static object.
    for attribute in (
        "translateX", "translateY", "translateZ",
        "rotateX", "rotateY", "rotateZ",
        "scaleX", "scaleY", "scaleZ",
    ):
        plug = "%s.%s" % (duplicate, attribute)
        for source in cmds.listConnections(plug, source=True, destination=False, plugs=True) or []:
            cmds.disconnectAttr(source, plug)
        cmds.setAttr(plug, lock=False)
    for attribute in ("rotateX", "rotateY", "rotateZ"):
        cmds.setAttr("%s.%s" % (duplicate, attribute), 0.0)
    cmds.setAttr("%s.translateY" % duplicate, -ground_offset)
    cmds.makeIdentity(duplicate, apply=True, translate=True, rotate=True, scale=True)
    cmds.delete(duplicate, constructionHistory=True)
    return cmds.ls(duplicate, long=True)[0]


def main():
    skeleton_path, base_path, output_path, report_path = _arguments()
    cmds, mel = _start_maya()
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc")

    _import_into_namespace(cmds, skeleton_path, TARGET_NAMESPACE)
    characters = [
        node for node in cmds.ls(type="HIKCharacterNode") or []
        if node.startswith("%s:" % TARGET_NAMESPACE)
    ]
    if len(characters) != 1:
        raise RuntimeError("Expected one HumanIK character, found %r" % characters)
    character = characters[0]

    _import_into_namespace(cmds, base_path, SOURCE_NAMESPACE, file_type="FBX")
    meshes = cmds.ls("%s:*" % SOURCE_NAMESPACE, type="mesh", noIntermediate=True, long=True)
    if len(meshes) != 1:
        raise RuntimeError("Expected one base mesh, found %d" % len(meshes))
    base_mesh = cmds.listRelatives(meshes[0], parent=True, fullPath=True)[0]
    cluster = _skin_cluster(cmds, base_mesh)
    weights, influences = _read_weights(cmds, base_mesh, cluster)
    vertex_count = cmds.polyEvaluate(base_mesh, vertex=True)

    # The skeleton was generated standing on y = 0; the base import is not.
    hips = _slot_node(cmds, mel, character, "Hips")
    pelvis = cmds.ls("%s:pelvis" % SOURCE_NAMESPACE, long=True)[0]
    raw = cmds.xform(pelvis, q=True, ws=True, t=True)
    ground_offset = -raw[2] - cmds.xform(hips, q=True, ws=True, t=True)[1]

    mesh = _upright_duplicate(cmds, base_mesh, ground_offset)

    slot_nodes, macap_to_target = {}, {}
    for macap_name, slot in HIK_MAPPING.items():
        node = slot_nodes.setdefault(slot, _slot_node(cmds, mel, character, slot))
        macap_to_target[macap_name] = node
    head = slot_nodes.setdefault(FACE_FALLBACK, _slot_node(cmds, mel, character, FACE_FALLBACK))
    for macap_name in FACE_JOINTS:
        macap_to_target[macap_name] = head

    bind_joints = sorted(set(slot_nodes.values()))
    new_cluster = cmds.skinCluster(
        bind_joints, mesh, toSelectedBones=True, bindMethod=0, skinMethod=0,
        normalizeWeights=1, maximumInfluences=MAX_INFLUENCES, obeyMaxInfluences=True,
        name="%s_Skin" % MESH_NAME,
    )[0]

    from maya.api.OpenMaya import (  # noqa: E402
        MDoubleArray, MFn, MFnSingleIndexedComponent, MIntArray, MSelectionList,
    )
    from maya.api.OpenMayaAnim import MFnSkinCluster  # noqa: E402

    selection = MSelectionList()
    selection.add(new_cluster)
    skin = MFnSkinCluster(selection.getDependNode(0))
    target_influences = [path.partialPathName() for path in skin.influenceObjects()]
    target_index = {name: index for index, name in enumerate(target_influences)}

    columns = []
    unmapped = []
    for name in influences:
        short = name.split(":")[-1]
        target = macap_to_target.get(short)
        if target is None:
            unmapped.append(short)
            columns.append(None)
            continue
        columns.append(target_index[target.split("|")[-1].split(":")[-1]]
                       if target.split("|")[-1].split(":")[-1] in target_index
                       else target_index[target.split("|")[-1]])

    stride = len(target_influences)
    transferred = [0.0] * (vertex_count * stride)
    for vertex in range(vertex_count):
        base = vertex * len(influences)
        for slot, column in enumerate(columns):
            if column is None:
                continue
            transferred[vertex * stride + column] += weights[base + slot]

    selection = MSelectionList()
    selection.add(mesh)
    dag = selection.getDagPath(0)
    dag.extendToShape()
    component = MFnSingleIndexedComponent().create(MFn.kMeshVertComponent)
    MFnSingleIndexedComponent(component).setCompleteData(vertex_count)
    skin.setWeights(dag, component, MIntArray(range(stride)), MDoubleArray(transferred), False)

    sums = [
        sum(transferred[v * stride:(v + 1) * stride]) for v in range(vertex_count)
    ]
    worst_sum = max(abs(value - 1.0) for value in sums)
    worst_influences = max(
        sum(1 for value in transferred[v * stride:(v + 1) * stride] if value > 1e-6)
        for v in range(vertex_count)
    )

    cmds.delete(cmds.ls("%s:*" % SOURCE_NAMESPACE))
    roots = sorted({_top_ancestor(cmds, joint) for joint in bind_joints})
    roots.append(_top_ancestor(cmds, mesh))
    cmds.select(sorted(set(roots)), replace=True, hierarchy=True)
    mel.eval("FBXExportBakeComplexAnimation -v false")
    mel.eval("FBXExportInputConnections -v false")
    mel.eval("FBXExportSkins -v true")
    mel.eval("FBXExportShapes -v true")
    mel.eval("FBXExportConstraints -v false")
    mel.eval("FBXExportUpAxis y")
    mel.eval('FBXExport -f "%s" -s' % output_path.replace("\\", "/"))
    if not os.path.isfile(output_path):
        raise RuntimeError("Bind produced no FBX")

    scene_path = os.path.splitext(output_path)[0] + ".ma"
    cmds.file(rename=scene_path)
    cmds.file(save=True, type="mayaAscii", force=True)

    failures = []
    if unmapped:
        failures.append("base influences with no HumanIK slot: %s" % ", ".join(sorted(set(unmapped))))
    if worst_sum > WEIGHT_SUM_TOLERANCE:
        failures.append("skin weights sum off by %.6f" % worst_sum)
    if worst_influences > MAX_INFLUENCES:
        failures.append("a vertex has %d influences" % worst_influences)

    report = {
        "schema_version": "autoanim.macap-hik-bind/1.0",
        "status": "passed" if not failures else "failed",
        "maya_version": cmds.about(version=True),
        "character": character,
        "skeleton": skeleton_path,
        "base_model": base_path,
        "output_fbx": output_path,
        "output_scene": scene_path,
        "vertices": vertex_count,
        "bind_joints": len(bind_joints),
        "base_influences": len(influences),
        "face_joints_folded_into_head": list(FACE_JOINTS),
        "worst_weight_sum_error": worst_sum,
        "max_influences_per_vertex": worst_influences,
        "ground_offset_cm": ground_offset,
        "skeleton_sha256": _sha256(skeleton_path),
        "base_sha256": _sha256(base_path),
        "output_sha256": _sha256(output_path),
        "failures": failures,
    }
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "vertices", "bind_joints", "worst_weight_sum_error",
        "max_influences_per_vertex", "failures")}))
    if failures:
        raise RuntimeError("macap HumanIK bind failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
