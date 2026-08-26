#!/usr/bin/env python
"""Assemble both performers of a MAMMA take into one Maya scene.

Each animated export is rebased to its own origin so a character can be used on
its own. Reassembling the pair means putting that offset back: the two are
placed where the capture had them, 1.86 m apart, rather than on top of one
another.

Run through mayapy:

    mayapy scripts/build_macap_scene.py OUTPUT.ma REPORT.json base|hik \
        FBX_00 X,Y,Z  FBX_01 X,Y,Z
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_macap_to_maya_hik import _sha256, _start_maya  # noqa: E402


# The animated exports are authored in metres; Maya works in centimetres.
OFFSET_SCALE = 100.0
# The check is that a human-scale character actually arrived, not that it is
# standing: the pair are measured at the take's first frame, where a performer
# may already be crouched.
MIN_FIGURE_EXTENT_CM = 100.0


def _arguments():
    values = sys.argv[1:]
    if len(values) != 7 or values[2] not in ("base", "hik"):
        raise SystemExit(
            "Expected OUTPUT.ma REPORT.json base|hik FBX X,Y,Z[,NAME] FBX X,Y,Z[,NAME]"
        )
    output, report = (os.path.abspath(value) for value in values[:2])
    kind = values[2]
    takes = []
    for index, (name, spec) in enumerate(((values[3], values[4]), (values[5], values[6]))):
        path = os.path.abspath(name)
        if not os.path.isfile(path):
            raise SystemExit("Missing take: %s" % path)
        parts = spec.split(",")
        offset = [float(v) for v in parts[:3]]
        label = parts[3] if len(parts) > 3 else "id%02d" % index
        takes.append((path, offset, label))
    if not output.lower().endswith(".ma"):
        raise SystemExit("Output must be a .ma scene")
    for path in (output, report):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    return output, report, kind, takes


def _place(cmds, namespace, offset, tilted, frame):
    """Move one character by offsetting its root joint's animation.

    Parenting a character under a group and moving the group shifts the skinned
    mesh twice - once through its own transform and again through the skeleton it
    is bound to - which leaves the body floating away from its skeleton.
    Offsetting the root joint's translation keys moves the whole character
    rigidly, mesh included, and adds no nodes to the scene.
    """

    roots = [
        joint for joint in cmds.ls(type="joint", long=True) or []
        if ("%s:" % namespace) in joint
        and not (cmds.listRelatives(joint, parent=True, fullPath=True, type="joint") or [])
    ]
    if not roots:
        raise RuntimeError("No root joint found for %s" % namespace)
    root = sorted(roots, key=lambda name: name.count("|"))[0]

    cmds.currentTime(frame)
    wanted = [offset[0], offset[2], -offset[1]] if tilted else list(offset)
    wanted = [value * OFFSET_SCALE for value in wanted]
    current = cmds.xform(root, q=True, worldSpace=True, translation=True)
    delta_world = [wanted[index] - current[index] for index in range(3)]

    # The root's translation lives in its parent's space, which for a base-rig
    # export carries the exporter's rotation and its metre-to-centimetre scale.
    parents = cmds.listRelatives(root, parent=True, fullPath=True) or []
    if parents:
        from maya.api.OpenMaya import MMatrix, MVector

        inverse = MMatrix(cmds.xform(parents[0], q=True, ws=True, matrix=True)).inverse()
        vector = MVector(delta_world) * inverse
        delta_local = [vector.x, vector.y, vector.z]
    else:
        delta_local = delta_world

    for axis, value in zip("XYZ", delta_local):
        plug = "%s.translate%s" % (root, axis)
        if cmds.keyframe(plug, q=True, keyframeCount=True):
            cmds.keyframe(plug, edit=True, relative=True, valueChange=value)
        else:
            cmds.setAttr(plug, cmds.getAttr(plug) + value)
    return root


def main():
    output, report_path, kind, takes = _arguments()
    cmds, mel = _start_maya()
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc")

    placed = []
    for index, (path, offset, label) in enumerate(takes):
        namespace = label
        if not cmds.namespace(exists=namespace):
            cmds.namespace(add=namespace)
        cmds.namespace(set=":%s" % namespace)
        try:
            cmds.file(path, i=True, type="FBX", options="fbx", ignoreVersion=True, prompt=False)
        finally:
            cmds.namespace(set=":")
        placed.append((namespace, path, offset))

    frames = sorted(set(cmds.keyframe(cmds.ls(type="joint"), q=True, timeChange=True) or []))
    start, end = int(round(min(frames))), int(round(max(frames)))
    roots = {}
    for namespace, path, offset in placed:
        roots[namespace] = _place(cmds, namespace, offset, kind == "base", start)
    cmds.playbackOptions(
        minTime=start, maxTime=end, animationStartTime=start, animationEndTime=end
    )
    cmds.currentTime(start)

    heights, separation = {}, None
    anchors, drift, root_positions = [], {}, []
    for namespace, path, offset in placed:
        meshes = [
            node for node in cmds.ls(type="mesh", noIntermediate=True, long=True) or []
            if ("%s:" % namespace) in node
        ]
        if not meshes:
            raise RuntimeError("No mesh imported for %s" % namespace)
        box = cmds.exactWorldBoundingBox(meshes[0])
        # A base-rig export lies along -Z, so the figure's long axis is not Y.
        heights[namespace] = max(box[3] - box[0], box[4] - box[1], box[5] - box[2])
        centre = [0.5 * (box[0] + box[3]), 0.5 * (box[1] + box[4]), 0.5 * (box[2] + box[5])]
        anchors.append(centre)
        # If the body and its skeleton ever come apart, this is where it shows.
        rootpos = cmds.xform(roots[namespace], q=True, ws=True, t=True)
        root_positions.append(rootpos)
        drift[namespace] = sum((centre[i] - rootpos[i]) ** 2 for i in range(3)) ** 0.5
    separation = sum((anchors[0][i] - anchors[1][i]) ** 2 for i in range(3)) ** 0.5
    root_separation = sum(
        (root_positions[0][i] - root_positions[1][i]) ** 2 for i in range(3)
    ) ** 0.5
    wanted = [
        [o[0], o[2], -o[1]] if kind == "base" else list(o) for _, _, o in placed
    ]
    intended = sum(
        ((wanted[0][i] - wanted[1][i]) * OFFSET_SCALE) ** 2 for i in range(3)
    ) ** 0.5

    cmds.file(rename=output)
    cmds.file(save=True, type="mayaAscii", force=True)

    failures = []
    for namespace, height in heights.items():
        if height < MIN_FIGURE_EXTENT_CM:
            failures.append("%s measures only %.1f cm on its longest axis" % (namespace, height))
    for namespace, value in drift.items():
        if value > 60.0:
            failures.append(
                "%s body sits %.1f cm from its own root - mesh and skeleton are apart"
                % (namespace, value)
            )
    for namespace, value in drift.items():
        if value > 60.0:
            failures.append(
                "%s body sits %.1f cm from its own root - mesh and skeleton are apart"
                % (namespace, value)
            )
    # Verify the placement did what was asked rather than assuming a minimum
    # gap: a comparison scene deliberately puts two characters in one spot.
    if abs(root_separation - intended) > 1.0:
        failures.append(
            "roots are %.1f cm apart, %.1f cm was asked for" % (root_separation, intended)
        )

    report = {
        "schema_version": "autoanim.macap-scene/1.0",
        "status": "passed" if not failures else "failed",
        "kind": kind,
        "output_scene": output,
        "frame_range": [start, end],
        "characters": [
            {"namespace": ns, "source": p, "offset_m": o, "source_sha256": _sha256(p)}
            for ns, p, o in placed
        ],
        "mesh_heights_cm": heights,
        "mesh_to_root_distance_cm": drift,
        "mesh_to_root_distance_cm": drift,
        "separation_cm": separation,
        "root_separation_cm": root_separation,
        "intended_separation_cm": intended,
        "output_sha256": _sha256(output),
        "failures": failures,
    }
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("status", "kind", "frame_range", "mesh_heights_cm",
                       "mesh_to_root_distance_cm", "root_separation_cm",
                       "intended_separation_cm", "failures")}))
    if failures:
        raise RuntimeError("scene assembly failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
