"""D3 -- the exporter honours the track's own rest: GLB joints = the track's FK.

NEW file (the brief forbids editing existing tests).  Uses the real MPFB asset the
delivery is built from when it is on disk, and skips otherwise; the instrument version of
this check is `tools/compare/d3_skeleton_gate.py` (CLOSURE band).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import autoanim_gnm
from autoanim_gnm.body import DETAILED_HUMANOID, BodyTrack, forward_kinematics_positions
from autoanim_gnm.body_export import export_animated_body_glb

ROOT = Path(__file__).resolve().parents[1]
BODY_RUN = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands-fbd9784b"
TICKS = 48_000


def _glb_world_positions(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    json_length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + json_length])
    binary_start = 20 + json_length + 8
    count = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def accessor(index: int) -> np.ndarray:
        a = doc["accessors"][index]
        view = doc["bufferViews"][a["bufferView"]]
        n = count[a["type"]]
        return np.frombuffer(raw, dtype=np.float32, count=a["count"] * n,
                             offset=binary_start + view["byteOffset"] + a.get("byteOffset", 0)).reshape(a["count"], n)

    joints = len(DETAILED_HUMANOID.joints)
    nodes = doc["nodes"][:joints]
    parents = [j.parent for j in DETAILED_HUMANOID.joints]
    rotations = {}
    root = None
    animation = doc["animations"][0]
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        if channel["target"]["path"] == "rotation":
            rotations[channel["target"]["node"]] = accessor(sampler["output"])
        else:
            root = accessor(sampler["output"])
    frames = len(root)
    out = np.zeros((frames, joints, 3))
    for f in range(frames):
        world_p = np.zeros((joints, 3)); world_r = [None] * joints
        for j in range(joints):
            t = np.asarray(nodes[j]["translation"]) if j else root[f]
            q = Rotation.from_quat(rotations[j][f])
            p = parents[j]
            if p == -1:
                world_p[j] = t; world_r[j] = q
            else:
                world_p[j] = world_p[p] + world_r[p].apply(t); world_r[j] = world_r[p] * q
        out[f] = world_p
    return out


def _sized_track(frames: int = 4) -> BodyTrack:
    rng = np.random.default_rng(3)
    rest = np.array(DETAILED_HUMANOID.rest_translations_m)
    for name in ("LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand", "LeftLowerLeg",
                 "RightLowerLeg", "LeftFoot", "RightFoot", "Spine", "Chest", "UpperChest", "Neck"):
        rest[DETAILED_HUMANOID.index(name)] *= rng.uniform(0.7, 1.3)
    for name in ("LeftShoulder", "LeftUpperArm", "RightShoulder", "RightUpperArm",
                 "LeftUpperLeg", "RightUpperLeg"):
        rest[DETAILED_HUMANOID.index(name)][0] *= 0.7
    joints = len(DETAILED_HUMANOID.joints)
    rotations = Rotation.from_rotvec(rng.normal(0.0, 0.25, (frames * joints, 3))).as_quat().reshape(frames, joints, 4)
    rotations[:, 0] = (0.0, 0.0, 0.0, 1.0)
    eyes = np.zeros((frames, 2, 4), dtype=np.float32); eyes[..., 3] = 1.0
    ticks = np.arange(frames, dtype=np.int64) * (TICKS // 30)
    return BodyTrack(
        duration_ticks=int(ticks[-1]), ticks_per_second=TICKS, sample_rate_hz=30,
        joint_names=DETAILED_HUMANOID.names, ticks=ticks,
        root_translation_m=rng.normal(0.0, 0.5, (frames, 3)).astype(np.float32),
        local_rotations_xyzw=rotations.astype(np.float32),
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=np.broadcast_to(np.asarray((0.0, 0.0, 1.0), np.float32), (frames, 3)),
        gaze_strength=np.zeros(frames, dtype=np.float32), gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256="0" * 64, rest_translations_m=rest,
    )


@pytest.mark.skipif(not (BODY_RUN / "neutral-body.npz").exists(), reason="the delivery's MPFB asset is not on disk")
def test_the_exported_glb_reproduces_the_tracks_own_forward_kinematics(tmp_path):
    assert Path(autoanim_gnm.__file__).resolve().is_relative_to(ROOT / "src")
    track = _sized_track()
    out = tmp_path / "sized.glb"
    export_animated_body_glb(out, body_manifest_path=BODY_RUN / "neutral-body.json",
                             body_asset_path=BODY_RUN / "neutral-body.npz", track=track,
                             mapping_path=tmp_path / "sized-mapping.npz")
    from autoanim_gnm.body import skeleton_for_track
    expected = forward_kinematics_positions(track.root_translation_m, track.local_rotations_xyzw,
                                            skeleton=skeleton_for_track(track))
    got = _glb_world_positions(out)
    assert np.abs(got - expected).max() < 1.0e-4                    # float32, every frame, every joint
    # the same track read as a CANONICAL body is the pre-D3 defect: tens of millimetres
    canonical = forward_kinematics_positions(track.root_translation_m, track.local_rotations_xyzw,
                                             skeleton=DETAILED_HUMANOID)
    assert np.abs(got - canonical).max() > 0.02
    mapping = np.load(tmp_path / "sized-mapping.npz")
    assert "rest_translations_m" in mapping.files
    assert np.array_equal(mapping["rest_translations_m"], track.rest_translations_m)
