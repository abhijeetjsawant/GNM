"""D1 (fix): the left/right naming mirror, stated as properties the repair must hold.

Every assertion here FAILS on the build of 2026-09-01 and passes after the repair. They
are deliberately artefact-free -- no capture, no reference, no GLB -- so that the fix can
be developed without a rebuild and so that a regression is caught by `pytest` rather than
by a 20-minute instrument run.

The sites, one test each:

  1. `body.CANONICAL_HUMANOID` / `DETAILED_HUMANOID` -- the rest skeleton's own declared
     contract. `HumanoidSkeleton.as_dict()` says `handedness: right, up_axis: +Y,
     forward_axis: +Z`. For that frame the anatomical LEFT is `up x forward = +X`. The
     shipped skeleton puts its `Left` joints at -X, so it contradicts the contract it
     publishes, with no reference to any capture.
  2. `body_provider.DEFAULT_MPFB_JOINT_MAP` -- where the mirror was born. It maps
     AutoAnim `Left*` onto MPFB `.R` bones deliberately, to make (1) true.
  3. `commercial_multiview.positions_to_body_track` -- the torso frame it produces must
     put the rig's face axis (+Z) on the direction the performer actually faces.
  4. `head_orientation.CANONICAL_HEAD_AXES` -- the head reaches the same wrong answer by
     its own route and no skeleton or mesh change touches it.

Plus a fifth site found by grep during the repair (`soma_motion`, which COMPENSATED for
the mirror), the asset derivation, and the rejected geometry route.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "compare"))

ASSET = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands/neutral-body.npz"


def _rest_positions(skeleton) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for joint in skeleton.joints:
        offset = np.asarray(joint.rest_translation_m, dtype=np.float64)
        positions[joint.name] = offset if joint.parent == -1 else (
            offset + positions[skeleton.joints[joint.parent].name]
        )
    return positions


def _to_rig(v: np.ndarray) -> np.ndarray:
    """capture (Z up, the head solve's convention) -> rig (Y up)."""
    return np.array([v[0], v[2], -v[1]])


# ---------------------------------------------------------------------------- site 1
def test_the_rest_skeleton_obeys_the_convention_it_publishes() -> None:
    """`as_dict()` declares right-handed, up +Y, forward +Z. In that frame the anatomical
    left is `up x forward` = +X. Nothing external is consulted: this is the skeleton
    against its own published contract."""
    from autoanim_gnm.body import CANONICAL_HUMANOID, DETAILED_HUMANOID

    declared = CANONICAL_HUMANOID.as_dict()["coordinate_system"]
    assert declared == {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "linear_unit": "meter",
        "rotation": "local quaternion [x,y,z,w]",
    }
    up, forward = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    left_axis = np.cross(up, forward)
    assert np.allclose(left_axis, [1.0, 0.0, 0.0])

    for skeleton in (CANONICAL_HUMANOID, DETAILED_HUMANOID):
        at = _rest_positions(skeleton)
        # the face and the toes are at +Z -- this half is already right and must not move
        assert at["LeftEye"][2] > at["Head"][2]
        assert at["LeftToes"][2] > at["LeftFoot"][2]
        # ...so every joint named Left must be on +X
        for name, position in at.items():
            if name.startswith("Left") and abs(position[0]) > 1e-9:
                assert position[0] > 0.0, f"{name} is named Left and sits at x={position[0]}"
            if name.startswith("Right") and abs(position[0]) > 1e-9:
                assert position[0] < 0.0, f"{name} is named Right and sits at x={position[0]}"


def test_the_rest_skeletons_handedness_triple_product_matches_a_real_human() -> None:
    """`sign((right - left) x up . forward)`. Every real human measured in this lane reads
    -1; the shipped rest skeleton reads +1, which is the mirror stated as one bit."""
    from autoanim_gnm.body import CANONICAL_HUMANOID, DETAILED_HUMANOID

    for skeleton in (CANONICAL_HUMANOID, DETAILED_HUMANOID):
        at = _rest_positions(skeleton)
        across = at["RightUpperArm"] - at["LeftUpperArm"]
        up = at["Neck"] - at["Hips"]
        forward = at["LeftToes"] - at["LeftFoot"]
        assert np.sign(np.dot(np.cross(across, up), forward)) == -1.0


def test_the_left_and_right_rest_skeletons_stay_exact_mirrors_of_each_other() -> None:
    """The repair must be a pure sign change, not a re-authoring: after it, every Left
    joint must sit at exactly minus the X of its Right twin, and share the other two."""
    from autoanim_gnm.body import DETAILED_HUMANOID

    at = _rest_positions(DETAILED_HUMANOID)
    pairs = [n for n in at if n.startswith("Left")]
    assert len(pairs) >= 20
    for name in pairs:
        twin = "Right" + name.removeprefix("Left")
        assert twin in at, twin
        left, right = at[name], at[twin]
        assert np.allclose(left * (-1.0, 1.0, 1.0), right, atol=1e-12), name


# ---------------------------------------------------------------------------- site 2
def test_the_mpfb_joint_map_keeps_each_side_on_its_own_side() -> None:
    """The mirror's birth certificate. The map deliberately sends AutoAnim `Left*` to
    MPFB `.R` so that `Left` would land at negative X; MPFB's `.L` IS the anatomical
    left. With the rest skeleton corrected, the map must stop swapping."""
    from autoanim_gnm.body_provider import DEFAULT_MPFB_JOINT_MAP, DETAILED_MPFB_JOINT_MAP

    for joint_map in (DEFAULT_MPFB_JOINT_MAP, DETAILED_MPFB_JOINT_MAP):
        for name, mpfb in joint_map.items():
            if name.startswith("Left"):
                assert mpfb.endswith(".L"), f"{name} -> {mpfb}"
            elif name.startswith("Right"):
                assert mpfb.endswith(".R"), f"{name} -> {mpfb}"
            else:
                assert "." not in mpfb, f"{name} -> {mpfb}"


# ---------------------------------------------------------------------------- site 3
def _performer_facing(direction: np.ndarray, frames: int = 8) -> np.ndarray:
    """A synthetic 19-joint capture, Z-up, of a person facing `direction` in the ground
    plane, with the anatomy built from that facing rather than assumed. No rig convention
    enters, so this fixture cannot inherit the defect it is testing for."""
    from autoanim_gnm.commercial_multiview import JOINT_INDEX, JOINT_NAMES

    forward = np.asarray(direction, dtype=np.float64)
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)          # right-handed body: left = up x forward
    output = np.zeros((frames, len(JOINT_NAMES), 3), dtype=np.float64)
    for frame in range(frames):
        root = np.array([0.0, 0.0, 1.0 + 0.01 * frame])
        values = {
            "root": root,
            "neck": root + 0.62 * up,
            "nose": root + 0.78 * up + 0.10 * forward,
            "left_eye": root + 0.76 * up + 0.09 * forward + 0.035 * left,
            "right_eye": root + 0.76 * up + 0.09 * forward - 0.035 * left,
            "left_ear": root + 0.73 * up + 0.09 * left,
            "right_ear": root + 0.73 * up - 0.09 * left,
            "left_shoulder": root + 0.52 * up + 0.19 * left,
            "right_shoulder": root + 0.52 * up - 0.19 * left,
            "left_elbow": root + 0.34 * up + 0.24 * left,
            "right_elbow": root + 0.34 * up - 0.24 * left,
            "left_wrist": root + 0.16 * up + 0.26 * left + 0.02 * frame * forward,
            "right_wrist": root + 0.16 * up - 0.26 * left,
            "left_hip": root + 0.09 * left,
            "right_hip": root - 0.09 * left,
            "left_knee": root - 0.43 * up + 0.09 * left,
            "right_knee": root - 0.43 * up - 0.09 * left,
            "left_ankle": root - 0.86 * up + 0.09 * left,
            "right_ankle": root - 0.86 * up - 0.09 * left,
        }
        for name, value in values.items():
            output[frame, JOINT_INDEX[name]] = value
    return output


@pytest.mark.parametrize(
    "direction", [(0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.6, -0.8, 0.0)]
)
def test_the_torso_frame_points_the_rigs_face_axis_where_the_performer_faces(
    direction: tuple[float, float, float],
) -> None:
    """The defect, at the size of one function. `positions_to_body_track` builds the torso
    frame from `_frame_alignment`, whose source secondary axis states which rig axis is
    the subject's left. On the shipped build the rig's +Z -- the axis the mesh's nose,
    eyes and toes all lie on -- comes out pointing at the performer's BACK, and this
    reads -1."""
    from scipy.spatial.transform import Rotation

    from autoanim_gnm.commercial_multiview import DETAILED_HUMANOID, positions_to_body_track

    capture = _performer_facing(np.asarray(direction, dtype=np.float64))
    track = positions_to_body_track(
        capture, sample_rate_hz=30, provenance_sha256=sha256(b"d1-fix").hexdigest()
    )
    names = list(track.joint_names)
    parents = [joint.parent for joint in DETAILED_HUMANOID.joints]
    local = track.local_rotations_xyzw.astype(np.float64)
    world = np.zeros_like(local)
    for index, parent in enumerate(parents):
        rotation = Rotation.from_quat(local[:, index])
        world[:, index] = (
            rotation if parent == -1 else Rotation.from_quat(world[:, parent]) * rotation
        ).as_quat()

    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    for joint in ("Hips", "Spine", "Chest", "UpperChest"):
        face = Rotation.from_quat(world[:, names.index(joint)]).apply(
            np.tile((0.0, 0.0, 1.0), (len(local), 1))
        )
        # rig (x, y, z) -> capture (x, -z, y)
        face = np.stack([face[:, 0], -face[:, 2], face[:, 1]], axis=-1)
        assert np.median(face @ forward) > 0.99, joint
        side = Rotation.from_quat(world[:, names.index(joint)]).apply(
            np.tile((1.0, 0.0, 0.0), (len(local), 1))
        )
        side = np.stack([side[:, 0], -side[:, 2], side[:, 1]], axis=-1)
        # rig +X is where the bones named Left now sit, so it must be the performer's left
        assert np.median(side @ left) > 0.99, joint


# ---------------------------------------------------------------------------- site 4
def test_the_canonical_head_axes_face_the_way_the_asset_does() -> None:
    """The mirror's site no skeleton or mesh change reaches. The head is set as an
    ABSOLUTE world rotation from its own rigid fit, so it is backwards by its own route.
    The frame must stay right-handed AND agree with the geometry it drives: the asset's
    nose, eyes and toes are at rig +Z and the track schema's own `gaze.direction_body` is
    [0, 0, 1]."""
    from autoanim_gnm.head_orientation import CANONICAL_HEAD_AXES

    axes = np.asarray(CANONICAL_HEAD_AXES, dtype=np.float64)
    left, up, forward = axes[:, 0], axes[:, 1], axes[:, 2]
    assert np.allclose(np.cross(left, up), forward), "left x up = forward"
    assert np.isclose(np.linalg.det(axes), 1.0), "a proper rotation, never a reflection"
    assert np.allclose(_to_rig(forward), [0.0, 0.0, 1.0])
    assert np.allclose(_to_rig(left), [1.0, 0.0, 0.0])
    assert np.allclose(_to_rig(up), [0.0, 1.0, 0.0])


def test_the_shipped_head_axes_were_a_yaw_of_the_right_frame_and_not_a_mirror() -> None:
    """Stated for the record and as a guard on the repair: the frame that shipped had
    determinant +1, so it was a 180 degree yaw about the skull's own long axis -- not a
    reflection. Both its left and its forward were negated and its up was not, which is
    exactly a yaw. A repair that reflects instead of rotating would also satisfy the
    forward assertion above and would put the eyes on the wrong sides."""
    shipped = np.asarray(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))).T
    assert np.isclose(np.linalg.det(shipped), 1.0)

    from autoanim_gnm.head_orientation import CANONICAL_HEAD_AXES

    repaired = np.asarray(CANONICAL_HEAD_AXES, dtype=np.float64)
    # `relative` is the shipped frame read in the repaired frame's own axes, so its fixed
    # direction is a canonical axis INDEX: 0 is left, 1 is up, 2 is forward.
    relative = repaired.T @ shipped
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    assert np.isclose(angle, 180.0, atol=1e-9)
    assert np.allclose(relative, np.diag([-1.0, 1.0, -1.0]), atol=1e-12)
    # ...a half turn about the skull's own long axis: up is fixed, left and forward reverse
    assert np.allclose(relative @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


# ------------------------------------------------------- site 5: a lane that compensated
def test_the_soma_delta_mapping_no_longer_compensates_for_the_mirror() -> None:
    """The blast radius, found by grep and not by a gate. `soma_motion._DELTA_MAPPING`
    deliberately sent our `Left*` joints onto SOMA's `Right*` ones, with the reason
    written beside it: "SOMA anatomical left is positive source X, while AutoAnim's
    reviewed canonical skeleton defines Left on negative X". SOMA declares the same
    coordinate system we do -- right-handed, up +Y, forward +Z -- so once our Left is +X
    the compensation becomes the error, and the SOMA lane would ship every performer's
    arms and legs exchanged. There is no capture fixture on that lane, so nothing else
    would have caught it."""
    from autoanim_gnm.soma_motion import _DELTA_MAPPING, _DETAILED_DELTA_MAPPING

    for mapping in (_DELTA_MAPPING, _DETAILED_DELTA_MAPPING):
        for target, source in mapping.items():
            if target.startswith("Left"):
                assert source.startswith("Left"), f"{target} -> {source}"
            elif target.startswith("Right"):
                assert source.startswith("Right"), f"{target} -> {source}"


def test_the_speech_motion_smplx_mapping_keeps_each_side_on_its_own_side() -> None:
    """The other retarget boundary, asserted so that the two lanes cannot drift apart.
    SMPL-X's anatomical left is positive X, exactly like ours now is, and the bundled map
    already sends `Left*` to `left_*` -- so this lane was MIRRORED before the repair and is
    correct after it, which is the opposite of the SOMA lane above. Both are stated here
    because the pair is the whole reason a blast-radius grep was needed."""
    from autoanim_gnm.speech_motion import SMPLX55_TARGET_TO_SOURCE

    for target, source in SMPLX55_TARGET_TO_SOURCE.items():
        if target.startswith("Left"):
            assert source.startswith("left_"), f"{target} -> {source}"
        elif target.startswith("Right"):
            assert source.startswith("right_"), f"{target} -> {source}"


# ---------------------------------------------------------------------------- site 6
@pytest.mark.skipif(not ASSET.exists(), reason="the MPFB body-provider run is not present")
def test_the_asset_relabel_moves_no_geometry() -> None:
    """The route shipped is a RELABEL, not a mirror of geometry: the derivation permutes
    which AutoAnim name each MPFB bone answers to and touches no vertex, no triangle and
    no weight value. That is the whole argument for preferring it, so it is asserted."""
    from d1_asset_relabel import relabel_asset  # noqa: PLC0415

    source = dict(np.load(ASSET, allow_pickle=True))
    fixed = relabel_asset(source)

    assert np.array_equal(fixed["vertices_m"], source["vertices_m"])
    assert np.array_equal(fixed["triangles"], source["triangles"])
    assert np.array_equal(np.sort(fixed["joint_weights"], axis=1),
                          np.sort(source["joint_weights"], axis=1))
    assert list(fixed["joint_names"]) == list(source["joint_names"])
    assert np.array_equal(fixed["parents"], source["parents"])

    names = [str(n) for n in fixed["joint_names"].tolist()]
    world = np.zeros((len(names), 4, 4))
    for index, parent in enumerate(fixed["parents"].tolist()):
        world[index] = (
            fixed["local_rest_matrices"][index]
            if parent == -1
            else world[parent] @ fixed["local_rest_matrices"][index]
        )
    at = {name: world[i][:3, 3] for i, name in enumerate(names)}
    assert at["LeftUpperArm"][0] > 0 > at["RightUpperArm"][0]
    across = at["RightUpperArm"] - at["LeftUpperArm"]
    up = at["Neck"] - at["Hips"]
    forward = at["LeftToes"] - at["LeftFoot"]
    assert np.sign(np.dot(np.cross(across, up), forward)) == -1.0

    dominant = fixed["joint_indices"][
        np.arange(len(fixed["vertices_m"])), fixed["joint_weights"].argmax(1)
    ]
    left_flesh = fixed["vertices_m"][dominant == names.index("LeftUpperArm")]
    right_flesh = fixed["vertices_m"][dominant == names.index("RightUpperArm")]
    assert left_flesh[:, 0].mean() > 0 > right_flesh[:, 0].mean()


# ---------------------------------------------------------------------------- site 6
@pytest.mark.skipif(not ASSET.exists(), reason="the MPFB body-provider run is not present")
def test_the_geometry_route_turns_the_mesh_inside_out() -> None:
    """The route REJECTED, rejected by construction rather than by opinion. The review
    proposed negating every vertex's X. That is a reflection: it reverses every triangle's
    winding and inverts every normal, so the closed mesh's signed volume changes sign and
    the character renders inside out -- and no joint gate, no forward-dot and no
    silhouette IoU can see it."""
    source = np.load(ASSET, allow_pickle=True)
    vertices = source["vertices_m"].astype(np.float64)
    triangles = source["triangles"].astype(np.int64)

    def signed_volume(v: np.ndarray) -> float:
        a, b, c = v[triangles[:, 0]], v[triangles[:, 1]], v[triangles[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    original = signed_volume(vertices)
    mirrored = vertices * (-1.0, 1.0, 1.0)
    assert original * signed_volume(mirrored) < 0.0
    assert np.isclose(signed_volume(mirrored), -original, rtol=1e-9)
