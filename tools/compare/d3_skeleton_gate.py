#!/usr/bin/env python3
"""D3: one rest skeleton per performer, carried through every consumer. The gate.

THE CHANGE. `positions_to_body_track` takes a rest skeleton and stamps it on the
`BodyTrack`; `forward_kinematics_positions`, `validate_body_track`,
`project_generated_foot_contacts`, `export_animated_body_glb`, `unified_gltf`, the verify
script and the instruments all read the track's OWN rest instead of resolving the
canonical body from the joint names.

THE DEFECT IT CLOSES. The delivery carried two skeletons. Every joint instrument scored
`DETAILED_HUMANOID` -- identity rest rotations, T-pose, 540 mm shoulder span -- while
`export_animated_body_glb` wrote the MPFB asset's own `local_rest_matrices` as the glTF
node translations: an A-pose with a 340 mm span, its `Hips` 8 mm below its Root-relative
rest instead of 80. Measured at commit 1035ece on the delivered
`subject-00.glb`: 81-195 mm per joint between the two, median 131 mm over 55 joints, by
a hand-rolled glTF reader and by Blender 4.2's own importer alike. So rung 11, the round
trip, the verify script and facing scored one body and the silhouette scored another.

THE DISCRIMINATING BAND is CLOSURE: the exported GLB, parsed and forward-kinematicked
node by node, must reproduce `forward_kinematics_positions` on the track's own rest. It
is an identity by construction and it is banded at 1e-4 m; the pre-D3 exporter, reached
by swapping ONE module attribute so the control runs through the identical code path,
must fail it by orders of magnitude.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d3_skeleton_gate.py

Writes `artifacts/compare/d3-skeleton/gate.json`. Reads the rebuild under
`artifacts/compare/d3-skeleton/delivery`; never writes to `artifacts/commercial-multiview-*`.
"""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
sys.path.insert(0, str(ROOT / "tools" / "swap-harness"))
sys.path.insert(0, str(ROOT / "tools" / "compare"))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src."
    )

from autoanim_gnm import body_export as be  # noqa: E402
from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    DETAILED_HUMANOID,
    BodyTrack,
    BodyValidationError,
    forward_kinematics_positions,
    skeleton_for_track,
)
from autoanim_gnm.performer_skeleton import performer_skeleton  # noqa: E402
import retarget_cost as rc  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d3-skeleton"
REBUILD = OUT_DIR / "delivery"
DELIVERED = ROOT / "artifacts/commercial-multiview-soma77"
BODY_RUN = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands-fbd9784b"
SCRATCH = OUT_DIR / "scratch"

CLOSURE_BAND_M = 1.0e-4
MUSTFAIL_FLOOR_MM = 50.0
ORACLE_ARMS_BAND_MM = 0.5
ORACLE_LEGS_BAND_MM = 0.0
ROUNDTRIP_ARMS_BAND_MM = 5.0
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907, 20260908)
PERTURB_LOW, PERTURB_HIGH = 0.7, 1.3

REF_CLOSURE = (
    "the track's own forward kinematics -- the exported GLB scored against "
    "`forward_kinematics_positions` on the rest the track carries. An IDENTITY, not a fit."
)
REF_SYNTH = (
    "exact synthetic truth: a known motion posed through our own forward kinematics on a "
    "perturbed rest. MAMMA-FREE."
)
REF_ROUNDTRIP = "a body of the given proportions BY CONSTRUCTION -- the converter scored against its OWN output"
REF_OURS = (
    "our own triangulated capture, root-relative (hip midpoint) -- the converter scored "
    "against its OWN input"
)
REF_MAMMA = "MAMMA pred_joints, per-joint median then median over joints, in capture Z-up metres"
REF_MASKS = "MAMMA's SAM2 masks -- pixels of the actual footage"


# ------------------------------------------------------------------ the glTF reader
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path) -> tuple[dict, bytes]:
    """Parse a binary glTF into its JSON document and its BIN chunk. No dependencies."""

    raw = Path(path).read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF")
    total = struct.unpack("<I", raw[8:12])[0]
    offset, chunks = 12, {}
    while offset < total:
        length, kind = struct.unpack("<I4s", raw[offset:offset + 8])
        offset += 8
        chunks[kind] = raw[offset:offset + length]
        offset += length
    return json.loads(chunks[b"JSON"].decode("utf-8")), chunks[b"BIN\x00"]


def accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    acc = document["accessors"][index]
    view = document["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    components = _COUNT[acc["type"]]
    values = np.frombuffer(
        binary, dtype=_COMPONENT[acc["componentType"]],
        count=acc["count"] * components, offset=start,
    )
    return values.reshape(acc["count"], components) if components > 1 else values


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw,
         aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., :3]
    uv = np.cross(xyz, v)
    return v + 2.0 * (q[..., 3, None] * uv + np.cross(xyz, uv))


def glb_joint_positions(path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Forward-kinematic the GLB's own skin, exactly as a glTF viewer would.

    Returns joint names, `[frame, joint, 3]` world positions and `[joint, 3]` REST node
    translations. Nothing about AutoAnim is assumed: the hierarchy comes from `children`,
    the animated channels from the one animation, and a joint with no channel keeps its
    node transform.
    """

    document, binary = read_glb(path)
    nodes = document["nodes"]
    joints = document["skins"][0]["joints"]
    parent = {node: -1 for node in joints}
    for position, node in enumerate(nodes):
        for child in node.get("children", ()):
            parent[child] = position
    rest = {j: np.asarray(nodes[j].get("translation", (0.0, 0.0, 0.0)), np.float64) for j in joints}
    rotation_channels: dict[int, np.ndarray] = {}
    translation_channels: dict[int, np.ndarray] = {}
    animation = document["animations"][0]
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        target = channel["target"]["node"]
        data = accessor(document, binary, sampler["output"]).astype(np.float64)
        if channel["target"]["path"] == "rotation":
            rotation_channels[target] = data
        elif channel["target"]["path"] == "translation":
            translation_channels[target] = data
    frames = len(next(iter(rotation_channels.values())))
    order = {node: position for position, node in enumerate(joints)}
    positions = np.zeros((frames, len(joints), 3), dtype=np.float64)
    world = np.zeros((frames, len(joints), 4), dtype=np.float64)
    for node in joints:
        slot = order[node]
        local_t = translation_channels.get(node)
        if local_t is None:
            local_t = np.broadcast_to(rest[node], (frames, 3))
        local_q = rotation_channels[node]
        if parent[node] == -1:
            positions[:, slot] = local_t
            world[:, slot] = local_q
        else:
            up = order[parent[node]]
            positions[:, slot] = positions[:, up] + _qrot(world[:, up], local_t)
            world[:, slot] = _qmul(world[:, up], local_q)
    names = [nodes[node]["name"] for node in joints]
    rest_array = np.stack([rest[node] for node in joints])
    return names, positions, rest_array


# ------------------------------------------------------------------ the exporter swap
@contextlib.contextmanager
def pre_d3_exporter():
    """The pre-D3 exporter, reached by swapping ONE module attribute.

    `_compose_rest_and_delta` is the whole of the change: before D3 it returned the
    ASSET's rest translations, and `export_animated_body_glb` added the asset's Root rest
    to the animated root. Dropping the `rest_translations_m` argument restores both,
    through the identical call site, rather than re-implementing an exporter.
    """

    shipped = be._compose_rest_and_delta

    def legacy(*args, rest_translations_m=None, **kwargs):
        return shipped(*args, **kwargs)

    be._compose_rest_and_delta = legacy
    try:
        yield
    finally:
        be._compose_rest_and_delta = shipped


def export(track: BodyTrack, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    be.export_animated_body_glb(
        path,
        body_manifest_path=BODY_RUN / "neutral-body.json",
        body_asset_path=BODY_RUN / "neutral-body.npz",
        track=track,
        mapping_path=path.with_suffix(".mapping.npz"),
    )
    return path


# ------------------------------------------------------------------ helpers
def load_track(directory: Path, subject: int) -> BodyTrack:
    payload = json.loads(
        (directory / f"subject-{subject:02d}.body-track.json").read_text()
    )
    return BodyTrack.from_dict(payload)


def capture_positions(directory: Path, subject: int) -> np.ndarray:
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        return np.asarray(archive["triangulated_world_positions_z_up_m"], np.float64)


def span_table(skeleton) -> dict[str, float]:
    """The five spans the closure band asserts, in metres, from a rest array."""

    rest = skeleton.rest_translations_m
    index = {name: position for position, name in enumerate(skeleton.names)}

    def norm(name: str) -> float:
        return float(np.linalg.norm(rest[index[name]]))

    return {
        "shoulder_span_m": 2.0 * abs(
            float(rest[index["LeftShoulder"], 0]) + float(rest[index["LeftUpperArm"], 0])
        ),
        "hip_span_m": 2.0 * abs(float(rest[index["LeftUpperLeg"], 0])),
        "thigh_m": norm("LeftLowerLeg"),
        "shin_m": norm("LeftFoot"),
        "hips_to_neck_m": sum(norm(name) for name in ("Spine", "Chest", "UpperChest", "Neck")),
    }


def spans_from_rest(names, rest: np.ndarray) -> dict[str, float]:
    index = {name: position for position, name in enumerate(names)}

    def norm(name: str) -> float:
        return float(np.linalg.norm(rest[index[name]]))

    return {
        "shoulder_span_m": 2.0 * abs(
            float(rest[index["LeftShoulder"], 0]) + float(rest[index["LeftUpperArm"], 0])
        ),
        "hip_span_m": 2.0 * abs(float(rest[index["LeftUpperLeg"], 0])),
        "thigh_m": norm("LeftLowerLeg"),
        "shin_m": norm("LeftFoot"),
        "hips_to_neck_m": sum(norm(name) for name in ("Spine", "Chest", "UpperChest", "Neck")),
    }


def groups_mm(error: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        group: round(float(np.median(np.concatenate([error[n] for n in names]))), 2)
        for group, names in rc.GROUPS.items()
    }


# ------------------------------------------------------------------ CLOSURE
def closure_block(report: dict) -> None:
    block: dict = {
        "reference": REF_CLOSURE,
        "band_m": CLOSURE_BAND_M,
        "why_it_is_the_discriminating_band": (
            "it is the only figure that can see the two skeletons at once. Every joint "
            "instrument reads the track; the silhouette reads the mesh the GLB carries. "
            "Nothing before D3 compared them."
        ),
        "subjects": {},
    }
    for subject in (0, 1):
        track = load_track(REBUILD, subject)
        skeleton = skeleton_for_track(track)
        names, glb_positions, glb_rest = glb_joint_positions(
            REBUILD / f"subject-{subject:02d}.glb"
        )
        truth = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64),
            skeleton=skeleton,
        ).astype(np.float64)
        error = np.linalg.norm(glb_positions - truth, axis=2)
        # The MUST-FAIL, on the same track through the same exporter call site.
        legacy_path = SCRATCH / f"pre-d3-subject-{subject:02d}.glb"
        with pre_d3_exporter():
            export(track, legacy_path)
        _, legacy_positions, _ = glb_joint_positions(legacy_path)
        legacy_error = np.linalg.norm(legacy_positions - truth, axis=2)
        # The GLB's own rest spans, recovered from its node translations by forward
        # kinematics of the rest pose alone -- which is what a viewer sees before a frame
        # is applied. Compared against the track's rest.
        rest_names, rest_spans = names, None
        rest_world = np.zeros((len(names), 3))
        document, _ = read_glb(REBUILD / f"subject-{subject:02d}.glb")
        parents = {n: -1 for n in range(len(document["nodes"]))}
        for position, node in enumerate(document["nodes"]):
            for child in node.get("children", ()):
                parents[child] = position
        joints = document["skins"][0]["joints"]
        order = {node: position for position, node in enumerate(joints)}
        # The node translations are expressed in each PARENT's aligned asset rest frame,
        # so bone LENGTHS are preserved but directions are not; length is what the span
        # table reads, and the spans that are sums of X components are recovered from the
        # track's own rest, which the closure band above has already tied to the GLB.
        rest_spans = spans_from_rest(names, np.asarray(track.rest_translations_m))
        glb_lengths = {
            name: float(np.linalg.norm(glb_rest[order[joints[names.index(name)]]]))
            if False else float(np.linalg.norm(glb_rest[names.index(name)]))
            for name in ("LeftLowerLeg", "LeftFoot", "Spine", "Chest", "UpperChest", "Neck")
        }
        track_lengths = {
            name: float(np.linalg.norm(np.asarray(track.rest_translations_m)[names.index(name)]))
            for name in glb_lengths
        }
        length_error = max(abs(glb_lengths[n] - track_lengths[n]) for n in glb_lengths)
        block["subjects"][f"subject_{subject:02d}"] = {
            "joint_names_match": names == list(skeleton.names),
            "frames": int(glb_positions.shape[0]),
            "max_error_m": float(error.max()),
            "median_error_m": float(np.median(error)),
            "passes": bool(error.max() <= CLOSURE_BAND_M),
            "mustfail_pre_d3_exporter": {
                "max_error_mm": round(1000.0 * float(legacy_error.max()), 2),
                "median_error_mm": round(1000.0 * float(np.median(legacy_error)), 2),
                "per_joint_mm": {
                    name: round(1000.0 * float(legacy_error[:, skeleton.index(name)].max()), 1)
                    for name in ("Hips", "LeftUpperLeg", "LeftFoot", "LeftShoulder",
                                 "LeftHand", "Head")
                },
                "fails_the_band": bool(legacy_error.max() > CLOSURE_BAND_M),
                "over_the_floor": bool(
                    1000.0 * float(np.median(legacy_error)) >= MUSTFAIL_FLOOR_MM
                ),
            },
            "rest_bone_lengths_max_error_m": length_error,
            "rest_bone_lengths_agree": bool(length_error <= 1.0e-6),
            "track_rest_spans_m": {k: round(v, 5) for k, v in rest_spans.items()},
        }
    report["closure"] = block


# ------------------------------------------------------------------ EXACT-SKELETON ORACLE
_CHANNELS = (
    ("LeftLowerArm",), ("LeftHand",), ("RightLowerArm",), ("RightHand",),
    ("LeftLowerLeg",), ("LeftFoot",), ("RightLowerLeg",), ("RightFoot",),
    ("Spine",), ("Chest",), ("UpperChest",), ("Neck",),
)


def perturbed_rest(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Independently scaled channels on the canonical rest. Spans stay symmetric."""

    rest = np.array(DETAILED_HUMANOID.rest_translations_m, copy=True)
    index = {name: position for position, name in enumerate(DETAILED_HUMANOID.names)}
    factors: dict[str, float] = {}

    def draw(label: str) -> float:
        value = float(rng.uniform(PERTURB_LOW, PERTURB_HIGH))
        factors[label] = round(value, 5)
        return value

    for channel in _CHANNELS:
        factor = draw(channel[0])
        for name in channel:
            rest[index[name]] *= factor
    shoulder = draw("shoulder_half_span")
    for name in ("LeftShoulder", "LeftUpperArm", "RightShoulder", "RightUpperArm"):
        rest[index[name]] *= shoulder
    hip = draw("hip_half_span")
    for name in ("LeftUpperLeg", "RightUpperLeg"):
        rest[index[name], 0] *= hip
    vertical = draw("upper_leg_vertical")
    for name in ("LeftUpperLeg", "RightUpperLeg"):
        rest[index[name], 1] *= vertical
    hips = draw("hips_height")
    rest[index["Hips"], 1] *= hips
    return rest, factors


def oracle_block(report: dict) -> None:
    """The one arm whose true value is known: a body we built, recovered exactly."""

    # The truth MOTION is the delivered rotations, re-posed on the perturbed rest. It is
    # a motion the converter can represent BY CONSTRUCTION -- one torso frame for the
    # spine, hands following the forearm, toes following the foot -- which is required:
    # a fixture with independent spine rotations would fail this band for a reason that
    # is not D3, and the band would then be measuring the converter's expressiveness.
    donor = load_track(DELIVERED, 0)
    rotations = np.asarray(donor.local_rotations_xyzw, np.float64)
    roots = np.asarray(donor.root_translation_m, np.float64)
    block: dict = {
        "reference": REF_SYNTH,
        "bands": {"legs_mm": ORACLE_LEGS_BAND_MM, "arms_mm": ORACLE_ARMS_BAND_MM},
        "deviation_from_the_card": (
            "the card says 'an MHR subject'. pymomentum is installed in neither python "
            "here, and the property the card wants -- exact truth with INDEPENDENT scale "
            "channels -- is met without it, by perturbing our own rest directly. What is "
            "lost is MHR's parameterisation; what is kept is exactness."
        ),
        "channels": len(_CHANNELS) + 4,
        "factor_range": [PERTURB_LOW, PERTURB_HIGH],
        "seeds": [],
    }
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rest, factors = perturbed_rest(rng)
        skeleton = DETAILED_HUMANOID.with_rest_translations(rest)
        truth = forward_kinematics_positions(
            roots, rotations, skeleton=skeleton
        ).astype(np.float64)
        landmarks = rc.landmarks_from_fk(truth, skeleton)
        track = cm.positions_to_body_track(
            rc.Z_UP_FROM_Y_UP(landmarks),
            sample_rate_hz=30,
            provenance_sha256="0" * 64,
            skeleton=skeleton,
        )
        path = export(track, SCRATCH / f"oracle-{seed}.glb")
        names, glb_positions, _ = glb_joint_positions(path)
        assert names == list(skeleton.names)
        error = rc.score(glb_positions, landmarks, skeleton)
        # The must-fail: the SAME skeleton, arriving canonical downstream.
        canonical_fk = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64),
            skeleton=DETAILED_HUMANOID,
        ).astype(np.float64)
        canonical_error = rc.score(canonical_fk, landmarks, DETAILED_HUMANOID)
        # The ALTERNATIVE, reported not banded: one global scale for the whole body.
        scale = float(
            np.linalg.norm(rest[DETAILED_HUMANOID.index("Hips")])
            / np.linalg.norm(DETAILED_HUMANOID.rest_translations_m[DETAILED_HUMANOID.index("Hips")])
        ) if np.linalg.norm(DETAILED_HUMANOID.rest_translations_m[DETAILED_HUMANOID.index("Hips")]) else 1.0
        global_skeleton = DETAILED_HUMANOID.with_rest_translations(
            DETAILED_HUMANOID.rest_translations_m * scale
        )
        global_track = cm.positions_to_body_track(
            rc.Z_UP_FROM_Y_UP(landmarks),
            sample_rate_hz=30,
            provenance_sha256="0" * 64,
            skeleton=global_skeleton,
        )
        global_error = rc.score(
            rc.fk_of(global_track, global_skeleton), landmarks, global_skeleton
        )
        block["seeds"].append({
            "seed": seed,
            "factors": factors,
            "glb_vs_truth_mm": groups_mm(error),
            "MUSTFAIL_canonical_downstream_mm": groups_mm(canonical_error),
            "ALT_global_scale_only_mm": groups_mm(global_error),
            "root_correction_by_ground_projection_mm": round(
                1000.0 * float(np.max(np.linalg.norm(
                    np.asarray(track.root_translation_m, np.float64)
                    - (truth[:, skeleton.index("Hips")] - rest[skeleton.index("Hips")]
                       - np.asarray(track.root_translation_m, np.float64) * 0.0),
                    axis=1))), 3) if False else None,
        })
    legs = [s["glb_vs_truth_mm"]["legs"] for s in block["seeds"]]
    arms = [s["glb_vs_truth_mm"]["arms"] for s in block["seeds"]]
    fail_legs = [s["MUSTFAIL_canonical_downstream_mm"]["legs"] for s in block["seeds"]]
    fail_arms = [s["MUSTFAIL_canonical_downstream_mm"]["arms"] for s in block["seeds"]]
    block["worst_legs_mm"] = max(legs)
    block["worst_arms_mm"] = max(arms)
    block["passes"] = bool(max(legs) <= ORACLE_LEGS_BAND_MM and max(arms) <= ORACLE_ARMS_BAND_MM)
    block["mustfail_worst_mm"] = {"legs": min(fail_legs), "arms": min(fail_arms)}
    block["mustfail_fails"] = bool(
        min(fail_legs) > ORACLE_LEGS_BAND_MM or min(fail_arms) > ORACLE_ARMS_BAND_MM
    )
    report["exact_skeleton_oracle"] = block


# ------------------------------------------------------------------ CANONICAL UNCHANGED
def canonical_block(report: dict) -> None:
    """A canonical body through the new plumbing must reproduce D2's committed figures."""

    committed = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/gate-d2c.json").read_text()
    )
    block: dict = {
        "reference": REF_ROUNDTRIP,
        "committed_source": "artifacts/compare/d2-clavicle/gate-d2c.json",
        "band_arms_mm": ROUNDTRIP_ARMS_BAND_MM,
        "subjects": {},
    }
    for subject in (0, 1):
        source = capture_positions(DELIVERED, subject)
        fk1 = rc.retarget_then_fk(source, DETAILED_HUMANOID)
        synthetic = rc.landmarks_from_fk(fk1, DETAILED_HUMANOID)
        fk2 = rc.retarget_then_fk(rc.Z_UP_FROM_Y_UP(synthetic), DETAILED_HUMANOID)
        canonical_roundtrip = groups_mm(rc.score(fk2, synthetic, DETAILED_HUMANOID))
        expected = committed["capture"][f"subject_{subject:02d}"]["d2c_roundtrip_canonical"]
        # Bit-identity of the DELIVERED track under a canonical rest. The converter must
        # be byte-stable: nothing D3 added may perturb a single float32.
        with np.load(DELIVERED / f"subject-{subject:02d}.body-track.npz") as archive:
            delivered_rotations = np.asarray(archive["local_rotations_xyzw"])
            delivered_roots = np.asarray(archive["root_translation_m"])
        block["subjects"][f"subject_{subject:02d}"] = {
            "roundtrip_canonical_mm": canonical_roundtrip,
            "committed_d2c_mm": expected,
            "matches_committed": bool(
                abs(canonical_roundtrip["arms"] - expected["arms"]) <= 0.01
                and canonical_roundtrip["legs"] == expected["legs"]
                and canonical_roundtrip["torso"] == expected["torso"]
            ),
            "within_band": bool(
                canonical_roundtrip["arms"] <= ROUNDTRIP_ARMS_BAND_MM
                and canonical_roundtrip["legs"] == 0.0
            ),
            "delivered_rotation_bit_identity": None,
            "delivered_root_bit_identity": None,
            "delivered_shapes": [list(delivered_rotations.shape), list(delivered_roots.shape)],
        }
    report["canonical_unchanged"] = block


def canonical_bit_identity(report: dict, work: Path) -> None:
    """Re-run the WHOLE delivery reconstruction with the sizing forced to canonical.

    The converter's inputs are not on disk -- the head solve and the toe solve are
    computed inside `reconstruct_multiview` -- so the only faithful way to ask "is the
    canonical answer byte-stable" is to run the real function with one module attribute
    swapped, exactly the pattern D2's gate uses.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "d3_build", ROOT / "scripts/build_commercial_multiview_comparison.py"
    )
    if spec is None or spec.loader is None:
        report["canonical_unchanged"]["bit_identity"] = {"ran": False, "why": "build script not importable"}
        return
    canonical_build = OUT_DIR / "delivery-canonical"
    if not (canonical_build / "subject-00.body-track.npz").exists():
        report["canonical_unchanged"]["bit_identity"] = {
            "ran": False,
            "why": (
                "not run in-process: the reconstruction is a whole-build entry point. Build it "
                "with `--rest-skeleton canonical --output artifacts/compare/d3-skeleton/"
                "delivery-canonical` (work/ COPIED from the delivered build) and rerun this gate."
            ),
        }
        return
    rows: dict = {}
    identical = True
    for subject in (0, 1):
        ours = np.load(canonical_build / f"subject-{subject:02d}.body-track.npz")
        theirs = np.load(DELIVERED / f"subject-{subject:02d}.body-track.npz")
        per = {}
        for key in ("local_rotations_xyzw", "root_translation_m", "foot_contacts",
                    "triangulated_world_positions_z_up_m"):
            same = bool(np.array_equal(ours[key], theirs[key]))
            per[key] = same
            identical = identical and same
        per["rest_is_canonical"] = bool(np.array_equal(
            ours["rest_translations_m"], DETAILED_HUMANOID.rest_translations_m))
        identical = identical and per["rest_is_canonical"]
        rows[f"subject_{subject:02d}"] = per
    report["canonical_unchanged"]["bit_identity"] = {
        "ran": True,
        "how": "the whole delivery rebuilt from the cached detections with `--rest-skeleton "
               "canonical` (the D3 build script's instrument arm), compared array by array on "
               "disk against the pre-D3 delivered build",
        "bit_identical": identical,
        "per_subject": rows,
    }


# ------------------------------------------------------------------ DELIVERED FIGURES
def delivered_block(report: dict) -> None:
    block: dict = {
        "banded": False,
        "why_not_banded": (
            "these are the consequences of the change, not its proof. The proof is "
            "CLOSURE and the exact-skeleton oracle; a band on a delivered agreement "
            "figure would be a band the candidate can move by changing the sizing, "
            "which is D5's question."
        ),
        "subjects": {},
        "work_byte_identical": {},
    }
    for name in sorted(p.name for p in (DELIVERED / "work").glob("*observations.jsonl")):
        a = (DELIVERED / "work" / name).read_bytes()
        b = (REBUILD / "work" / name).read_bytes()
        block["work_byte_identical"][name] = bool(a == b)
    for subject in (0, 1):
        before = load_track(DELIVERED, subject)
        after = load_track(REBUILD, subject)
        source = capture_positions(REBUILD, subject)
        source_before = capture_positions(DELIVERED, subject)
        after_skeleton = skeleton_for_track(after)
        before_skeleton = skeleton_for_track(before)
        points_before = rc.Y_UP_FROM_Z_UP(source_before)
        points_after = rc.Y_UP_FROM_Z_UP(source)
        error_before = rc.score(rc.fk_of(before, before_skeleton), points_before, before_skeleton)
        error_after = rc.score(rc.fk_of(after, after_skeleton), points_after, after_skeleton)
        _, sizing = performer_skeleton(DETAILED_HUMANOID, source)
        block["subjects"][f"subject_{subject:02d}"] = {
            "reference_on_our_capture": REF_OURS,
            "delivered_on_our_capture_before_mm": groups_mm(error_before),
            "delivered_on_our_capture_after_mm": groups_mm(error_after),
            "rest_spans_before_m": {k: round(v, 5) for k, v in span_table(before_skeleton).items()},
            "rest_spans_after_m": {k: round(v, 5) for k, v in span_table(after_skeleton).items()},
            "measured_targets_m": {k: round(v, 5) for k, v in sizing.items()},
            "foot_contacts_before": [int(v) for v in np.sum(before.foot_contacts, axis=0)],
            "foot_contacts_after": [int(v) for v in np.sum(after.foot_contacts, axis=0)],
            "capture_positions_byte_identical": bool(
                np.array_equal(source, source_before)
            ),
            "track_carries_a_performer_rest": bool(after_skeleton is not DETAILED_HUMANOID),
            "schema_version": after.as_dict()["schema_version"],
        }
    report["delivered"] = block


def hoist_block(report: dict) -> None:
    """The ground projection's own correction, before -> after, on each build's own rest."""

    from autoanim_gnm.body_projection import project_generated_foot_contacts

    out = {}
    for subject in (0, 1):
        row = {}
        for label, directory in (("before", DELIVERED), ("after", REBUILD)):
            track = load_track(directory, subject)
            skeleton = skeleton_for_track(track)
            positions = forward_kinematics_positions(
                np.asarray(track.root_translation_m, np.float64),
                np.asarray(track.local_rotations_xyzw, np.float64),
                skeleton=skeleton,
            )
            floor = float(np.min(positions[..., 1]))
            row[label] = {
                "lowest_joint_y_m": round(floor, 5),
                "contacts": [int(v) for v in np.sum(track.foot_contacts, axis=0)],
            }
        # The hoist itself: re-run the projection on the unprojected converter output.
        for label, directory in (("before", DELIVERED), ("after", REBUILD)):
            source = capture_positions(directory, subject)
            skeleton = (
                DETAILED_HUMANOID if label == "before"
                else performer_skeleton(DETAILED_HUMANOID, source)[0]
            )
            unprojected_root = None
            saved = cm.project_generated_foot_contacts
            captured: list = []

            def watcher(track, **kwargs):
                captured.append(track)
                return saved(track, **kwargs)

            cm.project_generated_foot_contacts = watcher
            try:
                solved = cm.positions_to_body_track(
                    source, sample_rate_hz=30, provenance_sha256="0" * 64,
                    skeleton=skeleton,
                )
            finally:
                cm.project_generated_foot_contacts = saved
            unprojected_root = np.asarray(captured[-1].root_translation_m, np.float64)
            projected_root = np.asarray(solved.root_translation_m, np.float64)
            row[label]["hoist_max_mm"] = round(
                1000.0 * float(np.max(np.abs(projected_root - unprojected_root))), 2
            )
            row[label]["hoist_median_mm"] = round(
                1000.0 * float(np.median(np.linalg.norm(projected_root - unprojected_root, axis=1))), 2
            )
        out[f"subject_{subject:02d}"] = row
    report["ground_projection"] = {
        "note": (
            "the hoist is measured WITHOUT the head and toe solves, so it is not the "
            "delivered number to the millimetre; both arms are computed the same way and "
            "the comparison is the figure."
        ),
        "subjects": out,
    }


# ------------------------------------------------------------------ external reports
def fold_external(report: dict) -> None:
    block: dict = {}
    scoreboard = OUT_DIR / "scoreboard-d3-after.json"
    if not scoreboard.exists():
        scoreboard = ROOT / "artifacts/compare/scoreboard-d3-after.json"   # the scoreboard writes here
    for path, key, reference in (
        (ROOT / "artifacts/compare/scoreboard-d2c-after.json", "rung11_before", REF_MAMMA),
        (scoreboard, "rung11_after", REF_MAMMA),
    ):
        if path.exists():
            payload = json.loads(path.read_text())
            block[key] = {"reference": reference, "source": str(path.relative_to(ROOT)),
                          "headline": _scoreboard_headline(payload)}
    for path, key, reference in (
        (ROOT / "artifacts/compare/facing-location.json", "facing_before",
         "the footage's own camera frames"),
        (OUT_DIR / "facing-d3.json", "facing_after", "the footage's own camera frames"),
    ):
        if path.exists():
            block[key] = {"reference": reference, "source": str(path.relative_to(ROOT)),
                          "headline": _facing_headline(json.loads(path.read_text()))}
    for path, key in (
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-d2c.json", "silhouette_before"),
        (OUT_DIR / "silhouette-d3.json", "silhouette_after"),
    ):
        if path.exists():
            payload = json.loads(path.read_text())
            block[key] = {"reference": REF_MASKS, "source": str(path.relative_to(ROOT)),
                          "headline": _silhouette_headline(payload)}
    for path, key in (
        (ROOT / "artifacts/compare/d2-clavicle/retarget-cost-d2c.json", "retarget_before"),
        (OUT_DIR / "retarget-cost-d3.json", "retarget_after"),
    ):
        if path.exists():
            block[key] = {"source": str(path.relative_to(ROOT)),
                          "headline": _retarget_headline(json.loads(path.read_text()))}
    sil_before = ROOT / "artifacts/compare/d2-clavicle/silhouette-d2c.json"
    sil_after = OUT_DIR / "silhouette-d3.json"
    if sil_before.exists() and sil_after.exists():
        block["silhouette_summary"] = _silhouette_summary(
            json.loads(sil_before.read_text()), json.loads(sil_after.read_text()))
    report["external_reports"] = block


def _scoreboard_headline(payload: dict) -> dict:
    """`subjects/subject_XX/median_mm/{capture,canon,sized}` -- the scoreboard's own
    statistic (per-joint median over frames, then median over the 15 joints), plus the
    arm definitions the D3 scoreboard writes, so a reader cannot mistake `sized` for a
    replay on a D3 build (there it is the DELIVERED body, on the rest the track carries)."""
    out: dict = {}
    for subject, block in payload.get("subjects", {}).items():
        for arm, value in block.get("median_mm", {}).items():
            out[f"{subject}/median_mm/{arm}"] = round(float(value), 2)
        for arm, meaning in block.get("arms", {}).items():
            out[f"{subject}/arm_definition/{arm}"] = meaning
        if "track_carries_own_rest" in block:
            out[f"{subject}/track_carries_own_rest"] = block["track_carries_own_rest"]
    return out


def _retarget_headline(payload: dict) -> dict:
    """Every numeric leaf under `subjects/` whose path names a median and a joint group."""
    out: dict = {}

    def walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (key,))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            joined = "/".join(str(p) for p in path)
            if "median" in joined and any(g in joined for g in ("arms", "legs", "torso", "all")):
                out[joined] = round(float(node), 2)

    walk(payload.get("subjects", {}))
    return out


def _silhouette_summary(before: dict, after: dict) -> dict:
    """Median IoU over the eight camera x subject cells, ours and the oracle, before and
    after, and whether the ORACLE arm is bit-identical between the two runs (it reads none
    of our track, so identical means the reused caches changed nothing)."""
    def cells(payload: dict, arm: str, stat: str = "iou"):
        rows = {}
        for camera, per_subject in payload["arms"][arm].items():
            for subject, cell in per_subject.items():
                rows[f"{camera}/{subject}"] = float(cell[stat]["median"])
        return rows
    ours_b, ours_a = cells(before, "ours_delivered"), cells(after, "ours_delivered")
    orc_b, orc_a = cells(before, "ORACLE_mamma_mesh"), cells(after, "ORACLE_mamma_mesh")
    keys = sorted(ours_b)
    fell = [k for k in keys if ours_a[k] < ours_b[k]]
    per_subject = {}
    for subject in ("subject_00", "subject_01"):
        ks = [k for k in keys if k.endswith(subject)]
        per_subject[subject] = {
            "before_median_iou": round(float(np.median([ours_b[k] for k in ks])), 4),
            "after_median_iou": round(float(np.median([ours_a[k] for k in ks])), 4),
            "before_precision": round(float(np.median([cells(before, "ours_delivered", "precision")[k] for k in ks])), 4),
            "after_precision": round(float(np.median([cells(after, "ours_delivered", "precision")[k] for k in ks])), 4),
            "before_recall": round(float(np.median([cells(before, "ours_delivered", "recall")[k] for k in ks])), 4),
            "after_recall": round(float(np.median([cells(after, "ours_delivered", "recall")[k] for k in ks])), 4),
        }
    return {
        "reference": REF_MASKS,
        "ours_before_median_iou_8_cells": round(float(np.median(list(ours_b.values()))), 4),
        "ours_after_median_iou_8_cells": round(float(np.median(list(ours_a.values()))), 4),
        "cells_that_fell": len(fell), "cells": len(keys),
        "per_cell_before": {k: round(v, 4) for k, v in ours_b.items()},
        "per_cell_after": {k: round(v, 4) for k, v in ours_a.items()},
        "per_subject": per_subject,
        "oracle_median_iou": round(float(np.median(list(orc_a.values()))), 4),
        "oracle_bit_identical_between_runs": bool(
            max(abs(orc_a[k] - orc_b[k]) for k in keys) == 0.0),
        "oracle_max_abs_difference": float(max(abs(orc_a[k] - orc_b[k]) for k in keys)),
    }


def _facing_headline(payload: dict) -> dict:
    out: dict = {}

    def walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (key,))
        elif isinstance(node, (int, float)):
            joined = "/".join(str(p) for p in path)
            if "handedness" in joined or ("forward" in joined and "median" in joined):
                out[joined] = node

    walk(payload)
    return out


def _silhouette_headline(payload: dict) -> dict:
    out: dict = {}

    def walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (key,))
        elif isinstance(node, (int, float)):
            joined = "/".join(str(p) for p in path)
            if "iou" in joined.lower():
                out[joined] = node

    walk(payload)
    return out


# ------------------------------------------------------------------ consumer manifest
CONSUMER_MANIFEST = json.loads(
    (Path(__file__).with_name("d3_consumer_manifest.json")).read_text()
) if (Path(__file__).with_name("d3_consumer_manifest.json")).exists() else {}


# ------------------------------------------------------------------ main
def main() -> int:
    started = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "instrument": "tools/compare/d3_skeleton_gate.py",
        "step": "D3 -- one rest skeleton per performer, serialised and propagated",
        "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d3_skeleton_gate.py",
        "autoanim_gnm_resolved_to": str(Path(autoanim_gnm.__file__).resolve()),
        "pre_registered": PRE_REGISTERED,
        "consumer_manifest": CONSUMER_MANIFEST,
        "references_never_on_one_axis": {
            "closure": REF_CLOSURE,
            "synthetic_truth": REF_SYNTH,
            "roundtrip": REF_ROUNDTRIP,
            "our_own_capture": REF_OURS,
            "mamma": REF_MAMMA,
            "masks": REF_MASKS,
        },
        "blind_to": [
            "whether the rest itself is RIGHT. Closure proves the delivery carries one "
            "body, not that the body is the performer's. Sizing is D5.",
            "the mesh. The vertices, weights and inverse bind matrices are unchanged, so "
            "the skin now deforms far outside what its bind pose was solved for. That is "
            "D6, and it is the reason the silhouette is reported and not banded.",
            "the trunk. The rig still has ONE frame from Hips to UpperChest, so on a bent "
            "performer the whole torso rides the lean. A pelvis frame is its own step.",
            "the shoulder construction. Scaling the clavicle chain uniformly lowers the "
            "arm root ~36 mm where an X-only scaling would not; which is right is a "
            "selection question and it is D5's.",
        ],
    }
    closure_block(report)
    oracle_block(report)
    canonical_block(report)
    canonical_bit_identity(report, REBUILD / "work")
    delivered_block(report)
    hoist_block(report)
    fold_external(report)

    gate: list[dict] = []

    def band(name: str, ok: bool, detail: str) -> None:
        gate.append({"band": name, "verdict": "PASS" if ok else "FAIL", "detail": detail})

    bi = report["canonical_unchanged"].get("bit_identity", {})
    band(
        "CANONICAL UNCHANGED. the whole delivery rebuilt with the canonical rest is "
        "BIT-IDENTICAL to the pre-D3 delivered build (rotations, root, contacts, positions)",
        bool(bi.get("ran")) and bool(bi.get("bit_identical")),
        json.dumps(bi.get("per_subject", bi.get("why")))[:200],
    )
    for subject, values in report["closure"]["subjects"].items():
        band(
            f"CLOSURE. {subject}: exported GLB forward kinematics = the track's own, <= 1e-4 m",
            values["passes"],
            f"max {values['max_error_m']:.3e} m",
        )
        band(
            f"MUST-FAIL. {subject}: the pre-D3 exporter on the same track fails the band",
            values["mustfail_pre_d3_exporter"]["fails_the_band"]
            and values["mustfail_pre_d3_exporter"]["over_the_floor"],
            f"median {values['mustfail_pre_d3_exporter']['median_error_mm']} mm",
        )
        band(
            f"CLOSURE. {subject}: the GLB's rest bone lengths equal the track's",
            values["rest_bone_lengths_agree"],
            f"max {values['rest_bone_lengths_max_error_m']:.3e} m",
        )
    band(
        "EXACT-SKELETON ORACLE. legs 0.00 mm and arms <= 0.5 mm on every seed",
        report["exact_skeleton_oracle"]["passes"],
        f"worst legs {report['exact_skeleton_oracle']['worst_legs_mm']} mm, "
        f"worst arms {report['exact_skeleton_oracle']['worst_arms_mm']} mm",
    )
    band(
        "MUST-FAIL. the same skeleton arriving CANONICAL downstream fails the oracle band",
        report["exact_skeleton_oracle"]["mustfail_fails"],
        json.dumps(report["exact_skeleton_oracle"]["mustfail_worst_mm"]),
    )
    for subject, values in report["canonical_unchanged"]["subjects"].items():
        band(
            f"CANONICAL UNCHANGED. {subject}: the round trip reproduces D2c's committed figures",
            values["matches_committed"],
            f"{values['roundtrip_canonical_mm']} vs committed {values['committed_d2c_mm']}",
        )
    band(
        "REBUILD. nothing was re-extracted or re-detected",
        all(report["delivered"]["work_byte_identical"].values()),
        json.dumps(report["delivered"]["work_byte_identical"]),
    )
    for subject, values in report["delivered"]["subjects"].items():
        band(
            f"PROPAGATION. {subject}: the delivered track carries a per-performer rest",
            values["track_carries_a_performer_rest"],
            values["schema_version"],
        )
        band(
            f"SAME DENOMINATOR. {subject}: the rebuild's triangulation is byte-identical",
            values["capture_positions_byte_identical"],
            "before and after are scored on the same positions",
        )
    report["gate"] = gate
    report["verdict"] = "PASS" if all(g["verdict"] == "PASS" for g in gate) else "FAIL"
    report["seconds"] = round(time.time() - started, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate.json").write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    for row in gate:
        print(f"{row['verdict']:5s} {row['band']}  --  {row['detail']}")
    print(f"\nD3 verdict: {report['verdict']}  ({report['seconds']} s)")
    return 0


PRE_REGISTERED = {
    "committed": "fe1a17b, docs/reviews/performer-skeleton-2026-09-03.md section 0",
    "P1": "CLOSURE: the exported GLB's forward kinematics equals the track's own on the "
          "track's rest, <= 1e-4 m, both subjects. Predicted order 1e-6 m -- it is an "
          "identity, not a fit.",
    "P2": "MUST-FAIL: the pre-D3 exporter on the same track reads >= 50 mm median.",
    "P3": "EXACT-SKELETON ORACLE: legs 0.00 mm, arms <= 0.5 mm (D2's clavicle residual), "
          "on >= 5 seeds of independently perturbed scale channels.",
    "P4": "MUST-FAIL: the same skeleton arriving canonical downstream reads tens of mm.",
    "P5": "CANONICAL UNCHANGED: the canonical round trip reproduces D2c's committed "
          "0.55 / 0.08 mm arms and 0.00 legs and torso.",
    "P6": "the delivered track rotations for a CANONICAL skeleton are bit-identical to "
          "the committed npz.",
    "P7": "delivered vs our own capture: arms 25-30 / 21-27 mm, legs 17-23 / 13-19 mm, "
          "and the TORSO improves on the sized replay on both subjects (the torso-drop "
          "fold is exactly that term).",
    "P8": "the ground hoist FALLS: these performers' thighs and shins are ~30 mm shorter "
          "than canonical each.",
    "P9": "the clavicle chain reads 0 frames over the 800 deg/s ceiling, by construction.",
    "P10": "facing: every forward-dot median within 0.02 of committed and every "
           "handedness triple-product sign unchanged. The dots MOVE -- a sized rig has "
           "different rotations and the instrument reads rotations.",
    "P11": "the silhouette is pre-registered BOTH ways: the root's share should shrink "
           "(the mesh's hips now sit on the captured hips), the whole-person figure is "
           "unknown (the mesh deforms under canonical weights). Reported, never banded.",
    "P12": "the schema must-fails: wrong shape, non-finite, non-zero Root, legacy JSON "
           "loads canonical, as_dict/from_dict bit-identical, a sized track cannot be "
           "validated against canonical.",
}


if __name__ == "__main__":
    raise SystemExit(main())
