#!/usr/bin/env python3
"""D1 (locate): WHERE does the facing defect live? Not "does it exist" -- where.

The 2026-08-30 finding was "every delivered character faces 180 degrees the wrong way".
On 2026-09-02 the surface instrument (I6) turned our delivered mesh 180 degrees and it
got WORSE on every camera, decisively on seven of eight cells -- which, read plainly,
says the shipped facing is already right. Both cannot be true, and the overlay script
the original diagnosis ran through had a timebase bug (fixed in the commit before this
one), so the original evidence is suspect too. This instrument re-locates the defect
before anything under `src/` is touched.

WHAT IT MEASURES, and why each piece exists

  1. THE ASSET'S OWN HANDEDNESS, read off the shipped `neutral-body.npz`. A skeleton's
     naming is mirrored if the bones called Left are on the opposite side from the mesh's
     anatomical left, and the mesh's anatomical left follows from which way its face
     points. One signed number states it: the HANDEDNESS TRIPLE PRODUCT
     sign((across x up) . forward) with across = right - left BY NAME. Every real human
     gives one sign; a mirrored rig gives the other. This is the only figure here that
     needs no capture at all.

  2. THE HANDEDNESS TRIPLE PRODUCT on five arms -- our triangulated capture, MAMMA's
     `pred_joints`, the delivered rig read two different ways, and the asset at rest.
     A yaw cannot change it and a mirror must; I1 scores joints by name and I6 scores an
     outline, and neither can see a mirror at all.

  3. THE FORWARD-DOT, the quantity the D1 gate card actually asks for: the direction each
     part of the delivered character faces, against the direction the footage says the
     performer faces, per bone group and per camera.

  4. THE SURFACE, from `facing_surface_probe.py`: where the delivered MESH's nose points,
     measured on the shipped GLB through the real skinning, with vertex sets picked out
     of the bind pose by geometry so that no joint name enters. This is the reading that
     answers I6 in I6's own currency, and the one no by-name score can fake.

  5. WHY I6 SAW WHAT IT SAW. Quantified, not argued.

STANDING RULES. MAMMA is an instrument and never selects anything; it appears here as a
second, independent opinion on which way the performers face, beside our own capture,
and the two are reported separately. `body_id-00` is our subject 1, resolved through
`tools/head/subject_map.py`. Nothing here writes under `artifacts/commercial-multiview-*`
and nothing here changes `src/`.

    .venv/bin/python tools/compare/facing_location.py
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names  # noqa: E402
from autoanim_gnm.commercial_multiview import DETAILED_HUMANOID, JOINT_INDEX  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

MA3D = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
ASSET = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands/neutral-body.npz"
PROBE = ROOT / "artifacts/compare/d1-facing/surface-probe.json"
OUT = ROOT / "artifacts/compare/facing-location.json"

# MAMMA `pred_joints` (SMPL-X) indices. Feet 10/11 are the ball of the foot; the
# ankle->ball direction is anterior, which I4 measured rather than assumed.
MA = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "neck": 12,
      "left_shoulder": 16, "right_shoulder": 17,
      "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11}

BLOCK, RESAMPLES, SEED = 15, 2000, 20260902


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rig_to_capture(v: np.ndarray) -> np.ndarray:
    """Rig space (Y up) -> the capture's Z-up world. The inverse of the change of basis
    `positions_to_body_track` applies, asserted numerically in `basis_assertion`."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def world_quaternions(local_xyzw: np.ndarray, parents: list[int]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    world = np.zeros_like(local_xyzw)
    for index, parent in enumerate(parents):
        rotation = Rotation.from_quat(local_xyzw[:, index])
        world[:, index] = (
            rotation if parent == -1 else Rotation.from_quat(world[:, parent]) * rotation
        ).as_quat()
    return world


def rotate(quaternion_xyzw: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quaternion_xyzw).apply(np.tile(vector, (len(quaternion_xyzw), 1)))


def triple(across: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """sign((across x up) . forward), per frame. across = RIGHT minus LEFT, by name.

    Invariant under any proper rotation and under uniform scale; it flips under any
    reflection. That is the whole point: a 180 degree yaw leaves it alone and a
    left/right mirror inverts it, so it separates the two failures that a silhouette
    and a by-name joint score both confuse."""
    return np.einsum("fj,fj->f", np.cross(across, up), forward)


def block_bootstrap_median(values: np.ndarray, seed: int = SEED) -> dict:
    """Moving-block bootstrap CI of the median. Per-frame series here are strongly
    autocorrelated (lag-1 ~0.99 in this lane), so ordinary resampling is invalid."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < BLOCK * 2:
        return {"median": float(np.median(values)) if len(values) else None, "ci95": None}
    rng = np.random.default_rng(seed)
    starts = len(values) - BLOCK + 1
    blocks = int(np.ceil(len(values) / BLOCK))
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picks = rng.integers(0, starts, blocks)
        draws[i] = np.median(
            np.concatenate([values[p:p + BLOCK] for p in picks])[:len(values)]
        )
    lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if len(values) > 2 else None
    return {"median": float(np.median(values)),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "block_length": BLOCK, "resamples": RESAMPLES, "lag1_autocorrelation": lag1}


def digest(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None


def label_path(path: Path) -> str:
    """`artifacts/` and `.cache/` are symlinks in a worktree, so `relative_to(ROOT)`
    raises on a resolved path. Report a repo-relative label without resolving."""
    import os

    return os.path.relpath(path, ROOT)


# ---------------------------------------------------------------------------------------
# 1. the asset's own handedness, and the SHA chain that says which asset it is
# ---------------------------------------------------------------------------------------

def rest_convention(asset_path: Path = ASSET) -> dict:
    """Read the handedness off the shipped asset. No capture, no reference, no camera."""
    asset = np.load(asset_path)
    names = [str(v) for v in asset["joint_names"].tolist()]
    vertices = asset["vertices_m"].astype(np.float64)
    parents = np.asarray(asset["parents"], dtype=np.int64)
    local_rest = asset["local_rest_matrices"].astype(np.float64)
    world = np.zeros((len(names), 4, 4))
    for index, parent in enumerate(parents.tolist()):
        world[index] = local_rest[index] if parent == -1 else world[parent] @ local_rest[index]
    at = {name: world[i][:3, 3] for i, name in enumerate(names)}
    dominant = asset["joint_indices"][np.arange(len(vertices)), asset["joint_weights"].argmax(1)]
    head = vertices[dominant == names.index("Head")]
    nose = head[head[:, 2].argmax()]
    occiput = head[head[:, 2].argmin()]

    def side(name: str) -> float:
        return float(np.mean(vertices[dominant == names.index(name)][:, 0]))

    out = {}
    for label, joints in (("asset_mpfb_neutral_body", at), ("code_DETAILED_HUMANOID", None)):
        if joints is None:
            positions, joints = {}, {}
            for joint in DETAILED_HUMANOID.joints:
                offset = np.asarray(joint.rest_translation_m, dtype=np.float64)
                name = joint.name
                positions[name] = offset if joint.parent == -1 else (
                    offset + positions[DETAILED_HUMANOID.joints[joint.parent].name]
                )
            joints = positions
        across = joints["RightUpperArm"] - joints["LeftUpperArm"]   # BY NAME: right - left
        up = joints["Neck"] - joints["Hips"]
        forward = joints["LeftToes"] - joints["LeftFoot"]           # the skeleton's own toes
        sign = float(np.sign(np.dot(np.cross(across, up), forward)))
        out[label] = {
            "left_named_joints_x": float(joints["LeftUpperArm"][0]),
            "right_named_joints_x": float(joints["RightUpperArm"][0]),
            "toes_minus_foot_z": float(forward[2]),
            "eye_z_minus_head_z": float(joints["LeftEye"][2] - joints["Head"][2]),
            "handedness_triple_product_sign": sign,
        }
    out["mesh_surface"] = {
        "most_anterior_head_vertex_xyz_m": [float(v) for v in nose],
        "most_posterior_head_vertex_xyz_m": [float(v) for v in occiput],
        "reading": ("the extreme +Z head vertex sits on the midline at eye height and "
                    "protrudes: it is the nose, so the MESH faces rig +Z"),
        "mean_x_of_vertices_bound_to_LeftUpperArm": side("LeftUpperArm"),
        "mean_x_of_vertices_bound_to_RightUpperArm": side("RightUpperArm"),
    }
    out["verdict"] = (
        "The mesh faces rig +Z (nose at +Z) with up +Y. A right-handed body facing +Z with "
        "up +Y has its anatomical LEFT at +X. The bones named Left sit at -X and carry the "
        "mesh's -X flesh. So the asset's Left-named bones drive the mesh's anatomical "
        "RIGHT: the naming is mirrored relative to the geometry, in BOTH the shipped MPFB "
        "asset and the DETAILED_HUMANOID skeleton in code, identically. The triple product "
        "sign below is the opposite of every real human measured here."
    )
    return out


def head_convention() -> dict:
    """The mirror's THIRD site, and the one a skeleton fix would not reach.

    The head is not carried by the torso frame: since 981e437 `positions_to_body_track`
    sets it from its own rigid fit as an absolute world rotation. It is still backwards
    (-0.89 / -0.94), so it must be reaching the same wrong answer by its own route -- and
    it is. `head_orientation.CANONICAL_HEAD_AXES` is a hardcoded frame declaring, in the
    capture convention, that the subject's LEFT is rig -X and FORWARD is rig -Z. It is a
    perfectly good right-handed frame, and it is the mirror image of the geometry it
    drives: the asset's nose, eyes and toes are all at rig +Z and the track schema's own
    `gaze.direction_body` is [0, 0, 1].

    The head template itself is innocent -- it is learned from the subject's own landmarks
    (`_initialise` takes the median of the centred targets) and `_anatomical_gauge` fixes
    the zero from that template's own eyes and `HeadEnd`, consulting no reference. The
    error is only in the constant it is mapped ONTO.

    This matters for the fix: mirroring the skeleton and the mesh does NOT move this
    literal, so a fix that stops there would leave the head pointing backwards on a body
    that had just been turned round -- and the gate's head band would fail for a reason
    that has nothing to do with the torso."""
    from autoanim_gnm.head_orientation import CANONICAL_HEAD_AXES

    axes = np.asarray(CANONICAL_HEAD_AXES, dtype=np.float64)
    left, up, forward = axes[:, 0], axes[:, 1], axes[:, 2]

    def to_rig(v: np.ndarray) -> list[float]:
        return [float(v[0]), float(v[2]), float(-v[1])]

    return {
        "where": "src/autoanim_gnm/head_orientation.py :: CANONICAL_HEAD_AXES",
        "columns_are": "(subject's left, up through the skull, forward), capture convention",
        "declared_left_in_rig_axes": to_rig(left),
        "declared_forward_in_rig_axes": to_rig(forward),
        "is_right_handed": bool(np.allclose(np.cross(left, up), forward)),
        "asset_says_forward_is_rig": [0.0, 0.0, 1.0],
        "asset_evidence": ("nose, eyes and toes all at rig +Z; the body-track schema's "
                           "gaze.direction_body is [0, 0, 1]"),
        "agrees_with_the_asset": bool(np.allclose(to_rig(forward), [0.0, 0.0, 1.0])),
        "reading": (
            "The head solve is doing exactly what it was told: it puts rig -Z on the "
            "subject's forward, and the mesh's face is at rig +Z, so the face comes out "
            "backwards. Same mirror, independent site."),
        "what_it_must_become_after_the_asset_is_mirrored": {
            "left": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "forward": [0.0, -1.0, 0.0],
            "note": ("in capture coordinates: rig +X (the new Left) is capture +X, rig +Y "
                     "is capture +Z, rig +Z (the face) is capture -Y. left x up = forward "
                     "still holds, so the frame stays right-handed."),
        },
    }


def sha_chain(delivery: Path = DELIVERY, asset_path: Path = ASSET) -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    src = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%h %ad %s", "--date=iso", "--", "src/"],
        capture_output=True, text=True).stdout.strip()
    files = {
        "subject-00.glb": delivery / "subject-00.glb",
        "subject-01.glb": delivery / "subject-01.glb",
        "subject-00.body-track.npz": delivery / "subject-00.body-track.npz",
        "subject-01.body-track.npz": delivery / "subject-01.body-track.npz",
        "camera-rig.json": delivery / "camera-rig.json",
        "run-report.json": delivery / "run-report.json",
        "neutral-body.npz": asset_path,
    }
    shas = {name: digest(path) for name, path in files.items()}
    silhouette = json.loads((ROOT / "artifacts/compare/silhouette.json").read_text())
    feet = json.loads((ROOT / "artifacts/feet-lane/mamma-feet-bar.json").read_text())
    report = json.loads((delivery / "run-report.json").read_text())
    i6_inputs = silhouette.get("input_sha256", {})
    return {
        "delivery": label_path(delivery),
        "asset": label_path(asset_path),
        "src_head_commit": head,
        "last_commit_touching_src": src,
        "delivery_mtime_utc": __import__("datetime").datetime.utcfromtimestamp(
            (delivery / "subject-00.glb").stat().st_mtime).isoformat() + "Z",
        "sha256": shas,
        "i6_rendered_these_glbs": {
            "subject-00": i6_inputs.get("subject-00.glb") == shas["subject-00.glb"],
            "subject-01": i6_inputs.get("subject-01.glb") == shas["subject-01.glb"],
        },
        "i4_scored_these_tracks": {
            f"subject-{s:02d}": feet.get("input_sha256", {}).get(
                f"artifacts/commercial-multiview-soma77/subject-{s:02d}.body-track.npz"
            ) == shas[f"subject-{s:02d}.body-track.npz"] for s in (0, 1)
        },
        "delivered_build_carries_the_head_solve": report["head_orientation"][0]["status"],
        "delivered_build_carries_the_toe_solve": report["toe_triangulation"][0]["status"],
        "finding": (
            "The delivery was written 2026-09-01 15:15:33, THIRTY SECONDS before f6a4973 "
            "was committed and eighteen hours after 981e437 put the head solve on the "
            "delivery path. `run-report.json` records head_orientation 'solved' and "
            "toe_triangulation 'solved', and no commit has touched src/ since f6a4973. "
            "The delivered tracks therefore DO reflect current code and DO postdate the "
            "head fix. The parity board's claim that they predate it is wrong. I6 and I4 "
            "both scored these exact bytes, so all three instruments share one build."
        ),
        "the_2026_08_30_build_is_not_on_disk": (
            "The facing diagnosis was committed 2026-08-31 00:28 (fd8f7373), before "
            "981e437 (head solve, 21:15 the same day) and before f6a4973 (feet, 09-01). "
            "The GLBs it was made from were overwritten by the 09-01 rebuild and cannot "
            "be re-measured. On that build the head inherited the torso frame and the "
            "feet were welded to it, so the whole character -- face included -- was "
            "yawed; today the feet are solved separately and the head is solved but "
            "still lands backwards (see forward_dot). That is why it was obvious on "
            "sight then and needs an instrument now."
        ),
        "no_facing_instrument_ever_existed": (
            "The -0.42 / -0.90 facing dots in BODY_LANE_PLAN have no script behind them "
            "in any commit -- searched across all history. A number that lives only in a "
            "document is 'instrument missing'. This file is that instrument; its figures "
            "supersede those two, which should be struck rather than reconciled."
        ),
    }


# ---------------------------------------------------------------------------------------
# 2. the arms
# ---------------------------------------------------------------------------------------

def capture_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """across (right - left) and up, from our 19-joint triangulated contract."""
    left, right = joints[:, JOINT_INDEX["left_hip"]], joints[:, JOINT_INDEX["right_hip"]]
    pelvis = 0.5 * (left + right)
    across = joints[:, JOINT_INDEX["right_shoulder"]] - joints[:, JOINT_INDEX["left_shoulder"]]
    return across, joints[:, JOINT_INDEX["neck"]] - pelvis


def capture_forward(joints: np.ndarray) -> np.ndarray:
    """unit(left x up): the performer's facing, from the pelvis and the neck.

    This is the reference the forward-dot is scored against, and it is DERIVED FROM THE
    FOOTAGE only in the sense that the joints were triangulated from it. It cannot itself
    be mirrored-and-undetected, because MAMMA's independent answer is reported beside it
    and the two agree (see `oracle_capture_vs_mamma_forward`)."""
    left, right = joints[:, JOINT_INDEX["left_hip"]], joints[:, JOINT_INDEX["right_hip"]]
    up = joints[:, JOINT_INDEX["neck"]] - 0.5 * (left + right)
    return unit(np.cross(left - right, up))


def mamma_arrays(body_id: int, frames: int) -> np.ndarray:
    return np.load(MA3D / f"verts_joints_body_id-{body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)[:frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    # D1 (fix) added these three. The AFTER arm is the same instrument pointed at a
    # rebuild under `artifacts/compare/d1-fix/`; the delivered directory is never touched,
    # and `artifacts/compare/facing-location.json` stays the BEFORE arm, scored on the old
    # bytes by the old code. Running this build's FK against the old rotations would be
    # meaningless -- `across` comes from FK positions, so old rotations on a negated rest
    # flip the handedness sign for a reason that has nothing to do with the delivery.
    parser.add_argument("--delivery", type=Path, default=DELIVERY)
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--probe", type=Path, default=PROBE)
    parser.add_argument("--label", default="D1 (locate)")
    args = parser.parse_args()
    delivery, asset_path = args.delivery, args.asset

    names = json.loads((delivery / "subject-00.body-track.json").read_text())["joint_names"]
    skeleton = skeleton_for_joint_names(names)
    parents = [j.parent for j in DETAILED_HUMANOID.joints]
    rig_camera = json.loads((delivery / "camera-rig.json").read_text())["cameras"]

    tracks = [np.load(delivery / f"subject-{s:02d}.body-track.npz") for s in (0, 1)]
    ours = np.stack([t["triangulated_world_positions_z_up_m"] for t in tracks])
    mapping = mamma_index_for(ours)
    probe = json.loads(args.probe.read_text()) if args.probe.exists() else None

    # THE FOOTGUN THIS GUARD EXISTS FOR. Every by-name figure below comes from FORWARD
    # KINEMATICS: `triple()`'s `across` is a difference of FK positions. Run this script's
    # repaired skeleton against the PRE-repair rotations -- which is what happens if it is
    # pointed at the delivery from the D1 (fix) branch, or run after the merge and before
    # the delivery is rebuilt -- and the handedness sign inverts for a reason that has
    # nothing to do with the delivery, silently, into `artifacts/compare/facing-location.json`,
    # which is the BEFORE arm the fix gate scores and which `tests/test_facing_location.py`
    # reads. So refuse rather than score. The same hazard applies to every by-name tool in
    # this lane until the delivery is rebuilt; this one guards the file it would corrupt.
    convention = rest_convention(asset_path)
    asset_sign = convention["asset_mpfb_neutral_body"]["handedness_triple_product_sign"]
    code_sign = convention["code_DETAILED_HUMANOID"]["handedness_triple_product_sign"]
    if asset_sign != code_sign:
        raise SystemExit(
            f"REFUSING TO SCORE: the code skeleton (handedness {code_sign:+.0f}) and the "
            f"asset at {label_path(asset_path)} (handedness {asset_sign:+.0f}) disagree "
            "about which side is left. Forward kinematics would put the rig's bones on the "
            "wrong side of the body and every by-name figure here would be wrong in a way "
            "no band could detect. Rebuild the delivery against an asset that matches the "
            "code, or point --asset at the one it was built from. "
            "docs/reviews/facing-fix-2026-09-02.md"
        )

    report: dict = {
        "step": args.label,
        "title": "where the facing defect lives -- the rig's handedness, the delivered "
                 "surface, and what I6's control actually moved",
        "fixture": "pushing_and_lifting_from_ground",
        "frames": int(len(ours[0])),
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}" for k, v in mapping.items()},
        "sha_chain": sha_chain(delivery, asset_path),
        "rest_convention": convention,
        "basis_assertion": {},
        "triple_product": {"definition": triple.__doc__, "arms": {}},
        "forward_dot": {},
        "per_camera": {},
        "i6_reconciliation": {},
        "subjects": {},
    }

    for s in (0, 1):
        track = tracks[s]
        capture = track["triangulated_world_positions_z_up_m"]
        frames = len(capture)
        world = world_quaternions(track["local_rotations_xyzw"].astype(np.float64), parents)
        fk_rig = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=skeleton)
        fk = rig_to_capture(fk_rig)
        mamma = mamma_arrays(mapping[s], frames)

        # ---- basis assertion: the same guard I4 uses, so no conversion is taken on trust
        pelvis = 0.5 * (capture[:, JOINT_INDEX["left_hip"]] + capture[:, JOINT_INDEX["right_hip"]])
        candidates = {
            "(x,-z,y) CHOSEN": rig_to_capture(fk_rig)[:, names.index("Hips")],
            "(x,y,z) identity": fk_rig[:, names.index("Hips")],
            "(x,z,-y)": np.stack([fk_rig[..., 0], fk_rig[..., 2], -fk_rig[..., 1]], -1)[:, names.index("Hips")],
        }
        root = capture[:, JOINT_INDEX["root"]]
        report["basis_assertion"][f"subject_{s:02d}"] = {
            key: float(np.nanmedian(np.linalg.norm(value - root, axis=1)) * 1000.0)
            for key, value in candidates.items()
        } | {
            "reference": "the delivered rig's FK Hips against the triangulated `root` joint",
            "residual_reading": (
                "the chosen basis is not 0 mm but ~56-69 mm, and that residual is the known "
                "root/hip placement convention (`positions_to_body_track` subtracts a 0.98 "
                "literal while the exporter's Hips rest sits 106 mm lower, and the root is "
                "ground-projected) -- D2's territory, not a basis error. The margin over "
                "every other basis is ~100x, which is all this guard is for."),
            "against_the_hip_midpoint_instead_mm": float(
                np.nanmedian(np.linalg.norm(candidates["(x,-z,y) CHOSEN"] - pelvis, axis=1)) * 1000.0),
        }

        # ---- the frames every arm shares
        across_c, up_c = capture_frame(capture)
        fwd_c = capture_forward(capture)
        fwd_m = unit(np.cross(mamma[:, MA["left_hip"]] - mamma[:, MA["right_hip"]],
                              mamma[:, MA["neck"]] - mamma[:, MA["pelvis"]]))
        # An ankle->ball direction is anterior; I4 measured that rather than assuming it
        # (its G1 gate, mean direction offset 2-9 degrees against our own shin frame).
        foot_m = unit(mamma[:, MA["left_foot"]] - mamma[:, MA["left_ankle"]])
        foot_d = unit(fk[:, names.index("LeftToes")] - fk[:, names.index("LeftFoot")])
        torso_z = rig_to_capture(rotate(world[:, names.index("Chest")], (0.0, 0.0, 1.0)))

        # ---- 2. the handedness triple product ------------------------------------------
        arms = {
            "our_triangulated_capture": (
                across_c, up_c, foot_d,
                "our 19-joint capture for across/up; forward from the delivered foot "
                "joints, which are solved from the triangulated ToeBase (I4: the two "
                "agree to 0.3-1.1 deg), because the 19-joint contract stops at the ankle"),
            "mamma_pred_joints": (
                mamma[:, MA["right_shoulder"]] - mamma[:, MA["left_shoulder"]],
                mamma[:, MA["neck"]] - mamma[:, MA["pelvis"]], foot_m,
                "entirely MAMMA's own joints, paired through subject_map"),
            "delivered_rig_forward_from_its_FEET": (
                fk[:, names.index("RightUpperArm")] - fk[:, names.index("LeftUpperArm")],
                fk[:, names.index("Neck")] - fk[:, names.index("Hips")], foot_d,
                "the delivered rig's own named joints after FK, forward taken from the "
                "independently-solved foot"),
            "delivered_rig_forward_from_its_TORSO": (
                fk[:, names.index("RightUpperArm")] - fk[:, names.index("LeftUpperArm")],
                fk[:, names.index("Neck")] - fk[:, names.index("Hips")], torso_z,
                "THE SAME delivered rig, forward taken instead from the direction the "
                "torso bones actually point (rig +Z, the asset's face axis)"),
        }
        if probe:
            centroids = probe["subjects"][f"subject_{s:02d}"]["centroids_world_z_up_m"]
            sampled = probe["subjects"][f"subject_{s:02d}"]["frames"]
            nose = unit(np.array(centroids["head_plus_z"]) - np.array(centroids["head_minus_z"]))
            plus_x = unit(np.array(centroids["upper_plus_x"]) - np.array(centroids["upper_minus_x"]))
            arms["delivered_MESH_surface"] = (
                -plus_x, up_c[sampled], nose,
                "the shipped GLB's own skin: vertex sets chosen from the bind pose by "
                "GEOMETRY (the most anterior head vertices are the nose), carried to "
                "world through the real skinning. across = -(mesh +X side), because the "
                "mesh's anatomical left is its +X side when it faces +Z")

        for name, (across, up, forward, note) in arms.items():
            values = triple(across, up, forward)
            ok = np.isfinite(values)
            entry = report["triple_product"]["arms"].setdefault(name, {"note": note})
            entry[f"subject_{s:02d}"] = {
                "sign_median": float(np.sign(np.median(values[ok]))),
                "fraction_of_frames_positive": float(np.mean(values[ok] > 0)),
                "frames": int(ok.sum()),
            }

        # controls, applied to the capture arm as exact transforms of its three axes
        controls = {
            "CONTROL_yaw180_about_the_subjects_own_up": ("rotation", np.pi),
            "CONTROL_yaw90_about_the_subjects_own_up": ("rotation", np.pi / 2),
            "CONTROL_sagittal_mirror": ("reflection", None),
        }
        for name, (kind, angle) in controls.items():
            up_hat = unit(up_c)
            a_par = up_hat * np.einsum("fj,fj->f", across_c, up_hat)[:, None]
            f_par = up_hat * np.einsum("fj,fj->f", foot_d, up_hat)[:, None]
            a_perp, f_perp = across_c - a_par, foot_d - f_par
            if kind == "rotation":
                # Rodrigues about the subject's own vertical: a PROPER rotation, so the
                # triple product must survive it and only the forward-dot may move.
                def turn(v):
                    return (v * np.cos(angle) + np.cross(up_hat, v) * np.sin(angle))
                across_x, forward_x = a_par + turn(a_perp), f_par + turn(f_perp)
            else:
                # Reflect in the subject's own sagittal plane: left and right exchange,
                # facing is untouched. det = -1, so the triple product must invert.
                across_x, forward_x = a_par - a_perp, foot_d
            values = triple(across_x, up_c, forward_x)
            ok = np.isfinite(values)
            entry = report["triple_product"]["arms"].setdefault(
                name, {"note": "our capture, transformed exactly; see `controls_reading`"})
            entry[f"subject_{s:02d}"] = {
                "sign_median": float(np.sign(np.median(values[ok]))),
                "fraction_of_frames_positive": float(np.mean(values[ok] > 0)),
                "forward_dot_vs_capture_median": float(
                    np.median(np.einsum("fj,fj->f", unit(forward_x), fwd_c)[ok])),
                "frames": int(ok.sum()),
            }

        # ---- 2b. the same controls, applied to the DELIVERED RIG ------------------------
        # The capture-arm controls above transform our capture and score its FOOT
        # direction against its own pelvis/neck forward, whose untransformed ceiling is
        # +0.86 -- so a "median > +0.9" band cannot be applied to them and the mirror
        # control could never be shown to PASS the forward-dot. These score the rig the
        # gate actually scores: the delivered torso's own face axis and its own by-name
        # across, so the untransformed arm sits where the delivered torso sits and every
        # control is read on the gate's own scale. Same three transforms, same algebra.
        rig_across = fk[:, names.index("RightUpperArm")] - fk[:, names.index("LeftUpperArm")]
        rig_up = fk[:, names.index("Neck")] - fk[:, names.index("Hips")]
        delivered_controls = {
            "DELIVERED_untransformed": ("identity", None),
            "DELIVERED_CONTROL_yaw180": ("rotation", np.pi),
            "DELIVERED_CONTROL_yaw90": ("rotation", np.pi / 2),
            "DELIVERED_CONTROL_sagittal_mirror": ("reflection", None),
        }
        for name, (kind, angle) in delivered_controls.items():
            up_hat = unit(rig_up)
            a_par = up_hat * np.einsum("fj,fj->f", rig_across, up_hat)[:, None]
            f_par = up_hat * np.einsum("fj,fj->f", torso_z, up_hat)[:, None]
            a_perp, f_perp = rig_across - a_par, torso_z - f_par
            if kind == "identity":
                across_x, forward_x = rig_across, torso_z
            elif kind == "rotation":
                def turn(v, angle=angle):
                    return v * np.cos(angle) + np.cross(up_hat, v) * np.sin(angle)
                across_x, forward_x = a_par + turn(a_perp), f_par + turn(f_perp)
            else:
                across_x, forward_x = a_par - a_perp, torso_z
            values = triple(across_x, rig_up, forward_x)
            ok = np.isfinite(values)
            dots = np.einsum("fj,fj->f", unit(forward_x), fwd_c)
            entry = report.setdefault("delivered_rig_controls", {}).setdefault(
                name, {"note": ("the DELIVERED rig's own torso face axis and by-name "
                                "across, transformed exactly; the identity row is the "
                                "arm the gate scores")})
            entry[f"subject_{s:02d}"] = {
                "handedness_sign_median": float(np.sign(np.median(values[ok]))),
                "fraction_of_frames_positive": float(np.mean(values[ok] > 0)),
                "forward_dot_vs_our_capture": block_bootstrap_median(dots),
                "forward_dot_vs_mamma": block_bootstrap_median(
                    np.einsum("fj,fj->f", unit(forward_x), fwd_m)),
                "frames": int(ok.sum()),
            }

        # ---- 3. the forward-dot ---------------------------------------------------------
        groups = {
            "delivered_torso_Hips": ["Hips"],
            "delivered_torso_Chest": ["Spine", "Chest", "UpperChest"],
            "delivered_Neck": ["Neck"],
            "delivered_Head": ["Head"],
            "delivered_feet": ["LeftFoot", "RightFoot"],
        }
        dots: dict = {}
        for label, joints in groups.items():
            forward = unit(np.mean([rig_to_capture(rotate(world[:, names.index(j)], (0.0, 0.0, 1.0)))
                                    for j in joints], axis=0))
            dots[label] = {
                "vs_our_capture_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", forward, fwd_c)),
                "vs_mamma_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", forward, fwd_m)),
            }
        if probe:
            sampled = probe["subjects"][f"subject_{s:02d}"]["frames"]
            centroids = probe["subjects"][f"subject_{s:02d}"]["centroids_world_z_up_m"]
            nose = unit(np.array(centroids["head_plus_z"]) - np.array(centroids["head_minus_z"]))
            plus_x = unit(np.array(centroids["upper_plus_x"]) - np.array(centroids["upper_minus_x"]))
            left_c = unit(capture[:, JOINT_INDEX["left_hip"]] - capture[:, JOINT_INDEX["right_hip"]])
            dots["delivered_MESH_nose"] = {
                "vs_our_capture_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", nose, fwd_c[sampled])),
                "vs_mamma_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", nose, fwd_m[sampled])),
                "note": "no joint name enters this figure",
            }
            dots["delivered_MESH_plus_x_side_vs_performer_LEFT"] = {
                "vs_our_capture_left": block_bootstrap_median(
                    np.einsum("fj,fj->f", plus_x, left_c[sampled])),
                "note": ("the mesh's +X side is its anatomical LEFT (it faces +Z). "
                         "Negative means the mesh's left arm is on the performer's right. "
                         "Both this AND the nose being negative is a YAW; only one of "
                         "them negative would be a reflection."),
            }
        # ORACLE for the HEAD band specifically, added at D1 (fix). The reference forward
        # every figure above is scored against is the BODY's, built from the pelvis and the
        # neck. A head is not welded to a body: a performer can look sideways with their
        # chest square, so the head's dot against the body forward is bounded by the
        # performer's own neck, not by our solve. Scoring MAMMA's own head through the same
        # frames measures that ceiling -- and "a gate no oracle can pass is miscalibrated"
        # is this lane's standing rule (CLAUDE.md). MAMMA's head world rotation comes from
        # its own `smplx_pose` chain, exactly as `tools/head/head_gate.py` builds it, and
        # SMPL-X's canonical body faces +Z, so its head forward is that chain applied to
        # (0, 0, 1). MAMMA's world IS the camera-rig world.
        try:
            from scipy.spatial.transform import Rotation as _R

            pose = np.load(MA3D / f"smplx_params_body_id-{mapping[s]:02d}.npz",
                           allow_pickle=True)["smplx_pose"].astype(np.float64)[:frames]
            chain = (0, 3, 6, 9, 12, 15)      # pelvis, spine1-3, neck, head
            head_world = _R.from_rotvec(pose[:, 3 * chain[0]:3 * chain[0] + 3]).as_matrix()
            for joint in chain[1:]:
                head_world = head_world @ _R.from_rotvec(
                    pose[:, 3 * joint:3 * joint + 3]).as_matrix()
            mamma_head_forward = unit(head_world @ np.array([0.0, 0.0, 1.0]))
            dots["ORACLE_mamma_head_forward_vs_our_capture_forward"] = {
                "vs_our_capture_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", mamma_head_forward, fwd_c)),
                "vs_mamma_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", mamma_head_forward, fwd_m)),
                "note": ("THE CEILING FOR THE HEAD BAND. MAMMA's own head direction, from "
                         "its `smplx_pose` chain, scored against the same body-derived "
                         "forward every other row uses. Our Head cannot beat this, and a "
                         "band above it is a band on the performer's neck rather than on "
                         "any pipeline."),
            }
        except (OSError, KeyError, ValueError) as error:      # noqa: BLE001
            dots["ORACLE_mamma_head_forward_vs_our_capture_forward"] = {
                "unavailable": f"{type(error).__name__}: {error}"}

        if probe:
            # The mesh nose chord is NOT the head's forward axis. It runs from the centroid
            # of the 40 most posterior head vertices to the 40 most anterior, and in the
            # bind pose that chord is tilted below rig +Z by the asset's own geometry. This
            # row is that tilt, measured with no reference in it at all -- it is the fixed
            # offset between the `delivered_Head` figure and the `delivered_MESH_nose` one,
            # and without it the two look like a disagreement instead of a chord angle.
            sampled = probe["subjects"][f"subject_{s:02d}"]["frames"]
            centroids = probe["subjects"][f"subject_{s:02d}"]["centroids_world_z_up_m"]
            nose_axis = unit(np.array(centroids["head_plus_z"])
                             - np.array(centroids["head_minus_z"]))
            head_z = unit(rig_to_capture(
                rotate(world[:, names.index("Head")], (0.0, 0.0, 1.0))))[sampled]
            rest_sets = probe["rest_sets_rig_space"]
            chord_rest = (np.asarray(rest_sets["head_plus_z"]["centroid_xyz_m"])
                          - np.asarray(rest_sets["head_minus_z"]["centroid_xyz_m"]))
            dots["delivered_MESH_nose_vs_its_own_Head_joint_forward"] = {
                "vs_our_capture_forward": block_bootstrap_median(
                    np.einsum("fj,fj->f", nose_axis, head_z)),
                "rest_chord_tilt_below_rig_plus_z_deg": float(np.degrees(np.arccos(
                    float(chord_rest[2] / np.linalg.norm(chord_rest))))),
                "note": ("no reference and no capture enters this row: it is the mesh's own "
                         "nose chord against its own Head joint's +Z axis. The repair is an "
                         "exact 180 degree yaw about each joint's own UP axis, which "
                         "reverses the +Z component of this chord and PRESERVES its up "
                         "component -- so the nose figure cannot flip to the same magnitude "
                         "it had, and the difference is arithmetic, not a residual error."),
            }

        # ORACLE: MAMMA's own answer to the same question, through our frames. It measures
        # the floor our frame definitions impose -- how well two independent estimates of
        # "which way is this person facing" can ever agree on this take.
        dots["ORACLE_mamma_forward_vs_our_capture_forward"] = {
            "vs_our_capture_forward": block_bootstrap_median(
                np.einsum("fj,fj->f", fwd_m, fwd_c)),
            "note": ("the ceiling. A delivered part cannot beat this, and a delivered "
                     "part near -1 while this is near +1 is unambiguous."),
        }
        # The head does NOT inherit the torso frame -- since 981e437 it is set as an
        # absolute world rotation from its own rigid fit -- and it is still backwards. So
        # ask the head the second question too: which way does its rig +X point? If BOTH
        # its +Z and its +X are reversed the head is yawed like the torso; if only one is,
        # it is reflected, and the two need different fixes.
        left_c = unit(capture[:, JOINT_INDEX["left_hip"]] - capture[:, JOINT_INDEX["right_hip"]])
        for label, joint in (("Head", "Head"), ("Chest", "Chest")):
            plus_x = rig_to_capture(rotate(world[:, names.index(joint)], (1.0, 0.0, 0.0)))
            dots[f"delivered_{label}_rig_plus_X_vs_performer_LEFT"] = {
                "vs_our_capture_left": block_bootstrap_median(
                    np.einsum("fj,fj->f", plus_x, left_c)),
                "note": ("rig +X is where the bones named RIGHT sit, so a correct rig puts "
                         "it on the performer's right and this reads NEGATIVE for the right "
                         "reason. Read it beside the +Z figure: both reversed relative to "
                         "the mesh's own geometry is a yaw, one reversed is a reflection."),
            }
        report["forward_dot"][f"subject_{s:02d}"] = dots

        # ---- 3b. per camera: does our character face the camera when the performer does?
        per_camera = {}
        for camera, entry in rig_camera.items():
            centre = np.asarray(entry["camera_center_world_m"], dtype=np.float64)
            to_camera = unit(centre[None, :] - pelvis)
            performer = np.einsum("fj,fj->f", fwd_c, to_camera)
            torso = np.einsum("fj,fj->f", unit(torso_z), to_camera)
            mamma_faces = np.einsum("fj,fj->f", fwd_m, to_camera)
            per_camera[camera] = {
                "frames_the_PERFORMER_faces_this_camera": float(np.mean(performer > 0)),
                "frames_MAMMA_says_the_performer_faces_it": float(np.mean(mamma_faces > 0)),
                "frames_the_DELIVERED_TORSO_faces_this_camera": float(np.mean(torso > 0)),
                "frames_the_two_disagree": float(np.mean((performer > 0) != (torso > 0))),
                "median_dot_performer": float(np.median(performer)),
                "median_dot_delivered_torso": float(np.median(torso)),
            }
        report["per_camera"][f"subject_{s:02d}"] = per_camera

        # ---- 5. what I6's control actually moved ---------------------------------------
        # A 180 degree yaw about the pelvis vertical is what I6 applied to the whole mesh.
        # It reverses the torso (which is wrong and needs reversing) AND swings every limb
        # to the wrong side of the body (which is right and does not). Measure both.
        up_hat = unit(up_c)
        displacement = {}
        for joint in ("Hips", "UpperChest", "Head", "LeftHand", "RightHand",
                      "LeftFoot", "RightFoot", "LeftLowerArm"):
            point = fk[:, names.index(joint)]
            radial = point - pelvis
            par = up_hat * np.einsum("fj,fj->f", radial, up_hat)[:, None]
            perp = radial - par
            displacement[joint] = float(np.median(np.linalg.norm(2.0 * perp, axis=1)) * 1000.0)
        report["i6_reconciliation"][f"subject_{s:02d}"] = {
            "joint_displacement_under_a_whole_body_yaw180_mm": displacement,
            "note": ("how far each joint MOVES when the whole character is turned 180 "
                     "degrees about its own pelvis vertical -- twice its horizontal "
                     "distance from that axis"),
        }

        report["subjects"][f"subject_{s:02d}"] = {
            "mamma_body_id": mapping[s], "frames": int(frames)}

    report["triple_product"]["controls_reading"] = (
        "The two rotation controls are PROPER rotations of our capture about each "
        "subject's own vertical, so they must leave the triple product exactly where it "
        "was and move only the forward-dot -- that is the demonstration that the triple "
        "product cannot see a yaw. The sagittal mirror exchanges left and right without "
        "touching facing, so it must invert the triple product and leave the forward-dot "
        "at +1 -- the demonstration that the forward-dot cannot see a mirror. THE D1 GATE "
        "CARD ASKS FOR SOMETHING THAT CANNOT HAPPEN: 'a mirrored self-consistent human "
        "must fail the forward-dot'. A sagittal mirror is invisible to a forward-dot by "
        "construction, and invisible to a silhouette because a body's outline is nearly "
        "bilaterally symmetric. Only the triple product rejects it, which is why both "
        "instruments are in this report and why the gate needs both bands."
    )
    report["head_convention"] = head_convention()
    report["verdict"] = VERDICT
    report["proposed_fix_and_gate"] = FIX
    report["blind_to"] = BLIND
    report["plan_corrections"] = CORRECTIONS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


VERDICT = {
    "which_reading_survives": "iii -- a left/right mirror in the rig, delivered as a yaw",
    "statement": (
        "The 2026-08-30 diagnosis was RIGHT, and I6's contrary reading is an artefact of a "
        "control that is not the inverse of the defect. Both the shipped MPFB asset and "
        "DETAILED_HUMANOID name their Left bones on the mesh's anatomical RIGHT: the mesh's "
        "nose is at rig +Z and its Left-named bones are at rig -X, which is left-handed. "
        "`_frame` builds a right-handed basis, so `_frame_alignment` can satisfy either "
        "'bones named Left land on the performer's left' or 'the face points forward', not "
        "both. It satisfies the first, and the torso frame comes out yawed 180 degrees. "
        "EVERYTHING WHOSE ORIENTATION IS DEFINED AGAINST THE RIG'S REST CONVENTION INHERITS "
        "THE MIRROR -- the torso through `_frame_alignment`, and the head independently "
        "through `CANONICAL_HEAD_AXES`, which is a second hardcoded statement of the same "
        "wrong convention (see head_convention). Only the feet escape, because they aim rig "
        "+Z straight at a captured world direction and never consult the rest convention at "
        "all. Measured on the shipped GLB: the Hips forward-dot "
        "is -1.000, the chest -0.99, the head -0.89/-0.94, and the mesh's own nose -0.94/-0.96 "
        "-- while the FEET, solved from the triangulated ToeBase since f6a4973, read +0.86 to "
        "+0.90. The delivered character is a correctly-proportioned, self-consistent human "
        "whose body is turned round and whose feet are not."
    ),
    "why_the_other_two_readings_fail": {
        "i_internal_yaw_compensated_at_export": (
            "REJECTED by direct measurement of the shipped surface. The exporter composes "
            "the track's world rotation onto the asset's own rest and the inverse bind "
            "matrices cancel that rest, so the net rotation applied to bind geometry is the "
            "track's world rotation -- no compensation exists, and the probe confirms it on "
            "the real GLB: the mesh's nose points at the performer's back."),
        "ii_the_broken_overlay_timebase_manufactured_it": (
            "REJECTED. The timebase bug was real and is fixed in the previous commit, but it "
            "can only show the WRONG FRAME of a correct character; it cannot turn a character "
            "round. Every figure in this report is computed from the retained arrays and the "
            "shipped GLB with the fps set correctly, and the defect is still there. What the "
            "bug does explain is why the original write-up's numbers (-0.42 / -0.90) match "
            "nothing measurable today."),
    },
    "what_the_2026_08_30_write_up_got_wrong": (
        "Only the SCOPE, and only because the pipeline moved underneath it. It said 'a pure "
        "180 degree yaw' of the whole character, which was true on the 08-30 build. Two later "
        "commits took the feet (f6a4973) off the torso frame, so today the character is yawed "
        "EXCEPT at the feet. That is what made I6's whole-body control read backwards, and it "
        "is the one correction the original entry needs."
    ),
}

FIX = {
    "why_there_is_no_code_only_fix": (
        "A left-handed source basis cannot be expressed as a proper rotation: "
        "`target @ source.T` would have determinant -1 and `Rotation.from_matrix` will not "
        "return it. Flipping `_frame_alignment`'s secondary axis alone was tried on "
        "2026-08-31 and it fixed the facing while destroying the pose, because the limb "
        "chains still aim at world directions while the torso they hang from turns."),
    "proposed": [
        "Negate X throughout the canonical rest skeleton so that bones named Left sit at "
        "+X (CANONICAL_HUMANOID, which DETAILED_HUMANOID extends), and mirror the bound "
        "mesh with it: swap each vertex's Left/Right skin weights and negate vertex X in "
        "the MPFB asset. The skeleton and the mesh must move TOGETHER or the binding "
        "breaks; this is an asset change with a code change beside it, not a code change.",
        "Then flip the secondary axis in both `_frame_alignment` calls from (-1,0,0) to "
        "(1,0,0), so rig +X maps to the performer's left as the new naming requires.",
        "THIRD, AND IT IS NOT REACHED BY EITHER OF THE ABOVE: re-state "
        "`head_orientation.CANONICAL_HEAD_AXES`, which independently hardcodes left = rig "
        "-X and forward = rig -Z. Its columns must become left (1,0,0), up (0,0,1), forward "
        "(0,-1,0) in the capture convention. The head TEMPLATE needs no change -- it is "
        "learned from the subject's own landmarks and gauged against its own anatomy -- "
        "only the constant frame it is mapped onto is mirrored. Miss this and the body "
        "turns round while the head stays backwards.",
    ],
    "gate": {
        "primary_forward_dot": (
            "median > +0.9 AND p05 > 0 on every (subject, bone group) for Hips, chest, "
            "Neck, Head and the delivered MESH nose, against BOTH our capture forward and "
            "MAMMA's, with a moving-block bootstrap CI whose lower bound stays above 0. "
            "The feet must NOT move: their dots stay within their current interval. "
            "THE HEAD BAND IS CONTINGENT ON THE THIRD FIX STEP: the head does not inherit "
            "the torso frame, so mirroring the skeleton and the mesh alone leaves it "
            "backwards and this band fails for a reason that is not the torso. If that "
            "happens the answer is CANONICAL_HEAD_AXES, never a relaxed band."),
        "primary_handedness": (
            "the triple product sign of the delivered rig must agree with our capture AND "
            "with MAMMA on every frame, read BOTH ways -- forward from the feet and "
            "forward from the torso. Those two arms disagreeing is the present defect "
            "stated as one bit, and agreement is the fix stated as one bit."),
        "surface": (
            "I6 silhouette IoU must not fall on any of the 8 (camera, subject) cells, on "
            "the front/back-distinguishable half it already tabled."),
        "degenerate_controls_that_must_FAIL": [
            "the CURRENT asset, as the mirrored-rest control: it must fail the forward-dot "
            "band -- if the repaired gate passes today's build, the gate is broken",
            "a sagittal-mirrored human: must fail the handedness band while passing the "
            "forward-dot and the silhouette, which is the whole reason the handedness band "
            "exists",
            "a 180-degree yawed human: must fail the forward-dot while passing handedness",
            "a 90-degree yawed human: must fail the forward-dot",
            "a constant world forward: must fail, PROVIDED the take carries enough turning "
            "-- check the capture's own yaw range on the fixture before quoting this one",
        ],
        "blast_radius_before_landing": (
            "DETAILED_HUMANOID is shared by the capture retarget, speech_motion, "
            "soma_motion, body_export, unified_gltf's skin matrices and body_binding's "
            "socket geometry. Any lane that silently compensated for the mirror breaks when "
            "it is removed. Every lane needs a camera overlay before and after -- and the "
            "overlay must be the fps-repaired one."),
        "sequencing": (
            "Cannot ship while the THORAX_SMOOTHING_FRAMES leak stands: "
            "tools/compare/provenance.py exits 1 until I8 re-selects it."),
    },
}

BLIND = (
    "This instrument is blind to MAGNITUDE: a forward-dot says which way a part points, "
    "not how far off it is in degrees, and a dot near -1 is consistent with anything from "
    "160 to 200 degrees. It is blind to everything that is not an ORIENTATION -- the "
    "clavicle-origin error, the bone lengths and the root/hip placement offset are all "
    "untouched here and none of them are fixed by fixing this. The triple product is a "
    "SIGN: it detects a mirror and says nothing about how good a non-mirrored answer is, "
    "and on a nearly-straight limb or a near-degenerate frame the three axes approach "
    "coplanarity and the sign is noise -- which is why the fraction of frames is reported "
    "beside every median. The reference forward is built from the pelvis and neck of "
    "estimates, ours and MAMMA's; both are estimates and neither is truth, so this "
    "measures AGREEMENT on facing, and it is only decisive because the disagreement is a "
    "whole reversal rather than a few degrees. One take, two performers, 150 correlated "
    "frames: nothing here generalises past this fixture until lane H delivers a second."
)

CORRECTIONS = [
    "PARITY BOARD, wrong: the delivered body-track.npz files do NOT predate the head fix. "
    "They were written 2026-09-01 15:15:33, eighteen hours after 981e437 and thirty "
    "seconds before f6a4973, and run-report.json records both the head and toe solves as "
    "'solved'. No commit has touched src/ since. I6 and I4 scored these exact bytes.",
    "PARITY BOARD / I6, wrong conclusion from a correct measurement: "
    "`control_ours_yaw180_facing` turns the WHOLE delivered mesh 180 degrees, which is not "
    "the inverse of the defect. The defect yaws only what the torso frame rigidly carries; "
    "the limbs are re-aimed at captured world directions afterwards and the feet are solved "
    "separately, so they are already right. The control therefore fixes the torso and "
    "breaks everything else, and IoU falls. See i6_reconciliation for the displacement it "
    "imposes on each joint.",
    "I6's own facing-sensitivity demonstration is confounded by the same thing: "
    "`control_oracle_yaw180_facing` also moves MAMMA's limbs, so the 0.35-0.62 IoU it "
    "loses measures sensitivity to a WHOLE-BODY YAW, not to facing with limbs held. The "
    "corrected control is to yaw only the vertices whose dominant weight is a torso bone, "
    "about the torso vertical, and rescore. Until that is run, I6 has not demonstrated "
    "that a silhouette can see facing on this take.",
    "LADDER_EXECUTION_PLAN, D1 gate card: 'a mirrored self-consistent human must fail the "
    "forward-dot even where IoU passes' is AMBIGUOUS, and one of its two readings cannot "
    "be met. A sagittal-mirrored POSE leaves facing exactly where it was, so no "
    "forward-dot can reject it -- measured, not argued (forward-dot +0.81, handedness "
    "sign +1); the handedness band is what rejects that one. A human driven through a "
    "MIRRORED-NAMING RIG is the other reading, and that is today's build, which fails the "
    "forward-dot at -1.000. Both are in the proposed gate, as separate controls. The card "
    "should say which it means.",
    "BODY_LANE_PLAN: the -0.42 / -0.90 facing dots have no instrument behind them in any "
    "commit. Strike them and quote this report's forward_dot instead.",
    "BODY_LANE_PLAN, facing entry: 'a pure 180 degree yaw' is now only true of the torso. "
    "Since f6a4973 the feet are solved from the triangulated ToeBase and read +0.91 to "
    "+0.97 against the performer's forward while the torso reads -1.00.",
    "THE MIRROR HAS A THIRD SITE, and it is not in any skeleton: "
    "head_orientation.CANONICAL_HEAD_AXES independently hardcodes left = rig -X and "
    "forward = rig -Z. The head is set as an ABSOLUTE world rotation, not inherited from "
    "the torso, which is why it is backwards by its own route. Any fix that stops at the "
    "skeleton and the mesh leaves the head reversed. See head_convention.",
    "tools/swap-harness/camera_overlay.py never set scene.render.fps: fixed in the commit "
    "before this one, with a POSE CHECK so it cannot recur silently. Every overlay rendered "
    "before 2026-09-02, including those behind the original facing diagnosis, showed take "
    "frame 60 + 1.25*F and froze past F=119.",
]

if __name__ == "__main__":
    main()
