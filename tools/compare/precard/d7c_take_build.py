"""Pre-card D7c, the real take: rebuild the shipped delivery through the REAL build script with
the candidate pelvis fit substituted (the rig's own rest offsets about the leg-root midpoint,
no constant), and dump every converter input on the way so the take-level measurements can be
made offline against the delivered bytes.

  --mode shipped    today's `_pelvis_world_frames` (a hygiene arm; must reproduce the shipped
                    delivery byte for byte -- D9b's hygiene clause, rerun here)
  --mode candidate  the C branch with the template read from the caller's skeleton rest and
                    the observed set taken about the captured HIP MIDPOINT, not the `root`
                    landmark (SOMA-77's `Hips`, a pelvis-interior joint whose place relative to
                    the hip joints is SOMA's convention, drops out of the fit entirely)

Nothing under `src/` changes; the substitution is by module attribute, which is the reason
`_pelvis_world_frames` is called by bare name.
"""
import argparse, runpy, shutil, sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim")
sys.path.insert(0, str(ROOT / "src"))
import autoanim_gnm.commercial_multiview as cm      # noqa: E402

SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
BUILD = ROOT / "scripts/build_commercial_multiview_comparison.py"
CURRENT: dict = {"rest": None, "calls": 0}


def candidate_pelvis_world_frames(points_rig_y_up_m, spine1_rig_y_up_m, *, mode=None, smoothing_frames=None):
    """`_pelvis_world_frames`' C branch on the rig's own rest, about the hip midpoint."""
    rest = CURRENT["rest"]
    if rest is None:
        raise RuntimeError("no skeleton rest recorded for this converter call")
    points = np.asarray(points_rig_y_up_m, dtype=np.float64)
    spine = np.asarray(spine1_rig_y_up_m, dtype=np.float64)
    frames = len(points)
    valid = np.isfinite(spine).all(axis=1)
    resolved = float(valid.mean())
    if resolved < cm.PELVIS_MINIMUM_RESOLVED_FRACTION or int(valid.sum()) < 2:
        return None, {"status": "fell_back_to_torso_frame", "resolved_fraction": resolved,
                      "mode": "C_kabsch_pelvis_rig_rest"}
    filled = spine.copy()
    axis = np.arange(frames, dtype=np.float64)
    for component in range(3):
        filled[:, component] = np.interp(axis, axis[valid], spine[valid, component])
    left_hip = points[:, cm.JOINT_INDEX["left_hip"]]
    right_hip = points[:, cm.JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left_hip + right_hip)
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    template = np.stack((rest["Spine"] - mid, rest["LeftUpperLeg"] - mid, rest["RightUpperLeg"] - mid))
    quaternions = np.zeros((frames, 4), dtype=np.float64)
    residual = np.zeros(frames)
    for frame in range(frames):
        observed = np.stack((filled[frame] - hip_mid[frame], left_hip[frame] - hip_mid[frame],
                             right_hip[frame] - hip_mid[frame]))
        u, _, vt = np.linalg.svd(template.T @ observed)
        sign = float(np.sign(np.linalg.det(vt.T @ u.T)))
        rotation = vt.T @ np.diag((1.0, 1.0, sign)) @ u.T
        quaternions[frame] = Rotation.from_matrix(rotation).as_quat()
        residual[frame] = np.linalg.norm(observed - template @ rotation.T, axis=1).mean()
    for frame in range(1, frames):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions, {"status": "solved", "mode": "C_kabsch_pelvis_rig_rest",
                         "resolved_fraction": resolved, "smoothing_frames": 0,
                         "kabsch_residual_mm_median": round(1e3 * float(np.median(residual)), 4),
                         "template_m": template.tolist()}


def hipline_primary_frames(points_rig_y_up_m, spine1_rig_y_up_m, *, mode=None, smoothing_frames=None):
    """Candidate (b): hip line exact primary (rig rest L - R), pitch from Spine1 - hip midpoint
    against rest[Spine] - mid. No constant."""
    rest = CURRENT["rest"]
    points = np.asarray(points_rig_y_up_m, dtype=np.float64)
    spine = np.asarray(spine1_rig_y_up_m, dtype=np.float64)
    frames = len(points)
    valid = np.isfinite(spine).all(axis=1)
    filled = spine.copy(); axis = np.arange(frames, dtype=np.float64)
    for component in range(3):
        filled[:, component] = np.interp(axis, axis[valid], spine[valid, component])
    left = points[:, cm.JOINT_INDEX["left_hip"]]; right = points[:, cm.JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left + right)
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    src_across = rest["LeftUpperLeg"] - rest["RightUpperLeg"]; src_up = rest["Spine"] - mid
    q = np.zeros((frames, 4))
    for f in range(frames):
        q[f] = cm._frame_alignment(src_across, src_up, left[f] - right[f], filled[f] - hip_mid[f])
    for f in range(1, frames):
        if float(np.dot(q[f], q[f - 1])) < 0.0:
            q[f] *= -1.0
    return q, {"status": "solved", "mode": "hipline_primary_rig_rest", "resolved_fraction": float(valid.mean()), "smoothing_frames": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shipped", "candidate", "hipline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(SHIPPED / "work", output / "work")      # never symlinked
    dumps = output / "converter-inputs"; dumps.mkdir()

    real_converter = cm.positions_to_body_track
    real_pelvis = cm._pelvis_world_frames

    def converter(positions, **kwargs):
        skeleton = kwargs.get("skeleton")
        names = list(skeleton.names)
        rest_arr = np.asarray(skeleton.rest_translations_m, np.float64)
        CURRENT["rest"] = {n: rest_arr[k] for k, n in enumerate(names)}
        n = CURRENT["calls"]; CURRENT["calls"] += 1
        np.savez(dumps / f"call-{n:02d}.npz",
                 positions_world_z_up_m=np.asarray(positions, np.float64),
                 toe_world_z_up_m=np.asarray(kwargs.get("toe_world_z_up_m"), np.float64) if kwargs.get("toe_world_z_up_m") is not None else np.zeros(0),
                 spine_world_z_up_m=np.asarray(kwargs.get("spine_world_z_up_m"), np.float64) if kwargs.get("spine_world_z_up_m") is not None else np.zeros(0),
                 head_world_rotations=np.asarray(kwargs.get("head_world_rotations"), np.float64) if kwargs.get("head_world_rotations") is not None else np.zeros(0),
                 rest_translations_m=rest_arr, joint_names=np.array(names))
        (dumps / f"call-{n:02d}.json").write_text(json.dumps(
            {k: (v if isinstance(v, (int, float, str, bool)) else str(type(v))) for k, v in kwargs.items()}, indent=1))
        return real_converter(positions, **kwargs)

    cm.positions_to_body_track = converter
    if args.mode == "candidate":
        cm._pelvis_world_frames = candidate_pelvis_world_frames
    elif args.mode == "hipline":
        cm._pelvis_world_frames = hipline_primary_frames
    sys.argv = ["build_commercial_multiview_comparison.py",
                "--videos", str(ROOT / ".cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos"),
                "--calibration-yaml", str(ROOT / ".cache/mamma/configs/examples/calib/iphones_outdoors.yaml"),
                "--detector", "soma77", "--output", str(output)]
    try:
        runpy.run_path(str(BUILD), run_name="__main__")
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise
    finally:
        cm.positions_to_body_track = real_converter
        cm._pelvis_world_frames = real_pelvis
    print("converter calls:", CURRENT["calls"], "output:", output)


if __name__ == "__main__":
    main()
