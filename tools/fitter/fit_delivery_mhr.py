"""D4 pre-card: fit MHR (momentum, Apache/MIT) to the delivered take's OWN raw triangulated
landmarks and write one skinned glTF per performer, so the existing silhouette instrument can
score a fitted body against the SAM2 masks beside the D7c rig delivery.

Same denominator as every instrument: `raw_triangulated_world_positions_z_up_m` from the delivered
`subject-XX.body-track.npz`. Capture Z-up metres -> MHR Y-up centimetres by (x, z, -y) * 100 (the
inverse of `tools/fitter/rootcheck.py`'s `tocap`). Pinned locator offsets (limit_weight 10): the
honest setting from FITTER_PLAN section 7 -- offsets free let the locators absorb the performer and
leave the skeleton at the mean. MHR from its own release under .cache/mhr/assets, 68 raw scale
channels, never the SAM-licensed PCA. MAMMA enters nowhere.

Usage: /tmp/momenv/bin/python tools/fitter/fit_delivery_mhr.py OUT_DIR [--mean-body]
  --mean-body   the CONTROL: identity scales held at zero (MHR's mean body), pose tracked only.
"""
import sys, json, os
from pathlib import Path
import numpy as np
import pymomentum.geometry as g
import pymomentum.marker_tracking as mt

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / ".cache/mhr/assets"
DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
MAP = {'root': 'root', 'neck': 'c_neck', 'nose': 'c_head',
       'left_shoulder': 'l_uparm', 'right_shoulder': 'r_uparm', 'left_elbow': 'l_lowarm', 'right_elbow': 'r_lowarm',
       'left_wrist': 'l_wrist', 'right_wrist': 'r_wrist', 'left_hip': 'l_upleg', 'right_hip': 'r_upleg',
       'left_knee': 'l_lowleg', 'right_knee': 'r_lowleg', 'left_ankle': 'l_foot', 'right_ankle': 'r_foot',
       'left_eye': 'l_eye', 'right_eye': 'r_eye'}
NAMES = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
         "left_wrist", "root", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
         "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]


def to_mhr_cm(p_zup_m: np.ndarray) -> np.ndarray:
    return np.stack([p_zup_m[..., 0], p_zup_m[..., 2], -p_zup_m[..., 1]], axis=-1) * 100.0


def main() -> None:
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    mean_body = "--mean-body" in sys.argv
    report = {"arm": "mean_body_control" if mean_body else "fitted_pinned_offsets", "subjects": {}}
    for subj in (0, 1):
        z = np.load(DELIVERY / f"subject-{subj:02d}.body-track.npz")
        raw = np.asarray(z["raw_triangulated_world_positions_z_up_m"], np.float64)
        arr = to_mhr_cm(raw)
        c = g.Character.load_fbx(str(A / "lod6.fbx")).load_model_definition(str(A / "compact_v6_1.model"))
        jn = list(c.skeleton.joint_names)
        c = c.with_locators([g.Locator(name=k, parent=jn.index(v), offset=np.zeros(3, np.float32),
                                       limit_origin=np.zeros(3, np.float32), limit_weight=np.full(3, 10.0, np.float32))
                             for k, v in MAP.items()])
        md = [[g.Marker(name=lm, pos=(arr[f, NAMES.index(lm)] if np.isfinite(arr[f, NAMES.index(lm)]).all() else np.zeros(3)),
                        occluded=not np.isfinite(arr[f, NAMES.index(lm)]).all()) for lm in MAP] for f in range(arr.shape[0])]
        pt = c.parameter_transform; zero = np.zeros(len(pt.names), np.float32)
        cfgB = mt.CalibrationConfig(); cfgB.calib_frames = 100; cfgB.locators_only = False
        cfgB.global_scale_only = False; cfgB.loss_alpha = 2.0; cfgB.max_iter = 30
        if mean_body:
            ident = zero.copy()
        else:
            cfgA = mt.CalibrationConfig(); cfgA.calib_frames = 100; cfgA.locators_only = True; cfgA.loss_alpha = 2.0; cfgA.max_iter = 30
            mt.calibrate_markers(c, zero.copy(), md, cfgA)
            ident, _, _ = mt.calibrate_markers(c, zero.copy(), md, cfgB)
            ident = np.asarray(ident, np.float32)
        tcfg = mt.TrackingConfig(); tcfg.smoothing = 0.0; tcfg.max_iter = 30; tcfg.loss_alpha = 2.0
        motion = np.asarray(mt.process_markers(c, ident, md, tcfg, cfgB, calibrate=False), np.float32)
        # closure on the fed markers: locator-to-marker residual per frame (cm -> mm)
        names_pt = list(pt.names)
        res = []
        for f in range(motion.shape[0]):
            st = np.asarray(g.model_parameters_to_skeleton_state(c, motion[f]))[..., :3]
            d = [np.linalg.norm(st[jn.index(MAP[lm])] - arr[f, NAMES.index(lm)]) for lm in MAP
                 if np.isfinite(arr[f, NAMES.index(lm)]).all()]
            res.append(float(np.median(d)) * 10.0)
        np.savez(out / f"motion_{subj}.npz", motion=motion, identity=ident, names=np.array(names_pt))
        b = g.GltfBuilder(fps=30.0); b.add_motion(c, fps=30.0, motion=(names_pt, motion))
        path = out / f"subject-{subj:02d}.glb"; b.save(str(path))  # GltfBuilder(fps=) or the FBX's 120 fps wins and the take runs 4x fast
        scales = {n: float(v) for n, v in zip(names_pt, ident) if abs(float(v)) > 1e-6}
        report["subjects"][f"subject_{subj:02d}"] = {
            "frames": int(motion.shape[0]), "gltf": str(path),
            "joint_to_landmark_residual_mm_median_over_frames": round(float(np.median(res)), 2),
            "joint_to_landmark_residual_mm_p95": round(float(np.percentile(res, 95)), 2),
            "nonzero_scale_channels": len(scales), "scale_channels": scales}
        print(subj, report["subjects"][f"subject_{subj:02d}"]["joint_to_landmark_residual_mm_median_over_frames"], "mm median residual;", len(scales), "scales")
    (out / "fit-report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
