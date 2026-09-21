"""D4: the body model in the delivery path -- MHR fitted by momentum and delivered as the body.

Called by `scripts/build_commercial_multiview_comparison.py --body mhr` as a SUBPROCESS: the
build runs on `.venv` (no pymomentum) and this runs on `/tmp/momenv/bin/python` (pymomentum-cpu).
It reads ONLY the arrays the build hands it under `--inputs` (the converter-input dump, which is
B2's artefact), fits MHR's identity and per-frame pose, and writes the delivery:

    subject-XX.glb              skinned, 30 fps, MHR's own mesh, no marker spheres
    subject-XX.body-track.npz   schema autoanim.body-track/2.0-mhr
    subject-XX.body-track.json  the same, without the per-frame vertex-scale arrays
    subject-XX.markers.npz      exactly what momentum received (B2's second half)
    fit-report.json

MHR comes from its own release under `.cache/mhr/assets` (Apache), momentum is MIT. The 68 raw
`scale_*` channels only -- never SOMA's SAM-licensed 28-dim scale PCA. MAMMA enters nowhere.

Capture Z-up metres -> MHR Y-up centimetres is `(x, z, -y) * 100`; the inverse (the frame every
instrument downstream reads) is `(X, -Z, Y) / 100`. Verified by closure, never by reading: the
report carries the GLB-independent FK of the delivered motion against the fed landmarks.

Locator offsets are PINNED (limit_weight 10), FITTER_PLAN section 7's honest setting: free offsets
let the locators absorb the performer and leave the skeleton at the mean. `--free-offsets` is the
must-fail arm and exists only to be scored.

ONE SUBJECT PER PROCESS. A `calibrate_markers` call mutates whatever a LATER
`Character.load_fbx(...).load_model_definition(...)` returns in the same process -- the same 204
model parameters then put the skeleton somewhere else (measured 2026-09-22; `export_glb`'s
docstring carries the numbers). The pre-card fitted both performers in one process, so its
performer 1 was fitted on a model performer 0's calibration had moved.

Usage:
  /tmp/momenv/bin/python tools/fitter/mhr_delivery.py --inputs DIR --out DIR --subject N \
      [--lod 2] [--landmarks smoothed|raw] [--mean-body] [--free-offsets]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymomentum.geometry as g
import pymomentum.marker_tracking as mt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / ".cache/mhr/assets"

# The declared mapping: our 19-joint body contract -> MHR joints. 17 of the 19; SOMA-77 emits no
# ears, so `left_ear`/`right_ear` are schema-only and populated on zero frames (CLAUDE.md).
MAP = {'root': 'root', 'neck': 'c_neck', 'nose': 'c_head',
       'left_shoulder': 'l_uparm', 'right_shoulder': 'r_uparm', 'left_elbow': 'l_lowarm',
       'right_elbow': 'r_lowarm', 'left_wrist': 'l_wrist', 'right_wrist': 'r_wrist',
       'left_hip': 'l_upleg', 'right_hip': 'r_upleg', 'left_knee': 'l_lowleg',
       'right_knee': 'r_lowleg', 'left_ankle': 'l_foot', 'right_ankle': 'r_foot',
       'left_eye': 'l_eye', 'right_eye': 'r_eye'}

SCHEMA = "autoanim.body-track/2.0-mhr"


def to_mhr_cm(p_zup_m: np.ndarray) -> np.ndarray:
    """Capture Z-up metres -> MHR Y-up centimetres."""
    return np.stack([p_zup_m[..., 0], p_zup_m[..., 2], -p_zup_m[..., 1]], axis=-1) * 100.0


def to_capture_m(p_mhr_cm: np.ndarray) -> np.ndarray:
    """MHR Y-up centimetres -> capture Z-up metres (the inverse of `to_mhr_cm`)."""
    return np.stack([p_mhr_cm[..., 0], -p_mhr_cm[..., 2], p_mhr_cm[..., 1]], axis=-1) / 100.0


def load_character(lod: int) -> g.Character:
    return g.Character.load_fbx(str(ASSETS / f"lod{lod}.fbx")).load_model_definition(
        str(ASSETS / "compact_v6_1.model"))


def with_pinned_locators(character: g.Character, *, free: bool) -> g.Character:
    joint_names = list(character.skeleton.joint_names)
    weight = 0.0 if free else 10.0
    return character.with_locators([
        g.Locator(name=landmark, parent=joint_names.index(joint), offset=np.zeros(3, np.float32),
                  limit_origin=np.zeros(3, np.float32),
                  limit_weight=np.full(3, weight, np.float32))
        for landmark, joint in MAP.items()])


def marker_data(array_cm: np.ndarray, joint_names: list[str]) -> list[list[g.Marker]]:
    """The declared mapping applied: one momentum Marker per mapped landmark per frame.

    A landmark with any non-finite coordinate is passed as OCCLUDED at the origin, which is
    momentum's contract for a missing marker -- never as a zero the solver would chase.
    """
    index = {name: joint_names.index(name) for name in MAP}
    frames = array_cm.shape[0]
    out = []
    for frame in range(frames):
        row = []
        for landmark in MAP:
            position = array_cm[frame, index[landmark]]
            finite = bool(np.isfinite(position).all())
            row.append(g.Marker(name=landmark,
                                pos=(position if finite else np.zeros(3)),
                                occluded=not finite))
        out.append(row)
    return out


def marker_arrays(markers: list[list[g.Marker]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = [m.name for m in markers[0]]
    positions = np.asarray([[np.asarray(m.pos, np.float64) for m in frame] for frame in markers])
    occluded = np.asarray([[bool(m.occluded) for m in frame] for frame in markers])
    return positions, occluded, names


def fk_positions_cm(character: g.Character, motion: np.ndarray) -> np.ndarray:
    """Per-frame world joint positions (centimetres) from the model parameters alone."""
    out = np.empty((motion.shape[0], len(character.skeleton.joint_names), 3), np.float64)
    for frame in range(motion.shape[0]):
        out[frame] = np.asarray(
            g.model_parameters_to_skeleton_state(character, motion[frame]))[..., :3]
    return out


def fit_one(array_zup_m: np.ndarray, joint_names: list[str], *, lod: int, mean_body: bool,
            free_offsets: bool, calib_frames: int = 100, max_iter: int = 30,
            loss_alpha: float = 2.0, smoothing: float = 0.0,
            freeze_flexible: bool = False) -> dict:
    """Two-stage calibration then per-frame tracking. Returns everything the delivery needs."""
    array_cm = to_mhr_cm(np.asarray(array_zup_m, np.float64))
    character = with_pinned_locators(load_character(lod), free=free_offsets)
    markers = marker_data(array_cm, joint_names)
    transform = character.parameter_transform
    parameter_names = list(transform.names)
    zero = np.zeros(len(parameter_names), np.float32)

    calibration = mt.CalibrationConfig()
    calibration.calib_frames = calib_frames
    calibration.locators_only = False
    calibration.global_scale_only = False
    calibration.loss_alpha = loss_alpha
    calibration.max_iter = max_iter

    if mean_body:
        identity = zero.copy()
    else:
        stage_a = mt.CalibrationConfig()
        stage_a.calib_frames = calib_frames
        stage_a.locators_only = True
        stage_a.loss_alpha = loss_alpha
        stage_a.max_iter = max_iter
        mt.calibrate_markers(character, zero.copy(), markers, stage_a)
        identity, _, _ = mt.calibrate_markers(character, zero.copy(), markers, calibration)
        identity = np.asarray(identity, np.float32)

    tracking = mt.TrackingConfig()
    tracking.smoothing = smoothing
    tracking.max_iter = max_iter
    tracking.loss_alpha = loss_alpha
    if freeze_flexible:
        # O1 only: MHR duplicates `scale_spine_length` as `spine_length_flexible` among the pose,
        # so an oracle that left the 26 `*_flexible` channels free would be exact for the wrong
        # reason. They are frozen at zero in the truth AND excluded from the solve here.
        active = np.asarray(transform.pose_parameters).copy()
        flexible = np.asarray([name.endswith("_flexible") for name in parameter_names])
        active &= ~flexible
        tracking.active_params = active

    motion = np.asarray(mt.process_markers(character, identity, markers, tracking, calibration,
                                           calibrate=False), np.float32)
    positions_cm = fk_positions_cm(character, motion)
    return {"character": character, "identity": identity, "motion": motion, "lod": lod,
            "parameter_names": parameter_names, "markers": markers,
            "array_cm": array_cm, "positions_cm": positions_cm}


def locator_residual_mm(fit: dict, joint_names: list[str]) -> np.ndarray:
    """Per-frame median distance from each mapped MHR joint to the landmark it was fed."""
    skeleton = list(fit["character"].skeleton.joint_names)
    array_cm, positions = fit["array_cm"], fit["positions_cm"]
    index = {name: joint_names.index(name) for name in MAP}
    out = np.full(positions.shape[0], np.nan)
    for frame in range(positions.shape[0]):
        distances = [float(np.linalg.norm(positions[frame, skeleton.index(joint)]
                                          - array_cm[frame, index[landmark]]))
                     for landmark, joint in MAP.items()
                     if np.isfinite(array_cm[frame, index[landmark]]).all()]
        if distances:
            out[frame] = float(np.median(distances)) * 10.0
    return out


def export_glb(fit: dict, export_character: g.Character, path: Path, fps: float = 30.0) -> None:
    """The skinned GLB. The marker spheres are stripped by exporting a LOCATOR-FREE character.

    Two traps, both measured here (2026-09-22), both of which ship a wrong file silently:

    * `Character.with_locators([])` does NOT clear locators (17 in, 17 out), so the export
      character cannot be derived from the fitted one. It is loaded separately.
    * **a character loaded AFTER any `calibrate_markers` call is a DIFFERENT model.** The same
      204 model parameters put its skeleton somewhere else (skeleton-state sum -53291.87 against
      -57778.49; the exported mesh moved up to 0.76 m and the glTF animation dropped from 98
      channels to 79 while still importing cleanly at the right frame range). A character loaded
      BEFORE the first calibration is unaffected by it -- its export is byte-identical before and
      after -- so `export_character` is loaded once at the top of the run, and the guard below
      compares its skeleton state against the FITTED character's on every frame.

    `GltfBuilder(fps=...)` is not optional: without it the builder inherits the FBX's 120 fps and
    the take runs 4x fast (Blender then reads an action range of 0..37 instead of 0..149).
    """
    if list(export_character.parameter_transform.names) != fit["parameter_names"]:
        raise SystemExit("the export character's parameters differ from the fitted character's")
    motion = fit["motion"]
    worst = 0.0
    for frame in range(motion.shape[0]):
        a = np.asarray(g.model_parameters_to_skeleton_state(export_character, motion[frame]))
        b = np.asarray(g.model_parameters_to_skeleton_state(fit["character"], motion[frame]))
        worst = max(worst, float(np.abs(a - b).max()))
    if worst > 1e-4:
        raise SystemExit(f"the export character is not the fitted character's model: skeleton "
                         f"states differ by {worst} (see the docstring: load it before calibrating)")
    builder = g.GltfBuilder(fps=fps)
    builder.add_motion(export_character, fps=fps, motion=(fit["parameter_names"], motion))
    builder.save(str(path))
    print(f"    export guard: skeleton states agree to {worst:.3e}", flush=True)


def write_track(fit: dict, prefix: Path, *, subject: int, consumed: dict, lod: int,
                landmarks: str, settings: dict, joint_names: list[str]) -> dict:
    character = fit["character"]
    skeleton = list(character.skeleton.joint_names)
    names = fit["parameter_names"]
    identity_mask = [name.startswith("scale_") for name in names]
    pose_mask = np.asarray(character.parameter_transform.pose_parameters)
    identity_names = [n for n, keep in zip(names, identity_mask) if keep]
    pose_names = [n for n, keep in zip(names, pose_mask) if keep]
    motion = fit["motion"]
    positions_capture = to_capture_m(fit["positions_cm"])
    frames = int(motion.shape[0])
    rest = np.asarray(g.model_parameters_to_skeleton_state(
        character, np.zeros(len(names), np.float32)))[..., :3]

    track = {
        "schema_version": SCHEMA,
        "body_model": {"name": "MHR", "release": "meta open release", "lod": lod,
                       "assets": str(ASSETS), "mesh_vertices": int(character.mesh.vertices.shape[0]),
                       "joint_count": len(skeleton), "licence": "MHR Apache-2.0, momentum MIT"},
        "fps": 30.0,
        "frame_count": frames,
        "joint_names": skeleton,
        "identity_channel_names": identity_names,
        "identity_values": [float(v) for v, keep in zip(fit["identity"], identity_mask) if keep],
        "pose_channel_names": pose_names,
        "landmark_to_joint": MAP,
        "consumed_landmark_array": landmarks,
        "consumed_joint_names": joint_names,
        "settings": settings,
    }
    json_track = dict(track)
    json_track["pose_values"] = np.asarray(motion[:, pose_mask], np.float64).round(6).tolist()
    prefix.with_suffix(".body-track.json").write_text(json.dumps(json_track), encoding="utf-8")

    np.savez(
        prefix.with_suffix(".body-track.npz"),
        schema_version=np.array(SCHEMA),
        joint_names=np.array(skeleton),
        parameter_names=np.array(names),
        model_parameters=motion,
        identity_channel_names=np.array(identity_names),
        identity_values=np.asarray(fit["identity"])[identity_mask],
        pose_channel_names=np.array(pose_names),
        pose_values=motion[:, pose_mask],
        rest_positions_z_up_m=to_capture_m(rest),
        joint_positions_z_up_m=positions_capture,
        root_translation_m=positions_capture[:, skeleton.index("root")],
        ticks=consumed["ticks"],
        # carried so every downstream instrument reads the SAME array this body was fitted to
        triangulated_world_positions_z_up_m=consumed["triangulated_world_positions_z_up_m"],
        raw_triangulated_world_positions_z_up_m=consumed["raw_triangulated_world_positions_z_up_m"],
        consumed_joint_names=np.array(joint_names),
        consumed_landmark_array=np.array(landmarks),
    )
    return {"identity_names": identity_names, "pose_names": pose_names,
            "joint_positions_z_up_m": positions_capture}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lod", type=int, default=2)
    parser.add_argument("--landmarks", choices=("smoothed", "raw"), default="smoothed")
    parser.add_argument("--mean-body", action="store_true")
    parser.add_argument("--free-offsets", action="store_true")
    parser.add_argument("--subject", type=int, required=True,
                        help="ONE subject per process. A calibration mutates whatever a LATER "
                             "Character.load_fbx returns in the same process (see export_glb), so "
                             "two subjects fitted in one process are not fitted on the same model.")
    arguments = parser.parse_args()
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    key = ("triangulated_world_positions_z_up_m" if arguments.landmarks == "smoothed"
           else "raw_triangulated_world_positions_z_up_m")
    report = {"subject": arguments.subject, "arm": ("mean_body" if arguments.mean_body else
                      "free_offsets" if arguments.free_offsets else "fitted_pinned_offsets"),
              "lod": arguments.lod, "landmarks": arguments.landmarks,
              "consumed_array_key": key,
              "module": str(Path(g.__file__).resolve()),
              "settings": {"calib_frames": 100, "loss_alpha": 2.0, "max_iter": 30,
                           "smoothing": 0.0, "locator_limit_weight": 0.0 if arguments.free_offsets else 10.0},
              "subjects": {}}
    # Loaded BEFORE any calibration: see `export_glb`'s docstring.
    export_character = load_character(arguments.lod)
    for subject in (arguments.subject,):
        # allow_pickle: the file is the build's own converter-input dump from this same run
        # (object arrays; CLAUDE.md), never an outside artifact.
        source = np.load(arguments.inputs / f"subject-{subject:02d}-consumed.npz", allow_pickle=True)
        consumed = {k: source[k] for k in source.files}
        joint_names = [str(n) for n in consumed["joint_names"]]
        array = np.asarray(consumed[key], np.float64)
        fit = fit_one(array, joint_names, lod=arguments.lod, mean_body=arguments.mean_body,
                      free_offsets=arguments.free_offsets)
        prefix = out / f"subject-{subject:02d}"
        export_glb(fit, export_character, prefix.with_suffix(".glb"))
        written = write_track(fit, prefix, subject=subject, consumed=consumed, lod=arguments.lod,
                              landmarks=arguments.landmarks, settings=report["settings"],
                              joint_names=joint_names)
        positions, occluded, names = marker_arrays(fit["markers"])
        # B2's second half: exactly what momentum received, beside the array it was derived from.
        np.savez(out / f"subject-{subject:02d}.markers.npz",
                 marker_names=np.array(names), marker_positions_mhr_cm=positions,
                 marker_occluded=occluded, source_array_zup_m=array,
                 source_array_key=np.array(key), source_joint_names=np.array(joint_names))
        residual = locator_residual_mm(fit, joint_names)
        identity = np.asarray(fit["identity"])
        nonzero = {n: float(v) for n, v in zip(fit["parameter_names"], identity)
                   if n.startswith("scale_") and abs(float(v)) > 1e-6}
        locators = fit["character"].locators
        offsets = np.asarray([np.asarray(l.offset, np.float64) for l in locators])
        report["subjects"][f"subject_{subject:02d}"] = {
            "frames": int(fit["motion"].shape[0]),
            "mesh_vertices": int(fit["character"].mesh.vertices.shape[0]),
            "joint_to_landmark_residual_mm_median_over_frames": round(float(np.nanmedian(residual)), 3),
            "joint_to_landmark_residual_mm_p95": round(float(np.nanpercentile(residual, 95)), 3),
            # counted over the 68 `scale_*` identity channels ONLY -- the pre-card's field counted
            # pose entries too and read 75/73 for 13/11 (Astra, 2026-09-21).
            "nonzero_identity_channels": len(nonzero),
            "identity_channels": nonzero,
            "locator_offset_mm_median": round(float(np.median(np.linalg.norm(offsets, axis=1)) * 10.0), 3),
            "locator_offset_mm_max": round(float(np.max(np.linalg.norm(offsets, axis=1)) * 10.0), 3),
            "occluded_marker_fraction": round(float(occluded.mean()), 6),
        }
        print(f"subject {subject:02d}: "
              f"{report['subjects'][f'subject_{subject:02d}']['joint_to_landmark_residual_mm_median_over_frames']}"
              f" mm median residual; {len(nonzero)} identity channels; "
              f"{written['joint_positions_z_up_m'].shape[0]} frames", flush=True)
    path = out / f"fit-report-subject-{arguments.subject:02d}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("WROTE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
