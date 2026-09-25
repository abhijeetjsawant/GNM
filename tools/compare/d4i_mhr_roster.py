"""D4i: the MHR-scope roster entries. Each wraps one UNCHANGED instrument and adds what the card requires of it.

For every entry, in this order:
  1. SCOPE. `scripts/body_delivery_schema.require_scope(delivery, "mhr")` reads each track's `schema_version` and
     nothing else; a rig-schema track, or a mixed directory, is REFUSED before the instrument runs or any payload
     field is read. (`root_translation_m` shares its key across the two schemas with a different frame.)
  2. The instrument runs as it stands, as a subprocess, its full output logged beside the entry report.
  3. EXECUTION is separated from the scientific result: a non-zero exit or a missing report is CRASH, whatever
     the instrument printed.
  4. THE FROZEN POPULATION: 2 subjects x 150 frames x the declared 17-landmark map (B4: its 15 SMPL-X pairs), and for
     B1 and the silhouette 150 frames x 4 cameras x 2 performers minus the cells the MASK cache puts below the mask
     floor. EXPECTED is derived here from the files; OBSERVED is read from the instrument's own report. Any
     mismatch in subjects, frames, mapping or exclusions is FAIL. The reference is bound by content: the delivery's
     captured landmarks must be byte-equal to the array the fit was handed (`converter-inputs/`), and a mesh must
     carry a binding sidecar naming the delivery's own GLB hashes.
  5. THE INSTRUMENT'S OWN VERDICT FIELD is read (never its exit code). B3, B4 and the silhouette figures have none:
     they are REPORTED, and their entry verdict is execution and population only -- no quality threshold is
     written here. B1's own reading is D4's registered band (candidate minus the D7c rig, lower CI > 0 on both
     performers); B2's is its `verdict`; the closure's `all_within_band`; B5's `verdict_bytes`.

The entry report's top-level `verdict` is one of PASS, FAIL, REFUSED, CRASH.

    .venv/bin/python tools/compare/d4i_mhr_roster.py ENTRY --delivery DIR --out OUT.json [entry options]

ENTRY: silhouette, b1, b2, b3, b4, closure, b5, verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools/compare"))
import body_delivery_schema as schema  # noqa: E402

PY = sys.executable
SUBJECTS = (0, 1)
FRAMES = 150
CAMERAS = ("A001", "B001", "C001", "D001")
# The declared landmark-to-joint map, frozen from D4d's gated delivery (subject-XX.body-track.json).
MAPPING = {"root": "root", "neck": "c_neck", "nose": "c_head", "left_shoulder": "l_uparm",
           "right_shoulder": "r_uparm", "left_elbow": "l_lowarm", "right_elbow": "r_lowarm",
           "left_wrist": "l_wrist", "right_wrist": "r_wrist", "left_hip": "l_upleg", "right_hip": "r_upleg",
           "left_knee": "l_lowleg", "right_knee": "r_lowleg", "left_ankle": "l_foot", "right_ankle": "r_foot",
           "left_eye": "l_eye", "right_eye": "r_eye"}
# B4 composes the map with retarget_cost's PAIRS: the 15 names both tables cover (the eyes have no SMPL-X pair).
B4_NAMES = ("root", "neck", "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
            "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
# MAMMA's subject indices are not ours (CLAUDE.md): body_id-00 is our subject 1.
B4_SUBJECT_MAP = {"0": 1, "1": 0}
MHR_JOINTS = 127
# The D7c rig mesh every B1 since D4 has used, bound by content (docs/reviews/body-model-flip-records/i6-baseline.json).
D7C_BASELINE_SHA256 = "f67f05d45bb4b0766712386ef49f5a7dd4c690c13f8a7f114ed3efa0bcc9561b"
DEFAULT_BASELINE = ROOT / "artifacts/compare/d4i-flip/i6-baseline-archive/delivered-mesh.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Entry:
    def __init__(self, name: str, delivery: Path, out: Path) -> None:
        self.name, self.delivery, self.out = name, delivery.resolve(), out.resolve()
        self.report: dict[str, Any] = {"entry": name, "scope": "mhr", "delivery": str(self.delivery),
                                       "population": {"expected": {}, "observed": {}, "mismatches": []}}

    def expect(self, key: str, expected: Any, observed: Any) -> None:
        self.report["population"]["expected"][key] = expected
        self.report["population"]["observed"][key] = observed
        if expected != observed:
            self.report["population"]["mismatches"].append(key)

    def run(self, command: list[str], raw: Path) -> dict | None:
        log = self.out.with_suffix(".instrument.log")
        raw.unlink(missing_ok=True)
        started = time.time()
        with log.open("w") as handle:
            handle.write(f"# command: {' '.join(command)}\n")
            handle.flush()
            code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        self.report["instrument"] = {"command": command, "exit_status": code, "log": str(log),
                                     "seconds": round(time.time() - started, 1), "raw_report": str(raw)}
        if code != 0 or not raw.is_file():
            self.report["instrument"]["crash"] = log.read_text(errors="replace")[-3000:]
            return None
        self.report["instrument"]["raw_report_sha256"] = sha256(raw)
        return json.loads(raw.read_text())

    def finish(self, verdict: str, own: dict[str, Any] | None = None, scope_note: str = "") -> int:
        self.report["instrument_verdict_field"] = own or {}
        self.report["verdict_scope"] = scope_note
        self.report["verdict"] = verdict
        self.out.write_text(json.dumps(self.report, indent=1), encoding="utf-8")
        print(f"{self.name}: {verdict}" + (f"  population mismatches {self.report['population']['mismatches']}"
                                           if self.report["population"]["mismatches"] else ""))
        return 0


def capture_reference(entry: Entry, subject: int) -> tuple[np.ndarray, list[str]]:
    """The delivery's captured landmarks, bound by content to the array the fit was HANDED."""
    track = np.load(entry.delivery / f"subject-{subject:02d}.body-track.npz", allow_pickle=False)
    markers = np.load(entry.delivery / f"subject-{subject:02d}.markers.npz", allow_pickle=False)
    key = str(markers["source_array_key"])
    handed = np.load(entry.delivery / "converter-inputs" / f"subject-{subject:02d}-consumed.npz",
                     allow_pickle=False)[key]
    reference = np.asarray(track["triangulated_world_positions_z_up_m"])
    entry.expect(f"subject_{subject:02d}/reference_is_the_array_the_fit_was_handed", True,
                 bool(reference.dtype == handed.dtype and reference.shape == handed.shape
                      and reference.tobytes() == handed.tobytes()))
    entry.expect(f"subject_{subject:02d}/capture_frames", FRAMES, int(reference.shape[0]))
    names = [str(n) for n in track["consumed_joint_names"]]
    return reference, names


def glb_frames_and_joints(path: Path) -> tuple[list[str], np.ndarray]:
    from d4_glb_closure import glb_joint_positions, to_capture
    names, positions = glb_joint_positions(path)
    return names, to_capture(positions)


def declared_map(entry: Entry, subject: int) -> dict:
    declared = json.loads((entry.delivery / f"subject-{subject:02d}.body-track.json").read_text())["landmark_to_joint"]
    entry.expect(f"subject_{subject:02d}/declared_map", MAPPING, declared)
    return declared


def mesh_bound(entry: Entry, mesh: Path) -> None:
    """A mesh must name, by content, the GLBs of THIS delivery and the MHR exporter."""
    sidecar = mesh.with_name("delivered-mesh.binding.json")
    recorded = json.loads(sidecar.read_text()) if sidecar.is_file() else None
    glbs = {f"subject-{s:02d}.glb": sha256(entry.delivery / f"subject-{s:02d}.glb") for s in SUBJECTS}
    entry.expect("mesh_binding/present", True, recorded is not None)
    entry.expect("mesh_binding/glb_sha256", glbs, (recorded or {}).get("glb_sha256"))
    entry.expect("mesh_binding/scope", "mhr", (recorded or {}).get("scope"))
    entry.expect("mesh_binding/exporter", "tools/compare/blender_export_mesh_momentum.py",
                 (recorded or {}).get("exporter"))
    entry.expect("mesh_binding/mesh_sha256", sha256(mesh) if mesh.is_file() else None,
                 (recorded or {}).get("mesh_sha256"))
    entry.report["mesh"] = {"path": str(mesh), "sha256": sha256(mesh) if mesh.is_file() else None,
                            "binding": recorded}


def mask_exclusions() -> dict[int, list]:
    """The cells the MASK cache puts below the floor: a property of the masks alone, identical for every arm."""
    import d4_silhouette_paired as paired
    import silhouette as si
    si.WORK = ROOT / "artifacts/compare/i6"          # read-only: the mask cache exists and is only read
    return paired.excluded_cells(si.MaskStore(4, si.CAMERAS))


# ------------------------------------------------------------------------------------------------ entries

def entry_b2(entry: Entry, args) -> int:
    rig_build = args.rig_build.resolve()
    try:
        schema.require_scope(rig_build, "rig")
    except schema.SchemaScopeError as error:
        entry.report["refused"] = f"--rig-build is not a rig-schema build: {error}"
        return entry.finish("REFUSED")
    raw = entry.out.with_suffix(".instrument.json")
    got = entry.run([PY, "tools/compare/d4_b2_same_denominator.py", "--delivery", str(entry.delivery),
                     "--rig-build", str(rig_build), "--out", str(raw)], raw)
    if got is None:
        return entry.finish("CRASH")
    entry.expect("subjects", ["subject_00", "subject_01"], sorted(got.get("subjects", {})))
    for subject in SUBJECTS:
        row = got["subjects"].get(f"subject_{subject:02d}", {})
        capture_reference(entry, subject)
        entry.expect(f"subject_{subject:02d}/declared_map", MAPPING, row.get("declared_landmark_to_joint"))
        entry.expect(f"subject_{subject:02d}/landmarks_consumed", len(MAPPING), row.get("landmarks_consumed"))
    own = {"field": "verdict", "value": got.get("verdict")}
    ok = got.get("verdict") == "PASS" and not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", own, "B2's own verdict (its nine numbered checks) AND the population")


def entry_b3(entry: Entry, args) -> int:
    raw = entry.out.with_suffix(".instrument.json")
    got = entry.run([PY, "tools/compare/d4_b3_placement.py", "--delivery", str(entry.delivery), "--out", str(raw)],
                    raw)
    if got is None:
        return entry.finish("CRASH")
    entry.expect("subjects", ["subject_00", "subject_01"], sorted(got.get("subjects", {})))
    for subject in SUBJECTS:
        row = got["subjects"].get(f"subject_{subject:02d}", {})
        reference, names = capture_reference(entry, subject)
        declared = declared_map(entry, subject)
        joint_names, world = glb_frames_and_joints(entry.delivery / f"subject-{subject:02d}.glb")
        entry.expect(f"subject_{subject:02d}/glb_frames", FRAMES, int(world.shape[0]))
        per = row.get("per_landmark", {})
        entry.expect(f"subject_{subject:02d}/landmarks_scored", sorted(MAPPING), sorted(per))
        for landmark, joint in MAPPING.items():
            if landmark not in names or joint not in joint_names:
                entry.expect(f"subject_{subject:02d}/{landmark}/resolvable", True, False)
                continue
            # EXPECTED frames: every one of the frozen 150 where both ends are finite. B3 itself takes the shorter
            # sequence and drops nonfinite distances, so a truncation or a NaN'd reference shows up only here.
            full = np.full(FRAMES, np.nan)
            n = min(FRAMES, world.shape[0], reference.shape[0])
            full[:n] = np.linalg.norm(world[:n, joint_names.index(joint)] - reference[:n, names.index(landmark)],
                                      axis=1)
            expected_frames = int(np.isfinite(full).sum()) if n == FRAMES else f"{FRAMES} (the take is {n})"
            entry.expect(f"subject_{subject:02d}/{landmark}/frames", expected_frames,
                         per.get(landmark, {}).get("frames"))
            entry.expect(f"subject_{subject:02d}/{landmark}/joint", joint, per.get(landmark, {}).get("joint"))
        entry.report.setdefault("reported", {})[f"subject_{subject:02d}"] = {
            "all_landmarks_median_mm": row.get("all_landmarks_median_mm"),
            "segment_mean_abs_error_mm": row.get("segment_mean_abs_error_mm")}
    ok = not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", {"field": None, "value": "REPORTED (B3 has no verdict)"},
                        "execution and the frozen population only; B3's figures are REPORTED, no threshold")


def entry_b4(entry: Entry, args) -> int:
    raw = entry.out.with_suffix(".instrument.json")
    got = entry.run([PY, "tools/compare/d4_b4_mamma_arm.py", "--delivery", str(entry.delivery), "--out", str(raw)],
                    raw)
    if got is None:
        return entry.finish("CRASH")
    import d4_b4_mamma_arm as b4
    entry.expect("subjects", ["subject_00", "subject_01"], sorted(got.get("subjects", {})))
    entry.expect("subject_to_mamma_body_id", B4_SUBJECT_MAP, got.get("subject_to_mamma_body_id"))
    entry.expect("mapped_joints", len(B4_NAMES), got.get("mapped_joints"))
    for subject in SUBJECTS:
        row = got["subjects"].get(f"subject_{subject:02d}", {})
        capture_reference(entry, subject)
        declared_map(entry, subject)
        _, world = glb_frames_and_joints(entry.delivery / f"subject-{subject:02d}.glb")
        entry.expect(f"subject_{subject:02d}/glb_frames", FRAMES, int(world.shape[0]))
        body = B4_SUBJECT_MAP[str(subject)]
        # allow_pickle: MAMMA's own retained output under the gitignored artifacts/ tree (object arrays, CLAUDE.md),
        # the same file B4 itself loads; nothing third-party.
        theirs = np.load(b4.MA3D / f"verts_joints_body_id-{body:02d}.npz", allow_pickle=True)["pred_joints"]
        entry.expect(f"subject_{subject:02d}/mamma_frames", FRAMES, int(theirs.shape[0]))
        entry.expect(f"subject_{subject:02d}/joints_scored", sorted(B4_NAMES), sorted(row.get("absolute", {})))
        finite = all(np.isfinite(v["absolute_median_mm"]) for v in row.get("absolute", {}).values())
        entry.expect(f"subject_{subject:02d}/figures_finite", True, bool(finite))
        entry.report.setdefault("reported", {})[f"subject_{subject:02d}"] = {
            k: row.get(k) for k in ("absolute_all_joint_median_mm", "root_relative_all_joint_median_mm")}
    ok = not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", {"field": None, "value": "REPORTED, never selected (B4)"},
                        "execution and the frozen population only; B4 is REPORTED, never selected, no threshold")


def entry_closure(entry: Entry, args) -> int:
    raw = entry.out.with_suffix(".instrument.json")
    command = [PY, "tools/compare/d4_glb_closure.py"]
    for subject in SUBJECTS:
        prefix = entry.delivery / f"subject-{subject:02d}"
        command += ["--pair", f"subject_{subject:02d}={prefix}.glb,{prefix}.body-track.npz"]
    got = entry.run(command + ["--out", str(raw)], raw)
    if got is None:
        return entry.finish("CRASH")
    entry.expect("subjects", ["subject_00", "subject_01"], sorted(got.get("pairs", {})))
    for subject in SUBJECTS:
        row = got["pairs"].get(f"subject_{subject:02d}", {})
        prefix = entry.delivery / f"subject-{subject:02d}"
        entry.expect(f"subject_{subject:02d}/glb_sha256", sha256(prefix.with_suffix(".glb")), row.get("glb_sha256"))
        entry.expect(f"subject_{subject:02d}/track_sha256", sha256(prefix.with_suffix(".body-track.npz")),
                     row.get("track_sha256"))
        entry.expect(f"subject_{subject:02d}/frames_in_glb", FRAMES, row.get("frames_in_glb"))
        entry.expect(f"subject_{subject:02d}/frames_in_track", FRAMES, row.get("frames_in_track"))
        entry.expect(f"subject_{subject:02d}/joints_compared", MHR_JOINTS, row.get("joints_compared"))
        entry.expect(f"subject_{subject:02d}/joints_missing", [], row.get("joints_missing_from_the_glb"))
    own = {"field": "all_within_band", "value": got.get("all_within_band"),
           "worst_max_abs_m": got.get("worst_max_abs_m"), "band_m": got.get("band_m")}
    ok = got.get("all_within_band") is True and not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", own, "the closure's own band (1e-4 m, D3/D4) AND the population")


def entry_b5(entry: Entry, args) -> int:
    mesh = args.mesh.resolve()
    mesh_bound(entry, mesh)
    raw = entry.out.with_suffix(".instrument.json")
    got = entry.run([PY, "tools/compare/d4_b5_delivered_bytes.py", "--delivery", str(entry.delivery),
                     "--reference", str(args.reference.resolve()), "--mesh", str(mesh), "--out", str(raw)], raw)
    if got is None:
        return entry.finish("CRASH")
    entry.expect("subjects", ["subject_00", "subject_01"], sorted(got.get("subjects", {})))
    for subject in SUBJECTS:
        row = got["subjects"].get(f"subject_{subject:02d}", {})
        reference, names = capture_reference(entry, subject)
        entry.expect(f"subject_{subject:02d}/sampler_samples", FRAMES, row.get("sampler_samples"))
        entry.expect(f"subject_{subject:02d}/mesh_frames", FRAMES, row.get("mesh_frames"))
        eyes = [names.index(n) for n in ("left_eye", "right_eye", "nose")]
        expected_scored = int(np.isfinite(reference[:, eyes]).all(axis=(1, 2)).sum()) \
            if reference.shape[0] == FRAMES else f"{FRAMES} (the take is {reference.shape[0]})"
        entry.expect(f"subject_{subject:02d}/facing_frames_scored", expected_scored,
                     row.get("facing_dot_frames_scored"))
    own = {"field": "verdict_bytes", "value": got.get("verdict_bytes"),
           "facing_positive_on_every_frame_both_performers (REPORTED)":
               got.get("facing_positive_on_every_frame_both_performers")}
    ok = got.get("verdict_bytes") == "PASS" and not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", own,
                        "B5's own byte verdict AND the population; its facing clause stays REPORTED")


def entry_silhouette(entry: Entry, args) -> int:
    work = args.work.resolve()
    raw = entry.out.with_suffix(".instrument.json")
    got = entry.run([PY, "tools/compare/silhouette.py", "--scope", "mhr", "--delivery", str(entry.delivery),
                     "--work", str(work), "--out", str(raw)], raw)
    if got is None:
        return entry.finish("CRASH")
    mesh_bound(entry, work / "delivered-mesh.npz")
    entry.expect("scope", "mhr", got.get("scope"))
    entry.expect("mesh_cache_bound_to_this_delivery",
                 {f"subject-{s:02d}.glb": sha256(entry.delivery / f"subject-{s:02d}.glb") for s in SUBJECTS},
                 (got.get("mesh_cache", {}).get("binding") or {}).get("glb_sha256"))
    excluded = mask_exclusions()
    arms = got.get("arms", {}).get("ours_delivered", {})
    for cam in CAMERAS:
        for subject in SUBJECTS:
            cut = sum(1 for c, _ in excluded[subject] if c == cam)
            entry.expect(f"{cam}/subject_{subject:02d}/frames_scored", FRAMES - cut,
                         arms.get(cam, {}).get(f"subject_{subject:02d}", {}).get("frames_scored"))
    entry.report["mesh_cache"] = got.get("mesh_cache")
    entry.report["reported"] = {cam: {f"subject_{s:02d}": arms.get(cam, {}).get(f"subject_{s:02d}", {})
                                      .get("iou", {}).get("median") for s in SUBJECTS} for cam in CAMERAS}
    ok = not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL",
                        {"field": "frame alignment (the instrument exits unless 'aligned')", "value": "aligned"},
                        "execution, scope, the bound mesh and the population; the per-arm figures are REPORTED")


def entry_b1(entry: Entry, args) -> int:
    mesh = (args.work.resolve() / "delivered-mesh.npz")
    mesh_bound(entry, mesh)
    baseline = args.baseline.resolve()
    entry.expect("baseline_sha256", D7C_BASELINE_SHA256, sha256(baseline) if baseline.is_file() else None)
    raw = entry.out.with_suffix(".instrument.json")
    candidate = args.candidate_name
    command = [PY, "tools/compare/d4_silhouette_paired.py", "--out", str(raw),
               "--arm", f"baseline_D7c_rig={baseline}"]
    for extra in args.extra_arm:
        command += ["--arm", extra]
    command += ["--arm", f"{candidate}={mesh}", "--pair", f"{candidate},baseline_D7c_rig"]
    for extra in args.extra_pair:
        command += ["--pair", extra]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    got = entry.run(command, raw)
    if got is None:
        return entry.finish("CRASH")
    excluded = mask_exclusions()
    population = got.get("population", {})
    for subject in SUBJECTS:
        key = f"subject_{subject:02d}"
        required = len(CAMERAS) * FRAMES - len(excluded[subject])
        entry.expect(f"{key}/cells_required", required, population.get("cells_required", {}).get(key))
        entry.expect(f"{key}/cells_excluded", [list(c) for c in excluded[subject]],
                     population.get("cells_excluded_by_the_mask_cache", {}).get(key))
        row = got.get("paired", {}).get(f"{candidate}_minus_baseline_D7c_rig_{key}", {})
        entry.expect(f"{key}/n", required, row.get("n"))
        entry.expect(f"{key}/population_as_named", True, row.get("population_as_named"))
        entry.expect(f"{key}/frames_consumed", FRAMES,
                     population.get("frames_consumed_per_arm", {}).get(candidate, {}).get(key))
    entry.expect("arm_file_sha256/baseline", D7C_BASELINE_SHA256,
                 got.get("arm_file_sha256", {}).get("baseline_D7c_rig"))
    entry.expect("arm_file_sha256/candidate", sha256(mesh), got.get("arm_file_sha256", {}).get(candidate))
    lower = {key: got.get("paired", {}).get(f"{candidate}_minus_baseline_D7c_rig_{key}", {})
             .get("ci95_of_the_median_difference", [None])[0] for key in ("subject_00", "subject_01")}
    band = {key: (value is not None and value > 0.0) for key, value in lower.items()}
    own = {"field": "paired/<candidate>_minus_baseline_D7c_rig_subject_XX/ci95_of_the_median_difference[0] > 0 "
                    "(D4's registered B1 band, read from the instrument's rows)",
           "lower_ci": lower, "band_holds": band}
    ok = all(band.values()) and not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", own, "D4's B1 band (lower CI > 0, both performers) AND the population")


def entry_verifier(entry: Entry, args) -> int:
    raw = entry.out.with_suffix(".instrument.json")
    log = entry.out.with_suffix(".instrument.log")
    command = [PY, "scripts/verify_commercial_multiview_artifact.py", str(entry.delivery), "--scope", "mhr"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log.write_text(f"# command: {' '.join(command)}\n{completed.stdout}{completed.stderr}")
    try:
        got = json.loads(completed.stdout)
    except json.JSONDecodeError:
        entry.report["instrument"] = {"command": command, "exit_status": completed.returncode, "log": str(log),
                                      "crash": (completed.stdout + completed.stderr)[-3000:]}
        return entry.finish("CRASH")
    raw.write_text(json.dumps(got, indent=1))
    entry.report["instrument"] = {"command": command, "exit_status": completed.returncode, "log": str(log),
                                  "raw_report": str(raw)}
    if got.get("status") == "pass":
        entry.expect("tracks", [0, 1], [t.get("subject") for t in got.get("tracks", [])])
        for track in got.get("tracks", []):
            entry.expect(f"subject_{track['subject']:02d}/frames", FRAMES, track.get("frames"))
            entry.expect(f"subject_{track['subject']:02d}/joints", MHR_JOINTS, track.get("joints"))
            entry.expect(f"subject_{track['subject']:02d}/landmarks_mapped", len(MAPPING),
                         track.get("landmarks_mapped"))
    own = {"field": "status", "value": got.get("status"), "error": got.get("error")}
    ok = got.get("status") == "pass" and not entry.report["population"]["mismatches"]
    return entry.finish("PASS" if ok else "FAIL", own, "the verifier's own status AND the population")


ENTRIES = {"silhouette": entry_silhouette, "b1": entry_b1, "b2": entry_b2, "b3": entry_b3, "b4": entry_b4,
           "closure": entry_closure, "b5": entry_b5, "verifier": entry_verifier}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entry", choices=sorted(ENTRIES))
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rig-build", type=Path, help="b2: the explicit --body rig build of the same take")
    parser.add_argument("--work", type=Path, help="silhouette and b1: the MHR-scope silhouette work directory")
    parser.add_argument("--mesh", type=Path, help="b5: the bound delivered-mesh.npz (the silhouette's export)")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "artifacts/compare/d4-body/mhr-reference-lod2.npz",
                        help="b5: MHR's own model dump (mhr_delivery.py --dump-reference)")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="b1: the D7c rig mesh, bound by sha256")
    parser.add_argument("--candidate-name", default="delivered_MHR",
                        help="b1: the arm name the delivered mesh is scored under")
    parser.add_argument("--extra-arm", action="append", default=[], help="b1: NAME=PATH, reported alongside")
    parser.add_argument("--extra-pair", action="append", default=[], help="b1: CAND,REF, reported alongside")
    parser.add_argument("--seed", type=int, default=None, help="b1: the bootstrap seed (the instrument's default)")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    entry = Entry(args.entry, args.delivery, args.out)
    try:
        entry.report["schema_declared"] = schema.require_scope(entry.delivery, "mhr")
    except schema.SchemaScopeError as error:
        entry.report["refused"] = str(error)
        return entry.finish("REFUSED")
    try:
        return ENTRIES[args.entry](entry, args)
    except Exception as error:  # an entry that cannot evaluate its own population fails closed, as CRASH
        entry.report["evaluation_crash"] = f"{type(error).__name__}: {error}"
        return entry.finish("CRASH")


if __name__ == "__main__":
    raise SystemExit(main())
