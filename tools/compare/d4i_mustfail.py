"""D4i must-fails i-v and the frozen-population mutations, each demonstrated BY MUTATION on a copy of the evidence.

Every case builds its own copy of the default build (APFS clones, `cp -Rc`) under `artifacts/compare/d4i-flip/mustfail/`,
mutates the UNDERLYING files (tracks, run report, GLB bytes, meshes), runs the unchanged entries, and records the
predicted outcome beside the observed one. Nothing here reads or sets a PASS flag: every verdict comes from running
the entry on the mutated files.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_mustfail.py [GROUP ...] --out RECORD.json

GROUPS: scope (i), mixed (ii), negative (iii), head (iv), positive (v), population. Default: all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
PY = sys.executable
BASE = ROOT / "artifacts/compare/d4i-flip"
DEFAULT = BASE / "default"
RIG = ROOT / "artifacts/compare/soma77-rig-d4i"
SIL_WORK = BASE / "default-silhouette-work"
ORACLE = BASE / "oracle"
MF = BASE / "mustfail"
ENTRIES = ("silhouette", "b1", "b2", "b3", "b4", "closure", "b5", "verifier")
FAST = ("b2", "b3", "b4", "closure", "b5", "verifier")
BUILD = [PY, "scripts/build_commercial_multiview_comparison.py", "--videos",
         ".cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos", "--calibration-yaml",
         ".cache/mamma/configs/examples/calib/iphones_outdoors.yaml", "--detector", "soma77"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True)
    return dst


def entry(name: str, delivery: Path, out: Path, *extra: str, work: Path | None = None) -> dict:
    command = [PY, "tools/compare/d4i_mhr_roster.py", name, "--delivery", str(delivery), "--out", str(out),
               "--rig-build", str(RIG), "--work", str(work or SIL_WORK),
               "--mesh", str((work or SIL_WORK) / "delivered-mesh.npz"), *extra]
    subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    report = json.loads(out.read_text()) if out.is_file() else {"verdict": "NO REPORT"}
    return {"verdict": report.get("verdict"), "instrument_ran": "instrument" in report,
            "mismatches": report.get("population", {}).get("mismatches", []),
            "refused": report.get("refused"), "evaluation_crash": report.get("evaluation_crash"),
            "report": str(out.relative_to(ROOT))}


def strip(obj: Any, drop: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [strip(v, drop) for v in obj]
    return obj


def case(record: dict, group: str, name: str, predicted: str, observed: Any, held: bool, **detail) -> None:
    record.setdefault(group, {})[name] = {"predicted": predicted, "observed": observed, "held": bool(held), **detail}
    print(f"[{group}] {name}: {'HELD' if held else 'FAILED'} -- observed {json.dumps(observed)[:160]}")


def resave_npz(path: Path, **changes: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays.update(changes)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


# ------------------------------------------------------------------------------------------------ GLB surgery

def shift_glb_root(glb: Path, delta_local: tuple[float, float, float]) -> dict:
    """Add `delta_local` to every key of the translation channel of the top-most animated node, in place.

    Returns what moved, measured from the GLB's own bytes before and after (the capture-frame displacement of every
    skin joint), so the control's size is a reading rather than an assumption.
    """
    import d3_skeleton_gate as d3
    from d4_glb_closure import glb_joint_positions, to_capture

    before_names, before = glb_joint_positions(glb)
    raw = bytearray(glb.read_bytes())
    json_len = struct.unpack_from("<I", raw, 12)[0]
    bin_start = 20 + json_len + 8
    document, _ = d3.read_glb(glb)
    nodes = document["nodes"]
    parent = {i: -1 for i in range(len(nodes))}
    for index, node in enumerate(nodes):
        for child in node.get("children", ()):
            parent[child] = index

    def depth(n: int) -> int:
        d = 0
        while parent[n] != -1:
            n, d = parent[n], d + 1
        return d

    animation = document["animations"][0]
    translations = [(depth(c["target"]["node"]), c) for c in animation["channels"]
                    if c["target"]["path"] == "translation"]
    _, channel = min(translations, key=lambda item: item[0])
    accessor = document["accessors"][animation["samplers"][channel["sampler"]]["output"]]
    view = document["bufferViews"][accessor["bufferView"]]
    assert accessor["componentType"] == 5126 and accessor["type"] == "VEC3"
    offset = bin_start + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    for k in range(accessor["count"]):
        at = offset + k * stride
        values = struct.unpack_from("<3f", raw, at)
        struct.pack_into("<3f", raw, at, *(v + d for v, d in zip(values, delta_local)))
    glb.write_bytes(bytes(raw))
    after_names, after = glb_joint_positions(glb)
    moved = np.linalg.norm(to_capture(after) - to_capture(before), axis=-1)
    return {"node": nodes[channel["target"]["node"]].get("name"), "delta_local": list(delta_local),
            "joint_displacement_m": {"min": float(moved.min()), "median": float(np.median(moved)),
                                     "max": float(moved.max())}}


# ------------------------------------------------------------------------------------------------ groups

def group_scope(record: dict) -> None:
    out = MF / "i-scope"
    out.mkdir(parents=True, exist_ok=True)
    for name in ENTRIES:
        got = entry(name, RIG, out / f"rig-build-{name}.json")
        case(record, "i_scope", f"{name}_on_the_rig_build", "REFUSED, before the instrument runs",
             {"verdict": got["verdict"], "instrument_ran": got["instrument_ran"]},
             got["verdict"] == "REFUSED" and not got["instrument_ran"], refused=got["refused"])
    swapped = clone(DEFAULT, out / "rig-track-in-an-mhr-directory")
    for suffix in (".body-track.json", ".body-track.npz"):
        shutil.copyfile(RIG / f"subject-00{suffix}", swapped / f"subject-00{suffix}")
    for name in ENTRIES:
        got = entry(name, swapped, out / f"swapped-{name}.json")
        case(record, "i_scope", f"{name}_on_a_rig_track_swapped_into_the_mhr_delivery",
             "REFUSED, before the instrument runs", {"verdict": got["verdict"], "instrument_ran": got["instrument_ran"]},
             got["verdict"] == "REFUSED" and not got["instrument_ran"], refused=got["refused"])
    runs = {
        "silhouette_mhr_scope_on_the_rig_build": ([PY, "tools/compare/silhouette.py", "--scope", "mhr", "--delivery",
                                                   str(RIG), "--work", str(out / "sil-w1")], "refuse"),
        "silhouette_rig_scope_on_the_mhr_build": ([PY, "tools/compare/silhouette.py", "--scope", "rig", "--delivery",
                                                   str(DEFAULT), "--work", str(out / "sil-w2")], "refuse"),
        "silhouette_rig_scope_on_the_rig_build_accepts": ([PY, "tools/compare/silhouette.py", "--scope", "rig",
                                                           "--delivery", str(RIG), "--work", str(out / "sil-w3"),
                                                           "--ids-only"], "accept"),
        "verifier_mhr_scope_on_the_rig_build": ([PY, "scripts/verify_commercial_multiview_artifact.py", str(RIG),
                                                 "--scope", "mhr"], "refuse"),
    }
    for name, (command, want) in runs.items():
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        text = done.stdout + done.stderr
        refused = ("REFUSED" in text or "Refused in MHR scope" in text) and done.returncode != 0
        accepted = done.returncode == 0 and "scope rig:" in text
        held = refused if want == "refuse" else accepted
        case(record, "i_scope", name, "refused by schema before any payload" if want == "refuse"
             else "accepted in rig scope (id resolution runs)", {"exit": done.returncode,
                                                                 "tail": text.strip().splitlines()[-1][:200]}, held)


def group_mixed(record: dict) -> None:
    out = MF / "ii-mixed"
    stale = clone(DEFAULT, out / "mhr-with-a-stale-mapping")
    shutil.copyfile(RIG / "subject-00.mapping.npz", stale / "subject-00.mapping.npz")
    before = {str(p.relative_to(stale)): sha256(p) for p in stale.rglob("*") if p.is_file()}
    done = subprocess.run([*BUILD, "--output", str(stale)], cwd=ROOT, capture_output=True, text=True)
    after = {str(p.relative_to(stale)): sha256(p) for p in stale.rglob("*") if p.is_file()}
    case(record, "ii_mixed", "the_default_build_refuses_a_stale_mapping", "refused, nothing written",
         {"exit": done.returncode, "untouched": before == after,
          "tail": (done.stdout + done.stderr).strip().splitlines()[-1][:240]},
         done.returncode != 0 and before == after and "refusing to build" in done.stdout + done.stderr)
    rig_into_mhr = clone(DEFAULT, out / "rig-build-into-mhr")
    before = {str(p.relative_to(rig_into_mhr)): sha256(p) for p in rig_into_mhr.rglob("*") if p.is_file()}
    done = subprocess.run([*BUILD, "--body", "rig", "--body-run", "artifacts/compare/d1-fix/body-run-regenerated",
                           "--output", str(rig_into_mhr)], cwd=ROOT, capture_output=True, text=True)
    after = {str(p.relative_to(rig_into_mhr)): sha256(p) for p in rig_into_mhr.rglob("*") if p.is_file()}
    case(record, "ii_mixed", "a_rig_build_refuses_an_mhr_directory", "refused, nothing written",
         {"exit": done.returncode, "untouched": before == after}, done.returncode != 0 and before == after
         and "refusing to build" in done.stdout + done.stderr)
    done = subprocess.run([PY, "scripts/verify_commercial_multiview_artifact.py", str(stale)], cwd=ROOT,
                          capture_output=True, text=True)
    got = json.loads(done.stdout)
    case(record, "ii_mixed", "the_verifier_fails_a_stale_mapping", "status fail, 'Mixed delivery directory'",
         got, got.get("status") == "fail" and "Mixed delivery directory" in got.get("error", ""))
    rig_mixed = clone(RIG, out / "rig-with-mhr-markers")
    shutil.copyfile(DEFAULT / "subject-00.markers.npz", rig_mixed / "subject-00.markers.npz")
    done = subprocess.run([PY, "scripts/verify_commercial_multiview_artifact.py", str(rig_mixed)], cwd=ROOT,
                          capture_output=True, text=True)
    got = json.loads(done.stdout)
    case(record, "ii_mixed", "the_verifier_fails_mhr_files_in_a_rig_directory", "status fail, mixed",
         got, got.get("status") == "fail" and "Mixed delivery directory" in got.get("error", ""))
    for name in ENTRIES:
        got = entry(name, stale, out / f"stale-{name}.json")
        case(record, "ii_mixed", f"{name}_refuses_the_stale_mapping_directory", "REFUSED",
             {"verdict": got["verdict"], "instrument_ran": got["instrument_ran"]},
             got["verdict"] == "REFUSED" and not got["instrument_ran"])


def _oracle_raw(name: str) -> dict:
    special = {"b2": "b2-vs-rig-d4i", "silhouette": "silhouette-rebound"}
    return json.loads((ORACLE / f"{special.get(name, name)}.instrument.json").read_text())


# Provenance and path fields that name WHICH files were read, never a figure. Everything else must be equal.
PROVENANCE = {"delivery", "glb", "track", "track_sha256", "reference", "arm_files", "mesh_cache", "rig_build",
              "artifact"}


def group_negative(record: dict) -> None:
    out = MF / "iii-negative"
    mutated = clone(DEFAULT, out / "mean-body-mutated")
    work = clone(SIL_WORK, out / "silhouette-work")
    shifts = {}
    for subject in (0, 1):
        path = mutated / f"subject-{subject:02d}.body-track.npz"
        rest = np.load(path)["rest_positions_z_up_m"]
        changed = (rest * np.float32(1.3) + np.float32(0.25)).astype(rest.dtype)
        resave_npz(path, rest_positions_z_up_m=changed)
        shifts[f"subject_{subject:02d}"] = float(np.abs(changed - rest).max())
    for name in ENTRIES:
        extra = (["--candidate-name", "D4d_fitted_MHR_lod2", "--extra-arm",
                  "D4c_fitted_MHR_lod2=artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz", "--extra-pair",
                  "D4d_fitted_MHR_lod2,D4c_fitted_MHR_lod2"] if name == "b1" else [])
        got = entry(name, mutated, out / f"{name}.json", *extra, work=work)
        raw = json.loads((out / f"{name}.instrument.json").read_text()) if (out / f"{name}.instrument.json").is_file() \
            else {}
        oracle = _oracle_raw(name)
        if name == "verifier":
            same = strip(raw, PROVENANCE) == strip(oracle, PROVENANCE)
        else:
            same = strip(raw, PROVENANCE) == strip(oracle, PROVENANCE)
        case(record, "iii_negative", f"{name}_reading_unchanged_under_a_mutated_mean_body",
             "the same verdict and every scored figure equal (provenance and path fields excluded)",
             {"verdict": got["verdict"], "figures_equal_to_the_default": same}, got["verdict"] == "PASS" and same,
             rest_positions_max_change_m=shifts)


def group_head(record: dict) -> None:
    out = MF / "iv-head"
    mutations = {
        "head_removed_from_the_not_consumed_list":
            lambda r: r["not_consumed_by_delivered_body"].remove("head_orientation"),
        "head_marked_consumed": lambda r: r["head_orientation"][0].__setitem__("consumed_by_delivered_body", True),
        "the_marks_stripped_as_D4d_wrote_it": lambda r: (r.pop("not_consumed_by_delivered_body"),
                                                         [e.pop("consumed_by_delivered_body", None)
                                                          for k in ("head_orientation", "toe_triangulation",
                                                                    "spine_triangulation", "pelvis_frame")
                                                          for e in r[k]]),
    }
    for name, mutate in mutations.items():
        copy = clone(DEFAULT, out / name)
        report = json.loads((copy / "run-report.json").read_text())
        mutate(report)
        (copy / "run-report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        done = subprocess.run([PY, "scripts/verify_commercial_multiview_artifact.py", str(copy), "--scope", "mhr"],
                              cwd=ROOT, capture_output=True, text=True)
        got = json.loads(done.stdout)
        roster = entry("verifier", copy, out / f"{name}-entry.json")
        case(record, "iv_head", name, "verifier status fail; the roster's verifier entry FAIL",
             {"verifier": got, "entry": roster["verdict"]},
             got.get("status") == "fail" and roster["verdict"] == "FAIL")


def group_positive(record: dict) -> None:
    out = MF / "v-positive"
    moved = clone(DEFAULT, out / "glb-root-shifted")
    shifts = {f"subject_{s:02d}": shift_glb_root(moved / f"subject-{s:02d}.glb", (0.0, 0.0, 0.10)) for s in (0, 1)}
    b3 = entry("b3", moved, out / "b3.json")
    new, old = json.loads((out / "b3.instrument.json").read_text()), _oracle_raw("b3")
    readings = {s: (old["subjects"][s]["all_landmarks_median_mm"], new["subjects"][s]["all_landmarks_median_mm"])
                for s in ("subject_00", "subject_01")}
    case(record, "v_positive", "b3_moves_when_the_scored_joints_move", "B3's all-landmark median changes",
         {"all_landmarks_median_mm (default, shifted)": readings, "entry": b3["verdict"]},
         all(a != b for a, b in readings.values()), glb_shift=shifts)
    closure = entry("closure", moved, out / "closure.json")
    case(record, "v_positive", "closure_fails_when_the_glb_leaves_its_track", "closure FAIL",
         {"entry": closure["verdict"]}, closure["verdict"] == "FAIL")
    work = clone(SIL_WORK, out / "silhouette-work")
    sil = entry("silhouette", moved, out / "silhouette.json", work=work)
    b1 = entry("b1", moved, out / "b1.json", "--candidate-name", "D4d_fitted_MHR_lod2", work=work)
    raw = json.loads((out / "b1.instrument.json").read_text()) if (out / "b1.instrument.json").is_file() else {}
    oracle = _oracle_raw("b1")
    key = "D4d_fitted_MHR_lod2"
    got = {s: (oracle["arms"][key][s]["pooled_median_iou"], raw.get("arms", {}).get(key, {}).get(s, {})
               .get("pooled_median_iou")) for s in ("subject_00", "subject_01")}
    mesh_fresh = json.loads((out / "silhouette.json").read_text()).get("mesh_cache", {}).get("fresh_export")
    case(record, "v_positive", "b1_moves_when_the_mesh_motion_moves",
         "the pooled IoU changes, on a FRESH export of the shifted GLBs",
         {"pooled_median_iou (default, shifted)": got, "silhouette_entry": sil["verdict"], "b1_entry": b1["verdict"],
          "mesh_fresh_export": mesh_fresh}, mesh_fresh is True and all(a != b for a, b in got.values()))


def group_population(record: dict) -> None:
    out = MF / "population"
    gone = clone(DEFAULT, out / "missing-subject")
    for path in gone.glob("subject-01.*"):
        path.unlink()
    for name in FAST:
        got = entry(name, gone, out / f"missing-subject-{name}.json")
        case(record, "population", f"missing_subject_{name}", "not PASS (REFUSED: the track is missing)",
             got["verdict"], got["verdict"] != "PASS")
    no_glb = clone(DEFAULT, out / "missing-glb")
    (no_glb / "subject-01.glb").unlink()
    got = entry("b3", no_glb, out / "missing-glb-b3.json")
    case(record, "population", "missing_glb_b3", "CRASH (the instrument cannot run)", got["verdict"],
         got["verdict"] == "CRASH")
    short = clone(DEFAULT, out / "truncated-frames")
    for subject in (0, 1):
        track = short / f"subject-{subject:02d}.body-track.npz"
        consumed = short / "converter-inputs" / f"subject-{subject:02d}-consumed.npz"
        with np.load(track) as archive:
            cut = {k: archive[k][:140] for k in ("triangulated_world_positions_z_up_m",
                                                 "raw_triangulated_world_positions_z_up_m")}
        resave_npz(track, **cut)
        with np.load(consumed) as archive:
            cut = {k: archive[k][:140] for k in ("triangulated_world_positions_z_up_m",
                                                 "raw_triangulated_world_positions_z_up_m")}
        resave_npz(consumed, **cut)
    for name in FAST:
        got = entry(name, short, out / f"truncated-{name}.json")
        case(record, "population", f"truncated_frames_{name}",
             "not PASS for every entry that reads the captured landmarks (b2 b3 b4 b5 verifier); the closure does "
             "not read them", {"verdict": got["verdict"], "mismatches": got["mismatches"][:6]},
             got["verdict"] != "PASS" if name != "closure" else got["verdict"] == "PASS")
    swapped = clone(DEFAULT, out / "substituted-reference")
    with np.load(swapped / "subject-01.body-track.npz") as archive:
        other = archive["triangulated_world_positions_z_up_m"]
    resave_npz(swapped / "subject-00.body-track.npz", triangulated_world_positions_z_up_m=other)
    nudged = clone(DEFAULT, out / "nudged-reference")
    with np.load(nudged / "subject-00.body-track.npz") as archive:
        ref = archive["triangulated_world_positions_z_up_m"].copy()
    finite = np.argwhere(np.isfinite(ref))[0]
    ref[tuple(finite)] += 0.001
    resave_npz(nudged / "subject-00.body-track.npz", triangulated_world_positions_z_up_m=ref)
    for label, directory in (("substituted_reference", swapped), ("reference_nudged_1mm_on_one_value", nudged)):
        for name in ("b3", "b4", "b5", "b2"):
            got = entry(name, directory, out / f"{label}-{name}.json")
            case(record, "population", f"{label}_{name}", "FAIL (the reference is not the array the fit was handed)",
                 {"verdict": got["verdict"], "mismatches": got["mismatches"][:4]}, got["verdict"] == "FAIL")
    # B1: a truncated mesh re-bound by a forged sidecar still cannot pass (the instrument refuses a short arm),
    # and a substituted baseline fails its sha256 binding.
    forged = clone(SIL_WORK, out / "forged-short-mesh")
    with np.load(forged / "delivered-mesh.npz") as archive:
        arrays = {k: (archive[k][:140] if k.startswith("verts") else archive[k]) for k in archive.files}
    np.savez_compressed(forged / "delivered-mesh.npz", **arrays)
    sidecar = json.loads((forged / "delivered-mesh.binding.json").read_text())
    sidecar["mesh_sha256"] = sha256(forged / "delivered-mesh.npz")
    (forged / "delivered-mesh.binding.json").write_text(json.dumps(sidecar, indent=1))
    got = entry("b1", DEFAULT, out / "b1-short-mesh.json", work=forged)
    case(record, "population", "b1_truncated_mesh_with_a_forged_binding", "CRASH (d4_silhouette_paired refuses a "
         "short arm)", got["verdict"], got["verdict"] == "CRASH")
    got = entry("b1", DEFAULT, out / "b1-substituted-baseline.json", "--baseline",
                str(ROOT / "artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz"))
    case(record, "population", "b1_substituted_baseline", "FAIL (baseline sha256 is not the D7c mesh)",
         {"verdict": got["verdict"], "mismatches": got["mismatches"]}, got["verdict"] == "FAIL")
    got = entry("b5", DEFAULT, out / "b5-unbound-mesh.json", "--mesh",
                str(ROOT / "artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz"))
    case(record, "population", "b5_mesh_not_bound_to_this_delivery", "FAIL (no binding naming these GLBs)",
         {"verdict": got["verdict"], "mismatches": got["mismatches"][:4]}, got["verdict"] == "FAIL")


GROUPS = {"scope": group_scope, "mixed": group_mixed, "negative": group_negative, "head": group_head,
          "positive": group_positive, "population": group_population}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", default=list(GROUPS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for group in args.groups:
        GROUPS[group](record)
    held = [(g, n) for g, cases in record.items() if isinstance(cases, dict) for n, c in cases.items()
            if isinstance(c, dict) and "held" in c]
    record["summary"] = {"cases": len(held),
                         "held": sum(record[g][n]["held"] for g, n in held),
                         "failed": [f"{g}/{n}" for g, n in held if not record[g][n]["held"]]}
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
