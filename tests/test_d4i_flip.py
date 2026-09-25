"""D4i -- the default flip to MHR and the schema-aware roster. Tests that need neither momentum nor artifacts.

Pinned here:
  * the delivery-schema module decides the schema from `schema_version` alone, finds a MIXED directory (a stale
    `mapping.npz` beside an MHR delivery, MHR files beside a rig one), ignores `work/`, and the build's guard and an
    instrument's scope check refuse accordingly;
  * the build's momentum interpreter defaults to the gitignored `.venv-mhr/`, and the bootstrap pins momentum;
  * an MHR run report that claims a capture-side solve (the head above all) as delivered fails the verifier;
  * the silhouette's mesh cache is bound by content, never overwrites an unbound mesh, refuses the D7c baseline as a
    work directory, and refuses a delivery outside its scope;
  * every MHR-scope roster entry refuses a rig delivery before running anything.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import body_delivery_schema as schema  # noqa: E402

MHR = schema.MHR_SCHEMA
RIG = "autoanim.body-track/1.3"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(directory: Path, subject: int, body: str, *, npz_schema: str | None = "same") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"subject-{subject:02d}"
    declared = MHR if body == "mhr" else RIG
    prefix.with_suffix(".body-track.json").write_text(json.dumps({"schema_version": declared}))
    payload = {"root_translation_m": np.zeros((3, 3))}
    value = declared if npz_schema == "same" else npz_schema
    if body == "mhr" and value is not None:
        payload["schema_version"] = np.array(value)
    elif body == "rig" and npz_schema not in ("same", None):
        payload["schema_version"] = np.array(npz_schema)
    np.savez(prefix.with_suffix(".body-track.npz"), **payload)


def _delivery(directory: Path, body: str) -> Path:
    for subject in (0, 1):
        _track(directory, subject, body)
        if body == "mhr":
            (directory / f"subject-{subject:02d}.markers.npz").write_bytes(b"x")
        else:
            (directory / f"subject-{subject:02d}.mapping.npz").write_bytes(b"x")
    (directory / "work").mkdir(exist_ok=True)
    return directory


# ------------------------------------------------------------------------------ the schema module

def test_clean_deliveries_are_not_mixed(tmp_path):
    assert schema.mixed_reasons(_delivery(tmp_path / "m", "mhr")) == []
    assert schema.mixed_reasons(_delivery(tmp_path / "r", "rig")) == []
    assert schema.require_scope(tmp_path / "m", "mhr") == {"subject_00": MHR, "subject_01": MHR}


def test_a_stale_mapping_beside_an_mhr_delivery_is_mixed(tmp_path):
    delivery = _delivery(tmp_path / "m", "mhr")
    (delivery / "subject-00.mapping.npz").write_bytes(b"stale")
    assert schema.mixed_reasons(delivery)
    with pytest.raises(schema.SchemaScopeError, match="mixed"):
        schema.require_scope(delivery, "mhr")


def test_mhr_files_beside_a_rig_delivery_are_mixed(tmp_path):
    delivery = _delivery(tmp_path / "r", "rig")
    (delivery / "converter-inputs").mkdir()
    assert schema.mixed_reasons(delivery)


def test_a_rig_npz_under_an_mhr_json_is_caught(tmp_path):
    _track(tmp_path, 0, "mhr", npz_schema=None)
    with pytest.raises(schema.SchemaScopeError, match="mixed track"):
        schema.track_schema(tmp_path, 0)


def test_work_is_never_inspected(tmp_path):
    delivery = _delivery(tmp_path / "m", "mhr")
    (delivery / "work" / "subject-00.mapping.npz").write_bytes(b"x")
    assert schema.mixed_reasons(delivery) == []


@pytest.mark.parametrize("scope, other", [("mhr", "rig"), ("rig", "mhr")])
def test_scope_refuses_the_other_schema(tmp_path, scope, other):
    delivery = _delivery(tmp_path / other, other)
    with pytest.raises(schema.SchemaScopeError, match="refused before any payload field"):
        schema.require_scope(delivery, scope)


def test_the_build_guard_refuses_the_other_schema_and_accepts_work_only(tmp_path):
    empty = tmp_path / "fresh"
    (empty / "work" / "frames").mkdir(parents=True)
    schema.refuse_other_schema(empty, "mhr")            # only work/: fine
    schema.refuse_other_schema(empty, "rig")
    rig = _delivery(tmp_path / "rig", "rig")
    with pytest.raises(schema.SchemaScopeError, match="refusing to build a --body mhr"):
        schema.refuse_other_schema(rig, "mhr")
    mhr = _delivery(tmp_path / "mhr", "mhr")
    with pytest.raises(schema.SchemaScopeError, match="refusing to build a --body rig"):
        schema.refuse_other_schema(mhr, "rig")
    schema.refuse_other_schema(mhr, "mhr")               # a rebuild of the same schema is allowed


# ------------------------------------------------------------------------------ the build script

def _arguments() -> dict:
    tree = ast.parse((ROOT / "scripts/build_commercial_multiview_comparison.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            found[node.args[0].value] = {k.arg: k.value for k in node.keywords}
    return found


def test_the_momentum_interpreter_defaults_off_tmp():
    source = (ROOT / "scripts/build_commercial_multiview_comparison.py").read_text()
    default = _arguments()["--mhr-python"]["default"]
    assert isinstance(default, ast.Name) and default.id == "DEFAULT_MHR_PYTHON"
    assert 'DEFAULT_MHR_PYTHON = ROOT / ".venv-mhr" / "bin" / "python"' in source
    assert "/tmp/momenv" not in source
    assert ".venv-mhr/" in (ROOT / ".gitignore").read_text().split()


def test_the_bootstrap_pins_momentum_and_records_its_environment():
    lock = (ROOT / "scripts/mhr-requirements.lock").read_text()
    assert "pymomentum-cpu==0.1.114.post0" in lock.split()
    script = (ROOT / "scripts/bootstrap_mhr.sh").read_text()
    for needle in ("--no-deps", "mhr-requirements.lock", "bootstrap-record.json", "platform", "3.12.13"):
        assert needle in script


def test_the_build_marks_the_five_capture_side_solves_not_consumed():
    source = (ROOT / "scripts/build_commercial_multiview_comparison.py").read_text()
    tree = ast.parse(source)
    value = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "NOT_CONSUMED_BY_MHR" for t in node.targets))
    assert {e.value for e in value.elts} == {"head_orientation", "toe_triangulation", "spine_triangulation",
                                             "pelvis_frame", "contact_frames"}
    assert "body_delivery_schema.refuse_other_schema(output, arguments.body)" in source


# ------------------------------------------------------------------------------ the verifier (must-fail iv)

@pytest.fixture(scope="module")
def verifier():
    return _module(ROOT / "scripts/verify_commercial_multiview_artifact.py", "verify_d4i")


def _marked_report() -> dict:
    report = {"not_consumed_by_delivered_body": ["head_orientation", "toe_triangulation", "spine_triangulation",
                                                 "pelvis_frame", "contact_frames"],
              "contact_frames": [[36, 36], [5, 18]]}
    for key in ("head_orientation", "toe_triangulation", "spine_triangulation", "pelvis_frame"):
        report[key] = [{"status": "solved", "consumed_by_delivered_body": False} for _ in range(2)]
    return report


def test_a_marked_mhr_report_passes(verifier):
    verifier.check_mhr_report_marks(_marked_report())


@pytest.mark.parametrize("mutation", ["unlisted", "consumed_true", "mark_missing", "no_list"])
def test_a_false_head_claim_fails_the_verifier(verifier, mutation):
    report = _marked_report()
    if mutation == "unlisted":
        report["not_consumed_by_delivered_body"].remove("head_orientation")
    elif mutation == "consumed_true":
        report["head_orientation"][0]["consumed_by_delivered_body"] = True
    elif mutation == "mark_missing":
        del report["head_orientation"][1]["consumed_by_delivered_body"]
    else:
        del report["not_consumed_by_delivered_body"]
    with pytest.raises(verifier.VerificationFailure):
        verifier.check_mhr_report_marks(report)


# ------------------------------------------------------------------------------ the silhouette

@pytest.fixture()
def silhouette(tmp_path, monkeypatch):
    module = _module(ROOT / "tools/compare/silhouette.py", "silhouette_d4i")
    delivery = _delivery(tmp_path / "delivery", "mhr")
    for subject in (0, 1):
        (delivery / f"subject-{subject:02d}.glb").write_bytes(b"glTF" + bytes([subject]))
    monkeypatch.setattr(module, "DELIVERY", delivery)
    monkeypatch.setattr(module, "WORK", tmp_path / "work")
    monkeypatch.setattr(module, "SCOPE", "mhr")
    return module


def test_the_mesh_binding_follows_the_glb_bytes_and_the_exporter(silhouette):
    first = silhouette.mesh_binding(silhouette.DELIVERY, "mhr")
    assert first["exporter"] == "tools/compare/blender_export_mesh_momentum.py"
    assert silhouette.mesh_binding(silhouette.DELIVERY, "rig")["exporter"] == "tools/compare/blender_export_mesh.py"
    (silhouette.DELIVERY / "subject-01.glb").write_bytes(b"glTF-changed")
    assert silhouette.mesh_binding(silhouette.DELIVERY, "mhr")["glb_sha256"] != first["glb_sha256"]


def test_a_bound_cache_is_used_and_an_unbound_one_is_never_overwritten(silhouette, monkeypatch):
    silhouette.WORK.mkdir()
    np.savez(silhouette.WORK / "delivered-mesh.npz", verts_00=np.zeros((1, 3, 3)))
    calls = []
    monkeypatch.setattr(silhouette.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit, match="never overwritten"):
        silhouette.delivered_mesh()
    binding = silhouette.mesh_binding(silhouette.DELIVERY, "mhr")
    mesh_sha = silhouette.sha256(silhouette.WORK / "delivered-mesh.npz")
    (silhouette.WORK / "delivered-mesh.binding.json").write_text(json.dumps({**binding, "mesh_sha256": mesh_sha}))
    silhouette.delivered_mesh()
    assert silhouette.MESH_CACHE["fresh_export"] is False and not calls
    # a mesh swapped in beside the copied sidecar is not a cache hit: it is re-exported
    np.savez(silhouette.WORK / "delivered-mesh.npz", verts_00=np.ones((1, 3, 3)))
    try:
        silhouette.delivered_mesh()
    except (FileNotFoundError, KeyError, OSError, ValueError):
        pass                                      # the stubbed export wrote nothing new; what matters is below
    assert calls, "a substituted mesh was accepted as a bound cache"


def test_the_d7c_baseline_is_refused_as_work_and_a_rig_delivery_in_mhr_scope(silhouette, tmp_path, monkeypatch):
    mhr = silhouette.DELIVERY            # main() rebinds the module global; keep the MHR path
    for work in ("artifacts/compare/i6", "artifacts/compare/d4i-flip/i6-baseline-archive/sub"):
        monkeypatch.setattr(sys, "argv", ["silhouette.py", "--delivery", str(mhr), "--work", str(ROOT / work)])
        with pytest.raises(SystemExit, match="read-only D7c silhouette baseline"):
            silhouette.main()
    rig = _delivery(tmp_path / "rig", "rig")
    monkeypatch.setattr(sys, "argv", ["silhouette.py", "--delivery", str(rig), "--work", str(tmp_path / "w")])
    with pytest.raises(SystemExit, match=r"REFUSED \(mhr scope\)"):
        silhouette.main()
    monkeypatch.setattr(sys, "argv", ["silhouette.py", "--scope", "rig", "--delivery", str(mhr),
                                      "--work", str(tmp_path / "w")])
    with pytest.raises(SystemExit, match=r"REFUSED \(rig scope\)"):
        silhouette.main()


# ------------------------------------------------------------------------------ the roster entries (must-fail i)

@pytest.mark.parametrize("entry", ["silhouette", "b1", "b2", "b3", "b4", "closure", "b5", "verifier"])
def test_every_mhr_roster_entry_refuses_a_rig_delivery_before_running(tmp_path, monkeypatch, entry):
    roster = _module(ROOT / "tools/compare/d4i_mhr_roster.py", "roster_d4i")
    ran = []
    monkeypatch.setattr(roster.subprocess, "run", lambda *a, **k: ran.append(a))
    rig = _delivery(tmp_path / "rig", "rig")
    out = tmp_path / f"{entry}.json"
    monkeypatch.setattr(sys, "argv", ["d4i_mhr_roster.py", entry, "--delivery", str(rig), "--out", str(out),
                                      "--work", str(tmp_path / "w"), "--mesh", str(tmp_path / "m.npz"),
                                      "--rig-build", str(rig)])
    roster.main()
    report = json.loads(out.read_text())
    assert report["verdict"] == "REFUSED" and "rig-schema" in report["refused"]
    assert not ran
