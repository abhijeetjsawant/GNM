"""D4i -- the gate reproduces its committed verdict from the evidence, reads the oracle literally, and turns under
mutation of the evidence. Needs the D4i artifacts (skipped when they are absent, as on a fresh checkout)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/compare"))


def _gate():
    spec = importlib.util.spec_from_file_location("d4i_gate_test", ROOT / "tools/compare/d4i_flip_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _gate()
pytestmark = pytest.mark.skipif(
    not all(Path(gate.PATHS[k]).exists() for k in ("default", "d4d_delivery", "final_roster", "mustfail")),
    reason="the D4i artifacts are not on this machine")


@pytest.fixture(scope="module")
def baseline():
    return gate.verdicts()


def test_the_gate_reproduces_its_committed_verdict(baseline):
    committed = json.loads((ROOT / "docs/reviews/body-model-flip-records/gate.json").read_text())
    assert baseline["line"] == committed["line"]
    assert baseline["line"] == "VERDICT: FAIL (failed: oracle, coordinator_migration_recorded)"
    for name, conjunct in committed["conjuncts"].items():
        assert baseline["conjuncts"][name]["legs"] == conjunct["legs"], name


def test_the_oracle_fails_on_exactly_one_leaf_of_the_two_track_jsons_and_nothing_else(baseline):
    oracle = baseline["conjuncts"]["oracle"]
    assert oracle["failing_leaves"] == {"subject-00.body-track.json": ["/body_model/assets"],
                                        "subject-01.body-track.json": ["/body_model/assets"]}
    assert sum(r["byte_identical"] for r in oracle["files"].values()) == 10
    assert all(v for k, v in oracle["legs"].items() if k != "per_subject_files_byte_identical")


def test_the_oracle_is_read_literally_no_normalisation(tmp_path):
    copy = tmp_path / "default"
    subprocess.run(["cp", "-Rc", str(gate.PATHS["default"]), str(copy)], check=True)
    for s in (0, 1):   # the counterfactual: D4d's own JSON bytes make the leg hold; nothing else does
        shutil.copyfile(Path(gate.PATHS["d4d_delivery"]) / f"subject-{s:02d}.body-track.json",
                        copy / f"subject-{s:02d}.body-track.json")
    assert gate.verdicts({"default": copy})["conjuncts"]["oracle"]["holds"] is True
    data = json.loads((copy / "subject-00.body-track.json").read_text())
    (copy / "subject-00.body-track.json").write_text(json.dumps(data, indent=1))   # same content, other bytes
    assert gate.oracle({**gate.PATHS, "default": copy})["files"]["subject-00.body-track.json"]["byte_identical"] is False


def test_a_delivered_byte_turns_the_oracle_and_the_population(tmp_path):
    copy = tmp_path / "default"
    subprocess.run(["cp", "-Rc", str(gate.PATHS["default"]), str(copy)], check=True)
    raw = bytearray((copy / "subject-00.glb").read_bytes())
    raw[-50] ^= 1
    (copy / "subject-00.glb").write_bytes(bytes(raw))
    report = gate.verdicts({"default": copy})
    assert report["conjuncts"]["oracle"]["files"]["subject-00.glb"]["byte_identical"] is False
    assert report["conjuncts"]["population_and_own_verdicts"]["holds"] is False    # the roster did not measure it


def test_the_b2_verdict_is_the_instruments_own_field(tmp_path):
    copy = tmp_path / "final-roster"
    subprocess.run(["cp", "-Rc", str(gate.PATHS["final_roster"]), str(copy)], check=True)
    data = json.loads((copy / "b2.instrument.json").read_text())
    data["verdict"] = "FAIL"
    (copy / "b2.instrument.json").write_text(json.dumps(data))
    assert gate.verdicts({"final_roster": copy})["conjuncts"]["B2_PASS"]["holds"] is False


def test_unreadable_evidence_is_a_crash_that_fails_closed(tmp_path):
    copy = tmp_path / "final-roster"
    subprocess.run(["cp", "-Rc", str(gate.PATHS["final_roster"]), str(copy)], check=True)
    (copy / "closure.instrument.json").unlink()
    report = gate.verdicts({"final_roster": copy})
    assert "closure_PASS" in report["crashed"] and report["conjuncts"]["closure_PASS"]["holds"] is False


def test_the_committed_fuzz_turned_every_targeted_leg():
    fuzz = json.loads((ROOT / "docs/reviews/body-model-flip-records/fuzz.json").read_text())["summary"]
    assert fuzz["turned"] == fuzz["targeted"] and not fuzz["unmoved"] and not fuzz["crash_in_targeted"]
    assert fuzz["crash_class"]["all_crash_and_fail_closed"] and fuzz["negative_controls_unmoved"]


def test_the_extractor_stub_resolves_every_visual_key():
    done = subprocess.run([sys.executable, str(ROOT / "tools/compare/extractors/d4i_flip.py")], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0 and "VISUALS keys resolved: True []" in done.stdout
