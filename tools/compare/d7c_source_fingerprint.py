#!/usr/bin/env python3
"""D7c: tie every build report to the SOURCE THAT PRODUCED IT, not to where it sat on disk.

WHY A PATH PREFIX IS NOT PROVENANCE. The gate used to accept a build report if its
`resolved_module` string started with this worktree's root. Astra's round 8 set that root to
the MAIN checkout in memory and the gate still read MERGE, because this worktree lives
*underneath* the main checkout -- so the check passed the wrong source tree, which is the exact
failure (the PYTHONPATH trap) it was written to catch. The converse is just as wrong: an
identical checkout somewhere else would have been rejected for its location alone.

A path says where a file was. A CONTENT HASH says which code ran. So each build report carries
the sha256 of every `autoanim_gnm` module loaded when it was produced, the combined fingerprint
over that map, and the pelvis mode the build ran in; the gate compares those against the module
it resolves at execution. The absolute path stays, as provenance and nothing more.

THE THREE STAGES ARE DIFFERENT SOURCE, AND THE GATE MUST NOT CONFLATE THEM:

  hygiene   the historical rebuild on the UNCHANGED converter -- the module BEFORE D7c's src
            change -- which is what makes "today's code reproduces the shipped delivery
            byte-identically" a statement about D9b's code and not about ours
  tripwire  the REFACTORED module with the mode held at C: one execution, read twice
  delivery  the REFACTORED module with E shipping -- and the control build beside it

Rebuilding the E candidate and demanding historical byte-identity would test the wrong stage.

FINGERPRINTS COMPUTED AFTER THE FACT ARE MARKED AS SUCH. The producer recorded only a path
until now, so the three existing reports get their fingerprint from the module bytes still on
this branch (`git show <commit>:<path>`), and each says so in `computed_after_the_fact` with
the commit named. A build run after this change records its own at build time.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/compare/d7c-pelvis-rest"
# the retained pre-change converter, so the historical stage's fingerprint is checkable from
# the branch rather than pinned as a literal in the gate
RETAINED = BASE / "provenance/commercial_multiview.pre-D7c.py"
SRC_CHANGE_COMMIT = "dec1354"          # "D7c stage 4: the src change"
CONVERTER = "src/autoanim_gnm/commercial_multiview.py"

# report file -> (stage, the pelvis mode that stage runs in)
BUILD_STAGES = {
    "delivery-hygiene-build.json": ("pre_change", "C_kabsch_pelvis"),
    "tripwire-mode-c-build.json": ("refactored", "C_kabsch_pelvis"),
    "delivery-build.json": ("refactored", "E_rig_rest_kabsch"),
    "control-clear-contacts-build.json": ("refactored", "E_rig_rest_kabsch"),
    # the instrument runs resolve the same converter and are stamped with it, so that a
    # measurement of the candidate cannot have been taken against another source tree
    "instrument-d7c.json": ("refactored", "E_rig_rest_kabsch"),
    "instrument-take.json": ("refactored", "E_rig_rest_kabsch"),
}


def hash_file(path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def combine(modules: dict) -> str:
    """One fingerprint over the whole module map, order-independent."""
    return sha256(json.dumps(modules, sort_keys=True).encode()).hexdigest()


def loaded_modules() -> dict:
    """Every `autoanim_gnm` module currently imported, name -> sha256 of its file."""
    out = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "autoanim_gnm" or name.startswith("autoanim_gnm.")):
            continue
        origin = getattr(module, "__file__", None)
        if origin and Path(origin).is_file():
            out[name] = hash_file(origin)
    return out


def fingerprint_now(mode: str) -> dict:
    """What a build records about its own source, at build time."""
    modules = loaded_modules()
    converter = modules.get("autoanim_gnm.commercial_multiview")
    if converter is None:
        raise SystemExit("autoanim_gnm.commercial_multiview is not imported")
    import autoanim_gnm.commercial_multiview as cm
    return {
        "converter_module": "autoanim_gnm.commercial_multiview",
        "converter_path": str(Path(cm.__file__).resolve()),
        "converter_sha256": converter,
        "modules": modules,
        "fingerprint": combine(modules),
        "pelvis_mode": mode,
        "computed_after_the_fact": None,
    }


def resolved_converter_sha() -> str:
    """The converter THIS process resolves -- what the gate compares against."""
    import importlib.util
    spec = importlib.util.find_spec("autoanim_gnm.commercial_multiview")
    if spec is None or not spec.origin:
        raise FileNotFoundError("autoanim_gnm.commercial_multiview does not resolve")
    return hash_file(spec.origin)


def main() -> int:
    """Stamp the existing build reports from the bytes still on this branch."""
    import subprocess
    RETAINED.parent.mkdir(parents=True, exist_ok=True)
    pre = subprocess.run(["git", "show", f"{SRC_CHANGE_COMMIT}^:{CONVERTER}"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    RETAINED.write_bytes(pre)
    pre_sha, now_sha = sha256(pre).hexdigest(), hash_file(ROOT / CONVERTER)
    if pre_sha == now_sha:
        raise SystemExit("the pre-change and refactored converters hash the same; the "
                         "retained copy is not the pre-change module")
    for name, (stage, mode) in BUILD_STAGES.items():
        path = BASE / name
        if not path.exists():
            print(f"  {name}: absent, skipped")
            continue
        report = json.loads(path.read_text())
        sha = pre_sha if stage == "pre_change" else now_sha
        report["source_fingerprint"] = {
            "converter_module": "autoanim_gnm.commercial_multiview",
            "converter_path": report.get("resolved_module"),
            "converter_sha256": sha,
            "stage": stage,
            "pelvis_mode": mode,
            "computed_after_the_fact": (
                f"YES -- the producer recorded only a path, so this hash is the converter's "
                f"bytes on this branch: "
                f"{SRC_CHANGE_COMMIT + '^' if stage == 'pre_change' else 'the working tree'}. "
                f"A build run after this change records its own at build time."),
            "retained_pre_change_copy": (str(RETAINED.relative_to(ROOT))
                                         if stage == "pre_change" else None),
        }
        path.write_text(json.dumps(report, indent=1))
        print(f"  {name}: {stage} {sha[:12]} mode {mode}")
    print(f"retained {RETAINED.relative_to(ROOT)} ({pre_sha[:12]}) against the working tree's "
          f"{now_sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
