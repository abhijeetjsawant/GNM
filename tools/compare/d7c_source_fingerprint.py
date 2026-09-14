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

from functools import lru_cache
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
STAGES = ("pre_change", "refactored")
CONVERTER = "src/autoanim_gnm/commercial_multiview.py"

# THE ONE INDEPENDENT SIGNAL THE AFTER-THE-FACT STAMP CANNOT MANUFACTURE. Every hash below
# was written by reading the same bytes the gate compares against, so "refactored == the
# executing converter" is true by construction today and becomes evidence only on the next
# build. What is NOT by construction is WHEN each stage ran: the historical hygiene arm's log
# must predate every refactored stage's log and the src-change commit itself. That is
# ordering evidence for the pre-change attribution -- not proof, since an uncommitted edit
# leaves no timestamp, and the report says so.
STAGE_LOGS = {
    "delivery-hygiene-build.json": "01-hygiene.log",
    "tripwire-mode-c-build.json": "08-tripwire-mode-c.log",
    "delivery-build.json": "10-delivery.log",
    "control-clear-contacts-build.json": "12-control-clear-contacts.log",
    "instrument-d7c.json": "09-oracle-d7c.log",
    "instrument-take.json": "14-take.log",
    "silhouette-partwise.json": "17-b1-silhouette.log",
}

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
    # the photographs' own instrument: it runs no converter arm, but it reads the delivered
    # meshes and is stamped with the tree it ran from like every other stage
    "silhouette-partwise.json": ("refactored", "E_rig_rest_kabsch"),
}


def hash_file(path) -> str:
    """Memoised on (path, mtime, size): the gate re-hashes the same mask cache on every one
    of the fuzz's tens of thousands of builds, and that file is hundreds of megabytes."""
    stamp = Path(path).stat()
    return _hash_file(str(Path(path).resolve()), stamp.st_mtime_ns, stamp.st_size)


@lru_cache(maxsize=None)
def _hash_file(path: str, _mtime: int, _size: int) -> str:
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


def head_commit() -> str:
    """The commit the working tree was on when this build ran. Unlike a clock reading it can
    be CHECKED: the gate asks git whether it is an ancestor of the src-change commit, which
    is what makes a pre-change claim verifiable rather than asserted."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              check=True, text=True).stdout.strip()
    except Exception:                                   # a build outside a checkout
        return ""


def fingerprint_now(mode: str, *, stage: str) -> dict:
    """What a build records about its own source, AT BUILD TIME and in full.

    THE STAGE IS AN INPUT, NEVER INFERRED. An earlier draft had the producer decide its own
    stage by comparing its converter against the retained pre-change copy -- which is the
    file the GATE uses as its reference, so a swapped or absent copy would have had every
    producer and the gate agreeing with each other about nothing. The operator knows which
    arm they are running; they say so, and the gate checks the claim against the hashes and
    against git. A producer never verifies its own claim.

    This emits exactly what `d7c_gate_report.built_here` requires: the converter's sha256,
    the module map it came from, the stage, the pelvis mode, and a build_order block. Astra's
    round 9 substituted this function's actual return into a report and the gate rejected it
    for missing `stage` and `build_order/log_mtime`; the contract is now one definition read
    by both sides, and a genuine stamp is gated on.
    """
    if stage not in STAGES:
        raise SystemExit(f"--src-stage must be one of {sorted(STAGES)}, not {stage!r}")
    import time
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
        "stage": stage,
        "pelvis_mode": mode,
        "build_order": {
            "stage_time": int(time.time()),
            "stage_time_source": "the producer's own clock as it wrote this report",
            "head_commit": head_commit(),
            "argv": list(sys.argv),
        },
        "retrospective": False,
    }


@lru_cache(maxsize=None)
def is_ancestor(earlier: str, later: str) -> bool:
    """Does `earlier` lie on `later`'s history? The one ordering claim git can settle.

    MEMOISED because the gate is rebuilt tens of thousands of times under the fuzz and this
    shells out to git: uncached it took a build from 9 ms to 39 ms, which is 43 minutes of
    fuzz. The answer for a pair of commits does not change while the process runs.
    """
    import subprocess
    if not earlier or not later:
        return False
    try:
        return subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later],
                              cwd=ROOT, capture_output=True).returncode == 0
    except Exception:
        return False


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
    commit_time = int(subprocess.run(["git", "log", "--format=%ct", "-1", SRC_CHANGE_COMMIT],
                                     cwd=ROOT, capture_output=True, check=True,
                                     text=True).stdout.strip())
    mtimes = {}
    for name, log in STAGE_LOGS.items():
        path = BASE / "logs" / log
        mtimes[name] = int(path.stat().st_mtime) if path.exists() else None
    if pre_sha == now_sha:
        raise SystemExit("the pre-change and refactored converters hash the same; the "
                         "retained copy is not the pre-change module")
    kept = filled = 0
    for name, (stage, mode) in BUILD_STAGES.items():
        path = BASE / name
        if not path.exists():
            print(f"  {name}: absent, skipped")
            continue
        report = json.loads(path.read_text())
        existing = report.get("source_fingerprint")
        # A RETROSPECTIVE STAMP MAY ONLY FILL AN ABSENT ONE. It used to overwrite
        # `source_fingerprint` wholesale, which meant a genuine build-time stamp -- the only
        # kind that is evidence rather than bookkeeping -- could be replaced by one this
        # script wrote from the branch's bytes. Astra's round 9 named that, and after the
        # close-out rebuild it would have erased exactly the evidence the rebuild produced.
        if isinstance(existing, dict) and existing.get("retrospective") is False:
            print(f"  {name}: GENUINE stamp kept "
                  f"({existing.get('converter_sha256', '')[:12]}, stage "
                  f"{existing.get('stage')}) -- not overwritten")
            kept += 1
            continue
        sha = pre_sha if stage == "pre_change" else now_sha
        report["source_fingerprint"] = {
            "converter_module": "autoanim_gnm.commercial_multiview",
            "converter_path": report.get("resolved_module"),
            "converter_sha256": sha,
            "stage": stage,
            "pelvis_mode": mode,
            "retrospective": True,
            "computed_after_the_fact": (
                f"YES -- the producer recorded only a path when this report was written, so "
                f"this hash is the converter's bytes on this branch: "
                f"{SRC_CHANGE_COMMIT + '^' if stage == 'pre_change' else 'the working tree'}. "
                f"A build run after this change records its own at build time and this "
                f"script will not overwrite it."),
            "retained_pre_change_copy": (str(RETAINED.relative_to(ROOT))
                                         if stage == "pre_change" else None),
            "build_order": {
                "stage_time": mtimes.get(name),
                "stage_time_source": (f"the mtime of logs/{STAGE_LOGS.get(name)}, filled "
                                      "after the fact -- the producer recorded no time"),
                "head_commit": "",
                "log": STAGE_LOGS.get(name),
                **({"src_change_commit": SRC_CHANGE_COMMIT,
                    "src_change_commit_time": commit_time} if stage == "pre_change" else {}),
                "what_it_is": (
                    "ORDERING EVIDENCE, not proof. This stamp's hash was taken from the "
                    "bytes the gate itself compares against, so a refactored stage agrees "
                    "with it by construction; what is not by construction is that the "
                    "historical hygiene arm ran BEFORE the src change. An uncommitted edit "
                    "leaves no timestamp, so this narrows the claim rather than closing it. "
                    "A GENUINE stamp records the build's own head commit instead, which the "
                    "gate checks against git."),
            },
        }
        path.write_text(json.dumps(report, indent=1))
        filled += 1
        print(f"  {name}: retrospective {stage} {sha[:12]} mode {mode}")
    print(f"retained {RETAINED.relative_to(ROOT)} ({pre_sha[:12]}) against the working tree's "
          f"{now_sha[:12]}; {filled} filled, {kept} genuine stamps kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
