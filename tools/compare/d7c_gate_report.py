#!/usr/bin/env python3
"""D7c's gate: every clause, its predicted value, its measured value and a verdict.

IT COMPUTES NOTHING AND IT ASSERTS NOTHING. It loads the reports each instrument wrote and
DERIVES every verdict from the numbers in them.

WHY THIS FILE HAS FOUR RULES RATHER THAN A LIST OF PATCHES. Astra's merge review broke it in
six successive rounds -- literal verdicts, then saved classifications, then partial
populations, then stored aggregates, then four unread leaves, then four more -- and each round
was answered hole by hole. The holes were never the problem; the absence of a rule was. So:

  1. EVERY VALUE IS DERIVED FROM NAMED CONSTITUENTS, OR CROSS-CHECKED AGAINST THEM. An
     aggregate the gate reads without recomputing is an aggregate an attacker can write.
     Where a report also stores a summary, the stored and the derived value must AGREE, and a
     disagreement is a FAIL -- a report that contradicts itself is corrupt whichever half
     would have passed.
  2. A MISSING FIELD OR SET MEMBER IS A FAIL, NEVER A NO-OP. `all()` over an empty map is
     True; `max()` over a subset says nothing about the whole. Every read goes through
     `Reader`, which raises on an absent path, and the clause that needed it FAILS with the
     path named.
  3. EVERY SET IS CHECKED BY IDENTITY, NOT BY COUNT. The eight delivered files, the six oracle
     seeds, the two performers, the eight B1 cells, the six G1/G2/follower bodies, the six
     oracle arms, the seven S arms, and the contact RUNS by their `(side, start, end)`
     identity taken from the frozen mask. A renamed cell, a duplicated run and a dropped seed
     all survive a count.
  4. EVERY MEASUREMENT LEAF IS READ OR JUSTIFIED BY NAME. Rules 1-3 say what the gate does
     with what it reads; round 7 was about what it does not read at all. So `coverage_audit`
     classifies every leaf no clause touches -- LABEL / PROVENANCE / DIAGNOSTIC / MEASUREMENT
     -- and every MEASUREMENT leaf under a report some clause reads must be named by a family
     in `UNREAD_MEASUREMENTS_JUSTIFIED`, or the gate reads NO MERGE. `saved_value_inventory`
     does the same for the other direction: every boolean and string the gate CONSUMES
     without deriving or cross-checking it, generated from the Reader's own record rather
     than written from memory, must be named in `TRUSTED_READ_JUSTIFICATIONS`. A
     justification that matches nothing is reported too -- a stale cover is a hole.

AND IT IS PROVED RATHER THAN ASSERTED. `tools/compare/d7c_gate_fuzz.py` walks every leaf of
every report this gate reads, mutates each one in turn, and requires NO MERGE from every leaf
any clause depends on -- reporting, with a justification, the leaves that are genuinely inert.
The mutation count in `gate.json` is the number of leaves visited, not a hand-picked table.

TWO CLAUSES READ **FAIL** AND STAY THAT WAY: S at the card's own fixture (sigma 1.0) and the
calibration under its own frozen monotonicity precondition. `selector.json` and
`selector-calibrated.json` are immutable, and the two reviewer amendments that followed are
recorded as POST HOC.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_gate_report.py
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/compare/d7c-pelvis-rest"

if str(ROOT / "tools/compare") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools/compare"))
from d7c_source_fingerprint import (  # noqa: E402
    BUILD_STAGES, RETAINED, hash_file, resolved_converter_sha)

# THE CONVERTER THIS RUN RESOLVES, hashed once. Every "refactored" build report must name it;
# the historical hygiene arm must name the retained pre-change module instead, and must NOT
# name this one. Both are content, not location.
try:
    EXECUTING_CONVERTER_SHA = resolved_converter_sha()
except FileNotFoundError:
    EXECUTING_CONVERTER_SHA = ""
PRE_CHANGE_CONVERTER_SHA = hash_file(RETAINED) if RETAINED.exists() else ""

REPORTS = {
    "hygiene": "delivery-hygiene-build.json",
    "tripwire": "tripwire-mode-c-build.json",
    "delivery": "delivery-build.json",
    "oracle": "instrument-d7c.json",
    "take": "instrument-take.json",
    "sigma1": "selector.json",
    "calibration": "selector-calibrated.json",
    "admissibility": "selector-calibrated-amended.json",
    "reread": "selector-reread-sigma0.335546875.json",
    "projection": "projection-preservation.json",
    "p_oracle": "projection-preservation-oracle.json",
    "control2": "projection-preservation-control-clear-contacts.json",
    "p1_controls": "p1-controls.json",
    "silhouette": "silhouette-partwise.json",
    "b1_attribution": "b1-attribution.json",
    "b2": "b2-delivered-vs-capture.json",
    "b3": "b3-hoist-and-contacts.json",
    "b6": "b6-delivered-bytes.json",
}

# The bands, restated beside the numbers they test. None is new and none is moved.
O1_TILT_DEG, O1_ORIGIN_MM, O1_RESIDUAL_M = 0.01, 0.01, 1.0e-6
O2_LEG_MM, O2_HOIST_MM = 0.1, 0.05
CONTACT_TOLERANCE_M = 1.0e-5
FOLLOWER_RATIO, FOLLOWER_FLOOR_DEG = 2.0, 2.0
CALIBRATION_TARGET_MM, CALIBRATION_TAU_MM = 8.7636, 0.05
CALIBRATION_BRACKET = (0.10, 1.00)
TIE_DEG, TIE_MM = 0.1, 0.1
AGREEMENT = 1.0e-4                      # stored-vs-derived agreement, in each row's own unit

ORACLE_SEEDS = ("20260903", "20260904", "20260905", "20260906", "20260907", "20260908")
# The oracle's arms BY NAME, the fixture's length, and the mode the delivery ships. `src` is
# measured as `src_default`; S chose `E_rig_rest_kabsch`; the gate requires the two to be the
# same arm leaf for leaf rather than the same label.
ORACLE_ARMS = ("src_default", "C_soma_template", "wrong_origin", "D_rig_rest_hipline",
               "E_rig_rest_kabsch", "frozen_upright")
ORACLE_FRAMES = 150
PELVIS_MODE_SHIPPED = "E_rig_rest_kabsch"
PERFORMERS = ("subject_00", "subject_01")
DELIVERED_FILES = tuple(
    f"subject-{s:02d}{suffix}" for s in (0, 1)
    for suffix in (".glb", ".body-track.json", ".body-track.npz", ".mapping.npz"))
# B1's own cut sizes. They happen to equal S's synthetic fixture sizes, and that is a
# COINCIDENCE of two 150-frame takes with a third bent: the photographs are the real take and
# S's are six synthetic bodies, so pinning one to the other's constant would tie two unrelated
# populations together.
B1_CUT_FRAMES = {"whole_take": 150, "bent_tercile": 50}
B1_CELLS = tuple(f"clause_{part}_{cut}_worsening_not_established_vs_D9b"
                 for part in ("arm", "torso")
                 for cut in ("whole_take", "bent_tercile"))
PROTECTED = ("Root", "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg",
             "RightLowerLeg", "LeftFoot", "RightFoot", "LeftToes", "RightToes")
SIDE_JOINTS = {0: ("LeftFoot", "LeftToes"), 1: ("RightFoot", "RightToes")}
POPULATIONS = ("whole_take", "bent_tercile")
# S's frozen population sizes, per body per arm. 150 frames whole take and 50 in the bent
# tercile; pairs are formed on the FULL sequence and belong to a population iff BOTH
# endpoints do, which gives 149 whole-take pairs and 47 bent ones -- 47 and not 49, because
# the tercile's 50 frames are not contiguous. (An earlier note here attributed a 49-pair
# requirement to Astra's round 6. THAT ATTRIBUTION WAS WRONG AND IS WITHDRAWN: round 6 made no
# such requirement, and 47 is right by the both-endpoints rule -- every one of the 84 body x
# arm x population rows in the reread carries the size this table states.)
S_POPULATION = {"whole_take": (150, 149), "bent_tercile": (50, 47)}
# S's arms BY NAME -- the two candidates, the unguarded variant, C-on-SOMA and the three
# controls -- and the sigma the calibration accepted. Every arm's population is validated,
# including the arms no clause bands.
REREAD_ARMS = ("b_hipline_guarded", "a_kabsch_guarded", "b_hipline_unguarded", "C_on_SOMA",
               "world_vertical", "thorax_as_pelvis", "frozen_pitch_follower")
ACCEPTED_SIGMA = 0.335546875
# What each P1 control must fail, BY NAME. "some nonempty list" is satisfied by a channel the
# control never touches.
CONTROL_CHANNELS = {
    "control_1_foot_locals_overwritten": {"local::LeftFoot", "local::RightFoot"},
    "control_2_contact_mask_cleared": {"foot_contacts"},
}
CONTROL_2_BUILT_CHANNELS = {"root_translation_m", "foot_contacts"}
# The P1 channel set, BY NAME and in full: the two whole-track channels and the ten protected
# locals. `channel_preservation` requires the report's `channels` map to BE this set.
P1_CHANNELS = ("root_translation_m", "foot_contacts", *(f"local::{j}" for j in PROTECTED))
PROTECTED_CHANNELS = ({"root_translation_m", "foot_contacts"}
                      | {f"local::{j}" for j in (
                          "Root", "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg",
                          "RightLowerLeg", "LeftFoot", "RightFoot", "LeftToes", "RightToes")})
METRICS = (("i_orientation_deg", TIE_DEG), ("ii_step_deg", TIE_DEG),
           ("iii_root_step_mm", TIE_MM))


def kind_of(node) -> str:
    """One naming of the kinds, shared by the Reader and by the fuzzer's walk."""
    if isinstance(node, dict):
        return "map"
    if isinstance(node, list):
        return "list"
    if isinstance(node, bool):
        return "bool"
    if isinstance(node, (int, float)):
        return "number"
    if isinstance(node, str):
        return "string"
    return "other"


class Missing(Exception):
    """An absent path, an absent set member, or a value of the wrong kind."""


class Reader:
    """Every read goes through here, so an absent path FAILS its clause instead of vanishing."""

    def __init__(self, reports: dict) -> None:
        self.reports = reports
        self.touched: set[tuple] = set()
        # what KIND each read leaf was, and which reads were cross-checked against a value
        # derived from the leaf's own constituents. The saved-boolean inventory is generated
        # from these two sets, so it is what the gate DOES and not what its author recalls.
        self.kinds: dict[tuple, str] = {}
        self.cross_checked: set[tuple] = set()

    def at(self, *path):
        node = self.reports
        for step in path:
            if isinstance(node, list):
                if not isinstance(step, int) or not -len(node) <= step < len(node):
                    raise Missing("/".join(map(str, path)))
                node = node[step]
            elif isinstance(node, dict):
                if step not in node:
                    raise Missing("/".join(map(str, path)))
                node = node[step]
            else:
                raise Missing("/".join(map(str, path)))
        key = tuple(map(str, path))
        self.touched.add(key)
        self.kinds[key] = kind_of(node)
        return node

    def checked(self, *path, derived, tolerance=None):
        """A stored summary READ AND CROSS-CHECKED against a value derived from the
        constituents beside it. A disagreement is a FAIL: a report that contradicts itself is
        corrupt whichever half would have passed. Recorded, so the inventory is mechanical."""
        value = self.at(*path)
        if tolerance is None:
            ok = value == derived
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Missing("/".join(map(str, path)) + " is not a number")
            ok = agrees(value, derived, tolerance)
        if not ok:
            raise Missing(f"{'/'.join(map(str, path))} stores {value!r} against {derived!r} "
                          "derived from its own constituents")
        self.cross_checked.add(tuple(map(str, path)))
        return value

    def note_cross_checked(self, *path) -> None:
        """Record a leaf that was checked by comparison with ANOTHER leaf rather than with a
        derived value -- the two halves of an equality are each other's cross-check."""
        self.cross_checked.add(tuple(map(str, path)))

    def num(self, *path) -> float:
        value = self.at(*path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Missing("/".join(map(str, path)) + " is not a number")
        return float(value)

    def flag(self, *path) -> bool:
        value = self.at(*path)
        if not isinstance(value, bool):
            raise Missing("/".join(map(str, path)) + " is not a boolean")
        return value

    def text(self, *path) -> str:
        value = self.at(*path)
        if not isinstance(value, str):
            raise Missing("/".join(map(str, path)) + " is not a string")
        return value

    def named(self, *path, expect):
        """A map whose keys must BE the named set -- not merely contain or count like it."""
        node = self.at(*path)
        if not isinstance(node, dict) or set(node) != set(expect):
            raise Missing(f"{'/'.join(map(str, path))} keys {sorted(node) if isinstance(node, dict) else node!r} != {sorted(expect)}")
        return node

    def listing(self, *path, minimum=1, elements=False):
        """A list. `elements=True` says the caller consumes the VALUES, not just the length,
        and records each scalar element as read -- otherwise a banded `ci95[1]` would look
        unread to the coverage audit while the clause bands it. Left False where a clause
        only counts the list, so that a count is never mistaken for a reading."""
        node = self.at(*path)
        if not isinstance(node, list) or len(node) < minimum:
            raise Missing(f"{'/'.join(map(str, path))} is not a list of >= {minimum}")
        if elements:
            self._record_scalars(node, tuple(map(str, path)))
        return node

    def _record_scalars(self, node, prefix) -> None:
        if isinstance(node, list):
            for index, value in enumerate(node):
                self._record_scalars(value, prefix + (str(index),))
        elif not isinstance(node, dict):
            self.touched.add(prefix)
            self.kinds[prefix] = kind_of(node)


def median(values):
    ordered = sorted(values)
    if not ordered:
        raise Missing("median of an empty population")
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle]))


def agrees(stored, derived, tolerance=AGREEMENT) -> bool:
    return abs(float(stored) - float(derived)) <= tolerance


def build(reports: dict) -> dict:
    """Every clause, derived. A pure function of the loaded reports."""
    r = Reader(reports)
    clauses: list[dict] = []

    def clause(name, predicted, note=""):
        """Run one clause; an absent path FAILS it with the path named."""
        def wrap(fn):
            try:
                detail, ok = fn()
                clauses.append({"clause": name, "predicted": predicted, "measured": detail,
                                "verdict": ("REPORT" if ok == "REPORT"
                                            else "PASS" if ok else "FAIL"),
                                **({"note": note} if note else {})})
            except Missing as absent:
                clauses.append({"clause": name, "predicted": predicted,
                                "measured": f"MISSING or malformed: {absent}",
                                "verdict": "FAIL",
                                "note": note or "a missing measurement is a FAIL, never a "
                                                "no-op"})
        return wrap

    # ------------------------------------------------------------- hygiene and the tripwire
    def built_here(report_key):
        """THE PYTHONPATH TRAP, checked BY CONTENT. `.venv` is shared with the main checkout
        and `autoanim_gnm` is installed editable there, so a report can be a perfect
        measurement OF THE WRONG SOURCE TREE.

        This used to compare `resolved_module` against this worktree's root as a string, and
        Astra's round 8 broke it both ways: with ROOT set to the MAIN checkout the nested
        worktree still starts with it, so the wrong tree passes; and an identical checkout
        anywhere else would fail for its location alone. A path says where a file was; a
        content hash says which code ran. So each build report carries the converter's
        sha256 and the stage it belongs to, and the three stages stay apart -- the historical
        hygiene arm ran the module BEFORE the src change, the tripwire and the candidate run
        the refactored one.
        """
        stage = r.text(report_key, "source_fingerprint", "stage")
        recorded = r.text(report_key, "source_fingerprint", "converter_sha256")
        mode = r.text(report_key, "source_fingerprint", "pelvis_mode")
        if stage not in ("pre_change", "refactored"):
            raise Missing(f"{report_key} names an unknown source stage {stage!r}")
        if stage == "refactored":
            if recorded != EXECUTING_CONVERTER_SHA:
                raise Missing(f"{report_key} was built from converter {recorded[:12]} and "
                              f"this run resolves {EXECUTING_CONVERTER_SHA[:12]}")
        else:
            if recorded == EXECUTING_CONVERTER_SHA:
                raise Missing(f"{report_key} claims the PRE-CHANGE converter but hashes the "
                              "same as the one this run resolves")
            if recorded != PRE_CHANGE_CONVERTER_SHA:
                raise Missing(f"{report_key} names pre-change converter {recorded[:12]}, "
                              f"which is not the retained "
                              f"{(PRE_CHANGE_CONVERTER_SHA or 'MISSING')[:12]}")
        expected = BUILD_STAGES.get(REPORTS[report_key])
        if expected is not None and (stage, mode) != expected:
            raise Missing(f"{report_key} is stage {(stage, mode)}, not the card's {expected}")
        # the path is kept as PROVENANCE and nothing else: it names the tree, it proves none.
        return f"{stage} {recorded[:12]} ({mode})"

    def eight_files(report_key, label):
        r.named(report_key, "hygiene", "delivered_files_vs_shipped", expect=DELIVERED_FILES)
        equal = {}
        for name in DELIVERED_FILES:
            base = (report_key, "hygiene", "delivered_files_vs_shipped", name)
            rebuilt, shipped = r.text(*base, "rebuild"), r.text(*base, "shipped")
            equal[name] = rebuilt == shipped
            # the per-file `identical` is a SUMMARY of the two hashes beside it.
            r.checked(*base, "identical", derived=equal[name])
        # and `all_delivered_files_identical` is a summary of those eight.
        r.checked(report_key, "hygiene", "all_delivered_files_identical",
                  derived=all(equal.values()))
        # the observations the build consumed, and the copy rule that keeps a build from
        # writing through a symlink into the shipped tree.
        premises = {
            "observations before/after": r.flag(report_key, "hygiene",
                                                "observations_byte_identical_before_and_after_the_build"),
            "observations vs shipped": r.flag(report_key, "hygiene",
                                              "observations_byte_identical_to_the_shipped_build"),
            "work copied never symlinked": r.flag(report_key, "work_copied_never_symlinked"),
        }
        failed = sorted(k for k, v in premises.items() if not v)
        if failed:
            raise Missing(f"{report_key} build premises failed: {failed}")
        r.checked(report_key, "verdict",
                  derived="PASS" if all(equal.values()) else "FAIL")
        return (f"{sum(equal.values())} of {len(DELIVERED_FILES)} named files equal {label}, "
                f"from converter {built_here(report_key)}",
                all(equal.values()))

    @clause("hygiene: today's code rebuilds the shipped delivery byte-identically",
            f"all {len(DELIVERED_FILES)} named files present and SHA-equal")
    def _():
        return eight_files("hygiene", "(SHA)")

    @clause("REFACTOR TRIPWIRE (i): mode C held, the refactored function reproduces D9b",
            f"all {len(DELIVERED_FILES)} named files equal, mode held at C_kabsch_pelvis")
    def _():
        detail, ok = eight_files("tripwire", "(SHA)")
        held = r.text("tripwire", "pelvis_mode_held")
        # the requested mode is cross-checked against what the CONVERTER recorded per subject
        recorded = [r.text("tripwire", "diagnostics", "pelvis_frame", i, "mode")
                    for i in range(len(r.listing("tripwire", "diagnostics", "pelvis_frame",
                                                 minimum=2)))]
        if any(mode != held for mode in recorded):
            raise Missing(f"tripwire pelvis_mode_held {held!r} against recorded {recorded}")
        return (f"{detail}, mode held {held!r} and recorded {recorded}",
                ok and held == "C_kabsch_pelvis")

    # ------------------------------------------------------------------------- the oracle
    def arm(seed, name, *path):
        return r.num("oracle", "oracle", "seeds", seed, "arms", name, *path)

    def over_seeds(name, *path):
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        return {seed: arm(seed, name, *path) for seed in ORACLE_SEEDS}

    def deep_equal(path_a, path_b, *, ignore=()):
        """Two subtrees compared LEAF BY LEAF THROUGH THE READER, so every leaf of both counts
        as read and a difference anywhere is named."""
        node = r.at(*path_a)
        if isinstance(node, dict):
            other = r.at(*path_b)
            keys = set(node) - set(ignore)
            if not isinstance(other, dict) or keys != set(other) - set(ignore):
                raise Missing(f"{'/'.join(map(str, path_a))} and {'/'.join(map(str, path_b))}"
                              f" do not carry the same fields")
            for key in sorted(keys):
                deep_equal((*path_a, key), (*path_b, key))
            return
        if isinstance(node, list):
            other = r.listing(*path_b, minimum=0)
            if len(other) != len(node):
                raise Missing(f"{'/'.join(map(str, path_a))} and {'/'.join(map(str, path_b))}"
                              f" differ in length ({len(node)} vs {len(other)})")
            for index in range(len(node)):
                deep_equal((*path_a, index), (*path_b, index))
            return
        if node != r.at(*path_b):
            raise Missing(f"{'/'.join(map(str, path_a))} is {node!r} and "
                          f"{'/'.join(map(str, path_b))} is {r.at(*path_b)!r}")
        r.note_cross_checked(*path_a)
        r.note_cross_checked(*path_b)

    @clause("O1/O2 PREMISES: the six named arms, their populations, and the shipping path's "
            "identity with the named estimator",
            f"all {len(ORACLE_ARMS)} arms present on all {len(ORACLE_SEEDS)} seeds, "
            f"{ORACLE_FRAMES} frames each, and `src_default` == `E_rig_rest_kabsch` leaf for "
            "leaf",
            "the arms the O clauses read are only as good as the fixture under them: a "
            "missing arm, a short take, or a src path that is no longer the estimator S "
            "chose would each leave every O band measuring something else")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        shipping = r.text("oracle", "pelvis_frame_source_in_src")
        built_here("oracle")
        for seed in ORACLE_SEEDS:
            r.named("oracle", "oracle", "seeds", seed, "arms", expect=ORACLE_ARMS)
            for name in ORACLE_ARMS:
                frames = int(r.num("oracle", "oracle", "seeds", seed, "arms", name,
                                   "pelvis_vs_truth_deg", "angle", "n"))
                if frames != ORACLE_FRAMES:
                    raise Missing(f"oracle/{seed}/{name} is scored over {frames} frames, not "
                                  f"the fixture's {ORACLE_FRAMES}")
            # THE SHIPPING PATH IS THE NAMED ESTIMATOR, leaf for leaf and not by label. Every
            # O band is measured on `src_default`; S chose `E_rig_rest_kabsch`. If those two
            # were ever to part company the O clauses would be scoring an arm S never ranked.
            deep_equal(("oracle", "oracle", "seeds", seed, "arms", "src_default"),
                       ("oracle", "oracle", "seeds", seed, "arms", "E_rig_rest_kabsch"),
                       ignore=("arm",))
            mode = r.text("oracle", "oracle", "seeds", seed, "arms", "src_default",
                          "pelvis_report", "mode")
            if mode != shipping:
                raise Missing(f"oracle/{seed}/src_default reports mode {mode!r} against "
                              f"pelvis_frame_source_in_src {shipping!r}")
            # the fixture's own landmark contract: the root landmark IS the hip midpoint and
            # the spine landmark IS the rig's `Spine` joint.
            offset = r.num("oracle", "oracle", "seeds", seed,
                           "root_landmark_is_the_hip_midpoint_to_mm")
            if offset > 1e-6 or not r.flag("oracle", "oracle", "seeds", seed,
                                           "spine_landmark_is_the_rigs_Spine_joint"):
                raise Missing(f"oracle/{seed} landmark contract: root offset {offset} mm")
        return (f"{len(ORACLE_ARMS)} arms x {len(ORACLE_SEEDS)} seeds at {ORACLE_FRAMES} "
                f"frames; src_default == E_rig_rest_kabsch leaf for leaf; ships {shipping!r}",
                shipping == PELVIS_MODE_SHIPPED)

    @clause("O1/O2 PREMISES: the bands the instrument recorded ARE the bands this gate states",
            "the five O bands equal, to the digit")
    def _():
        stated = {"O1_tilt_deg": O1_TILT_DEG, "O1_origin_mm": O1_ORIGIN_MM,
                  "O1_residual_m": O1_RESIDUAL_M, "O2_leg_mm": O2_LEG_MM,
                  "O2_hoist_mm": O2_HOIST_MM}
        r.named("oracle", "bands", expect=stated)
        for name, value in stated.items():
            if r.num("oracle", "bands", name) != value:
                raise Missing(f"oracle/bands/{name} is {r.num('oracle', 'bands', name)}, not "
                              f"the gate's {value}")
        return f"{len(stated)} bands equal", True

    @clause("REFACTOR TRIPWIRE (ii): the SAME six-body C execution read against exact rig truth",
            f"6.865 deg on every seed, still outside O1's {O1_TILT_DEG} deg band",
            "ONE execution, two references, two verdicts")
    def _():
        values = over_seeds("C_soma_template", "pelvis_vs_truth_deg", "angle", "median")
        return (f"{min(values.values()):.4f}-{max(values.values()):.4f} deg over "
                f"{len(values)} seeds", min(values.values()) > O1_TILT_DEG)

    @clause("O1 pelvis vs truth, every seed, every frame",
            f"<= {O1_TILT_DEG} deg (from 6.865) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "pelvis_vs_truth_deg", "angle", "max")
        return (f"max {max(values.values())} deg over {len(values)} seeds",
                max(values.values()) <= O1_TILT_DEG)

    @clause("O1 `Spine` origin miss, hoist-subtracted",
            f"<= {O1_ORIGIN_MM} mm (from 21-28) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "spine_origin_miss_mm", "hoist_subtracted", "max")
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 `Hips` origin miss, hoist-subtracted",
            f"<= {O1_ORIGIN_MM} mm (from 10) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "hips_origin_miss_mm", "hoist_subtracted", "max")
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 torso on the unhoisted frames, ABSOLUTE row", "0.00 (from 8.98-12.09)")
    def _():
        values = over_seeds("src_default", "ABSOLUTE_groups_mm", "unhoisted_frames", "torso")
        return (f"max {max(values.values())} over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 unnormalised three-point positional residual", f"<= {O1_RESIDUAL_M} m",
            "the clause that discriminates the wrong-origin control")
    def _():
        values = over_seeds("src_default", "three_point_residual_m", "max")
        return (f"max {max(values.values()):.3e} m over {len(values)} seeds",
                max(values.values()) <= O1_RESIDUAL_M)

    @clause("must-fail: the WRONG-ORIGIN template (a KNOWN BLINDNESS realised)",
            "invisible to the tilt band AND caught by the residual band",
            "with the residual zeroed this control is caught by NOTHING, and the residual "
            "band it is the sole evidence for means nothing either")
    def _():
        tilt = over_seeds("wrong_origin", "pelvis_vs_truth_deg", "angle", "max")
        residual = over_seeds("wrong_origin", "three_point_residual_m", "max")
        return (f"tilt max {max(tilt.values())} deg (inside {O1_TILT_DEG}), residual min "
                f"{1e3 * min(residual.values()):.1f} mm (outside {O1_RESIDUAL_M} m)",
                max(tilt.values()) <= O1_TILT_DEG
                and min(residual.values()) > O1_RESIDUAL_M)

    @clause("must-fail: a pelvis frozen upright (D7's control)", "fails O1's tilt band")
    def _():
        values = over_seeds("frozen_upright", "pelvis_vs_truth_deg", "angle", "median")
        return (f"{min(values.values()):.4f}-{max(values.values()):.4f} deg",
                min(values.values()) > O1_TILT_DEG)

    @clause("O2 legs, feet and toes vs the shipped build's FK",
            f"<= {O2_LEG_MM} mm on all {len(ORACLE_SEEDS)} seeds",
            "bit-identity is NOT claimed: a pelvis frame is whole-take")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {}
        for seed in ORACLE_SEEDS:
            base = ("oracle", "oracle", "seeds", seed, "O2_vs_baseline")
            # the comparison's own premises: which arm it scored, over how many samples, and
            # that it makes NO bit-identity claim (a pelvis frame is whole-take).
            scored = r.text(*base, "arm")
            if scored != "src_default":
                raise Missing(f"O2/{seed} scored arm {scored!r}, not the shipping src path")
            samples = int(r.num(*base, "leg_foot_toe_move_mm", "n"))
            if samples != ORACLE_FRAMES * 8:      # 8 joints: upper leg, lower leg, foot, toes
                raise Missing(f"O2/{seed} moved {samples} samples, not "
                              f"{ORACLE_FRAMES * 8} (8 joints x {ORACLE_FRAMES} frames)")
            if r.flag(*base, "bit_identity_claimed"):
                raise Missing(f"O2/{seed} claims bit identity; the card does not")
            values[seed] = r.checked(*base, "leg_foot_toe_max_mm",
                                     derived=r.num(*base, "leg_foot_toe_move_mm", "max"),
                                     # the summary is written to 5 dp and the block it
                                     # summarises to 4, so they agree to the rounding
                                     tolerance=1e-4)
            r.checked(*base, "within_0_1_mm", derived=values[seed] <= O2_LEG_MM)
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O2_LEG_MM)

    @clause("O2 contacts identical on the oracle bodies",
            f"identical on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {s: r.flag("oracle", "oracle", "seeds", s, "O2_vs_baseline",
                            "contacts_identical") for s in ORACLE_SEEDS}
        return f"{all(values.values())} over {len(values)} seeds", all(values.values())

    @clause("O2 hoist change", f"<= {O2_HOIST_MM} mm on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {}
        for seed in ORACLE_SEEDS:
            base = ("oracle", "oracle", "seeds", seed, "O2_vs_baseline", "hoist_change_mm")
            frames = int(r.num(*base, "n"))
            if frames != ORACLE_FRAMES:
                raise Missing(f"O2 hoist/{seed} is over {frames} frames, not {ORACLE_FRAMES}")
            values[seed] = r.num(*base, "max")
            r.checked("oracle", "oracle", "seeds", seed, "O2_vs_baseline",
                      "hoist_within_0_05_mm", derived=values[seed] <= O2_HOIST_MM)
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O2_HOIST_MM)

    @clause("O3 the D3 gate's own leg-root-ALIGNED gauge, arms (REPORTED)",
            "1.32-2.72 -> 0.07-0.60",
            "that gauge is blind to a root move; no band reads it")
    def _():
        ours = over_seeds("src_default", "ALIGNED_rc_score_groups_mm", "arms")
        before = over_seeds("C_soma_template", "ALIGNED_rc_score_groups_mm", "arms")
        return (f"{min(before.values())}-{max(before.values())} -> "
                f"{min(ours.values())}-{max(ours.values())}", "REPORT")

    # ------------------------------------------------------ the two recorded STOPs, derived
    def sigma1_metric(seed, arm_name, population, metric):
        """One body's number on the CARD'S OWN FIXTURE, population validated -- the same
        rule as `body_metric` applies to the reread, because the recorded STOP deserves the
        denominators the PROCEED gets."""
        frames, pairs = S_POPULATION[population]
        base = ("sigma1", "bodies", seed, "arms", arm_name, population)
        if (int(r.num(*base, "n_frames")) != frames
                or int(r.num(*base, "n_pairs")) != pairs):
            raise Missing(f"sigma1/{seed}/{arm_name}/{population} population is "
                          f"{int(r.num(*base, 'n_frames'))}/{int(r.num(*base, 'n_pairs'))}, "
                          f"not the frozen {frames}/{pairs}")
        return r.num(*base, metric)

    @clause("S at the CARD'S OWN FIXTURE (sigma 1.0): the frozen-pitch follower >= 2x on EVERY body",
            f">= {FOLLOWER_RATIO}x on all {len(ORACLE_SEEDS)} bodies",
            "the step STOPPED here; `selector.json` is immutable")
    def _():
        rows = r.named("sigma1", "frozen_pitch_follower_bent_tercile", expect=ORACLE_SEEDS)
        if r.num("sigma1", "sigma_scale") != 1.0:
            raise Missing("selector.json is not at sigma 1.0")
        winner_arm = r.text("sigma1", "winner", "arm")
        if {"a_kabsch_guarded": "E_rig_rest_kabsch",
                "b_hipline_guarded": "D_rig_rest_hipline"}.get(winner_arm) != r.text(
                    "sigma1", "winner", "mode"):
            raise Missing("sigma1 winner/arm and winner/mode disagree")
        ratios = {}
        for seed in ORACLE_SEEDS:
            base = ("sigma1", "frozen_pitch_follower_bent_tercile", seed)
            # BOTH TERMS FROM THE BODY ROWS, exactly as the reread's follower is read.
            # Astra's round 8 set the winner's own bent-tercile error to 1 deg on all six
            # bodies: the constituent ratios become 15.5-21.1x and clear the stop everywhere,
            # and the gate -- reading this duplicated table -- kept reporting the STOP. A
            # recorded stop read off a copy is a recorded copy, not a recorded stop.
            follower = sigma1_metric(seed, "frozen_pitch_follower", "bent_tercile",
                                     "i_orientation_deg")
            winner = sigma1_metric(seed, winner_arm, "bent_tercile", "i_orientation_deg")
            if winner <= 0.0:
                raise Missing(f"sigma1/{seed}/{winner_arm}/bent_tercile is not positive")
            ratios[seed] = follower / winner
            for label, derived in (("follower_i_deg", follower), ("winner_i_deg", winner),
                                   ("ratio", ratios[seed])):
                r.checked(*base, label, derived=derived, tolerance=1e-2)
            r.checked(*base, "ratio_at_least_2x", derived=ratios[seed] >= FOLLOWER_RATIO)
            r.checked(*base, "at_least_2_deg", derived=follower >= FOLLOWER_FLOOR_DEG)
        # EVERY ARM'S POPULATION on the pre-registered fixture too. The stop this clause
        # records is only as good as the denominators under it.
        for seed in ORACLE_SEEDS:
            r.named("sigma1", "bodies", seed, "arms", expect=REREAD_ARMS)
            for arm_name in REREAD_ARMS:
                for population in POPULATIONS:
                    sigma1_metric(seed, arm_name, population, "i_orientation_deg")
        below = [s for s, v in ratios.items() if v < FOLLOWER_RATIO]
        discriminated = not below and all(
            r.num("sigma1", "frozen_pitch_follower_bent_tercile", s, "follower_i_deg")
            >= FOLLOWER_FLOOR_DEG for s in ORACLE_SEEDS)
        r.checked("sigma1", "follower_discriminated_on_every_body", derived=discriminated)
        # S's verdict in this file, DERIVED from the clause that decided it. If the follower
        # ever discriminated here the recorded stop would have lost its cause, and the file
        # would be claiming a stop it no longer measures.
        r.checked("sigma1", "S_verdict", derived="PROCEED" if discriminated else "STOP")
        return (f"{min(ratios.values()):.3f}-{max(ratios.values()):.3f}x; {len(below)} of "
                f"{len(ratios)} below {FOLLOWER_RATIO}x", not below)

    def evaluation_median(index):
        """One evaluation's statistic, DERIVED from the six bodies it is a median of -- and
        cross-checked in three places.

        The stored `median_of_six_guard_kept_sd_mm` is what both calibration clauses banded,
        and Astra's round 7 moved one body's value in evaluation 9 from 8.9585 to 100: the
        stored aggregate was untouched, the derived median rises to 9.028, and the accepted
        sigma is then 0.264 mm OUTSIDE the tolerance the admissibility rule requires. The six
        values are also written independently into the amended file, so the two records must
        agree as well.
        """
        base = ("calibration", "calibration", "bisection", "evaluations", index)
        sigma = r.num(*base, "sigma_scale")
        bodies = r.named(*base, "per_body_mm", expect=ORACLE_SEEDS)
        values = {seed: r.num(*base, "per_body_mm", seed) for seed in ORACLE_SEEDS}
        mirror = r.at("admissibility", "all_six_body_values_per_evaluation_mm")
        if not isinstance(mirror, dict):
            raise Missing("admissibility/all_six_body_values_per_evaluation_mm")
        keys = [k for k in mirror if abs(float(k) - sigma) <= 5e-7]
        if len(keys) != 1:
            raise Missing(f"the amended file records {len(keys)} evaluations at sigma "
                          f"{sigma}, not exactly one")
        r.named("admissibility", "all_six_body_values_per_evaluation_mm", keys[0],
                expect=ORACLE_SEEDS)
        for seed in ORACLE_SEEDS:
            twin = r.num("admissibility", "all_six_body_values_per_evaluation_mm", keys[0],
                         seed)
            if not agrees(twin, values[seed], 1e-6):
                raise Missing(f"sigma {sigma} body {seed}: the calibration records "
                              f"{values[seed]} and the amended file {twin}")
        derived = median(list(values.values()))
        stored = r.num(*base, "median_of_six_guard_kept_sd_mm")
        if not agrees(stored, derived, 1e-3):
            raise Missing(f"evaluation {index} at sigma {sigma} stores {stored} mm against "
                          f"{derived} derived from its {len(bodies)} bodies")
        return sigma, derived

    @clause("the amended card's FIXTURE CALIBRATION, under its own frozen monotonicity precondition",
            "monotone across the evaluations",
            "the step STOPPED again; `selector-calibrated.json` is immutable")
    def _():
        evaluations = r.listing("calibration", "calibration", "bisection", "evaluations",
                                minimum=2)
        # the search's own parameters ARE the ones this gate states, before any statistic is
        # read out of it.
        for name, expect in (("target_mm", CALIBRATION_TARGET_MM),
                             ("tolerance_mm", CALIBRATION_TAU_MM)):
            if r.num("calibration", "calibration", "bisection", name) != expect:
                raise Missing(f"calibration/{name} is not the gate's {expect}")
        bracket = [float(x) for x in r.listing("calibration", "calibration", "bisection",
                                               "bracket", minimum=2, elements=True)]
        if tuple(bracket) != CALIBRATION_BRACKET:
            raise Missing(f"calibration bracket {bracket} is not {CALIBRATION_BRACKET}")
        budget = int(r.num("calibration", "calibration", "bisection", "maximum_evaluations"))
        if len(evaluations) > budget:
            raise Missing(f"{len(evaluations)} evaluations against a budget of {budget}")
        pairs = sorted(evaluation_median(i) for i in range(len(evaluations)))
        values = [value for _sigma, value in pairs]
        worst = max((values[i] - values[j] for i in range(len(values))
                     for j in range(i + 1, len(values)) if values[j] < values[i]), default=0.0)
        r.checked("calibration", "calibration", "bisection",
                  "monotone_across_the_evaluations", derived=worst <= 0.0)
        r.checked("calibration", "calibration", "bisection", "largest_violation_mm",
                  derived=worst, tolerance=1e-3)
        violations = r.listing("calibration", "calibration", "bisection",
                               "monotonicity_violations", minimum=0)
        if bool(violations) != (worst > 0.0):
            raise Missing(f"{len(violations)} recorded violations against a derived largest "
                          f"decrease of {worst}")
        # each recorded violation against the pairs it claims to be between
        by_sigma = dict(pairs)
        for index in range(len(violations)):
            spot = ("calibration", "calibration", "bisection", "monotonicity_violations",
                    index)
            lower, higher = r.num(*spot, "sigma_lower"), r.num(*spot, "sigma_higher")
            for sigma, field in ((lower, "sd_lower_mm"), (higher, "sd_higher_mm")):
                matched = [v for k, v in by_sigma.items() if abs(k - sigma) <= 5e-7]
                if len(matched) != 1:
                    raise Missing(f"a violation names sigma {sigma}, which is not one "
                                  "evaluation")
                r.checked(*spot, field, derived=matched[0], tolerance=1e-3)
            r.checked(*spot, "decrease_mm",
                      derived=r.num(*spot, "sd_lower_mm") - r.num(*spot, "sd_higher_mm"),
                      tolerance=1e-3)
        # the recorded STATUS of the search, and S's own verdict in the same file
        r.checked("calibration", "calibration", "bisection", "status",
                  derived="UNREACHABLE" if worst > 0.0 else "REACHED")
        r.checked("calibration", "S_verdict", derived="STOP" if worst > 0.0 else "PROCEED")
        # the sigma-ordered table beside the evaluations must BE the same pairs.
        table = r.listing("calibration", "calibration", "bisection",
                          "evaluated_in_sigma_order", minimum=2, elements=True)
        if len(table) != len(pairs) or any(
                abs(float(row[0]) - sigma) > 5e-7 or not agrees(float(row[1]), value, 1e-3)
                for row, (sigma, value) in zip(table, pairs)):
            raise Missing("evaluated_in_sigma_order disagrees with the evaluations it orders")
        return (f"largest earlier-to-later decrease {worst:.4f} mm over {len(values)} "
                "evaluations", worst <= 0.0)

    @clause("the SAME frozen evaluations under Astra round 7's amended admissibility rule",
            f"A <= {CALIBRATION_TAU_MM} mm, B <= 1 sign change, C an accepted sigma whose OWN "
            f"statistic is within {CALIBRATION_TAU_MM} mm of the target and inside the bracket",
            "an OBSERVED TOLERANCE MATCH, never monotonicity or uniqueness. POST HOC.")
    def _():
        evaluations = r.listing("calibration", "calibration", "bisection", "evaluations",
                                minimum=2)
        target = r.num("admissibility", "admissibility", "target_mm")
        bracket = r.listing("admissibility", "admissibility", "bracket", minimum=2,
                            elements=True)
        # the amended rule RETAINS the target, the tolerance, the bracket and the budget: all
        # four are read here, so an amendment that quietly moved one would be a FAIL.
        if (target != CALIBRATION_TARGET_MM
                or r.num("admissibility", "admissibility", "tolerance_mm") != CALIBRATION_TAU_MM
                or tuple(float(x) for x in bracket) != CALIBRATION_BRACKET):
            raise Missing("the amended rule does not retain the frozen target, tolerance and "
                          "bracket")
        budget = int(r.num("admissibility", "admissibility", "evaluation_budget"))
        order = r.listing("admissibility", "admissibility",
                          "replay_of_the_frozen_stopping_rule", "evaluation_order_exact",
                          minimum=2)
        if len(order) != len(evaluations) or len(order) > budget:
            raise Missing(f"the replay lists {len(order)} evaluations against "
                          f"{len(evaluations)} recorded and a budget of {budget}")
        if not r.flag("admissibility", "admissibility",
                      "replay_of_the_frozen_stopping_rule",
                      "replay_reproduced_every_recorded_evaluation"):
            raise Missing("the replay did not reproduce every recorded evaluation")
        # EVERY evaluation's statistic derived from its own six bodies, because the rule
        # bands all of them (A and B) and not only the accepted one (C).
        pairs = sorted(evaluation_median(i) for i in range(len(evaluations)))
        values = [value for _sigma, value in pairs]
        worst = max((values[i] - values[j] for i in range(len(values))
                     for j in range(i + 1, len(values)) if values[j] < values[i]), default=0.0)
        signs = [(-1 if v < target else 1) for v in values if v != target]
        changes = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
        accepted = r.num("admissibility", "admissibility",
                         "replay_of_the_frozen_stopping_rule", "accepted_sigma_scale_exact")
        display = r.num("admissibility", "admissibility",
                        "replay_of_the_frozen_stopping_rule", "accepted_sigma_scale_display")
        if abs(display - accepted) > 5e-7:
            raise Missing(f"the accepted sigma displays as {display} and is {accepted}")
        matched = [value for sigma, value in pairs if abs(sigma - accepted) <= 5e-7]
        if not matched:
            raise Missing("the accepted sigma is not among the frozen evaluations")
        inside = (abs(matched[0] - target) <= CALIBRATION_TAU_MM
                  and float(bracket[0]) <= accepted <= float(bracket[1]))
        # THE RULE'S OWN THREE CHECKS, against the three this clause just derived. The
        # amended file recomputes A, B and C itself; if its arithmetic and the gate's
        # disagree, one of them is wrong and the gate does not get to pick.
        checks = ("A_no_earlier_to_later_decrease_over_tau",
                  "B_at_most_one_sign_change_of_statistic_minus_target",
                  "C_the_unchanged_stopping_rule_found_a_value_within_tau")
        r.named("admissibility", "admissibility", "checks", expect=checks)
        spot = ("admissibility", "admissibility", "checks", checks[0])
        r.checked(*spot, "largest_decrease_mm", derived=worst, tolerance=1e-3)
        r.checked(*spot, "tau_mm", derived=CALIBRATION_TAU_MM)
        r.checked(*spot, "passes", derived=worst <= CALIBRATION_TAU_MM)
        spot = ("admissibility", "admissibility", "checks", checks[1])
        r.checked(*spot, "sign_changes", derived=changes)
        r.checked(*spot, "passes", derived=changes <= 1)
        spot = ("admissibility", "admissibility", "checks", checks[2])
        r.checked(*spot, "accepted_sigma_scale_exact", derived=accepted)
        r.checked(*spot, "passes", derived=inside)
        # the replay's recorded order IS the order the evaluations were made in, and the
        # top-level copy of it is the same list again
        for index in range(len(order)):
            sigma = r.num("calibration", "calibration", "bisection", "evaluations", index,
                          "sigma_scale")
            for where in (("admissibility", "admissibility",
                           "replay_of_the_frozen_stopping_rule", "evaluation_order_exact"),
                          ("admissibility", "evaluation_order_exact")):
                if abs(r.num(*where, index) - sigma) > 5e-7:
                    raise Missing(f"{'/'.join(where)}[{index}] is not the sigma evaluation "
                                  f"{index} was made at")
                r.note_cross_checked(*where, str(index))
        r.checked("admissibility", "the_accepted_sigma", "exact_evaluated_value",
                  derived=accepted)
        r.checked("admissibility", "the_accepted_sigma", "display_rounding_six_places",
                  derived=display)
        if not r.text("admissibility", "S_status").startswith("PENDING"):
            raise Missing("the amended calibration file claims something other than PENDING "
                          "for S; calibration REACHED is not S PROCEED")
        reached = "REACHED" if (worst <= CALIBRATION_TAU_MM and changes <= 1
                                and inside) else "NOT REACHED"
        r.checked("admissibility", "verdict", derived=reached)
        r.checked("admissibility", "admissibility", "verdict", derived=reached)
        seeds = [str(int(x)) for x in r.listing("admissibility", "provenance", "seeds",
                                                minimum=len(ORACLE_SEEDS), elements=True)]
        if tuple(seeds) != ORACLE_SEEDS:
            raise Missing(f"the amended file's provenance seeds {seeds} are not the six")
        return (f"A {worst:.4f} mm, B {changes} sign change(s), C sigma {accepted} -> "
                f"{matched[0]} mm against target {target}",
                worst <= CALIBRATION_TAU_MM and changes <= 1 and inside)

    # ------------------------------------------------------------------- S, the reread
    def body_metric(seed, arm_name, population, metric):
        """ONE body's number, with its POPULATION validated first. EVERY arm goes through
        here -- the winner, the loser, C-on-SOMA and the CONTROLS alike.

        A median says nothing without the population it is over. Astra's round 6 set one
        body's `n_frames` to 0 and the medians sailed through, which put the check in
        `per_body`; round 7 then set the FOLLOWER's `n_frames` and `n_pairs` to 0, and the
        follower clause -- which reaches into the body rows directly rather than through the
        aggregate -- never saw it. The check belongs at the single point where a body's
        number is read, not at one of its callers.
        """
        r.named("reread", "bodies", expect=ORACLE_SEEDS)
        frames, pairs = S_POPULATION[population]
        base = ("reread", "bodies", seed, "arms", arm_name, population)
        if (int(r.num(*base, "n_frames")) != frames
                or int(r.num(*base, "n_pairs")) != pairs):
            raise Missing(
                f"bodies/{seed}/{arm_name}/{population} population is "
                f"{int(r.num(*base, 'n_frames'))}/{int(r.num(*base, 'n_pairs'))}, not "
                f"the frozen {frames}/{pairs}")
        return r.num(*base, metric)

    def per_body(arm_name, population, metric):
        """THE AGGREGATE'S CONSTITUENTS, each with its population validated."""
        return {seed: body_metric(seed, arm_name, population, metric)
                for seed in ORACLE_SEEDS}

    @clause("S REREAD: at the calibration's EXACT accepted sigma, on every arm's full "
            "population",
            f"sigma == the accepted {ACCEPTED_SIGMA!r}, and {S_POPULATION['whole_take'][0]}/"
            f"{S_POPULATION['whole_take'][1]} and {S_POPULATION['bent_tercile'][0]}/"
            f"{S_POPULATION['bent_tercile'][1]} on all "
            f"{len(REREAD_ARMS)} named arms of all {len(ORACLE_SEEDS)} bodies",
            "S is decided in this file, so WHICH file it is matters as much as what it "
            "says. The reread is at the calibration's own accepted sigma and is NOT the "
            "card's pre-registered fixture; `selector.json` at sigma 1.0 is.")
    def _():
        accepted = r.num("admissibility", "admissibility",
                         "replay_of_the_frozen_stopping_rule", "accepted_sigma_scale_exact")
        sigma = r.num("reread", "sigma_scale")
        if sigma != accepted:
            raise Missing(f"the reread is at sigma {sigma!r} and the calibration accepted "
                          f"{accepted!r}")
        if r.text("reread", "sigma_scale_repr") != repr(sigma):
            raise Missing(f"reread/sigma_scale_repr disagrees with its own sigma {sigma!r}")
        r.checked("reread", "is_the_reread_at_the_calibrated_sigma", derived=True)
        r.checked("reread", "is_the_pre_registered_fixture", derived=False)
        if r.num("sigma1", "sigma_scale") != 1.0:
            raise Missing("selector.json is not at sigma 1.0")
        r.checked("sigma1", "is_the_pre_registered_fixture", derived=True)
        # EVERY ARM, THE CONTROLS INCLUDED. `body_metric` validates the population of the
        # arms the merge rule reads; this reads the rest, so an arm cannot be scored on a
        # population that was never checked merely because no clause happens to band it.
        for seed in ORACLE_SEEDS:
            r.named("reread", "bodies", seed, "arms", expect=REREAD_ARMS)
            for arm_name in REREAD_ARMS:
                for population in POPULATIONS:
                    body_metric(seed, arm_name, population, "i_orientation_deg")
        return (f"sigma {sigma!r}; {len(REREAD_ARMS)} arms x {len(ORACLE_SEEDS)} bodies x "
                f"{len(POPULATIONS)} populations at the frozen sizes", True)

    def aggregate(arm_name, population, metric):
        derived = median(list(per_body(arm_name, population, metric).values()))
        stored = r.num("reread", "aggregated_median_of_six", arm_name, population, metric)
        if not agrees(stored, derived, 1e-3):
            raise Missing(f"aggregated_median_of_six/{arm_name}/{population}/{metric} "
                          f"stores {stored} against {derived} derived from the six bodies")
        return derived

    @clause("S REREAD: (a) vs (b), all three metrics, both populations",
            "one better-or-tied everywhere and strictly better somewhere, else SPLIT",
            "the six cells are recomputed from the per-body medians, and the implied winner "
            "is cross-checked against the mode the file says it ships")
    def _():
        cells, rebuilt = [], {}
        for population in POPULATIONS:
            for metric, tie in METRICS:
                b_value = aggregate("b_hipline_guarded", population, metric)
                a_value = aggregate("a_kabsch_guarded", population, metric)
                cell = ("tied" if abs(b_value - a_value) <= tie
                        else "better" if b_value < a_value else "worse")
                cells.append(cell)
                rebuilt[f"b_vs_a__{population}__{metric}"] = cell
        stored = r.named("reread", "b_vs_a", expect=rebuilt)
        if stored != rebuilt:
            raise Missing(f"b_vs_a stores {stored} against {rebuilt} derived")
        b_wins = all(c in ("better", "tied") for c in cells) and "better" in cells
        a_wins = all(c in ("worse", "tied") for c in cells) and "worse" in cells
        implied = ("D_rig_rest_hipline" if b_wins
                   else "E_rig_rest_kabsch" if a_wins else None)
        shipped = r.text("reread", "winner", "mode")
        arm_name = r.text("reread", "winner", "arm")
        if {"a_kabsch_guarded": "E_rig_rest_kabsch",
            "b_hipline_guarded": "D_rig_rest_hipline"}.get(arm_name) != shipped:
            raise Missing(f"winner/arm {arm_name!r} and winner/mode {shipped!r} disagree")
        status = r.text("reread", "S_verdict")
        return (f"recomputed cells {cells}; implies {implied}; ships {shipped}; "
                f"S_verdict {status}",
                implied is not None and implied == shipped and status == "PROCEED")

    @clause("S REREAD: the winner strictly better than C-on-SOMA on (i), both populations",
            "strictly better on (i), better-or-tied on (ii) and (iii)")
    def _():
        winner = r.text("reread", "winner", "arm")
        beats, detail, cells = [], [], {}
        for population in POPULATIONS:
            for metric, tie in METRICS:
                ours = aggregate(winner, population, metric)
                theirs = aggregate("C_on_SOMA", population, metric)
                cells[f"{population}__{metric}"] = (
                    "tied" if abs(ours - theirs) <= tie
                    else "better" if ours < theirs else "worse")
                if metric == "i_orientation_deg":
                    beats.append(ours < theirs)
                    detail.append(f"{population} {ours:.5f} vs {theirs:.5f}")
                else:
                    beats.append(ours <= theirs + tie)
        # the stored comparison map is rebuilt cell by cell, and the summary it feeds with it
        stored = r.named("reread", "winner_vs_C_on_SOMA", expect=cells)
        if stored != cells:
            raise Missing(f"winner_vs_C_on_SOMA stores {stored} against {cells} derived")
        r.checked("reread", "winner_beats_the_constant_it_removes", derived=all(beats))
        return "; ".join(detail), all(beats)

    @clause("S REREAD: the frozen-pitch follower >= 2x the winner AND >= 2 deg, on EVERY body",
            f">= {FOLLOWER_RATIO}x and >= {FOLLOWER_FLOOR_DEG} deg on all "
            f"{len(ORACLE_SEEDS)} bodies",
            "the clause that stopped the step at sigma 1.0")
    def _():
        rows = r.named("reread", "frozen_pitch_follower_bent_tercile", expect=ORACLE_SEEDS)
        winner_arm = r.text("reread", "winner", "arm")
        ok, ratios = True, {}
        for seed in rows:
            # BOTH TERMS COME FROM THE BODIES, not from the follower table's own copies.
            # Astra's round 6 set the winner's bent-tercile error on one body to 100 deg:
            # the six-body median barely moved, the duplicated `winner_i_deg` was left alone,
            # and that body's true ratio fell to 0.146 unnoticed.
            # AND BOTH GO THROUGH THE POPULATION CHECK. Astra's round 7 set the follower's
            # own `n_frames` and `n_pairs` to 0 on one body: the ratio was unchanged, because
            # a median over an empty population is still whatever the file says it is.
            follower = body_metric(seed, "frozen_pitch_follower", "bent_tercile",
                                   "i_orientation_deg")
            winner = body_metric(seed, winner_arm, "bent_tercile", "i_orientation_deg")
            if winner <= 0.0:
                raise Missing(f"bodies/{seed}/{winner_arm}/bent_tercile is not positive")
            ratios[seed] = follower / winner
            base = ("reread", "frozen_pitch_follower_bent_tercile", seed)
            for label, derived in (("follower_i_deg", follower), ("winner_i_deg", winner),
                                   ("ratio", ratios[seed])):
                r.checked(*base, label, derived=derived, tolerance=1e-2)
            r.checked(*base, "ratio_at_least_2x", derived=ratios[seed] >= FOLLOWER_RATIO)
            r.checked(*base, "at_least_2_deg", derived=follower >= FOLLOWER_FLOOR_DEG)
            ok &= (ratios[seed] >= FOLLOWER_RATIO and follower >= FOLLOWER_FLOOR_DEG)
        r.checked("reread", "follower_discriminated_on_every_body", derived=ok)
        return (f"{len(rows)} bodies; recomputed ratios {min(ratios.values()):.3f}-"
                f"{max(ratios.values()):.3f}x", ok)

    @clause("G1 (missing-only): identical masks and retained samples => bit-identical ARRAYS",
            f"holds on all {len(ORACLE_SEEDS)} named bodies",
            "an EQUIVALENCE and an error measurement; no superiority claim")
    def _():
        rows = r.named("reread", "G1_missing_only", "bodies", expect=ORACLE_SEEDS)
        holds, differing, unconditional, with_rejection = True, 0, True, []
        for seed in ORACLE_SEEDS:
            base = ("reread", "G1_missing_only", "bodies", seed)
            masks = r.flag(*base, "effective_masks_identical")
            arrays = r.flag(*base, "interpolated_arrays_bit_identical")
            differing += (not masks)
            holds &= ((not masks) or arrays)
            unconditional &= arrays
            # every per-body summary derived from the two booleans and the two lists it
            # summarises, and the demoted count from the frames it counts.
            r.checked(*base, "claim_holds_as_amended", derived=(not masks) or arrays)
            demoted = r.listing(*base, "guard_additionally_demoted", minimum=0)
            r.checked(*base, "guard_demoted_count", derived=len(demoted))
            rejections = r.listing(*base, "additional_rejections", minimum=0)
            if len(rejections) != len(demoted):
                raise Missing(f"G1/{seed} demotes {len(demoted)} frames and diagnoses "
                              f"{len(rejections)} rejections")
            if rejections:
                with_rejection.append(seed)
        r.checked("reread", "G1_missing_only", "claim_holds_on_every_body_as_amended",
                  derived=holds)
        r.checked("reread", "G1_missing_only", "unconditional_identity_on_every_body",
                  derived=unconditional)
        if sorted(r.listing("reread", "G1_missing_only",
                            "bodies_with_an_additional_rejection", minimum=0)) != with_rejection:
            raise Missing("G1/bodies_with_an_additional_rejection disagrees with the bodies "
                          f"that carry one ({with_rejection})")
        return (f"{len(rows)} bodies; {differing} have a different effective mask; identity "
                f"holds wherever the masks agree; unconditional identity {unconditional}",
                holds)

    @clause("G2 (finite-only): the guard beats the unguarded winner on BOTH (i) and (ii), every body",
            f"both metrics, all {len(ORACLE_SEEDS)} bodies and the median over them",
            "where the guard EARNS its place")
    def _():
        rows = r.named("reread", "G2_finite_only", "bodies", expect=ORACLE_SEEDS)
        wins, samples = True, {}
        for seed in ORACLE_SEEDS:
            base = ("reread", "G2_finite_only", "bodies", seed)
            # the corruption's own population, by the card's law: 20 % of the fixture.
            corrupted = int(r.num(*base, "corrupted_frames"))
            if corrupted != ORACLE_FRAMES // 5:
                raise Missing(f"G2/{seed} corrupts {corrupted} frames, not the card's "
                              f"{ORACLE_FRAMES // 5}")
            if int(r.num(*base, "transition_pairs")) < 1:
                raise Missing(f"G2/{seed} has no transition pairs")
            missed = r.listing(*base, "guard_missed_corrupted_frames", minimum=0)
            r.checked(*base, "guard_miss_rate", derived=len(missed) / corrupted,
                      tolerance=1e-3)
            for key, metric in (("i", "i_on_corrupted_frames_deg"),
                                ("ii", "ii_on_transition_pairs_deg")):
                r.checked(*base, f"guard_better_on_{key}",
                          derived=r.num(*base, "guarded", metric)
                          < r.num(*base, "unguarded", metric))
        for key, metric in (("i", "i_on_corrupted_frames_deg"),
                            ("ii", "ii_on_transition_pairs_deg")):
            guarded = [r.num("reread", "G2_finite_only", "bodies", s, "guarded", metric)
                       for s in rows]
            unguarded = [r.num("reread", "G2_finite_only", "bodies", s, "unguarded", metric)
                         for s in rows]
            wins &= all(g < u for g, u in zip(guarded, unguarded))
            samples[key] = (median(guarded), median(unguarded))
            for label, derived in (("guarded", samples[key][0]),
                                   ("unguarded", samples[key][1])):
                stored = r.num("reread", "G2_finite_only", "median_of_six",
                               f"{label}_{key}_deg")
                if not agrees(stored, derived, 1e-3):
                    raise Missing(f"G2 median_of_six/{label}_{key}_deg stores {stored} "
                                  f"against {derived:.5f} derived from {len(rows)} bodies")
            wins &= samples[key][0] < samples[key][1]
            r.checked("reread", "G2_finite_only", f"guard_wins_{key}_on_the_median",
                      derived=samples[key][0] < samples[key][1])
            r.named("reread", "G2_finite_only", f"guard_wins_per_body_{key}",
                    expect=ORACLE_SEEDS)
            for seed in ORACLE_SEEDS:
                r.checked("reread", "G2_finite_only", f"guard_wins_per_body_{key}", seed,
                          derived=r.flag("reread", "G2_finite_only", "bodies", seed,
                                         f"guard_better_on_{key}"))
        r.checked("reread", "G2_finite_only", "guard_wins_both", derived=wins)
        return (f"{len(rows)} bodies; medians (i) {samples['i'][0]:.5f} vs "
                f"{samples['i'][1]:.5f} deg, (ii) {samples['ii'][0]:.5f} vs "
                f"{samples['ii'][1]:.5f} deg", wins)

    @clause("the world-vertical control against the truth PELVIS's own tilt", "REPORT",
            "S's stops are unchanged; the follower carries the argument")
    def _():
        base = ("reread", "world_vertical_vs_truth_tilt")
        control = r.num(*base, "world_vertical_i_bent_deg")
        tilt = r.num(*base, "truth_PELVIS_bent_tilt_median_deg")
        trunk = r.num(*base, "truth_trunk_bent_tilt_median_deg")
        winner = r.num(*base, "winner_i_bent_deg")
        r.checked(*base, "stated_limitation_if_within_2_deg_of_the_tilt",
                  derived=abs(control - tilt) < 2.0)
        return (f"{control} deg against the truth pelvis's {tilt} deg (trunk {trunk}); the "
                f"winner reads {winner} deg; limitation applies: {abs(control - tilt) < 2.0}",
                "REPORT")

    # ------------------------------------------------------------------- the delivery
    @clause("the delivery: BOTH landmark arrays byte-identical (the same denominator)",
            f"raw AND smoothed identical on {len(PERFORMERS)} named performers",
            "an ABSENT comparison is not a passing one")
    def _():
        # FOUR INSTRUMENTS CLAIM THIS INDEPENDENTLY and they must agree. The three build
        # reports each write it per performer, the silhouette writes it once for the pair,
        # and the take instrument writes its own per performer -- so a single corrupted
        # boolean cannot carry the clause, and a DISAGREEMENT between two of them is a FAIL
        # even if the one the gate used to read says True.
        claims, ok = {}, True
        for report_key in ("delivery", "hygiene", "tripwire"):
            for key in ("raw_triangulation_byte_identical_same_denominator",
                        "smoothed_triangulation_byte_identical"):
                r.named(report_key, "hygiene", key, expect=PERFORMERS)
                for performer in PERFORMERS:
                    claims[f"{report_key}/{key}/{performer}"] = r.flag(
                        report_key, "hygiene", key, performer)
        for key in ("raw_triangulation_byte_identical",
                    "smoothed_triangulation_byte_identical"):
            claims[f"silhouette/{key}"] = r.flag("silhouette", key)
        r.named("take", "take", "subjects", expect=PERFORMERS)
        for performer in PERFORMERS:
            claims[f"take/{performer}"] = r.flag(
                "take", "take", "subjects", performer, "vs_baseline",
                "landmarks_byte_identical_same_denominator")
            # AND THE REST SKELETON DID NOT MOVE. D7c is a converter-only change; a moved
            # rest would make every joint comparison a comparison of two skeletons, which is
            # the defect D3 shipped against.
            claims[f"take/{performer}/rest_skeleton_unmoved"] = not r.flag(
                "take", "take", "subjects", performer, "vs_baseline", "rest_skeleton_moved")
        # the delivery build's own hygiene premises, read where the landmarks are read.
        # THE CANDIDATE'S OWN FILES MUST DIFFER FROM THE SHIPPED ONES -- this report is the
        # D7c build against D9b's delivery, so eight identical files would mean the change
        # did nothing. Each per-file flag is cross-checked against its own two hashes.
        r.named("delivery", "hygiene", "delivered_files_vs_shipped", expect=DELIVERED_FILES)
        changed = 0
        for name in DELIVERED_FILES:
            base = ("delivery", "hygiene", "delivered_files_vs_shipped", name)
            same = r.text(*base, "rebuild") == r.text(*base, "shipped")
            r.checked(*base, "identical", derived=same)
            changed += not same
        r.checked("delivery", "hygiene", "all_delivered_files_identical", derived=not changed)
        claims["delivery/the candidate's files differ from D9b's"] = changed == len(
            DELIVERED_FILES)
        # the delivery build's verdict is about the BUILD, not about matching D9b
        r.checked("delivery", "verdict",
                  derived="PASS" if all(claims.values()) else "FAIL")
        claims["delivery/observations before/after"] = r.flag(
            "delivery", "hygiene", "observations_byte_identical_before_and_after_the_build")
        claims["delivery/observations vs shipped"] = r.flag(
            "delivery", "hygiene", "observations_byte_identical_to_the_shipped_build")
        claims["delivery/work copied never symlinked"] = r.flag(
            "delivery", "work_copied_never_symlinked")
        built_here("delivery")
        ok = all(claims.values())
        if not ok:
            raise Missing("the delivery's premises do not all hold: "
                          f"{sorted(k for k, v in claims.items() if not v)}")
        return (f"{len(claims)} independent claims from 5 instruments, all True", ok)

    @clause("the delivered run-report records the mode and the guard's demoted frames",
            "E_rig_rest_kabsch; 0 and 29 demoted")
    def _():
        rows = r.listing("delivery", "diagnostics", "pelvis_frame", minimum=2)
        modes = [r.text("delivery", "diagnostics", "pelvis_frame", i, "mode")
                 for i in range(len(rows))]
        demoted = [len(r.at("delivery", "diagnostics", "pelvis_frame", i, "lever_guard",
                            "demoted_frames")) for i in range(len(rows))]
        counts = [r.num("delivery", "diagnostics", "pelvis_frame", i, "lever_guard",
                        "demoted_count") for i in range(len(rows))]
        return (f"{modes}; demoted {demoted}",
                all(m == "E_rig_rest_kabsch" for m in modes) and demoted == [0, 29]
                and [int(c) for c in counts] == demoted)

    # --------------------------------------------------------------------------- P
    def channel_preservation(*path):
        """Each P1 channel's preservation DERIVED from its own constituents, with the stored
        `bit_identical` and `failing_channels` cross-checked against what it derives.

        `local::<joint>` carries `frames_that_differ`, `foot_contacts` carries the two
        per-side contact counts before and after; both are on disk. `root_translation_m`
        carries only its dtype -- there is NO constituent in the report -- so its boolean is
        trusted and named in the inventory. Astra's round 7 set `local::LeftFoot`'s
        `frames_that_differ` to 150 and the left delivered contact count to 0, and this
        clause, reading only `bit_identical`, passed both.
        """
        r.named(*path, "channels", expect=P1_CHANNELS)
        derived = {}
        for name in P1_CHANNELS:
            base = (*path, "channels", name)
            if name.startswith("local::"):
                value = r.num(*base, "frames_that_differ") == 0
                r.checked(*base, "bit_identical", derived=value)
            elif name == "foot_contacts":
                before = [int(x) for x in r.listing(*base, "snapshot_contacts", minimum=2,
                                                    elements=True)]
                after = [int(x) for x in r.listing(*base, "delivered_contacts", minimum=2,
                                                   elements=True)]
                if len(before) != 2 or len(after) != 2:
                    raise Missing(f"{'/'.join(map(str, base))} contact counts are not "
                                  f"per-side pairs: {before} and {after}")
                value = before == after
                r.checked(*base, "bit_identical", derived=value)
            else:
                # NO CONSTITUENT ON DISK, so this one is TRUSTED and must show up in the
                # generated inventory as trusted. Cross-checking it against itself would
                # always agree and would quietly move it out of that inventory -- which is
                # exactly the defect round 7 named, manufactured by the fix for it.
                value = r.flag(*base, "bit_identical")
            derived[name] = value
        failing = {name for name, ok in derived.items() if not ok}
        # `failing_channels` is CROSS-CHECKED against the per-channel values it summarises: a
        # list that disagrees with its own constituents is a corrupted report.
        stored_failing = set(r.at(*path, "failing_channels"))
        if stored_failing != failing:
            raise Missing(f"{'/'.join(map(str, path))}/failing_channels "
                          f"{sorted(stored_failing)} against {sorted(failing)} derived")
        return derived, failing

    def p1_verdicts(report_key, failing, *, expected):
        """A P-report's own verdict strings, DERIVED from the channels that decided them.
        `expected_p1` is what the arm was BUILT to do -- PASS for the delivery, FAIL for the
        control -- so `P1_as_expected` is a comparison and not a copy."""
        r.named(report_key, "P1_verdicts", expect=PERFORMERS)
        seen = {}
        for performer in PERFORMERS:
            seen[performer] = "FAIL" if failing[performer] else "PASS"
            r.checked(report_key, "P1_verdicts", performer, derived=seen[performer])
        r.checked(report_key, "expected_p1", derived=expected)
        r.checked(report_key, "P1_as_expected",
                  derived=all(v == expected for v in seen.values()))

    def measured_runs(*path, field):
        rows = r.listing(*path, field, minimum=1)
        seen, worst, holds = [], 0.0, True
        for index in range(len(rows)):
            side = int(r.num(*path, field, index, "side_index"))
            span = r.listing(*path, field, index, "run", minimum=2, elements=True)
            identity = (side, int(span[0]), int(span[1]))
            if identity in seen:
                raise Missing(f"{'/'.join(map(str, path))}/{field} repeats run {identity}")
            seen.append(identity)
            run_worst = 0.0
            for joint in SIDE_JOINTS[side]:          # BOTH named fields, per side
                run_worst = max(run_worst, r.num(*path, field, index, f"{joint}_max_m"))
            worst = max(worst, run_worst)
            # `holds` is CROSS-CHECKED against the run's own numbers rather than believed.
            stored_holds = r.checked(*path, field, index, "holds",
                                     derived=run_worst <= CONTACT_TOLERANCE_M)
            holds &= stored_holds
        return set(seen), worst, holds

    @clause("P1 channel preservation -- the delivery, both performers",
            f"every protected channel bit-identical on {len(PERFORMERS)} performers, on an "
            "AUTHENTICATED track")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        ok, failing = True, {}
        for performer in PERFORMERS:
            # AUTHENTICATION IS DERIVED FROM THE TWO HASHES, not read off the saved boolean.
            # Astra's round 6 set the GLB's stamped hash to 64 zeros, kept the recomputed one
            # and the boolean, and nothing moved. Both must be PRESENT and EQUAL.
            stamped = r.text("projection", "subjects", performer, "P1_channel_preservation",
                             "authentication", "glb_body_track_sha256")
            recomputed = r.text("projection", "subjects", performer,
                                "P1_channel_preservation", "authentication",
                                "recomputed_sha256")
            derived = bool(stamped) and stamped == recomputed
            r.checked("projection", "subjects", performer, "P1_channel_preservation",
                      "authentication", "authenticated", derived=derived)
            ok &= derived
            preserved, failed = channel_preservation(
                "projection", "subjects", performer, "P1_channel_preservation")
            ok &= all(preserved.values()) and not failed
            failing[performer] = sorted(failed)
        p1_verdicts("projection", failing, expected="PASS")
        for performer in PERFORMERS:
            r.checked("projection", "subjects", performer, "P1_channel_preservation",
                      "verdict", derived="PASS" if not failing[performer] else "FAIL")
        return f"{len(PERFORMERS)} performers; failing {failing}", ok

    @clause("P2 anchor lock -- the delivery, every accepted run, on the GLB's own arrays",
            f"<= {CONTACT_TOLERANCE_M} m at every run's first KEYED sample, runs matching "
            "the frozen mask by identity")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        ok, worst_all, detail = True, 0.0, {}
        for performer in PERFORMERS:
            path = ("projection", "subjects", performer, "P2_anchor_lock")
            # THE BAND THE INSTRUMENT USED IS THE BAND THIS GATE STATES. A report scored
            # against a looser band would pass every clause below while measuring less.
            if r.num(*path, "band_m") != CONTACT_TOLERANCE_M:
                raise Missing(f"P2/{performer} scored against band {r.num(*path, 'band_m')} "
                              f"m, not the gate's {CONTACT_TOLERANCE_M}")
            expected = {(int(a), int(b), int(c)) for a, b, c
                        in r.listing(*path, "mask_run_identities", minimum=1,
                                     elements=True)}
            seen, worst, holds = measured_runs(*path, field="runs")
            if seen != expected:
                raise Missing(f"P2/{performer} runs {sorted(seen)} != mask {sorted(expected)}")
            stored = r.num(*path, "worst_travel_m")
            if not agrees(stored, worst, 1e-12):
                raise Missing(f"P2/{performer}/worst_travel_m stores {stored} against "
                              f"{worst} derived from {len(seen)} runs")
            worst_all = max(worst_all, worst)
            ok &= holds and worst <= CONTACT_TOLERANCE_M
            detail[performer] = len(seen)
        r.named("projection", "P2_verdicts", expect=PERFORMERS)
        for performer in PERFORMERS:
            r.checked("projection", "P2_verdicts", performer,
                      derived="PASS" if ok else "FAIL")
            r.checked("projection", "subjects", performer, "P2_anchor_lock", "verdict",
                      derived="PASS" if ok else "FAIL")
        # the P report's own top-level verdict, over both contracts it carries
        r.checked("projection", "verdict", derived="PASS" if ok else "FAIL")
        return f"worst {worst_all:.3e} m; runs {detail}", ok

    @clause("P2 anchor lock on EVERY ORACLE BODY, from each exported GLB's own arrays",
            f"<= {CONTACT_TOLERANCE_M} m on all {len(ORACLE_SEEDS)} seeds, runs matching the "
            "frozen mask by identity")
    def _():
        r.named("projection", "P2_on_the_oracle_bodies", "seeds", expect=ORACLE_SEEDS)
        if r.num("projection", "P2_on_the_oracle_bodies", "band_m") != CONTACT_TOLERANCE_M:
            raise Missing("oracle P2 scored against a band this gate does not state")
        ok, worst_all, detail = True, 0.0, {}
        for seed in ORACLE_SEEDS:
            path = ("projection", "P2_on_the_oracle_bodies", "seeds", seed)
            expected = {(int(a), int(b), int(c)) for a, b, c
                        in r.listing(*path, "mask_run_identities", minimum=1,
                                     elements=True)}
            seen, worst, holds = measured_runs(*path, field="run_measurements")
            if seen != expected:
                raise Missing(f"oracle P2/{seed} runs {sorted(seen)} != mask "
                              f"{sorted(expected)}")
            stored = r.num(*path, "worst_travel_m")
            if not agrees(stored, worst, 1e-12):
                raise Missing(f"oracle P2/{seed}/worst_travel_m stores {stored} against "
                              f"{worst} derived")
            if int(r.num(*path, "runs")) != len(seen):
                raise Missing(f"oracle P2/{seed}/runs count disagrees with its rows")
            worst_all = max(worst_all, worst)
            ok &= holds and worst <= CONTACT_TOLERANCE_M
            detail[seed] = len(seen)
            r.checked(*path, "verdict",
                      derived="PASS" if holds and worst <= CONTACT_TOLERANCE_M else "FAIL")
        r.checked("projection", "P2_on_the_oracle_bodies", "verdict",
                  derived="PASS" if ok else "FAIL")
        global_stored = r.num("projection", "P2_on_the_oracle_bodies",
                              "worst_travel_m_over_all_seeds")
        if not agrees(global_stored, worst_all, 1e-12):
            raise Missing(f"oracle P2 global summary stores {global_stored} against "
                          f"{worst_all} derived from the seeds")
        return f"worst {worst_all:.3e} m re-derived from the runs; runs {detail}", ok

    @clause("P3 planted-foot travel on the frozen UNION of both builds' runs", "REPORT")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        total = sum(len(r.listing("projection", "subjects", s, "P3_travel_report",
                                  "intervals", minimum=1)) for s in PERFORMERS)
        return f"{total} intervals", "REPORT"

    @clause("P1 on EVERY ORACLE BODY (the card says the take AND every oracle body)",
            f"every protected channel bit-identical on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("p_oracle", "seeds", expect=ORACLE_SEEDS)
        ok, clean = True, 0
        for seed in ORACLE_SEEDS:
            ok &= r.flag("p_oracle", "seeds", seed, "root_bit_identical")
            ok &= r.flag("p_oracle", "seeds", seed, "contacts_bit_identical")
            for joint in PROTECTED:
                ok &= r.flag("p_oracle", "seeds", seed, f"local::{joint}")
            derived_failing = {j for j in PROTECTED
                               if not r.flag("p_oracle", "seeds", seed, f"local::{j}")}
            failing = set(r.at("p_oracle", "seeds", seed, "failing_channels"))
            if failing != derived_failing:
                raise Missing(f"oracle P1/{seed}/failing_channels {sorted(failing)} against "
                              f"{sorted(derived_failing)} derived from the flags")
            ok &= not failing
            clean += not failing
            r.checked("p_oracle", "seeds", seed, "verdict",
                      derived="PASS" if not failing else "FAIL")
        r.checked("p_oracle", "verdict", derived="PASS" if ok else "FAIL")
        return f"{clean} of {len(ORACLE_SEEDS)} clean", ok

    @clause("P1's CONTROL 1 -- the projection's foot locals overwritten", "must FAIL P1",
            "refused by the shipping path is STRONGER than caught by a gate")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing = {}
        for performer in PERFORMERS:
            base = ("p1_controls", "controls", performer,
                    "control_1_foot_locals_overwritten")
            failing[performer] = set(r.at(*base, "failing_channels"))
            # THE CONTROL'S OWN PREMISE: it must actually have changed something. A control
            # that mutated nothing would be "detected" by a report that says so.
            changed = int(r.num(*base, "samples_changed"))
            if changed < 1:
                raise Missing(f"control 1 on {performer} changed {changed} samples")
            r.checked(*base, "P1", derived="FAIL" if failing[performer] else "PASS")
            r.checked(*base, "detected_as_required", derived=bool(failing[performer]))
        # THE NAMED CHANNELS, not "some nonempty list". Astra's round 6 replaced performer
        # 0's failing list with ["local::Head"] -- a channel this control never touches, and
        # not even one P1 protects -- and the clause still passed.
        expected = CONTROL_CHANNELS["control_1_foot_locals_overwritten"]
        ok = all(seen == expected and seen <= PROTECTED_CHANNELS
                 for seen in failing.values())
        return (f"the shipping path REFUSES to build it; applied to the delivered bytes P1 "
                f"detects exactly {sorted(expected)}: "
                f"{ {s: sorted(v) for s, v in failing.items()} }", ok)

    @clause("P1's CONTROL 2 -- the nonempty contact mask cleared (offline)",
            "must FAIL P1 on the mask")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing, detected = {}, True
        for performer in PERFORMERS:
            base = ("p1_controls", "controls", performer, "control_2_contact_mask_cleared")
            failing[performer] = set(r.at(*base, "failing_channels"))
            # the control CLEARS a mask, so the mask must have been nonempty on both sides.
            nonempty = [int(x) for x in r.listing(*base, "mask_was_nonempty", minimum=2,
                                                  elements=True)]
            if len(nonempty) != 2 or min(nonempty) < 1:
                raise Missing(f"control 2 on {performer} cleared a mask that was {nonempty}")
            r.checked(*base, "P1", derived="FAIL" if failing[performer] else "PASS")
            r.checked(*base, "detected_as_required", derived=bool(failing[performer]))
            detected &= bool(failing[performer]) and bool(
                r.at("p1_controls", "controls", performer,
                     "control_1_foot_locals_overwritten", "failing_channels"))
        r.checked("p1_controls", "both_controls_detected_on_both_performers",
                  derived=detected)
        r.checked("p1_controls", "verdict", derived="PASS" if detected else "FAIL")
        expected = CONTROL_CHANNELS["control_2_contact_mask_cleared"]
        return (str({s: sorted(v) for s, v in failing.items()}),
                all(seen == expected and seen <= PROTECTED_CHANNELS
                    for seen in failing.values()))

    @clause("the UNMUTATED delivery through the same comparison", "PASS")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing = {}
        for performer in PERFORMERS:
            base = ("p1_controls", "controls", performer, "the_unmutated_delivery")
            failing[performer] = r.at(*base, "failing_channels")
            r.checked(*base, "P1", derived="FAIL" if failing[performer] else "PASS")
        return str(failing), not any(failing.values())

    @clause("P1's CONTROL 2, BUILT and run through the P instrument", "must FAIL P1",
            "an INSTRUMENT DEFECT was found by this very control")
    def _():
        r.named("control2", "subjects", expect=PERFORMERS)
        failing = {}
        for performer in PERFORMERS:
            # the SAME derivation as the delivery's P1: the control's detection is read from
            # the channels' own constituents, so a control that "fails" only in a saved
            # boolean cannot stand in for one the instrument actually caught.
            _preserved, failing[performer] = channel_preservation(
                "control2", "subjects", performer, "P1_channel_preservation")
        p1_verdicts("control2", failing, expected="FAIL")
        r.checked("control2", "verdict",
                  derived="PASS" if all(failing.values()) else "FAIL")
        return (str({s: sorted(v) for s, v in failing.items()}),
                all(seen == CONTROL_2_BUILT_CHANNELS and seen <= PROTECTED_CHANNELS
                    for seen in failing.values()))

    # -------------------------------------------------------------------------- B1, B2
    @clause("B1 the photographs: worsening NOT ESTABLISHED (ci95 upper bound >= 0), 8 cells",
            f"ci95[1] >= 0 on all {len(B1_CELLS) * len(PERFORMERS)} NAMED cells",
            "it does NOT establish non-worsening; a wide interval passes it for want of power")
    def _():
        ok, seen = True, 0
        # THE SILHOUETTE IS AN INSTRUMENT, NEVER A SELECTOR. It scores the MESH against the
        # photographs and no constant here was chosen on it.
        if not r.flag("silhouette", "instrument_only"):
            raise Missing("the silhouette report does not declare itself instrument-only")
        # AND THE PHOTOGRAPHS ARE THE SAME PHOTOGRAPHS: the mask cache this run read is the
        # one the earlier runs read, byte for byte. The MAMMA oracle's own agreement is the
        # other half of that argument and is banded in the clause below.
        masks = r.at("silhouette", "masks_copied_never_shared")
        if not isinstance(masks, dict) or not masks:
            raise Missing("silhouette/masks_copied_never_shared is empty")
        for name in sorted(masks):
            if not r.flag("silhouette", "masks_copied_never_shared", name, "byte_identical"):
                raise Missing(f"the silhouette read a mask cache that is not byte-identical: "
                              f"{name}")
        draws = int(r.num("silhouette", "statistics", "draws"))
        shortfall = {}
        r.named("silhouette", "subjects", expect=PERFORMERS)
        for performer in PERFORMERS:
            for name in B1_CELLS:
                base = ("silhouette", "preregistered_clause_verdicts", performer, name)
                cut = "whole_take" if "whole_take" in name else "bent_tercile"
                part = "arm" if name.startswith("clause_arm") else "torso"
                # THE CELL'S OWN MEASUREMENT, not the copy of it beside the verdict. The
                # producer copies `difference`, `ci95` and `cut_frames` out of
                # `subjects/<s>/cuts/<cut>/<part>_D7c_minus_D9b`; Astra's round 8 moved the
                # SOURCE interval's upper bound below zero -- worsening established -- and
                # the gate, reading the unchanged copy, passed the cell.
                source = ("silhouette", "subjects", performer, "cuts", cut,
                          f"{part}_D7c_minus_D9b")
                interval = r.listing(*source, "ci95", minimum=2, elements=True)
                difference = r.num(*source, "median_difference")
                frames = int(r.num("silhouette", "subjects", performer, "cuts", cut, "n"))
                if frames != B1_CUT_FRAMES[cut]:
                    raise Missing(f"B1/{performer}/{name} is over {frames} photographs, not "
                                  f"the cut's {B1_CUT_FRAMES[cut]}")
                # THE SAME DRAWS, MEASURED. Every part of a cut must report the same
                # `draws_used` -- that identity IS the card's identical-draws requirement
                # made observable -- and it can fall below the requested count when a
                # resample is degenerate, which is reported rather than banded.
                used = int(r.num(*source, "draws_used"))
                siblings = {int(r.num("silhouette", "subjects", performer, "cuts", cut,
                                      f"{other}_D7c_minus_D9b", "draws_used"))
                            for other in ("torso", "arm", "whole")}
                if len(siblings) != 1 or not 1 <= used <= draws:
                    raise Missing(f"B1/{performer}/{cut} used {sorted(siblings)} draws "
                                  f"across its parts, against the run's {draws}")
                shortfall[f"{performer}/{cut}"] = draws - used
                if not float(interval[0]) <= difference <= float(interval[1]):
                    raise Missing(f"B1/{performer}/{name} difference {difference} is outside "
                                  f"its own interval {interval}")
                # and the copies beside the verdict must AGREE with the measurement
                r.checked(*base, "difference", derived=difference, tolerance=1e-9)
                r.checked(*base, "cut_frames", derived=frames)
                copy = r.listing(*base, "ci95", minimum=2, elements=True)
                if [float(x) for x in copy] != [float(x) for x in interval]:
                    raise Missing(f"B1/{performer}/{name} copies interval {copy} against the "
                                  f"measurement's {interval}")
                r.checked(*base, "rose_with_ci_clear_of_zero",
                          derived=difference > 0.0 and float(interval[0]) > 0.0)
                r.checked(*base, "verdict",
                          derived="PASS" if float(interval[1]) >= 0.0 else "FAIL")
                ok &= float(interval[1]) >= 0.0
                seen += 1
        r.checked("silhouette", "verdict", derived="PASS" if ok else "FAIL")
        return (f"{seen} of {len(B1_CELLS) * len(PERFORMERS)} named cells checked on their "
                f"own subject/cut measurements; degenerate-draw shortfall "
                f"{min(shortfall.values())}-{max(shortfall.values())} of {draws}", ok)

    @clause("B1 IDENTICAL DRAWS -- the card's own requirement, enforced",
            f"every arm on the same {2000} draws, one moving-block bootstrap, one seed",
            "the card: `d7b_silhouette_partwise` D9b vs candidate ON IDENTICAL DRAWS. An "
            "exemption may never cover a measurement the card bands, and round 8 set this "
            "flag false with every clause unchanged")
    def _():
        if not r.flag("silhouette", "statistics", "every_arm_on_identical_draws"):
            raise Missing("the silhouette did not score every arm on identical draws")
        draws = int(r.num("silhouette", "statistics", "draws"))
        block = int(r.num("silhouette", "statistics", "moving_block"))
        seed = int(r.num("silhouette", "statistics", "seed"))
        lag1 = r.num("silhouette", "statistics", "lag1_autocorrelation_on_this_take")
        if draws < 1 or block < 1:
            raise Missing(f"the bootstrap ran {draws} draws with block {block}")
        # the block bootstrap exists BECAUSE the per-frame series is autocorrelated; a run
        # that measured no autocorrelation would not need one and would not be this run.
        if not 0.0 < lag1 <= 1.0:
            raise Missing(f"lag-1 autocorrelation {lag1} is not a correlation")
        # B2 is scored on identical draws too, by the same card line. It publishes no flag,
        # so its three bootstrap parameters are read and the missing boolean is stated.
        b2_draws = int(r.num("b2", "bootstrap", "draws"))
        b2_block = int(r.num("b2", "bootstrap", "block"))
        b2_seed = int(r.num("b2", "bootstrap", "seed"))
        if b2_draws < 1 or b2_block < 1:
            raise Missing(f"B2's bootstrap ran {b2_draws} draws with block {b2_block}")
        return (f"silhouette: {draws} draws, block {block}, seed {seed}, lag-1 {lag1}; "
                f"B2: {b2_draws} draws, block {b2_block}, seed {b2_seed} (B2 publishes no "
                f"identical-draws flag -- owed)", True)

    @clause("B1 the MAMMA mesh oracle bit-identical", "< 1e-9")
    def _():
        worst = r.num("silhouette", "preregistered_clause_verdicts",
                      "clause_mamma_mesh_oracle",
                      "this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference")
        r.checked("silhouette", "preregistered_clause_verdicts", "clause_mamma_mesh_oracle",
                  "verdict", derived="PASS" if worst < 1e-9 else "FAIL")
        return str(worst), worst < 1e-9

    @clause("B2 `delivered_vs_capture.py --reference smoothed`: the same-denominator clause",
            "PASS (landmarks byte-identical)",
            "CHANGED would mean the change did more than refit the pelvis")
    def _():
        # DERIVED FROM THE PER-SUBJECT, PER-BUILD CONSTITUENTS, which are on disk beside the
        # aggregate. Astra's round 7 set subject_00's D7c entry to false and this clause,
        # reading only `same_denominator`, still passed.
        r.named("b2", "triangulated_landmarks_byte_identical_across_arms",
                expect=PERFORMERS)
        identical = {}
        for performer in PERFORMERS:
            r.named("b2", "triangulated_landmarks_byte_identical_across_arms", performer,
                    expect=("D9b", "D7c"))
            for build_name in ("D9b", "D7c"):
                identical[f"{performer}/{build_name}"] = r.flag(
                    "b2", "triangulated_landmarks_byte_identical_across_arms", performer,
                    build_name)
        derived = all(identical.values())
        r.checked("b2", "same_denominator", derived=derived)
        return (f"{sum(identical.values())} of {len(identical)} subject x build arrays "
                f"byte-identical", derived)

    # --------------------------------------------------------------- the REPORT blocks
    @clause("B4 the pelvis and the root's motion (REPORTED, never banded)", "REPORT",
            "the 800 deg/s line is a physical REFERENCE, not a band")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        rows = [f"{s}: pitch "
                f"{r.num('take', 'take', 'subjects', s, 'vs_baseline', 'pelvis_change_deg', 'pitch_about_hip_line_signed_median')} deg, "
                f"root {r.num('take', 'take', 'subjects', s, 'vs_baseline', 'root_move_mm_hoist_subtracted', 'median')} mm, "
                f"step p95 {r.num('take', 'take', 'subjects', s, 'pelvis_step_deg_per_frame', 'p95')} deg, "
                f"{int(r.num('take', 'take', 'subjects', s, 'frames_over_800_deg_per_s'))} over 800 deg/s"
                for s in PERFORMERS]
        return "; ".join(rows), "REPORT"

    @clause("B4 the leg-root midpoint stays on the captured hip midpoint", "0.0 mm",
            "NOT (b)-specific: the card listed this as conditional on (b) in error")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: max "
            f"{r.num('take', 'take', 'subjects', s, 'leg_roots_on_captured_hip_midpoint_mm', 'max')} mm"
            for s in PERFORMERS), "REPORT")

    @clause("B2/B4 the hip residual under (a) -- a REPORT, and NO band may be made from it",
            "REPORT")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: full p95 "
            f"{r.num('take', 'take', 'subjects', s, 'vs_baseline', 'hip_residual_REPORT_never_a_band', 'baseline', 'full_positional_mm', 'p95')}"
            f" -> {r.num('take', 'take', 'subjects', s, 'vs_baseline', 'hip_residual_REPORT_never_a_band', 'candidate', 'full_positional_mm', 'p95')} mm"
            for s in PERFORMERS), "REPORT")

    @clause("B1 attribution of the three rising torso cells (DIAGNOSTIC)", "REPORT",
            "a POINT-ESTIMATE decomposition; only performer 0's ARTICULATION shares have "
            "intervals clear of zero")
    def _():
        r.named("b1_attribution", "subjects", expect=PERFORMERS)
        rows = []
        for performer in PERFORMERS:
            base = ("b1_attribution", "subjects", performer, "torso", "whole_take")
            rows.append(
                f"{performer}: both "
                f"{r.num(*base, 'candidate_minus_D9b__both_effects', 'median_difference'):+.5f}"
                f" = articulation "
                f"{r.num(*base, 'ablation_minus_D9b__the_ARTICULATION_alone', 'median_difference'):+.5f}"
                f" + root "
                f"{r.num(*base, 'candidate_minus_ablation__the_ROOT_TRANSLATION_alone', 'median_difference'):+.5f}"
                f" (root CI "
                f"{r.listing(*base, 'candidate_minus_ablation__the_ROOT_TRANSLATION_alone', 'ci95', minimum=2)})")
        return "; ".join(rows), "REPORT"

    @clause("B3 the hoist and the contacts (REPORTED)", "REPORT")
    def _():
        rows = []
        for label in ("D9b_shipped", "D7c_candidate"):
            r.named("b3", "arms", label, "subjects", expect=PERFORMERS)
            for performer in PERFORMERS:
                rows.append(
                    f"{label} {performer}: hoist p95 "
                    f"{r.num('b3', 'arms', label, 'subjects', performer, 'hoist_mm', 'p95')} mm, "
                    f"contacts {r.at('b3', 'arms', label, 'subjects', performer, 'contacts', 'count')}")
        return "; ".join(rows), "REPORT"

    @clause("B6 the delivered bytes (REPORT)", "REPORT")
    def _():
        base = ("b6", "builds", "D7c", "subject_00")
        return (f"LINEAR samplers, {int(r.num(*base, 'sampler_input_times', 'frames'))} "
                f"frames, {r.num(*base, 'duration_s'):.4f} s; track->GLB positional closure "
                f"max {r.num(*base, 'track_to_glb_closure', 'positional_mm', 'max')} mm; "
                f"rotational closure median "
                f"{r.num(*base, 'track_to_glb_closure', 'rotational_deg_frame_corrected', 'median')} deg",
                "REPORT")

    @clause("B6 the carried-tetrahedron PROXY (NOT an inversion count)", "REPORT",
            "a PROXY with two demonstrated failure modes -- it mis-classifies a proper rigid "
            "motion under varying weights and it is vertex-order dependent -- so NO inversion "
            "claim is made. The sound measurement is the skinning Jacobian with spatially "
            "varying weights (Kavan, direct methods eq. 17) and is D6's.")
    def _():
        base = ("b6", "builds", "D7c", "subject_00", "mesh_deformation_pelvis_hip_thigh")
        low = int(r.num(*base, "carried_tetrahedron_PROXY", "per_frame_min"))
        high = int(r.num(*base, "carried_tetrahedron_PROXY", "per_frame_max"))
        return (f"fires on {low}-{high} of {int(r.num(*base, 'triangles'))} triangles per "
                f"frame", "REPORT")

    @clause("B6 the `Root` / eye / finger local invariants vs D9b (a TRACK-ARRAY claim)",
            "bit-identical")
    def _():
        node = r.at("b6", "builds", "D7c", "subject_00",
                    "invariants_vs_the_other_build_TRACK_ARRAYS")
        return str(node), "REPORT"

    @clause("B5b the delivered `Head` WORLD rotation, from the GLB's own bytes", "REPORT",
            "NOT identical; the earlier claim of identity is withdrawn")
    def _():
        r.named("b6", "B5b_head_world_between_builds", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: {r.num('b6', 'B5b_head_world_between_builds', s, 'per_frame_difference_deg', 'median')} deg "
            f"median, {r.num('b6', 'B5b_head_world_between_builds', s, 'per_frame_difference_deg', 'max')} max"
            for s in PERFORMERS), "REPORT")

    # ------------------------------------------------------------ the card's merge rule
    def verdict_of(prefix):
        hits = [c for c in clauses if c["clause"].startswith(prefix)]
        return None if not hits else (
            "PASS" if all(c["verdict"] in ("PASS", "REPORT") for c in hits) else "FAIL")

    def all_of(prefixes):
        values = [verdict_of(p) for p in prefixes]
        return None if any(v is None for v in values) else (
            "PASS" if all(v == "PASS" for v in values) else "FAIL")

    conjuncts = {name: all_of(prefixes) for name, prefixes in CONJUNCTS}
    missing = [k for k, v in conjuncts.items() if v is None]
    # THE TWO RECORDED STOPS MUST STILL READ FAIL. `selector.json` and
    # `selector-calibrated.json` are immutable and each records a stop; a run in which one of
    # them PASSES means the immutable file no longer records what the step stopped on, which
    # is a corruption of the record and not a newly satisfied clause.
    by_name = {c["clause"]: c["verdict"] for c in clauses}
    measured = {c["clause"]: str(c["measured"]) for c in clauses}
    # AND IT MUST FAIL ON ITS BAND. A `Missing` raise also produces FAIL, so deleting
    # `selector.json`'s follower table would satisfy a naive check while destroying the
    # record the clause exists to preserve. The stop has to be MEASURED and failing.
    stops_held = all(by_name.get(name) == "FAIL"
                     and not measured.get(name, "MISSING").startswith("MISSING")
                     for name in RECORDED_STOPS)
    return {
        "clauses": clauses, "conjuncts": conjuncts, "not_yet_measured": missing,
        "touched": sorted("/".join(path) for path in r.touched),
        "kinds": dict(r.kinds), "cross_checked": set(r.cross_checked),
        "recorded_stops_still_fail": {name: by_name.get(name) for name in RECORDED_STOPS},
        "verdict": ("MERGE" if not missing and stops_held
                    and all(v == "PASS" for v in conjuncts.values())
                    else "INCOMPLETE" if missing else "NO MERGE"),
    }


# THE TWO CLAUSES THAT READ FAIL ON THE UNMUTATED REPORTS, BY NAME. The fuzzer pins its
# "historical FAIL" class to exactly these: deriving that class from "whichever clauses fail
# today" would let a NEW clause that accidentally fails at the baseline absorb every leaf it
# reads into a class that is excused by construction.
RECORDED_STOPS = (
    "S at the CARD'S OWN FIXTURE (sigma 1.0): the frozen-pitch follower >= 2x on EVERY body",
    "the amended card's FIXTURE CALIBRATION, under its own frozen monotonicity precondition",
)
S_STOPS = (
    "the SAME frozen evaluations under Astra round 7's amended admissibility rule",
    "S REREAD: at the calibration's EXACT accepted sigma",
    "S REREAD: (a) vs (b)",
    "S REREAD: the winner strictly better than C-on-SOMA",
    "S REREAD: the frozen-pitch follower",
    "G1 (missing-only)",
    "G2 (finite-only)",
)
MUST_FAILS = (
    "REFACTOR TRIPWIRE (ii)",
    "must-fail: the WRONG-ORIGIN template",
    "must-fail: a pelvis frozen upright",
    "P1's CONTROL 1",
    "P1's CONTROL 2 -- the nonempty contact mask cleared",
    "P1's CONTROL 2, BUILT",
)
CONJUNCTS = (
    ("hygiene", ("hygiene:",)),
    ("the refactor tripwire (both readings)",
     ("REFACTOR TRIPWIRE (i)", "REFACTOR TRIPWIRE (ii)")),
    ("the oracle's own premises (the arms, the populations, the bands)",
     ("O1/O2 PREMISES:",)),
    ("O1", ("O1 ",)),
    ("O2", ("O2 ",)),
    ("every must-fail still fails", MUST_FAILS),
    ("P1 on the take",
     ("P1 channel preservation", "the UNMUTATED delivery through the same comparison")),
    ("P2 on the take", ("P2 anchor lock -- the delivery",)),
    ("P1 on every oracle body", ("P1 on EVERY ORACLE BODY",)),
    ("P2 on every oracle body", ("P2 anchor lock on EVERY ORACLE BODY",)),
    ("S (every stop of the reread, G1 and G2 included)", S_STOPS),
    ("B1 on both performers, oracle included",
     ("B1 the photographs", "B1 IDENTICAL DRAWS", "B1 the MAMMA mesh oracle")),
    ("the same denominator (B2 and both landmark arrays)",
     ("B2 `delivered_vs_capture.py", "the delivery: BOTH landmark arrays byte-identical")),
)
# The hand-written inventory that used to stand here is GONE. It claimed "every other saved
# boolean is derived or cross-checked" and Astra's round 7 showed that was false -- the oracle
# P1 clause alone consumes ten more per body. An inventory written from memory is a claim about
# the author, not about the gate, so the inventory is now GENERATED from the Reader's own
# record by `saved_value_inventory`, and every family it finds must be named in
# `TRUSTED_READ_JUSTIFICATIONS` below or the gate reads NO MERGE.
# EVERY MEASUREMENT LEAF THE GATE DOES NOT READ, BY FAMILY, WITH THE REASON. `**` matches any
# tail; `*` matches one segment; `<seed>`, `<subject>` and `<i>` are the normalised forms.
# A family here is a family the CARD does not band -- not one the gate found inconvenient.
UNREAD_MEASUREMENTS_JUSTIFIED = (
    # --- the report blocks. The card: "O3, B3, B4, B5, B6 report".
    ("take/**", "B2/B4 REPORT blocks. THE CARD: \"O3, B3, B4, B5, B6 report\" -- it reports "
                "the take's pelvis and root motion and bands nothing in it; the clauses that "
                "read it carry verdict REPORT."),
    ("b3/**", "B3 REPORTED by the card's own line \"O3, B3, B4, B5, B6 report\": the hoist "
              "and the contacts. No band."),
    ("b6/**", "B5b/B6 REPORTED by the same card line: the delivered bytes, the closures, the "
              "head world rotation and the carried-tetrahedron PROXY, which makes no "
              "inversion claim."),
    ("b1_attribution/**", "B1's attribution is a DIAGNOSTIC decomposition, explicitly a "
                          "point estimate; only the three rising torso cells' shares are "
                          "quoted and the clause that reads them is REPORT."),
    ("b2/subjects/**", "B2's per-joint distances to MAMMA. B2 is a MAMMA-referenced "
                       "instrument: its one banded clause is the same-denominator one, and "
                       "no constant is selected on any of it."),
    # --- the oracle's controls and diagnostics
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_deg/angle/*",
     "the order statistics of an arm's tilt other than the one its clause bands. O1 bands "
     "the MAX on the shipping arm; each must-fail control is banded on the statistic the "
     "card names for it, and the rest are reported beside them."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_deg/pitch_signed_median",
     "the SIGNED pitch, reported so the direction of a tilt is legible; the bands are on "
     "the unsigned angle."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_worst_frame_deg",
     "the index of the worst frame: a pointer into the take, not a measurement of it."),
    ("oracle/oracle/seeds/<seed>/arms/*/penetration_before_mm",
     "the ground penetration before the hoist, reported; D9b's contact projection owns it."),
    ("oracle/oracle/seeds/<seed>/arms/*/hoisted_frames",
     "how many frames the contact projection lifted, reported; O2 bands the hoist CHANGE."),
    ("oracle/oracle/seeds/<seed>/arms/*/ALIGNED_rc_score_groups_mm/*",
     "the D3 gate's leg-root-ALIGNED gauge, which D9b established is blind to a root move. "
     "O3 reports the arms row and no band reads any of it."),
    ("oracle/oracle/seeds/<seed>/arms/*/ABSOLUTE_groups_mm/**",
     "the absolute-frame companion rows other than the torso one O1 bands."),
    ("oracle/oracle/seeds/<seed>/arms/*/hips_origin_miss_mm/**",
     "the order statistics other than the max O1 bands."),
    ("oracle/oracle/seeds/<seed>/arms/*/spine_origin_miss_mm/**",
     "the order statistics other than the max O1 bands."),
    ("oracle/oracle/seeds/<seed>/arms/*/three_point_residual_m/*",
     "the order statistics other than the max O1 bands and the median the wrong-origin "
     "control is quoted on."),
    ("oracle/oracle/seeds/<seed>/arms/*/neck_miss_mm/**",
     "the neck, which D7b owns; D7c neither moves nor bands it."),
    ("oracle/oracle/seeds/<seed>/arms/*/hoist_mm/**",
     "the hoist's own distribution per arm, reported; O2 bands the CHANGE between builds."),
    ("oracle/oracle/seeds/<seed>/arms/*/contacts/**",
     "the contact mask per arm, reported; O2 bands its identity between builds."),
    ("oracle/oracle/seeds/<seed>/arms/*/fit_geometry/**",
     "the template each arm fitted, reported so an arm's geometry is legible beside its "
     "score. The shipping arm's is read leaf for leaf by the src == E identity."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_report/**",
     "the converter's own run report per arm, reported; the shipping arm's mode is read and "
     "the delivery's guard counts are banded in their own clause."),
    ("oracle/oracle/seeds/<seed>/arms/D_rig_rest_hipline/**",
     "the mode S RANKED AND DID NOT CHOOSE. Its ranking is S's business and is made in the "
     "selector files on the noise fixture; the O bands are on the arm that ships."),
    ("oracle/oracle/seeds/<seed>/factors/**",
     "the per-seed sizing factors that MAKE the fixture: inputs to it, not measurements of "
     "the candidate."),
    ("oracle/oracle/seeds/<seed>/rig_rest_mm/**",
     "the fixture's own rest geometry, an input to it."),
    ("oracle/oracle/seeds/<seed>/O2_vs_baseline/*/median", "the order statistics other than "
     "the max O2 bands."),
    ("oracle/oracle/seeds/<seed>/O2_vs_baseline/*/p95", "the order statistics other than the "
     "max O2 bands."),
    # --- S's controls and its fourth metric
    ("reread/bodies/<seed>/arms/*/*/iii_rotational_compensation_step_mm",
     "a FOURTH metric, reported beside the card's three. S's rule is stated on (i), (ii) and "
     "(iii) and adding a fourth after the fact would be choosing the metric that wins."),
    ("reread/aggregated_median_of_six/*/*/iii_rotational_compensation_step_mm",
     "the same fourth metric's aggregate."),
    ("reread/bodies/<seed>/arms/world_vertical/**",
     "a CONTROL arm the merge rule does not read: report-only by the card. Its population is "
     "validated with every other arm's, and its one quoted number has its own REPORT clause."),
    ("reread/bodies/<seed>/arms/thorax_as_pelvis/**",
     "a CONTROL arm the merge rule does not read: report-only by the card."),
    ("reread/bodies/<seed>/arms/b_hipline_unguarded/**",
     "the unguarded variant, which G2 scores on its own corrupted fixture rather than on S's "
     "three metrics; its populations are validated with every other arm's."),
    ("reread/bodies/<seed>/arms/frozen_pitch_follower/*/ii_step_deg",
     "the follower is the control for metric (i); (ii) and (iii) are reported beside it."),
    ("reread/bodies/<seed>/arms/frozen_pitch_follower/*/iii_root_step_mm",
     "the follower is the control for metric (i); (ii) and (iii) are reported beside it."),
    ("reread/G1_missing_only/bodies/<seed>/recovery_error_on_missing_frames_i_deg",
     "G1's own text: the recovery error is REPORTED, never banded."),
    ("reread/G1_missing_only/bodies/<seed>/recovery_error_on_transition_pairs_ii_deg",
     "G1's own text: the recovery error is REPORTED, never banded."),
    ("reread/G1_missing_only/bodies/<seed>/i_deg_elsewhere",
     "the error away from the missing pattern, reported as the comparison's floor."),
    ("reread/G1_missing_only/bodies/<seed>/quaternions_bit_identical",
     "the amended G1 claim is about the INTERPOLATED ARRAYS; the quaternion identity is a "
     "stronger statement reported beside it and the amendment says why it does not hold "
     "unconditionally."),
    ("reread/G1_missing_only/bodies/<seed>/additional_rejections/**",
     "each additional rejection's own lever, median and threshold: the DIAGNOSIS of a "
     "rejection, which the card requires be diagnosed and never selected away. The count is "
     "cross-checked against the demoted list."),
    ("reread/G2_finite_only/bodies/<seed>/guard_demoted_count",
     "how many frames the guard demoted on the corrupted fixture; G2 bands the two errors, "
     "not the count, and the false-positive list beside it is reported."),
    ("reread/G2_finite_only/bodies/<seed>/*/i_elsewhere_deg",
     "the error away from the corrupted frames, reported as the comparison's floor."),
    ("reread/G2_finite_only/bodies/<seed>/runs/**",
     "where the corruption was placed, an input to the fixture."),
    ("reread/G2_finite_only/bodies/<seed>/guard_demoted_uncorrupted_frames/**",
     "the guard's false positives, reported; G2 bands the two errors."),
    ("reread/G2_finite_only/bodies/<seed>/guard_missed_corrupted_frames/**",
     "the guard's misses, reported; the miss RATE is cross-checked against this list."),
    ("reread/noise/**", "the noise model that MAKES the fixture: an input to S, not a "
                        "measurement of a candidate."),
    ("reread/fixture_attribution/**", "what the fixture is attributed to, a provenance "
                                      "block."),
    # --- the sigma-1.0 fixture, an immutable recorded STOP
    # --- the sigma-1.0 fixture: an IMMUTABLE RECORDED STOP, preserved and not re-banded
    ("sigma1/bodies/<seed>/arms/*/*/ii_step_deg",
     "`selector.json` records a STOP, and what stopped the step is the follower's "
     "separation on metric (i) -- which the gate now reads FROM THESE BODY ROWS, follower "
     "and winner alike, so metric (i) is no longer exempt at all. Metrics (ii) and (iii) "
     "are the frozen evidence beside that stop: preserved, not re-banded, because "
     "re-banding a recorded stop on the fixture it was recorded at is how a stop gets "
     "quietly relitigated. Every arm's population IS validated."),
    ("sigma1/bodies/<seed>/arms/*/*/iii_root_step_mm", "the same, metric (iii)."),
    ("sigma1/bodies/<seed>/arms/*/*/iii_rotational_compensation_step_mm",
     "the same, and a fourth metric the card's three-metric rule does not include."),
    ("sigma1/aggregated_median_of_six/**", "the same evidence, aggregated."),
    ("sigma1/bodies/<seed>/truth_trunk_tilt_deg/**",
     "the fixture's own truth tilt, an input to it."),
    ("sigma1/fixture_attribution/**",
     "WHY the pre-registered fixture failed to discriminate: the noiseless deficit per arm, "
     "the synthetic observation spread and the real take's own lever spread beside it. It is "
     "the DIAGNOSIS that led to the amended calibration, reported; the calibration it led to "
     "is read and banded in its own file."),
    ("sigma1/noise/sigma_px",
     "the pixel sigma that MAKES the fixture: an input to it. The scale applied to it is "
     "read and required to be 1.0 here."),
    ("sigma1/winner_beats_the_constant_it_removes",
     "the winner-vs-C-on-SOMA summary on the fixture that STOPPED. The reread's copy of it "
     "is derived cell by cell; this one is preserved with the rest of the stop's record."),
    # --- the calibration's frozen record
    ("calibration/calibration/take_target/**",
     "how the 8.7636 mm target was measured ON THE REAL TAKE -- the per-performer lever "
     "spreads it is the median of. The target is an INPUT to the calibration, read against "
     "the gate's own constant where the search uses it; its derivation is D8b's."),
    ("calibration/calibration/zero_noise_baseline/**",
     "the zero-noise floor, REPORTED FIRST AND NEVER SUBTRACTED by the card's own rule. It "
     "says what each estimator's error is with no observation noise at all, which is a "
     "property of the fixture's geometry and not of the candidate."),
    ("admissibility/take_target/**", "the same target derivation, copied into the amended "
                                     "file."),
    ("admissibility/zero_noise_baseline/**", "the same zero-noise floor, copied in."),
    ("admissibility/scope_of_the_match/**",
     "what the observed tolerance match does and does not cover, reported: the amendment is "
     "an OBSERVED TOLERANCE MATCH and never a monotonicity or uniqueness claim."),
    ("admissibility/admissibility/checks/*/decreases/**",
     "the rule's own listing of every earlier-to-later decrease it considered. Its LARGEST "
     "and its verdict are cross-checked against the gate's independent recomputation."),
    ("admissibility/admissibility/checks/*/sampled_crossings/**",
     "the rule's own listing of where the statistic crosses the target. Its sign-change "
     "COUNT and its verdict are cross-checked against the gate's."),
    ("admissibility/admissibility/checks/*/signs_in_sigma_order/<i>",
     "the sign of statistic-minus-target at each evaluation, the working behind that count."),
    ("admissibility/admissibility/checks/*/evaluations_inside_the_band/**",
     "which evaluations landed inside the tolerance band. The ACCEPTED one and its verdict "
     "are cross-checked against the gate's."),
    ("admissibility/admissibility/checks/*/tolerance_band_mm/<i>",
     "the band's two edges, which are the target plus and minus the tolerance the gate reads "
     "against its own constants."),
    ("admissibility/admissibility/replay_of_the_frozen_stopping_rule/"
     "evaluation_order_display/<i>",
     "the six-place display of each evaluated sigma; the EXACT order beside it is checked "
     "element by element against the evaluations, and the display rounding is the thing "
     "round 7 caught S being read at."),
    # --- the builds
    ("*/build_seconds", "wall-clock build time: a property of this machine, not of the "
                        "artifact."),
    ("control2/**", "the built control's remaining fields mirror the delivery report's; its "
                    "channels, its verdicts and its expectation are read."),
    # --- the photographs
    ("silhouette/subjects/**", "the per-subject overlap rows the eight cells summarise."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_deg/pitch_about_hip_line/*",
     "the tilt decomposed onto the hip line, reported so the DIRECTION of a control's error "
     "is legible; O1 and both must-fail clauses band the total angle. The shipping arm's "
     "decomposition is read leaf for leaf by the src == E identity."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_deg/roll/*",
     "the same decomposition, about the forward axis."),
    ("oracle/oracle/seeds/<seed>/arms/*/pelvis_vs_truth_deg/yaw/*",
     "the same decomposition, about the vertical."),
    ("projection/P2_on_the_oracle_bodies/seeds/<seed>/contacts/<i>",
     "how many contact frames each side carries on an oracle body, reported. P2 bands the "
     "travel at each run's first keyed sample, and the RUNS are checked by identity against "
     "the frozen mask -- which is the stronger statement."),
    ("reread/G1_missing_only/bodies/<seed>/guard_additionally_demoted/<i>",
     "which frames the guard additionally demoted; the COUNT is cross-checked against this "
     "list and each one is diagnosed in `additional_rejections`."),
    ("reread/G1_missing_only/pattern/<i>",
     "the missing-frame pattern that MAKES the G1 fixture: an input to it."),
    ("reread/aggregated_median_of_six/b_hipline_unguarded/**",
     "the aggregate of an arm outside the merge rule: report-only by the card."),
    ("reread/aggregated_median_of_six/world_vertical/**",
     "the aggregate of a CONTROL arm outside the merge rule: report-only by the card."),
    ("reread/aggregated_median_of_six/thorax_as_pelvis/**",
     "the aggregate of a CONTROL arm outside the merge rule: report-only by the card."),
    ("reread/aggregated_median_of_six/frozen_pitch_follower/**",
     "the follower's aggregates. Its clause is stated PER BODY on the bent tercile -- a "
     "median over six would hide the body that failed -- and is derived from the body rows."),
    ("reread/bodies/<seed>/truth_pelvis_tilt_deg/**",
     "the fixture's own truth tilt, an input to S and the reference the world-vertical "
     "control's stated limitation is read against."),
    ("reread/bodies/<seed>/truth_trunk_tilt_deg/**", "the same, for the trunk."),
    ("silhouette/preregistered_clause_verdicts/<subject>/reported_*/**",
     "the REPORTED silhouette cells, which the card does not band: B1 is stated on the eight "
     "pre-registered worsening-not-established cells and those are read by name."),
    ("silhouette/masks_copied_never_shared/**",
     "the silhouette's own copy rule per mask file, reported."),
    # --- P
    ("projection/subjects/<subject>/P3_travel_report/**",
     "P3 is a TRAVEL REPORT by the card and carries no band; its interval count is read."),
    ("projection/subjects/<subject>/P2_anchor_lock/runs/<i>/**",
     "each run's remaining per-joint maxima; both named joints per side are read and the "
     "run's own `holds` is cross-checked against them."),
    ("projection/P2_on_the_oracle_bodies/seeds/<seed>/run_measurements/<i>/**",
     "the same, on the oracle bodies."),
)
# EVERY BOOLEAN AND STRING THE GATE CONSUMES WITHOUT DERIVING OR CROSS-CHECKING IT, BY FAMILY.
# This table is checked against the Reader's own record on every run: a family that no longer
# appears is reported, and a trusted read with no entry here FAILS the gate.
TRUSTED_READ_JUSTIFICATIONS = (
    ("*/hygiene/delivered_files_vs_shipped/*/rebuild",
     "one of the two hashes the file-identity clause compares AGAINST EACH OTHER; neither is "
     "believed on its own."),
    ("*/hygiene/delivered_files_vs_shipped/*/shipped", "the other half of that comparison."),
    ("*/source_fingerprint/converter_sha256",
     "the sha256 of the converter that produced this report. It is not believed: it is "
     "COMPARED against the module this run resolves -- equal for the refactored stages, "
     "and for the historical hygiene arm required to differ from it and to equal the "
     "retained pre-change copy on this branch."),
    ("*/source_fingerprint/stage",
     "which source stage the report belongs to; compared against the card's own table of "
     "stages, and it decides which way the hash comparison must come out."),
    ("*/source_fingerprint/pelvis_mode",
     "the pelvis mode that stage runs in, compared against the same table. The hygiene arm "
     "and the tripwire both run C and are NOT the same stage: one is the module before the "
     "src change, the other the refactored module with the mode held."),
    ("projection/subjects/<subject>/P1_channel_preservation/authentication/glb_body_track_sha256",
     "one of the two hashes the authentication clause compares against each other."),
    ("projection/subjects/<subject>/P1_channel_preservation/authentication/recomputed_sha256",
     "the other half of that comparison."),
    ("*/subjects/<subject>/P1_channel_preservation/channels/root_translation_m/bit_identical",
     "an `np.array_equal` over two [frame, 3] root arrays. The report carries no per-frame "
     "constituent for this channel -- unlike every local, which carries `frames_that_differ` "
     "-- so the gate can believe it or drop the channel. Owed as instrument debt."),
    ("p_oracle/seeds/<seed>/*",
     "the oracle P1 report is BOOLEANS ONLY: ten `np.array_equal` results per body over the "
     "delivered arrays against the projection's own return. None of the arrays is in any "
     "report, so none can be recomputed here. `failing_channels` IS cross-checked against "
     "them, and the same comparison is made with constituents on the delivery, where the "
     "channels carry their own frame counts."),
    ("oracle/oracle/seeds/<seed>/O2_vs_baseline/contacts_identical",
     "an `np.array_equal` over two [frame, 2] contact masks held in the instrument's memory. "
     "P1 makes the same comparison on the same six bodies from the delivered bytes."),
    ("oracle/oracle/seeds/<seed>/spine_landmark_is_the_rigs_Spine_joint",
     "the fixture's own landmark contract, asserted by the instrument that built the "
     "fixture; the companion root offset is read as a number and required to be zero."),
    ("reread/G1_missing_only/bodies/<seed>/effective_masks_identical",
     "an `np.array_equal` over two [frame] masks that exist only inside the selector's run. "
     "The G1 clause is an EQUIVALENCE check and carries no band; the alternative to trusting "
     "it is not measuring G1 at all."),
    ("reread/G1_missing_only/bodies/<seed>/interpolated_arrays_bit_identical",
     "the same, over the [frame, 3] interpolated arrays."),
    ("*/hygiene/raw_triangulation_byte_identical_same_denominator/<subject>",
     "`np.array_equal` over two [frame, 19, 3] landmark arrays in the build's memory. FIVE "
     "instruments write this claim independently and the clause requires all of them; "
     "re-deriving it from the delivered `.npz` is owed as instrument debt."),
    ("*/hygiene/smoothed_triangulation_byte_identical/<subject>", "the same, smoothed."),
    ("silhouette/raw_triangulation_byte_identical", "the silhouette's own copy of it."),
    ("silhouette/smoothed_triangulation_byte_identical", "the silhouette's own copy of it."),
    ("take/take/subjects/<subject>/vs_baseline/landmarks_byte_identical_same_denominator",
     "the take instrument's own copy of it."),
    ("take/take/subjects/<subject>/vs_baseline/rest_skeleton_moved",
     "an equality over the two builds' rest translations, held in the instrument's memory; "
     "required to be FALSE, which is what makes D7c a converter-only change."),
    ("*/work_copied_never_symlinked",
     "the build's own copy rule: it records that the work tree was copied rather than "
     "symlinked into the shipped delivery. Required True."),
    ("*/hygiene/observations_byte_identical_before_and_after_the_build",
     "`np.array_equal` over the observation files the build consumed. Required True."),
    ("*/hygiene/observations_byte_identical_to_the_shipped_build", "the same, against the "
     "shipped build's own inputs. Required True."),
    ("silhouette/instrument_only",
     "the silhouette's own declaration that it selects nothing. Required True."),
    ("admissibility/admissibility/replay_of_the_frozen_stopping_rule/"
     "replay_reproduced_every_recorded_evaluation",
     "the replay's own claim that it reproduced every recorded evaluation. Required True; "
     "the evaluations it replays are read and cross-checked against the amended file's "
     "independent copy of all six body values."),
    ("oracle/pelvis_frame_source_in_src",
     "the mode the instrument resolved from src. It is not believed: it is compared against "
     "the mode the converter recorded on every seed AND against the arm the src path "
     "reproduces leaf for leaf."),
    ("tripwire/pelvis_mode_held",
     "the mode the tripwire requested; compared against the modes the converter recorded."),
    ("tripwire/diagnostics/pelvis_frame/<i>/mode", "the other half of that comparison."),
    ("delivery/diagnostics/pelvis_frame/<i>/mode",
     "the mode the delivered run report records, required to be the shipping mode on both "
     "performers."),
    ("reread/sigma_scale_repr",
     "the sigma's exact repr, compared against the sigma it spells."),
    ("reread/winner/arm", "compared against `winner/mode` through the named mapping, and "
                          "against the mode the six cells imply."),
    ("sigma1/winner/arm",
     "the arm the pre-registered fixture's follower is measured against. Compared against "
     "`sigma1/winner/mode` through the same named mapping, and the row it selects is read "
     "from the body constituents rather than from the follower table's copy."),
    ("sigma1/winner/mode", "the other half of that comparison."),
    ("reread/winner/mode", "the other half of that comparison."),
    ("reread/S_verdict", "S's own verdict string, required to be PROCEED beside the six "
                         "cells the gate recomputes."),
    ("silhouette/statistics/every_arm_on_identical_draws",
     "the silhouette's own declaration that every arm was scored on the same bootstrap "
     "draws. It is not believed on its own: the card BANDS identical draws, so its clause "
     "also measures the property -- every part of a cut must report the same `draws_used` -- "
     "and reads the run's block, seed and lag-1 autocorrelation beside it."),
    ("silhouette/masks_copied_never_shared/mask/*/byte_identical",
     "a byte comparison of the mask cache this run read against the cache the earlier runs "
     "read, made by the instrument over files that are not in any report. Required True: B1 "
     "compares two builds against the SAME photographs, and the MAMMA oracle's own agreement "
     "to 0.0 is the second, independent half of that argument."),
    ("admissibility/S_status",
     "the amended calibration file's statement of where S stands. It is not believed: it is "
     "REQUIRED to begin PENDING, because calibration REACHED is not S PROCEED and a file "
     "that said otherwise would be claiming a verdict it does not hold."),
    ("b2/triangulated_landmarks_byte_identical_across_arms/<subject>/*",
     "`np.array_equal` over the triangulated landmark arrays of one build, per subject. "
     "These ARE the constituents `same_denominator` is derived from, and four other "
     "instruments make the same claim in the clause beside this one."),
    ("oracle/oracle/seeds/<seed>/O2_vs_baseline/arm",
     "which arm O2 scored; required to be the shipping src path."),
    ("oracle/oracle/seeds/<seed>/O2_vs_baseline/bit_identity_claimed",
     "the instrument's own declaration that it claims no bit identity; required FALSE, "
     "because a pelvis frame is whole-take and O2 is a band, not an identity."),
)
OUTSIDE = {
    "the delivered run-report records the mode and the guard's demoted frames":
        "a REPORT clause; the card does not band the diagnostics block. Astra's round 2 "
        "accepted this exclusion explicitly.",
}


# --------------------------------------------------------------- the coverage audit
# Astra's round 7: INVERT THE UNREAD CLASS. Six rounds of review found stored summaries the
# gate read instead of their constituents, one family at a time, and each round was answered
# by reading that family. The class only ends when the gate can say, of EVERY leaf in every
# report it reads, either "a clause reads it" or "here is why it is not a measurement I band".
#
# So every leaf the gate does not read is classified:
#
#   LABEL        a string that names something (a title, a rule, a note)
#   PROVENANCE   a string that identifies an input or an output (a hash, a path, a mode)
#   DIAGNOSTIC   a number or boolean inside a subtree the card puts outside the predicate
#   MEASUREMENT  any other number or boolean -- something that summarises or constitutes a
#                measurement
#
# and every MEASUREMENT leaf under a report any clause reads is a GAP unless a justification
# below names it. The justifications are PATTERNS over normalised paths, so they are families
# and not a list of 9,000 leaves; a pattern that matches nothing is reported too, because a
# stale justification is a hole that looks like a cover.
DIAGNOSTIC_SEGMENTS = {"diagnostics", "blind_to", "note", "notes", "definitions",
                       "truth_motion_blind_spot", "sensitivity_note", "denominator_note",
                       "landmark_note", "keep_mask_diagnosis", "what_it_is", "watcher"}
PROVENANCE_WORDS = ("sha", "hash", "path", "module", "file", "dir", "source", "repr", "mode",
                    "arm", "seed", "version", "commit", "time", "stamp", "output")


def normalise(path) -> str:
    """One naming for a family of leaves: seeds, performers and list indices collapse."""
    out = []
    for step in map(str, path):
        if step in ORACLE_SEEDS:
            out.append("<seed>")
        elif step in PERFORMERS:
            out.append("<subject>")
        elif step.lstrip("-").isdigit():
            out.append("<i>")
        else:
            out.append(step)
    return "/".join(out)


def classify_leaf(path, kind) -> str:
    if kind in ("map", "list"):
        return "CONTAINER"
    if any(step in DIAGNOSTIC_SEGMENTS for step in map(str, path)):
        return "DIAGNOSTIC"
    if kind == "string":
        last = str(path[-1]).lower()
        # A SAVED VERDICT OR STATUS IS A CLASSIFICATION, not a name: round 2's whole attack
        # was a gate reading one instead of deriving it. Such a string is a MEASUREMENT.
        if last in ("verdict", "status", "s_verdict", "s_status", "p1", "p2"):
            return "MEASUREMENT"
        return ("PROVENANCE" if any(word in last for word in PROVENANCE_WORDS)
                else "LABEL")
    if kind in ("number", "bool"):
        return "MEASUREMENT"
    return "LABEL"


def matches(pattern, normalised) -> bool:
    """Segment by segment. `**` matches any remaining tail; within one segment the usual
    glob applies, so `reported_*` names a family of sibling keys."""
    want, have = pattern.split("/"), normalised.split("/")
    for index, step in enumerate(want):
        if step == "**":
            return True
        if index >= len(have) or not fnmatch.fnmatchcase(have[index], step):
            return False
    return len(have) == len(want)


def justification_for(normalised, table):
    for pattern, reason in table:
        if matches(pattern, normalised):
            return pattern, reason
    return None, None


def walk_leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk_leaves(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk_leaves(value, path + (str(index),))
    else:
        yield path, kind_of(node)


def coverage_audit(reports: dict, built: dict) -> dict:
    """Every leaf the gate does not read, classified, and every MEASUREMENT one justified."""
    touched = {tuple(p.split("/")) for p in built["touched"]}
    read_reports = {p[0] for p in touched}
    classes: dict[str, int] = {}
    per_subtree: dict[str, dict[str, int]] = {}
    gaps: dict[str, int] = {}
    used: set[str] = set()
    for path, kind in walk_leaves(reports):
        if not path or path in touched:
            continue
        label = classify_leaf(path, kind)
        classes[label] = classes.get(label, 0) + 1
        per_subtree.setdefault(path[0], {})
        per_subtree[path[0]][label] = per_subtree[path[0]].get(label, 0) + 1
        if label != "MEASUREMENT" or path[0] not in read_reports:
            continue
        pattern, _reason = justification_for(normalise(path), UNREAD_MEASUREMENTS_JUSTIFIED)
        if pattern is None:
            key = normalise(path)
            gaps[key] = gaps.get(key, 0) + 1
        else:
            used.add(pattern)
    dead = [pattern for pattern, _ in UNREAD_MEASUREMENTS_JUSTIFIED if pattern not in used]
    return {
        "rule": ("every leaf no clause reads is classified LABEL / PROVENANCE / DIAGNOSTIC / "
                 "MEASUREMENT; every MEASUREMENT leaf under a report any clause reads is a "
                 "GAP unless a justification names its family"),
        "reports_read_by_some_clause": sorted(read_reports),
        "leaves_read_by_a_clause": len(touched),
        "unread_by_class": classes,
        "unread_by_report_and_class": per_subtree,
        "justified_families": len(used),
        "justifications": [{"pattern": p, "why": w} for p, w in
                           UNREAD_MEASUREMENTS_JUSTIFIED],
        "justifications_matching_nothing": dead,
        "gaps": sorted(gaps),
        "gap_leaves": sum(gaps.values()),
        "verdict": ("COVERED" if not gaps and not dead else "GAPS"),
    }


def saved_value_inventory(built: dict) -> dict:
    """The inventory REGENERATED FROM THE GATE'S READS, never from memory.

    Astra's round 7: the hand-written list claimed "every other saved boolean is derived or
    cross-checked" and that was false -- the oracle P1 clause consumes ten more. So the
    inventory is now the set difference the Reader itself records: every boolean and string
    the gate consumed, minus every one it cross-checked against a derived value. Each
    surviving family must be named below, and one that is not is reported as unjustified.
    """
    trusted, used = {}, set()
    for path, kind in built["kinds"].items():
        if kind not in ("bool", "string") or path in built["cross_checked"]:
            continue
        key = normalise(path)
        trusted.setdefault(key, 0)
        trusted[key] += 1
    rows, unjustified = [], []
    for key in sorted(trusted):
        pattern, reason = justification_for(key, TRUSTED_READ_JUSTIFICATIONS)
        if pattern is None:
            unjustified.append(key)
        else:
            used.add(pattern)
            rows.append({"family": key, "leaves": trusted[key], "why": reason})
    dead = [pattern for pattern, _ in TRUSTED_READ_JUSTIFICATIONS if pattern not in used]
    return {
        "method": ("generated from the Reader's own record: every boolean and string the "
                   "gate read, minus every one it cross-checked against a value derived "
                   "from that leaf's own constituents"),
        "cross_checked_reads": len(built["cross_checked"]),
        "trusted_families": rows,
        "justifications_matching_nothing": dead,
        "unjustified": unjustified,
        "verdict": ("NAMED" if not unjustified and not dead
                    else "UNJUSTIFIED READS" if unjustified else "STALE JUSTIFICATIONS"),
    }


def load_all() -> dict:
    out = {}
    for key, name in REPORTS.items():
        path = BASE / name
        out[key] = json.loads(path.read_text()) if path.exists() else {}
    return out


def main() -> int:
    reports = load_all()
    built = build(reports)
    coverage = coverage_audit(reports, built)
    inventory = saved_value_inventory(built)
    fuzz_path = BASE / "gate-fuzz.json"
    fuzz = json.loads(fuzz_path.read_text()) if fuzz_path.exists() else {
        "status": "not run -- `tools/compare/d7c_gate_fuzz.py` has not been executed"}
    report = {
        "title": ("D7c -- the pelvis on the rig's own rest. Every clause, predicted / "
                  "measured / verdict, DERIVED from the reports."),
        "shipping_mode": "E_rig_rest_kabsch",
        "four_rules": [
            "every value is DERIVED from named constituents or CROSS-CHECKED against them; a "
            "stored summary that disagrees with its constituents is a FAIL",
            "a MISSING field or set member is a FAIL, never a no-op",
            "every set is checked by IDENTITY -- files, seeds, performers, cells, arms, and "
            "contact runs by their (side, start, end) identity from the frozen mask",
            "every MEASUREMENT leaf no clause reads is justified BY NAME, and every boolean "
            "or string the gate consumes without deriving it is named in an inventory "
            "GENERATED from the gate's own reads",
        ],
        "the_two_recorded_stops": (
            "S at the card's own fixture (sigma 1.0) and the calibration under its own frozen "
            "monotonicity precondition both read FAIL and stay that way. `selector.json` and "
            "`selector-calibrated.json` are immutable; the two amendments are POST HOC."),
        "clauses": built["clauses"],
        "merge_rule": {
            "source": ("the D7c card: hygiene AND the tripwire AND O1 AND O2 AND P1 and P2 on "
                       "the take and every seed AND S AND B1 on both performers AND B2's "
                       "same-denominator PASS; O3, B3, B4, B5, B6 report"),
            "conjuncts": built["conjuncts"],
            "recorded_stops_still_fail": built["recorded_stops_still_fail"],
            "not_yet_measured": built["not_yet_measured"],
            "verdict": built["verdict"],
        },
        "deliberately_outside_the_predicate": OUTSIDE,
        "measurement_coverage": coverage,
        "saved_value_inventory": inventory,
        "leaf_level_fuzz": fuzz,
    }
    if coverage["verdict"] != "COVERED" or inventory["verdict"] != "NAMED":
        report["merge_rule"]["verdict"] = "NO MERGE"
        report["merge_rule"]["coverage_blocked"] = (
            f"{coverage['gap_leaves']} unjustified MEASUREMENT leaves and "
            f"{len(inventory['unjustified'])} unjustified trusted reads")
    (BASE / "gate.json").write_text(json.dumps(report, indent=1))
    for entry in built["clauses"]:
        print(f"{entry['verdict']:7s} {entry['clause'][:74]:74s} "
              f"{str(entry['measured'])[:44]}")
    print()
    print("MERGE RULE:", json.dumps(built["conjuncts"], indent=1))
    print(f"coverage: {coverage['verdict']} -- {coverage['gap_leaves']} unjustified "
          f"MEASUREMENT leaves in {len(coverage['gaps'])} families, "
          f"{len(coverage['justifications_matching_nothing'])} justifications matching "
          f"nothing; {coverage['leaves_read_by_a_clause']} leaves read by a clause")
    print(f"saved values: {inventory['verdict']} -- {len(inventory['trusted_families'])} "
          f"trusted families named, {inventory['cross_checked_reads']} reads cross-checked, "
          f"{len(inventory['unjustified'])} unjustified")
    print("verdict:", report["merge_rule"]["verdict"],
          "| missing:", built["not_yet_measured"])
    print(f"fuzz: {fuzz.get('summary', fuzz.get('status'))}")
    print(f"\nwrote {BASE / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
