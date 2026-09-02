#!/usr/bin/env python3
"""D1 (fix): the gate on the repaired handedness. Same bands on the before and the after.

WHAT THIS IS. `tools/compare/facing_location.py` LOCATED the defect; this scores the
repair. It computes nothing itself -- every figure comes out of an instrument's own JSON
report -- and its whole job is to apply one set of bands to two arms and say which passed.

THE TWO ARMS, AND WHY THEY ARE NOT THE SAME BYTES SCORED TWICE
  * BEFORE: `artifacts/compare/facing-location.json`, the 2026-09-01 delivery scored by
    the pre-repair code. It is NOT recomputed here, and it must not be: `triple()`'s
    `across` comes from forward-kinematic positions, so running the repaired build's FK
    against the old rotations would flip the handedness sign for a reason that has
    nothing to do with the delivery. Old bytes, old code, quoted as they were measured.
  * AFTER: the same instrument pointed at a rebuild under `artifacts/compare/d1-fix/`.
    The delivered directory is never written to.
The "the current asset must FAIL the repaired gate" control is therefore not a separate
construction -- it is the BEFORE arm, put through the identical band function.

THE FOUR DEGENERATES. Three of them are exact transforms of the AFTER arm's own delivered
torso frame, computed by `facing_location.py` and read here:
  * a 180 degree yaw -- must FAIL the forward-dot, PASS handedness (a proper rotation)
  * a 90 degree yaw -- must FAIL the forward-dot
  * a sagittal mirror -- must PASS the forward-dot and FAIL handedness. It is the reason
    the handedness band exists at all: a mirrored human faces exactly where it faced, so
    no forward-dot and no silhouette can reject it.
  * the pre-repair build itself, which is the BEFORE arm.

WHAT THE BANDS ARE
  forward-dot   median > +0.9 AND the moving-block bootstrap's lower bound > 0, on both
                subjects, against BOTH references (our capture and MAMMA), for Hips, the
                chest group, Neck, Head and the mesh's own nose.
  feet          must NOT move: the after median stays inside the before interval. The feet
                were already right and the repair must not have touched them.
  handedness    the delivered rig's triple-product sign must agree with our capture AND
                with MAMMA, read BOTH ways -- forward from the feet, forward from the
                torso. Those two arms disagreeing was the defect stated as one bit.
  regressions   each named instrument's own figures, before against after.

MAMMA is an instrument here as everywhere: it reports a second independent opinion on
which way the performers face and selects nothing.

    .venv/bin/python tools/compare/facing_fix_gate.py
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "artifacts/compare/d1-fix"

BEFORE = ROOT / "artifacts/compare/facing-location.json"
AFTER = FIX / "facing-after.json"
OUT = FIX / "facing-fix-gate.json"

SUBJECTS = ("subject_00", "subject_01")
FORWARD_PARTS = ("delivered_torso_Hips", "delivered_torso_Chest", "delivered_Neck",
                 "delivered_Head", "delivered_MESH_nose")
REFERENCES = ("vs_our_capture_forward", "vs_mamma_forward")
MEDIAN_BAND = 0.9


def digest(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# --------------------------------------------------------------------------- the bands

def forward_dot_band(report: dict) -> dict:
    """median > +0.9 and bootstrap lower bound > 0, every subject, every reference."""
    rows, failures = {}, []
    for subject in SUBJECTS:
        dots = report.get("forward_dot", {}).get(subject, {})
        for part in FORWARD_PARTS:
            for reference in REFERENCES:
                entry = dots.get(part, {}).get(reference)
                key = f"{subject}/{part}/{reference}"
                if entry is None:
                    rows[key] = {"value": None, "verdict": "MISSING"}
                    failures.append(key)
                    continue
                median, ci = entry["median"], entry.get("ci95")
                ok = median > MEDIAN_BAND and ci is not None and ci[0] > 0.0
                rows[key] = {"median": median, "ci95": ci,
                             "band": f"median > +{MEDIAN_BAND} and ci95 lower bound > 0",
                             "verdict": verdict(ok)}
                if not ok:
                    failures.append(key)
    return {"cells": rows, "cells_failing": failures, "verdict": verdict(not failures)}


def feet_band(before: dict, after: dict) -> dict:
    """The feet were already right. The repair must not move them: the after median has
    to sit inside the before interval, on both subjects and both references."""
    rows, failures = {}, []
    for subject in SUBJECTS:
        for reference in REFERENCES:
            b = before["forward_dot"][subject]["delivered_feet"][reference]
            a = after["forward_dot"][subject]["delivered_feet"][reference]
            low, high = b["ci95"]
            ok = low <= a["median"] <= high
            key = f"{subject}/{reference}"
            rows[key] = {"before_median": b["median"], "before_ci95": [low, high],
                         "after_median": a["median"], "after_ci95": a["ci95"],
                         "band": "the after median stays inside the before interval",
                         "verdict": verdict(ok)}
            if not ok:
                failures.append(key)
    return {"cells": rows, "cells_failing": failures, "verdict": verdict(not failures)}


def handedness_band(report: dict) -> dict:
    """One sign, read six ways. The two delivered arms agreeing with each other AND with
    both references is the repair stated as one bit; a yaw cannot change any of them and
    a mirror must change all of them."""
    arms = report.get("triple_product", {}).get("arms", {})
    reference_arms = ("our_triangulated_capture", "mamma_pred_joints")
    delivered_arms = ("delivered_rig_forward_from_its_FEET",
                      "delivered_rig_forward_from_its_TORSO",
                      "delivered_MESH_surface")
    rows, failures = {}, []
    for subject in SUBJECTS:
        signs = {arm: arms.get(arm, {}).get(subject, {}).get("sign_median")
                 for arm in reference_arms + delivered_arms}
        agree = len({s for s in signs.values() if s is not None}) == 1 and None not in signs.values()
        rows[subject] = {
            "signs": signs,
            "band": ("every arm carries the same sign -- our capture, MAMMA, the delivered "
                     "rig read from its feet, the same rig read from its torso, and the "
                     "delivered mesh's own skin"),
            "verdict": verdict(agree),
        }
        if not agree:
            failures.append(subject)
    return {"per_subject": rows, "subjects_failing": failures, "verdict": verdict(not failures)}


def control_verdicts(after: dict, before: dict) -> dict:
    """The degenerates. Each must fail the band it is built to fail and pass the other,
    which is what makes the pair of bands non-redundant."""
    controls = after.get("delivered_rig_controls", {})

    def read(name: str) -> dict:
        rows = {}
        for subject in SUBJECTS:
            entry = controls.get(name, {}).get(subject)
            if entry is None:
                rows[subject] = None
                continue
            rows[subject] = {
                "forward_dot_median": entry["forward_dot_vs_our_capture"]["median"],
                "forward_dot_ci95": entry["forward_dot_vs_our_capture"]["ci95"],
                "handedness_sign": entry["handedness_sign_median"],
            }
        return rows

    def passes_forward(rows: dict) -> bool:
        return all(r is not None and r["forward_dot_median"] > MEDIAN_BAND
                   and r["forward_dot_ci95"][0] > 0.0 for r in rows.values())

    reference_sign = handedness_band(after)["per_subject"]["subject_00"]["signs"][
        "our_triangulated_capture"]

    def passes_handedness(rows: dict) -> bool:
        return all(r is not None and r["handedness_sign"] == reference_sign
                   for r in rows.values())

    out: dict = {}
    identity = read("DELIVERED_untransformed")
    out["the_repaired_build_itself"] = {
        "what_it_is": "the after arm, untransformed -- the row every control is measured "
                      "against, on the gate's own scale",
        "rows": identity,
        "expected": "PASS forward-dot, PASS handedness",
        "forward_dot": verdict(passes_forward(identity)),
        "handedness": verdict(passes_handedness(identity)),
        "verdict": verdict(passes_forward(identity) and passes_handedness(identity)),
    }
    for name, label, want_forward, want_hand in (
        ("DELIVERED_CONTROL_yaw180",
         "the repaired delivered torso turned 180 degrees about each subject's own "
         "vertical -- a PROPER rotation, so it cannot change the handedness sign",
         False, True),
        ("DELIVERED_CONTROL_yaw90",
         "the same, 90 degrees -- the forward-dot must collapse toward zero", False, True),
        ("DELIVERED_CONTROL_sagittal_mirror",
         "the repaired delivered torso reflected in each subject's own sagittal plane: "
         "left and right exchanged, FACING UNTOUCHED. It must PASS the forward-dot and "
         "FAIL handedness -- the whole reason the handedness band exists, and the reading "
         "the D1 gate card asked for that no forward-dot can deliver",
         True, False),
    ):
        rows = read(name)
        got_forward, got_hand = passes_forward(rows), passes_handedness(rows)
        out[name] = {
            "what_it_is": label,
            "rows": rows,
            "expected": f"forward-dot {verdict(want_forward)}, handedness {verdict(want_hand)}",
            "forward_dot": verdict(got_forward),
            "handedness": verdict(got_hand),
            "verdict": verdict(got_forward == want_forward and got_hand == want_hand),
        }

    # The pre-repair build, scored by the identical band function. If this passes, the
    # gate is broken and nothing else in this file means anything.
    before_forward = forward_dot_band(before)
    before_hand = handedness_band(before)
    out["the_pre_repair_build"] = {
        "what_it_is": ("the 2026-09-01 delivery as it stands, scored by the SAME band "
                       "function -- old bytes measured by the old code, never recomputed "
                       "under the repaired FK"),
        "delivery_sha256": before.get("sha_chain", {}).get("sha256", {}),
        "expected": "FAIL forward-dot; its two delivered handedness arms DISAGREE",
        "forward_dot": before_forward["verdict"],
        "handedness": before_hand["verdict"],
        "forward_dot_cells_failing": len(before_forward["cells_failing"]),
        "handedness_signs": {s: before_hand["per_subject"][s]["signs"] for s in SUBJECTS},
        "verdict": verdict(before_forward["verdict"] == "FAIL"),
    }
    return out


# ------------------------------------------------------------------------- regressions

def regressions(paths: dict[str, tuple[Path, Path]]) -> dict:
    """Each instrument's own report, before against after. Nothing is recomputed."""
    out: dict = {}

    def load(pair: tuple[Path, Path]) -> tuple[dict | None, dict | None]:
        return (json.loads(pair[0].read_text()) if pair[0].exists() else None,
                json.loads(pair[1].read_text()) if pair[1].exists() else None)

    # ---- I1: the canonical round trip. Its ORACLE arm feeds the rig's own output back
    # through the converter, so it is closed and must be invariant under a consistent
    # relabelling -- 0.00 mm on the legs and 36-47 mm on the arms, exactly, not nearly.
    before, after = load(paths["retarget_cost"])
    if before and after:
        rows = {}
        ok = True
        for subject in SUBJECTS:
            for group in ("legs", "arms", "torso"):
                b = before["subjects"][subject]["arms"]["ORACLE_roundtrip_canonical"][
                    "per_group"][group]["median_mm"]
                a = after["subjects"][subject]["arms"]["ORACLE_roundtrip_canonical"][
                    "per_group"][group]["median_mm"]
                same = abs(a - b) <= 0.01
                rows[f"{subject}/{group}"] = {
                    "before_mm": b, "after_mm": a, "delta_mm": round(a - b, 4),
                    "band": "identical to 0.01 mm -- a consistent relabel is a no-op here",
                    "verdict": verdict(same)}
                ok = ok and same
        out["I1_canonical_round_trip"] = {"cells": rows, "verdict": verdict(ok)}

    # ---- rungs 7 and 11: the scoreboard's by-name medians. A by-name score cannot see a
    # consistent relabelling either, so these must not move.
    before, after = load(paths["scoreboard"])
    if before and after:
        rows, ok = {}, True
        for subject in SUBJECTS:
            for arm in ("capture", "canon", "sized"):
                b = before["subjects"][subject]["median_mm"][arm]
                a = after["subjects"][subject]["median_mm"][arm]
                same = abs(a - b) <= 0.5
                rows[f"{subject}/{arm}"] = {
                    "before_mm": b, "after_mm": a, "delta_mm": round(a - b, 4),
                    "band": "within 0.5 mm; a by-name score is blind to the relabel",
                    "verdict": verdict(same)}
                ok = ok and same
        out["rungs_7_and_11_scoreboard"] = {"cells": rows, "verdict": verdict(ok)}

    # ---- rung 9: the shipped head. Read the report's own gated figures.
    before, after = load(paths["head_gate"])
    if before and after:
        out["rung_9_head_gate"] = {
            "before": before.get("summary", before),
            "after": after.get("summary", after),
            "band": "the gate's own verdict must not turn from pass to fail",
        }

    # ---- I6: the silhouette. Must not FALL on any of the 8 cells.
    before, after = load(paths["silhouette"])
    if before and after:
        out["I6_silhouette"] = _silhouette_rows(before, after)
    return out


def _silhouette_rows(before: dict, after: dict) -> dict:
    """IoU per (camera, subject) for the delivered arm, before against after."""
    def cells(report: dict) -> dict:
        found = {}
        for key, value in _walk(report):
            if isinstance(value, dict) and "iou" in value:
                found[key] = value["iou"]
        return found

    b, a = cells(before), cells(after)
    shared = sorted(set(b) & set(a))
    rows, ok = {}, True
    for key in shared:
        fell = a[key] < b[key] - 1e-9
        rows[key] = {"before": b[key], "after": a[key], "delta": a[key] - b[key],
                     "verdict": verdict(not fell)}
        ok = ok and not fell
    return {"cells": rows, "cells_compared": len(shared),
            "band": "IoU must not fall on any cell", "verdict": verdict(ok)}


def _walk(node, prefix: str = ""):
    if isinstance(node, dict):
        yield prefix, node
        for key, value in node.items():
            yield from _walk(value, f"{prefix}/{key}" if prefix else key)


BLIND = (
    "Everything `facing_location.py` is blind to, this inherits: it scores ORIENTATION "
    "and nothing else, so the clavicle-origin error, the bone lengths and the root/hip "
    "placement offset are untouched and none of them is fixed by fixing this. A "
    "forward-dot near +1 does not mean the facing is accurate to a degree -- it means it "
    "is not reversed; the residual is the two references' own disagreement, and the "
    "ORACLE row states that ceiling. The handedness figure is a SIGN: it detects a mirror "
    "and says nothing about how good a non-mirrored answer is. "
    "\n\nWhat is NEW and blind here, and belongs to this gate rather than to D1: "
    "(a) the repaired asset was DERIVED by relabelling the delivered one, not regenerated "
    "through the pinned Blender/MPFB worker, so it proves the relabel is consistent and "
    "does NOT prove the worker produces the same file -- a provider run is still owed; "
    "(b) no band here can see the finger CURL direction, which flipped sign with the "
    "skeleton and is a pose rather than a measurement; only a render shows it; "
    "(c) the SOMA and speech-motion retarget boundaries have no capture fixture on this "
    "footage, so their side convention is asserted by unit test and by nothing measured; "
    "(d) one take, two performers, 150 correlated frames -- as before, nothing here "
    "generalises past this fixture."
)

PLAN_CORRECTIONS = [
    "THE MIRROR HAD FIVE SITES, NOT THREE. The D1 (locate) review named "
    "`DETAILED_HUMANOID`, the MPFB asset, and `CANONICAL_HEAD_AXES`. Two more were found "
    "by grep during the repair. (4) `body_provider.DEFAULT_MPFB_JOINT_MAP` is where the "
    "mirror was BORN: it maps AutoAnim `Left*` onto MPFB `.R` bones deliberately, with "
    "the reason written beside it -- 'MPFB's anatomical .L is positive Blender X, whereas "
    "AutoAnim's canonical Left joints are negative X'. The asset is not an independent "
    "site; it is this map's output. (5) `soma_motion._DELTA_MAPPING` COMPENSATED for the "
    "mirror, sending our `Left*` joints to SOMA's `Right*` ones, with the same reason "
    "written beside it. Removing the mirror without removing that compensation would have "
    "shipped every SOMA performer with their arms and legs exchanged, and nothing on that "
    "lane has a capture fixture that would have caught it.",

    "THE SKELETON CONTRADICTED ITS OWN PUBLISHED CONTRACT, which is a defect readable with "
    "no capture, no reference and no asset. `HumanoidSkeleton.as_dict()` declares "
    "`handedness: right, up_axis: +Y, forward_axis: +Z`; in that frame the anatomical left "
    "is `up x forward` = +X, and every joint named Left sat at -X. The D1 (locate) argument "
    "reached the same conclusion through the mesh's nose; this is the shorter proof and it "
    "is now a unit test.",

    "`speech_motion` WAS MIRRORED TOO, and in the opposite direction, which is why a "
    "blast-radius grep was necessary rather than optional. Its bundled SMPL-X55 map sends "
    "`Left*` to `left_*`, and SMPL-X's anatomical left is +X -- so that lane was wrong "
    "BEFORE the repair and is right after it, while the SOMA lane was right before and "
    "needed changing. Two retarget boundaries, opposite errors, one convention.",

    "THE REVIEW'S PROPOSED FIX WOULD HAVE TURNED THE MESH INSIDE OUT. It proposed negating "
    "every vertex's X and swapping each vertex's Left/Right skin weights. On a bilaterally "
    "symmetric bind pose that reaches the same joint positions, and it is a REFLECTION of "
    "the surface: triangle winding reverses and every normal inverts. Measured, not argued "
    "-- the mesh's signed volume changes sign (`tests/test_facing_fix.py`). No band in this "
    "gate could see it: not a forward-dot, not a handedness sign, not a silhouette IoU, "
    "because an inside-out mesh has the same outline. The relabel route moves no vertex.",

    "THE ASSET IS NOT A SEPARATE ASSET CHANGE. Because the MPFB rig is bilaterally "
    "symmetric, regenerating under the repaired joint map is exactly a PERMUTATION of the "
    "existing asset's per-joint arrays plus a remap of its per-vertex bone indices. "
    "`tools/compare/d1_asset_relabel.py` applies that derivation and asserts the parent "
    "table survives it. What it does NOT do is prove the pinned worker produces the same "
    "bytes; `artifact.request_sha256` still binds the derived asset to the PRE-repair "
    "provider request, and a real provider run is owed before anything ships.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=BEFORE)
    parser.add_argument("--after", type=Path, default=AFTER)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    before = json.loads(arguments.before.read_text())
    after = json.loads(arguments.after.read_text())

    forward = forward_dot_band(after)
    feet = feet_band(before, after)
    hands = handedness_band(after)
    controls = control_verdicts(after, before)
    regression = regressions({
        "retarget_cost": (ROOT / "artifacts/compare/retarget-cost.json",
                          FIX / "retarget-cost.json"),
        "scoreboard": (ROOT / "artifacts/compare/scoreboard-commercial-multiview-soma77.json",
                       ROOT / "artifacts/compare/scoreboard-d1-fix.json"),
        "head_gate": (ROOT / "artifacts/head/head-gate-shipped.json",
                      FIX / "head-gate-shipped.json"),
        "silhouette": (ROOT / "artifacts/compare/silhouette.json",
                       FIX / "silhouette.json"),
    })

    all_pass = all(block["verdict"] == "PASS" for block in (forward, feet, hands)) and all(
        entry["verdict"] == "PASS" for entry in controls.values())

    report = {
        "step": "D1 (fix)",
        "title": "the left/right naming mirror, repaired and gated -- built on a branch, "
                 "not merged",
        "fixture": "pushing_and_lifting_from_ground",
        "route_shipped": {
            "name": "RELABEL",
            "what": ("negate X throughout the canonical rest skeleton so bones named Left "
                     "sit at +X, repair `DEFAULT_MPFB_JOINT_MAP` so each side maps to its "
                     "own MPFB side, derive the asset by permuting which AutoAnim name "
                     "each MPFB bone answers to, flip both `_frame_alignment` source "
                     "secondary axes from (-1,0,0) to (1,0,0), restate "
                     "`CANONICAL_HEAD_AXES`, flip the finger curl sign with the skeleton, "
                     "and remove `soma_motion`'s compensating side swap"),
            "why": ("the bound mesh is untouched -- no vertex, no triangle, no weight "
                    "value moves, and the joint name order is preserved, so tracks, GLB "
                    "node order and every downstream consumer keep their layout"),
        },
        "route_rejected": {
            "name": "GEOMETRY (negate vertex X and swap Left/Right skin weights)",
            "why": ("it is a REFLECTION of the surface. Triangle winding reverses and "
                    "every normal inverts, so the character renders inside out, and no "
                    "band in this gate can see it -- the joints land in the same places, "
                    "the forward-dot is unchanged, the handedness sign is unchanged and "
                    "the silhouette outline is unchanged."),
            "measured": ("the mesh's signed volume under the proposed transform, with the "
                         "original triangle order: it changes sign exactly. "
                         "tests/test_facing_fix.py::"
                         "test_the_geometry_route_turns_the_mesh_inside_out"),
        },
        "input_sha256": {
            "before_report": digest(arguments.before),
            "after_report": digest(arguments.after),
            "before_delivery": before.get("sha_chain", {}).get("sha256", {}),
            "after_delivery": after.get("sha_chain", {}).get("sha256", {}),
        },
        "gates": {
            "forward_dot": forward,
            "feet_must_not_move": feet,
            "handedness": hands,
        },
        "degenerate_controls": controls,
        "regressions": regression,
        "verdict": verdict(all_pass),
        "blind_to": BLIND,
        "plan_corrections": PLAN_CORRECTIONS,
        "sequencing": (
            "NOT MERGED and not mergeable yet. `tools/compare/provenance.py` gates the "
            "delivery on the THORAX_SMOOTHING_FRAMES leak, and the derived asset owes a "
            "real provider run. Merging is the registry owner's call, after review and "
            "after the delivery decision lands."
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {arguments.out}")
    for name, block in report["gates"].items():
        print(f"  {block['verdict']:4s}  {name}")
    for name, entry in controls.items():
        print(f"  {entry['verdict']:4s}  control: {name}")
    for name, block in regression.items():
        print(f"  {block.get('verdict', '----'):4s}  regression: {name}")
    print(f"  {report['verdict']}  overall")


if __name__ == "__main__":
    main()
