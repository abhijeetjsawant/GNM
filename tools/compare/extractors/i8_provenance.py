"""Ladder extractor for step **I8** -- the provenance audit and the thorax re-selection.

`tools/compare/ladder.py` is Fable's to edit (one registry owner, or the registrations
collide), so this ships as a standalone stub in the shape of that file's `x_*` functions:
one callable taking the ladder's context dict and returning `(figures, controls)`, each a
list of `fig(...)`-shaped dicts. `fig` and `_load` are redefined here rather than imported
so the stub is self-contained and cannot break `ladder.py` by importing it.

To register, add to the rung Fable chooses:

    from extractors.i8_provenance import x_provenance
    ... extract=x_provenance,
        reports=["artifacts/compare/provenance.json",
                 "artifacts/compare/thorax-window-sweep.json"],

**Read the references carefully; they are not one axis.** The leak and unknown counts are
counts over a manifest. The proposed window and its p95 are degrees against exact
synthetic truth under our own detector's noise. Nothing here shares an axis with any
figure quoted against MAMMA -- including the constant's own MAMMA-arm sweep, which the
report carries and which reports and never selects.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

LOWER = "lower is better"
HIGHER = "higher is better"
COUNT = "exposure count, not a score"


def _load(relative: str) -> dict | None:
    path = ROOT / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def fig(label: str, value: float | int | None, unit: str, reference: str, better: str = LOWER,
        key: str | None = None, note: str = "") -> dict:
    return {"key": key or label, "label": label, "value": value, "unit": unit,
            "reference": reference, "better": better, "note": note}


def x_provenance(_: dict) -> tuple[list, list]:
    provenance = _load("artifacts/compare/provenance.json")
    sweep = _load("artifacts/compare/thorax-window-sweep.json")
    if not provenance and not sweep:
        return [], []

    figures: list[dict] = []
    controls: list[dict] = []

    if provenance:
        counts = provenance["counts"]
        reference = "provenance manifest: every module-level constant and reachable keyword " \
                    "default on the delivery path, joined to code comments, docs and git history"
        figures.append(fig(
            "MAMMA-derived constants on the delivery path", counts["leaks"],
            f"of {counts['manifest']} manifest items", reference, LOWER, key="leaks",
            note="a constant whose SELECTION cites artifacts/mamma, ma_cap, SMPL-X outputs, "
                 "pred_joints/pred_vertices or a report computed from them. "
                 "provenance.py exits non-zero while this is above zero."))
        figures.append(fig(
            "constants with no stated origin anywhere", counts["unknown"],
            f"of {counts['manifest']}", reference, LOWER, key="unknown",
            note="not leaks -- unaudited. `unknown` is never guessed and never upgraded; "
                 "this count is the work that remains."))
        controls.append(fig(
            "declared-and-known MAMMA inputs (must stay listed, not zero)",
            counts["declared"], f"of {counts['manifest']}", reference, COUNT, key="declared",
            note="the camera rig IS MAMMA's ma_cap (ladder rung 0, 'not owned') and the "
                 "footage is its licence-scoped example take. Listing them is what stops "
                 "the audit condemning every build and burying the undeclared leaks. Lane H "
                 "retires both."))
        controls.append(fig(
            "trained weights fitted in this repo (a leak into weights needs weights)", 0,
            "sets of weights", "run-report.json runtime_dependencies", COUNT, key="own_weights",
            note="mamma=false, mamma_outputs=false, mamma_weights=false, smplx_model=false. "
                 "Nothing on the delivery path is trained here, so there is no training-"
                 "example manifest to audit."))

    if sweep:
        proposal = sweep["proposal"]
        reference = "exact synthetic truth (FK thorax frames) under our own detector's " \
                    "self-consistency noise -- no MAMMA output anywhere in the score"
        window = proposal.get("proposed_value")
        stride = str(proposal.get("headline_stride"))
        pooled = sweep["strides"][stride]["pooled_moving"]
        p95 = pooled.get(str(window), {}).get("pooled_p95_deg") if window is not None else None
        figures.append(fig(
            "THORAX_SMOOTHING_FRAMES re-selected without MAMMA", window, "frames",
            reference, COUNT, key="thorax_window",
            note=f"shipped value {sweep['constant']['current_value']}, selected on the MAMMA "
                 f"oracle arm. Bracket {proposal.get('bracket')}; argmin by playback stride "
                 f"{proposal.get('argmin_by_stride')}. {proposal.get('status')}"))
        figures.append(fig(
            "its pooled p95 thorax angular error at that window", p95, "deg p95",
            reference, LOWER, key="thorax_p95",
            note="the optimum is a bias-variance trade and moves with the RATIO of detector "
                 "noise to thorax angular speed, so the frame count is not transferable "
                 "between takes; the sweep reports that rather than hiding it. Blind to "
                 "temporally correlated detector error -- the injected noise is white."))
        heavy = sweep["strides"][stride].get("pooled_moving_heavy_tail", {})
        frozen = sweep["strides"][stride]["moving_clips"]
        squat = next((c for c in frozen if "squat" in c), None)
        controls.append(fig(
            "control: no smoothing at all, raw triangulated frame (must be worst)",
            pooled.get("0", {}).get("pooled_p95_deg"), "deg p95", reference, LOWER,
            key="thorax_none",
            note="the jitter the shipped docstring describes; its yaw-rate p95 runs "
                 "1.5-8.5x the truth's."))
        controls.append(fig(
            "control: the frozen mean frame, on the clip that turns most (must fail)",
            frozen.get(squat, {}).get("controls", {}).get("frozen_mean_frame", {})
                  .get("pooled_p95_deg") if squat else None,
            "deg p95", reference, LOWER, key="thorax_frozen",
            note="read PER CLIP: a freeze wins on a thorax that barely turns, so pooling it "
                 "would hide the control. Built as the take's mean frame, because "
                 "Savitzky-Golay at polyorder 2 never freezes however wide the window gets."))
        controls.append(fig(
            "sensitivity: the same selection under SOMA-77's measured heavy tail",
            proposal.get("heavy_tail_arm_argmin"), "frames", reference, COUNT,
            key="thorax_heavy_tail",
            note=f"p95 at that window {heavy.get(str(proposal.get('heavy_tail_arm_argmin')), {}).get('pooled_p95_deg')}. "
                 "Median-matched to the Gaussian arm so the arms differ only in tail."))
        controls.append(fig(
            "the MAMMA oracle sweep beside it: REPORTS, NEVER SELECTS",
            sweep["mamma_oracle_sweep"]["its_stated_choice"], "frames",
            "MAMMA's head orientation in our thorax frame -- a DIFFERENT quantity on "
            "DIFFERENT data; never on one axis with the figures above", COUNT,
            key="thorax_mamma_arm",
            note=sweep["mamma_oracle_sweep"]["note"]))
    return figures, controls
