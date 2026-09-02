#!/usr/bin/env python3
"""Ladder extractor for I7, the temporal stage.

Registry-ready but NOT registered: `tools/compare/ladder.py` has one owner, and two
agents editing it collide. Wire `x_temporal` into the rung-5 ("temporal") row there --
alongside `x_sequence_and_oracle`, not instead of it: I2 measures the stage's floor on a
take with no dropped views, and this measures what the stage does when views ARE
dropped, which I2's own `instrument_missing` note asks for by name.

**Two trajectory sources and they never share an axis.** `fk_synthetic` is MAMMA-free
forward kinematics and is the only arm that could ever select a shipped constant;
`mamma_pred_joints` is MAMMA-derived and reports only. Every key below is prefixed with
its source, and the proposed VISUALS entry puts them on separate charts.

The recovery figures are millimetres against EXACT truth; the held-out figures are
frames of lag against a camera the reconstruction never saw. Those two are not
comparable either.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOWER = "lower is better"
HIGHER = "higher is better"
REPORT = "artifacts/compare/temporal.json"


def _load(rel: str) -> dict | None:
    path = ROOT / rel
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


def _median(entry: dict | None) -> float | None:
    return entry.get("median_mm") if isinstance(entry, dict) else None


def x_temporal(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list[dict] = []
    ctrls: list[dict] = []

    pattern = r.get("real_run_occlusion_pattern", {})
    figs.append(fig("single-ray joint-slots on the reference footage -- how often the "
                    "sequence solve is exercised at all",
                    pattern.get("single_ray_slots"), "slots of "
                    f"{pattern.get('camera_support_histogram_0_to_4', [0])[0] is not None and sum(pattern.get('camera_support_histogram_0_to_4', [0])) or 0}",
                    "the delivered SOMA-77 run's own association, 17 emitted joints",
                    LOWER, key="real_single_ray_slots",
                    note="A001 and C001 see almost everything; B001 and D001 drop in "
                         "correlated bursts, so support stays at two or more nearly always. "
                         "The stage's own job barely occurs on this take, which is why the "
                         "figures below come from injected outages."))

    for source_key, tag in (("fk_synthetic", "fk"), ("mamma_pred_joints", "mamma")):
        source = (r.get("sources") or {}).get(source_key)
        if not source:
            continue
        ref = source.get("reference", source_key)
        short = ("exact FK truth, MAMMA-free" if tag == "fk"
                 else "MAMMA's own joints as a trajectory -- reports, never selects")

        oracle = source.get("oracle_no_drops", {})
        figs.append(fig(f"[{short}] what the temporal stage costs with nothing dropped",
                        _median(oracle.get("after_the_temporal_stage")), "mm median", ref,
                        key=f"{tag}_oracle_after_stage",
                        note="raw triangulation on the same run returns the projected "
                             f"trajectory to {oracle.get('raw_max_micrometres', 0):.3g} um, so "
                             "everything here is this stage"))

        arms = (source.get("recovery_5a") or {}).get("arms", {})
        for arm_key, arm_label in (("correlated_amplified", "correlated outages"),
                                   ("iid_same_rate", "independent outages, same rate")):
            arm = arms.get(arm_key)
            if not arm or arm.get("ran") is False:
                continue
            ray = arm.get("recovered_error_along_the_surviving_ray", {})
            figs.append(fig(f"[{short}] error on the joints only one camera saw, "
                            f"{arm_label}",
                            _median(arm.get("error_on_recovered_cells")), "mm median", ref,
                            key=f"{tag}_recovered_{arm_key}",
                            note=f"{arm['cells']['recovered']} recovered of "
                                 f"{arm['cells']['single_ray']} single-ray cells "
                                 f"({arm['cells']['single_ray_fraction'] * 100:.2f} % of "
                                 f"{arm['cells']['total']}); "
                                 f"{arm['cells']['demoted_to_interpolation']} demoted to the "
                                 "fill by the solve's own 14 px ray gate. Recovery THEN "
                                 "smoothing, not the solve alone. "
                                 f"{ray.get('median')} of the residual lies along the one "
                                 "surviving viewing ray (0.577 would be isotropic), which is "
                                 "the direction a single ray and a reprojection are both "
                                 "blind to"))
            figs.append(fig(f"[{short}] single-ray cells produced, {arm_label}",
                            arm["cells"]["single_ray_fraction"], "fraction of cells", ref,
                            HIGHER, key=f"{tag}_single_ray_fraction_{arm_key}",
                            note="an exposure count, not a quality figure: the gap between "
                                 "the correlated and independent arms is what view "
                                 "correlation does to starvation"))
            margin = arm.get("margin_on_recovered_cells", {})
            if margin.get("draws"):
                figs.append(fig(f"[{short}] P(the sequence solve beats interpolation) on "
                                f"identical block-bootstrap draws, {arm_label}",
                                margin["p_candidate_beats_control"], "probability", ref,
                                HIGHER, key=f"{tag}_p_beats_interp_{arm_key}",
                                note=f"block {margin['block_frames']} frames, "
                                     f"{margin['draws']} draws, same cells in both arms; "
                                     "lag-1 of this arm's own per-frame residual series is "
                                     f"{arm.get('lag1_autocorrelation_of_per_frame_recovered_median')}"))
            ctrls.append(fig(f"[{short}] control: the same cells with the sequence solve "
                             f"switched off (interpolation only), {arm_label}",
                             _median(arm.get(
                                 "control_interpolation_only_on_recovered_cells")),
                             "mm median", ref, HIGHER,
                             key=f"{tag}_interp_only_{arm_key}",
                             note="the real function replaced by a pass-through, so the fill "
                                  "draws a line through the very cells the solve recovered"))

        controls = (source.get("recovery_5a") or {}).get("controls", {})
        frozen = controls.get("frozen_trajectory")
        if frozen:
            ctrls.append(fig(f"[{short}] control: frame 0's pose on every frame",
                             _median(frozen.get("error_against_the_real_trajectory")),
                             "mm median", ref, HIGHER, key=f"{tag}_frozen",
                             note="it reconstructs and recovers perfectly and agrees with the "
                                  "trajectory not at all -- the figure a constant scores"))
        root = controls.get("root_starved_frames")
        if root:
            ctrls.append(fig(f"[{short}] diagnostic: frames whose ROOT was starved to one ray",
                             _median(root.get("error_on_starved_frames")), "mm median", ref,
                             HIGHER, key=f"{tag}_root_starved",
                             note="a starved root fails triangulation and the whole "
                                  "subject-frame is rejected, so no joint on it can be "
                                  "recovered from its own ray; "
                                  f"{_median(root.get('error_on_other_frames'))} mm elsewhere"))

        smoothing = source.get("smoothing_5b", {})
        clean = smoothing.get("clean_coverage", {})
        shipped = clean.get("shipped_window_9", {})
        over = clean.get("over_smoothed_window_27", {})
        identity = clean.get("identity_no_savitzky_golay", {})
        if shipped:
            figs.append(fig(f"[{short}] phase lag of the shipped smoothing window",
                            shipped.get("phase_lag", {}).get("argmin_subframe"), "frames", ref,
                            key=f"{tag}_lag_shipped",
                            note="shift sweep on ONE fixed frame set at every shift; the "
                                 "+/-1 entries are the time-shifted-truth control and both "
                                 "must be worse than 0, which they are: "
                                 f"{shipped.get('phase_lag', {}).get('rms_mm_by_shift')}"))
            figs.append(fig(f"[{short}] how much of a fast wrist or ankle the shipped window "
                            "removes",
                            shipped.get("peak_attenuation", {}).get("attenuation_median"),
                            "fraction of peak speed lost", ref, key=f"{tag}_attenuation_shipped",
                            note="fast events are the trajectory's own top-5 % frames by true "
                                 "displacement; true peak "
                                 f"{shipped.get('peak_attenuation', {}).get('true_peak_speed_m_s')}"
                                 " m/s"))
        for label, entry, key in (("3x the shipped window", over, "over"),
                                  ("no Savitzky-Golay at all (fill only)", identity, "identity")):
            if not entry:
                continue
            ctrls.append(fig(f"[{short}] control: {label} -- lag",
                             entry.get("phase_lag", {}).get("argmin_subframe"), "frames", ref,
                             HIGHER, key=f"{tag}_lag_{key}"))
            ctrls.append(fig(f"[{short}] control: {label} -- peak attenuation",
                             entry.get("peak_attenuation", {}).get("attenuation_median"),
                             "fraction of peak speed lost", ref, HIGHER,
                             key=f"{tag}_attenuation_{key}"))
            ctrls.append(fig(f"[{short}] control: {label} -- error with nothing dropped",
                             _median(entry.get("error")), "mm median", ref, HIGHER,
                             key=f"{tag}_error_{key}"))

        spikes = smoothing.get("spikes", {})
        two = spikes.get("2_camera", {})
        if two:
            figs.append(fig(f"[{short}] error at an injected outlier that reaches 3D, shipped "
                            "window",
                            _median((two.get("shipped_window_9") or {}).get(
                                "error_on_spiked_cells")), "mm median", ref,
                            key=f"{tag}_spike2_shipped",
                            note=f"spike amplitude {smoothing.get('spike_amplitude_px1280')} px "
                                 "at 1280 -- the p99 of OUR OWN detector's cross-view "
                                 "self-agreement, never MAMMA's residual"))
            ctrls.append(fig(f"[{short}] control: the same outlier with no Savitzky-Golay",
                             _median((two.get("identity_no_savitzky_golay") or {}).get(
                                 "error_on_spiked_cells")), "mm median", ref, HIGHER,
                             key=f"{tag}_spike2_identity",
                             note="the spike must survive when the filter is removed, or the "
                                  "filter is not what suppressed it"))
            ctrls.append(fig(f"[{short}] control: the same outlier with 3x the shipped window",
                             _median((two.get("over_smoothed_window_27") or {}).get(
                                 "error_on_spiked_cells")), "mm median", ref, HIGHER,
                             key=f"{tag}_spike2_over",
                             note="a wider window buries the outlier further and flattens "
                                  "everything else with it -- read it beside the peak "
                                  "attenuation control, never alone"))
        one = spikes.get("1_camera", {})
        if one:
            ctrls.append(fig(f"[{short}] control: an outlier in ONE camera only",
                             _median((one.get("shipped_window_9") or {}).get(
                                 "error_on_spiked_cells")), "mm median", ref, HIGHER,
                             key=f"{tag}_spike1_shipped",
                             note="absorbed by TRIANGULATION, not the smoother: "
                                  "triangulate_point keeps the largest inlier subset within "
                                  "14 px, so a lone outlier never reaches the temporal stage"))

        gaps = smoothing.get("gaps", {})
        for length in ("1_frame", "3_frame", "5_frame"):
            gap = gaps.get(length)
            if not gap or gap.get("ran") is False:
                continue
            figs.append(fig(f"[{short}] error inside a {length.split('_')[0]}-frame gap in "
                            "every camera",
                            _median(gap.get("error_inside_the_gap")), "mm median", ref,
                            key=f"{tag}_gap_{length}",
                            note="no ray at all, so the fill draws a line and the filter "
                                 "smooths it; recovery latency after the gap closes: "
                                 f"{gap.get('recovery_latency', {}).get('median_frames')} "
                                 "frames median"))

    held = r.get("held_out_camera_lag", {})
    probe = held.get("sync_hypothesis_probe", {})
    probe_note = ""
    if probe.get("ran"):
        probe_note = (f" One fold ({probe['suspect_camera']}) stands apart from the others by "
                      "most of a frame. Moving that camera's own content a frame each way and "
                      "rerunning all four folds puts the smallest mean minimum at shift "
                      f"{probe['shift_that_minimises_it']} -- a CANDIDATE sync offset for "
                      "rung 0, not a temporal finding, and one measurement short of a claim.")
    for name, fold in (held.get("folds") or {}).items():
        if not fold.get("ran"):
            ctrls.append(fig(f"held-out fold {name}: did not run",
                             None, "frames", "held-out camera", LOWER,
                             key=f"heldout_{name}", note=fold.get("reason", "")))
            continue
        figs.append(fig(f"lag of our smoothed 3D against the camera it never saw ({name})",
                        fold.get("argmin_subframe"), "frames", "held-out camera, real footage",
                        LOWER, key=f"heldout_{name}",
                        note="reconstructed from the other three cameras and reprojected; "
                             "one fixed frame set at every shift; "
                             f"{fold.get('scored_slots')} slots. BLIND TO DEPTH -- a residual "
                             "along the held-out camera's ray costs nothing, so this sees "
                             "transverse lag only." + probe_note))
    return figs, ctrls


def stamp() -> dict:
    path = ROOT / REPORT
    if not path.exists():
        return {"path": REPORT, "exists": False}
    return {"path": REPORT, "exists": True,
            "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")}


if __name__ == "__main__":
    figures, controls = x_temporal({})
    for row in figures:
        print(f"  FIG  {row['label']}: {row['value']} {row['unit']}")
    for row in controls:
        print(f"  CTRL {row['label']}: {row['value']} {row['unit']}")
