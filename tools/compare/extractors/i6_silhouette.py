#!/usr/bin/env python3
"""Ladder extractor for step I6 -- the surface rung (silhouette vs MAMMA's SAM2 masks).

Registry-ready stub. `tools/compare/ladder.py` owns the `RUNGS` table (one owner, or
registrations collide -- LADDER_EXECUTION_PLAN §2), so this file defines the extractor
and nothing else. To register it, add to `ladder.py`:

    from extractors.i6_silhouette import x_silhouette   # or paste the function in

and give the `masks` rung (n=1) `extract=x_silhouette`, `reports=[REPORT]`. The delivered
rung (n=11) may carry the same figures; they share a reference and may share an axis with
each other, and with NOTHING else on the page -- these are pixels against segmented
footage, not millimetres against `pred_joints`.

`fig`, `_load`, `LOWER` and `HIGHER` are imported lazily inside the function so that this
module can be imported from anywhere in `ladder.py` without depending on where in that
file the import line lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT = "artifacts/compare/silhouette.json"
REFERENCE = ("MAMMA's SAM2 person masks (`ma_masks`) -- the one retained reference on "
             "this fixture that is NOT model-mediated")
CAMERAS = ("A001", "B001", "C001", "D001")


def x_silhouette(_: dict) -> tuple[list, list]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ladder import HIGHER, LOWER, _load, fig  # noqa: PLC0415

    r = _load(REPORT)
    if not r:
        return [], []
    arms = r["arms"]

    def med(arm: str, cam: str, subject: int, metric: str) -> float | None:
        try:
            return arms[arm][cam][f"subject_{subject:02d}"][metric]["median"]
        except (KeyError, TypeError):
            return None

    def across(arm: str, metric: str) -> list[float]:
        return [v for cam in CAMERAS for s in (0, 1)
                if (v := med(arm, cam, s, metric)) is not None]

    figs, ctrls = [], []

    # Precision and recall are separate figures and never collapse into IoU alone: the
    # degenerate a mesh instrument actually produces is something too big, which buys
    # recall with precision. IoU carried per camera so a single bad view cannot average away.
    for metric in ("precision", "recall"):
        values = across("ours_delivered", metric)
        figs.append(fig(f"delivered mesh silhouette {metric}, median over 4 cameras x 2 subjects",
                        (sum(values) / len(values)) if values else None, "fraction", REFERENCE,
                        HIGHER, key=f"ours_{metric}",
                        note="reported beside its partner; never read IoU alone"))
    for cam in CAMERAS:
        for s in (0, 1):
            figs.append(fig(f"delivered mesh silhouette IoU, {cam} subject {s:02d}",
                            med("ours_delivered", cam, s, "iou"), "IoU median", REFERENCE,
                            HIGHER, key=f"ours_iou_{cam}_{s:02d}"))

    fs = r.get("facing_sensitivity", {})
    figs.append(fig("IoU the silhouette loses when the ORACLE is turned 180 degrees, on the "
                    "frames where front and back are distinguishable (how much a silhouette "
                    "can see a facing error AT ALL -- step D1's gate rests on this)",
                    fs.get("oracle_minus_yaw180_iou_on_distinguishable_frames", {}).get("median"),
                    "IoU difference", REFERENCE, HIGHER, key="facing_sensitivity",
                    note=fs.get("reading", "")))

    ctrls.append(fig("ORACLE: MAMMA's own mesh through the identical rasteriser against the "
                     "same masks -- the ceiling the masks and the rasteriser allow (both "
                     "meshes are nude bodies; the masks segment a clothed person)",
                     (lambda v: sum(v) / len(v) if v else None)(across("ORACLE_mamma_mesh", "iou")),
                     "IoU median", REFERENCE, HIGHER, key="oracle_iou"))
    for metric in ("precision", "recall"):
        values = across("ORACLE_mamma_mesh", metric)
        ctrls.append(fig(f"ORACLE {metric}", (sum(values) / len(values)) if values else None,
                         "fraction", REFERENCE, HIGHER, key=f"oracle_{metric}"))

    controls = [
        ("control_dilated_3px", "control: our silhouette dilated 3 px (recall must rise, "
                                "precision must fall -- that is why both are reported)"),
        ("control_dilated_9px", "control: our silhouette dilated 9 px"),
        ("control_dilated_25px", "control: our silhouette dilated 25 px"),
        ("control_mean_body", "control: MAMMA's own pose carrying the MEAN body (betas=0) "
                              "-- differs from the oracle by the shape term alone"),
        ("control_frozen_pose_tracked", "control: one frame's pose on every frame, centroid "
                                        "still tracking (the degenerate a pose solver produces)"),
        ("control_frozen_pose_static", "control: one frame's mesh, unmoved"),
        ("control_billboard", "control: a filled rectangle over our mesh's bounding box"),
        ("control_shuffled_subject", "control: subject 0's mesh against subject 1's mask"),
        ("control_oracle_yaw180_facing", "control: the ORACLE turned 180 degrees about its own "
                                         "pelvis -- a self-consistent human facing backwards"),
    ]
    for arm, label in controls:
        if arm not in arms:
            continue
        ctrls.append(fig(f"{label} [IoU]",
                         (lambda v: sum(v) / len(v) if v else None)(across(arm, "iou")),
                         "IoU median", REFERENCE, LOWER, key=f"{arm}_iou"))
        if arm.startswith("control_dilated"):
            for metric in ("precision", "recall"):
                values = across(arm, metric)
                ctrls.append(fig(f"{label} [{metric}]",
                                 (sum(values) / len(values)) if values else None, "fraction",
                                 REFERENCE, LOWER if metric == "precision" else HIGHER,
                                 key=f"{arm}_{metric}"))

    agreement = r.get("point_to_surface_agreement", {}).get("subjects", {})
    for s in (0, 1):
        entry = agreement.get(f"subject_{s:02d}", {})
        ctrls.append(fig(f"AGREEMENT ONLY (model-mediated, not the footage): symmetric "
                         f"point-to-surface, our mesh vs MAMMA's, subject {s:02d}",
                         entry.get("symmetric_median_mm"), "mm median",
                         "MAMMA's pred_vertices -- a fit, not truth; NOT on the mask axis",
                         LOWER, key=f"p2s_{s:02d}"))
    return figs, ctrls
