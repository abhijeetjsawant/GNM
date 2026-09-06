#!/usr/bin/env python3
"""Ladder extractor for D9b -- the foot-contact hoist re-aim.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d9b_hoist import x_hoist_reaim

and route its figures by what each REFERENCES: the `placement_` keys to rung 7 (the
converter, from the delivered file's own bytes, against the delivery's own captured
landmarks), the `oracle_` keys to rung 7 as well (the same stage against exact synthetic
truth) and the `silhouette_` keys to rung 1 (the masks). One extractor call, three
destinations.

It reads exactly two reports: `artifacts/compare/d9b-hoist/gate.json` and
`artifacts/compare/d9b-hoist/silhouette-partwise.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * OUR OWN DELIVERED FILE against ITS OWN captured landmarks. The perpendicular distance
    from a bone's landmark to the ray the bone actually points along, measured from the
    origin the delivered file gives that bone. Zero is exactly achievable and the candidate
    sets it directly -- this is a CLOSURE quantity, which is why on the ladder it is paired
    with the bit-identity clause and the photographs and never shown alone.
  * EXACT SYNTHETIC TRUTH: the D3 gate's six perturbed-rest bodies, posed through our own
    forward kinematics, recovered through the real converter. Millimetres against a known
    answer. MAMMA-free.
  * MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated.

THE HEADLINE. `project_generated_foot_contacts` translates the root AFTER every bone has
been aimed, so on the frames it moves (67 of 150 on one performer, 22 on the other) every
root-dependent bone pointed along a ray from an origin that no longer existed. The seven
of them missed their landmarks by 3.7-6.3 mm median and up to 14.4 at p95; they now miss by
0.0003 mm, which is the float32 floor of the delivered file itself. Nothing else in the
file moved: the root, the contacts, the hips, the legs, the feet and the toes are
bit-identical, and so is every joint on every frame the hoist did not touch.

WHAT THE CHARTS CANNOT SAY. The ray miss is a CLOSURE: it measures whether the bone points
where it was asked to point, not whether the ask was right, and a bone on its ray can still
be rolled about its own axis. The photographs are level -- 4 to 6 mm on a mesh rasterised
at a quarter of native resolution is under a pixel on most cameras -- so this step's
evidence is the bytes and the exact-truth oracle, with the masks there to catch a
regression and not to show the gain.

AND ONE CHART IS DELIBERATELY A DISAGREEMENT. On exactly the same six delivered oracle
bodies, the D3 gate's own gauge says the fix made the arms WORSE and the absolute gauge
says it made them BETTER. The D3 gauge subtracts each frame's leg-root midpoint -- it
"removes the ground-projection shift" by design -- so it cannot see a root move at all, and
a bone correctly re-aimed from the hoisted origin misses the ALIGNED target by the
perpendicular part of the hoist. Both bars are honest; they answer different questions, and
neither band is moved by this step. See `docs/reviews/hoist-reaim-2026-09-07.md`.

Self-check:  python3 tools/compare/extractors/d9b_hoist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d9b-hoist/gate.json"
SILHOUETTE = "artifacts/compare/d9b-hoist/silhouette-partwise.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py "
    "--out artifacts/compare/d9b-hoist/delivery-hygiene --expect-byte-identical && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py "
    "--out artifacts/compare/d9b-hoist/tripwire-zero-d8c --mode zero-hoist   # D8c src && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_gate.py --oracle "
    "--oracle-tag D8c --oracle-save artifacts/compare/d9b-hoist/oracle-d8c "
    "--out artifacts/compare/d9b-hoist/instrument.json   # before the src change && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py "
    "--out artifacts/compare/d9b-hoist/tripwire-zero-d9b --mode zero-hoist && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py "
    "--out artifacts/compare/d9b-hoist/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py "
    "--out artifacts/compare/d9b-hoist/degenerate-no-correction --mode no-correction && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_silhouette.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_gate.py "
    "--candidate artifacts/compare/d9b-hoist/delivery "
    "--tripwire artifacts/compare/d9b-hoist/tripwire-zero-d8c "
    "artifacts/compare/d9b-hoist/tripwire-zero-d9b "
    "--degenerate no_correction=artifacts/compare/d9b-hoist/degenerate-no-correction "
    "--degenerate lock_without_correction=artifacts/compare/d9b-hoist/tripwire-zero-d9b "
    "--oracle --oracle-tag D9b --oracle-save artifacts/compare/d9b-hoist/oracle-d9b "
    "--oracle-reference artifacts/compare/d9b-hoist/oracle-d8c "
    "--out artifacts/compare/d9b-hoist/gate.json"
)

REF_RAY = ("OUR OWN DELIVERED FILE against ITS OWN captured landmarks: the perpendicular "
           "distance from each bone's landmark to the ray that bone points along, measured "
           "from the origin the delivered GLB actually gives it. A CLOSURE -- zero is "
           "exactly achievable and the candidate sets it directly")
REF_TRUTH = ("EXACT SYNTHETIC TRUTH: the D3 gate's six perturbed-rest bodies, posed through "
             "our own forward kinematics and recovered through the real converter, scored "
             "in the WORLD frame with no per-frame alignment. MAMMA-free")
REF_ALIGNED = ("EXACT SYNTHETIC TRUTH, scored through `retarget_cost.score` -- the D3 "
               "gate's own gauge, which subtracts each frame's leg-root midpoint and is "
               "therefore blind to a root move BY CONSTRUCTION")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"

FAMILIES = {"arm": ("LeftUpperArm->left_elbow", "LeftLowerArm->left_wrist",
                    "RightUpperArm->right_elbow", "RightLowerArm->right_wrist"),
            "clavicle": ("LeftShoulder->left_shoulder", "RightShoulder->right_shoulder"),
            "trunk": ("Spine->neck",)}


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def _worst_ray(build: dict, subject: str, family: str) -> float | None:
    rows = ((build.get("subjects", {}).get(subject, {}).get("rays")) or {})
    values = [rows[name]["miss_from_delivered_origin_mm"]["hoisted"].get("median")
              for name in FAMILIES[family] if name in rows]
    values = [v for v in values if v is not None]
    return round(max(values), 4) if values else None


def x_hoist_reaim(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    # ------------------------------------------------ the delivered file's own bytes
    for subject in ("subject_00", "subject_01"):
        label = _subject_label(subject)
        for family, family_label in (("arm", "the arm bones"),
                                     ("clavicle", "the clavicles"),
                                     ("trunk", "the trunk")):
            for build, role, key in ((r.get("build_D9b") or {}, "ours", "d9b"),
                                     (r.get("build_D8c") or {}, "control", "d8c")):
                entry = fig(
                    f"D9b ray miss of {family_label} on the hoisted frames, {label}, "
                    f"{key.upper()}",
                    _worst_ray(build, subject, family), "mm median", REF_RAY, LOWER,
                    key=f"placement_ray_{family}_{key}_{subject}",
                    note=("the worst of the bones in this family. On the frames the hoist "
                          "did not move, both builds read 0.0003 mm"))
                (figs if role == "ours" else ctrls).append(entry)
        # The banded quantity: the excess over each build's OWN aim floor, worst joint.
        cell = ((r.get("B1_excess_over_the_aim_floor") or {}).get("subjects", {})
                .get(subject, {}).get("per_frame_excess_over_the_aim_floor_mm") or {})
        for key, role in (("D9b", "ours"), ("D8c", "control")):
            rows = cell.get(key) or {}
            worst = [row["whole"]["max"] for row in rows.values()
                     if row.get("whole", {}).get("max") is not None]
            entry = fig(
                f"D9b worst per-frame excess over the aim's own floor, {label}, {key}",
                round(max(worst), 4) if worst else None, "mm, worst frame", REF_RAY, LOWER,
                key=f"placement_excess_{key.lower()}_{subject}",
                note=("the floor is the bone LENGTH error an aim cannot remove, computed "
                      "per build from that build's own origins; the excess over it is the "
                      "part an aim owns"))
            (figs if role == "ours" else ctrls).append(entry)

    # ------------------------------------------------------------- exact synthetic truth
    oracle = (r.get("oracle") or {})
    seeds = oracle.get("seeds") or []
    if seeds:
        for stat, unit, note in (("absolute_fk_groups_hoisted_mm", "mm median",
                                  "the hoisted frames only, in the world frame"),
                                 ("absolute_fk_groups_p95_mm", "mm p95",
                                  "the whole take, in the world frame")):
            worst_arms = max(seed[stat]["arms"] for seed in seeds)
            figs.append(fig(
                f"D9b arms against exact truth, worst of six bodies, {stat.split('_')[-2]}",
                round(worst_arms, 3), unit, REF_TRUTH, LOWER,
                key=f"oracle_absolute_arms_{stat.split('_')[-2]}_d9b", note=note))
        ctrls.append(fig(
            "D9b control: the D3 gate's own ALIGNED gauge on the same six bodies, arms",
            round(max(seed["aligned_fk_groups_hoisted_mm"]["arms"] for seed in seeds), 3),
            "mm median", REF_ALIGNED, LOWER,
            key="oracle_aligned_arms_hoisted_d9b",
            note=("the gauge removes each frame's root offset, so a correctly re-aimed bone "
                  "misses the ALIGNED target by the perpendicular part of the hoist and "
                  "this bar RISES. It is not a regression; it is the alignment")))
        ctrls.append(fig(
            "D9b control: the legs on the same six bodies, unchanged by this step",
            round(max(seed["absolute_fk_groups_hoisted_mm"]["legs"] for seed in seeds), 3),
            "mm median", REF_TRUTH, LOWER,
            key="oracle_absolute_legs_hoisted_d9b",
            note=("the contact model's own cost on exact truth -- a foot planted where the "
                  "truth is still moving. Bit-identical before and after; it belongs to the "
                  "projection's own step, not to this one")))

    # ------------------------------------------------------------------ the photographs
    silhouette = _load(SILHOUETTE) or {}
    for subject, block in (silhouette.get("subjects") or {}).items():
        label = _subject_label(subject)
        cell = (block.get("cuts") or {}).get("hoisted_frames") or {}
        figs.append(fig(
            f"D9b the whole person in the photographs, hoisted frames, {label}, D9b",
            cell.get("whole_iou_D9b"), "IoU median", REF_MASKS, HIGHER,
            key=f"silhouette_whole_hoisted_d9b_{subject}",
            note="the frames the re-aim touches; expected LEVEL, and level is what it is"))
        ctrls.append(fig(
            f"D9b control: the same photographs before, {label}, D8c",
            cell.get("whole_iou_D8c"), "IoU median", REF_MASKS, HIGHER,
            key=f"silhouette_whole_hoisted_d8c_{subject}"))
        ctrls.append(fig(
            f"D9b: the reference fitter's own mesh, hoisted frames, {label}",
            cell.get("whole_iou_ORACLE"), "IoU median", REF_MASKS, HIGHER,
            key=f"silhouette_whole_hoisted_oracle_{subject}",
            note="the ceiling a body of the wrong proportions can reach, not a target"))
    return figs, ctrls


VISUALS = {
    "placement": [
        dict(title="D9b: does each bone point where it was told to point?",
             plain="After every bone had been aimed, the automatic floor-contact step slid "
                   "the whole character sideways to plant a foot -- and the bones kept "
                   "pointing along the lines they were aimed along from where they used to "
                   "be. This chart measures, on the frames that slide happens, how far each "
                   "bone's target sits off the line the bone actually points along. Lower "
                   "is better and zero is achievable exactly. The hatched bars are the "
                   "build before this step.",
             better="lower",
             bars=[dict(label="Arms after, performer 0", role="ours", key="placement_ray_arm_d9b_subject_00"),
                   dict(label="Arms before, performer 0", role="control", key="placement_ray_arm_d8c_subject_00"),
                   dict(label="Shoulders after, performer 0", role="ours", key="placement_ray_clavicle_d9b_subject_00"),
                   dict(label="Shoulders before, performer 0", role="control", key="placement_ray_clavicle_d8c_subject_00"),
                   dict(label="Trunk after, performer 0", role="ours", key="placement_ray_trunk_d9b_subject_00"),
                   dict(label="Trunk before, performer 0", role="control", key="placement_ray_trunk_d8c_subject_00")]),
        dict(title="D9b: the same question on the other performer",
             plain="The second performer's feet are planted on far fewer frames -- 22 of "
                   "150 against 67 -- so the change touches less of his take, and on the "
                   "frames it does touch the picture is the same. Lower is better.",
             better="lower",
             bars=[dict(label="Arms after, performer 1", role="ours", key="placement_ray_arm_d9b_subject_01"),
                   dict(label="Arms before, performer 1", role="control", key="placement_ray_arm_d8c_subject_01"),
                   dict(label="Shoulders after, performer 1", role="ours", key="placement_ray_clavicle_d9b_subject_01"),
                   dict(label="Shoulders before, performer 1", role="control", key="placement_ray_clavicle_d8c_subject_01"),
                   dict(label="Trunk after, performer 1", role="ours", key="placement_ray_trunk_d9b_subject_01"),
                   dict(label="Trunk before, performer 1", role="control", key="placement_ray_trunk_d8c_subject_01")]),
        dict(title="D9b: the worst single frame in the whole take",
             plain="Medians hide the bad frames, so this is the worst frame of 150 rather "
                   "than the middle one, and it counts only the part an aim can actually "
                   "fix -- a bone can be too short for its target, and no amount of aiming "
                   "cures that. Lower is better. After the change the worst frame of the "
                   "take is three ten-thousandths of a millimetre, which is the storage "
                   "precision of the delivered file itself.",
             better="lower",
             bars=[dict(label="After, performer 0", role="ours", key="placement_excess_d9b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="placement_excess_d8c_subject_00"),
                   dict(label="After, performer 1", role="ours", key="placement_excess_d9b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="placement_excess_d8c_subject_01")]),
    ],
    "oracle": [
        dict(title="D9b: measured against a known answer, and against a gauge that cannot "
                   "see the fix",
             plain="Six synthetic bodies of known proportions, posed through our own "
                   "maths and put back through the real pipeline, so the right answer is "
                   "known exactly. Lower is better. The solid bars are the arms measured "
                   "where they really are in the world. The first hatched bar is the SAME "
                   "delivered bodies scored through the older gauge, which lines each frame "
                   "up on the hips before measuring -- that removes exactly the sideways "
                   "slide this step is about, so it reports the corrected arms as worse. "
                   "Both readings are honest and they answer different questions; the older "
                   "band is left where it is. The last hatched bar is the legs, which this "
                   "step does not touch: their error is the cost of planting a foot the "
                   "truth is still moving, and it is handed to the step that owns it.",
             better="lower",
             bars=[dict(label="Arms, hoisted frames", role="ours", key="oracle_absolute_arms_hoisted_d9b"),
                   dict(label="Arms, whole take, p95", role="ours", key="oracle_absolute_arms_p95_d9b"),
                   dict(label="The alignment-blind gauge, arms", role="control", key="oracle_aligned_arms_hoisted_d9b"),
                   dict(label="Legs (not this step's)", role="control", key="oracle_absolute_legs_hoisted_d9b")]),
    ],
    "masks": [
        dict(title="D9b: how well the character covers the person in the photographs",
             plain="The reference fitter's person masks are the reference; higher is "
                   "better, on the frames this step changes. It is LEVEL, and level is the "
                   "honest answer: the bones move three to six millimetres on a character "
                   "rasterised at a quarter of the footage's resolution, which is under a "
                   "pixel on most of the four cameras. This chart is here to catch a "
                   "regression, not to show the gain -- the gain is in the two charts "
                   "above. The hatched bars are the build before this step and the "
                   "reference fitter's own mesh, which is the ceiling a body of the wrong "
                   "proportions can reach rather than a target.",
             better="higher",
             bars=[dict(label="After, performer 0", role="ours", key="silhouette_whole_hoisted_d9b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="silhouette_whole_hoisted_d8c_subject_00"),
                   dict(label="Reference fitter's mesh, performer 0", role="control", key="silhouette_whole_hoisted_oracle_subject_00"),
                   dict(label="After, performer 1", role="ours", key="silhouette_whole_hoisted_d9b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="silhouette_whole_hoisted_d8c_subject_01"),
                   dict(label="Reference fitter's mesh, performer 1", role="control", key="silhouette_whole_hoisted_oracle_subject_01")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_hoist_reaim({})
    for f in figs + ctrls:
        print(f"{f['key']:56s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
