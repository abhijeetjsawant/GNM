#!/usr/bin/env python3
"""Provenance manifest for every runtime constant on the DELIVERY path.

Ladder step **I8**, part 1. CLAUDE.md: *MAMMA is a measuring instrument, never in the
shipping path. Nothing it produces may enter a delivered artifact, trained weights, or a
shipped calibration constant.* `LADDER_EXECUTION_PLAN.md` §4 lists the leak paths and
names the containment: **provenance on every runtime constant**. This is that file.

**What it does.** It walks the delivery call graph from its entry points, mechanically
enumerates every module-level constant and every numeric keyword default it can reach,
joins each against a curated provenance table built from the code comments, `docs/`,
`git log -S` and `git blame`, and classifies:

    anatomy | literature | synthetic-truth | held-out-camera | own-capture
    engineering-limit | third-party | schema | asset | unknown
    declared-mamma | MAMMA-DERIVED

**The rule for the flag.** Any constant whose *selection* cites `artifacts/mamma`,
`ma_cap`, SMPL-X outputs, `pred_joints` / `pred_vertices`, or a report computed from
them is `MAMMA-DERIVED` and is a leak. It exits non-zero when `leaks` is non-empty, so a
delivery build can gate on it.

**"unknown" means unknown.** Where nothing in the code, the docs or the history states an
origin, the entry says `unknown`. It is never guessed and never upgraded to "probably
engineering judgement". An `unknown` is not a leak; it is an unaudited constant, and the
list of them is the work that remains.

**What this audit is blind to.** It enumerates module-level constants and keyword
defaults mechanically; *inline* numeric literals inside function bodies are curated by
hand from a read of the delivery functions and are NOT exhaustively enumerated, so an
inline magic number nobody noticed will not appear. It also cannot see a constant that
entered through a data file rather than through source -- which is exactly how the one
declared MAMMA dependency (the camera rig) enters. Both are stated in the report.

    .venv/bin/python tools/compare/provenance.py

Writes `artifacts/compare/provenance.json` and prints the summary. Exit code 1 if any
leak is present.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

ANATOMY = "anatomy"
LITERATURE = "literature"
SYNTHETIC = "synthetic-truth"
HELD_OUT = "held-out-camera"
OWN_CAPTURE = "own-capture"
ENGINEERING = "engineering-limit"
THIRD_PARTY = "third-party"
SCHEMA = "schema"
ASSET = "asset"
UNKNOWN = "unknown"
DECLARED_MAMMA = "declared-mamma"
MAMMA_DERIVED = "MAMMA-DERIVED"

CLEAN = {ANATOMY, LITERATURE, SYNTHETIC, HELD_OUT, OWN_CAPTURE, ENGINEERING,
         THIRD_PARTY, SCHEMA, ASSET}

# Modules the delivery path runs in, and the entry points into them. The call graph is
# walked from these; a function no entry point reaches is not delivery and is not audited.
DELIVERY_MODULES = (
    "src/autoanim_gnm/commercial_multiview.py",
    "src/autoanim_gnm/head_orientation.py",
    "src/autoanim_gnm/body_projection.py",
    "src/autoanim_gnm/soma_motion.py",
    "workers/commercial_multiview/soma77_pose.py",
)
ENTRY_POINTS = (
    "reconstruct_multiview", "positions_to_body_track", "_solve_head_for_subject",
    "solve_head_orientation", "project_generated_foot_contacts", "_thorax_frames",
    "_toe_world_for_subject", "_finger_rest_local", "solve_sequence_positions",
    "associate_frame_graph", "triangulate_point", "_fill_and_smooth_positions",
)

# --------------------------------------------------------------------------------------
# The curated table. Every entry carries its evidence: a comment, a document, or a commit.
# `evidence` is quoted or cited, never paraphrased into an assertion the source does not
# make. Where the source says nothing, `provenance` is `unknown` and `evidence` says so.
# --------------------------------------------------------------------------------------
CURATED: dict[str, dict] = {
    # ---------------------------------------------------------------- the known leak
    "THORAX_SMOOTHING_FRAMES": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py -- value 9 since 2026-09-02, with the "
                 "PROVENANCE block above it: selected by `tools/head/thorax_window_sweep.py` "
                 "(artifacts/compare/thorax-window-sweep.json) on exact FK thorax frames with our "
                 "own detector's self-agreement noise, interior p95 optimum at 9 (bracket 5-9), "
                 "an upper bound. HISTORY: the value was 15 from 2026-09-01 (commit 08a6c89), "
                 "chosen on the MAMMA oracle arm -- the leak I8 found. That sweep survives in "
                 "`_thorax_frames`'s docstring as a report and selects nothing. The user took "
                 "the decision to change it on 2026-09-02; the head gate was re-run after.",
        selected_against="synthetic truth (tracked FK fixture); the MAMMA arm reported beside it",
        remedy="none open. If the value moves again it must move on synthetic truth, a held-out "
               "camera or anatomy -- never on the MAMMA arm.",
    ),
    # ------------------------------------------------------------------- D7, the pelvis
    # ------------------------------------------------------------- D8, the occlusion repair
    "RAY_PAIR_CONDITIONING_CEILING_DEG": dict(
        provenance=ENGINEERING,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-05. A two-view slot "
                 "whose supporting rays meet beyond this angle (or inside its complement) is "
                 "depth-unconstrained along their common axis and is DEMOTED to the sequence "
                 "solve. DERIVED IN CLOSED FORM AND SELECTED ON NO TAKE, synthetic or real: two "
                 "rays meeting at theta determine a point across their common axis and only "
                 "weakly along it, with the along-axis error amplified by 1/|sin theta| "
                 "relative to a right-angled pair (90 deg = 1.0x, 150 deg = 2.0x, 172 deg = "
                 "7.2x). The value is the angle at which that amplification reaches 2x, and 2x "
                 "is the declared choice; the complement clause is the same expression from the "
                 "other side, so the rule is exactly |sin theta| < 0.5. THE D8 CARD SAID THIS "
                 "WOULD BE SELECTED ON SYNTHETIC TRUTH AND IT COULD NOT BE: "
                 "`tools/compare/d8_occlusion_synthetic.py` finds the sequence solve beating a "
                 "two-view triangulation at EVERY angle bin, including well-conditioned ones, "
                 "because the fixture's bones are exactly rigid and its motion exactly smooth "
                 "-- the recovery's own priors -- so the score's argmin is always 'never trust "
                 "two views', a capacity change rather than evidence. That refuted prediction is "
                 "recorded in docs/reviews/occlusion-repair-2026-09-05.md. The fixture DOES "
                 "confirm the closed form: measured two-view error rises with 1/|sin theta| "
                 "across its bins -- consistent with, and not quoted as a fit. That the fixture's "
                 "preference for demoting everything is an ARTEFACT is the codebase's own "
                 "measurement: `solve_sequence_positions` refuses to overwrite already-"
                 "triangulated slots because on real data the solve moved them by a median of "
                 "11-14 mm and up to 700 mm, the temporal and limb terms outvoting good "
                 "geometry. Those priors are true on the fixture and false on the footage. "
                 "ORDER OF DISCOVERY, disclosed: 150.0 was in src as a first guess before the "
                 "closed form was written down; k = 2 was recognised as the factor that yields "
                 "it, not chosen ahead of it.",
        remedy="none open. If it moves it must move on the closed form with a different "
               "declared amplification factor, or on a fixture whose limb lengths and motion "
               "do NOT satisfy the sequence solve's priors -- never on the real take and never "
               "on the MAMMA arm.",
    ),
    "REACHABILITY_SPEED_CEILING_M_S": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-05. Peak LINEAR speed "
                 "per landmark, used as a physical IMPOSSIBILITY bound and deliberately not a "
                 "plausibility one -- a ceiling tight enough to refuse what a body merely rarely "
                 "does would be a smoother wearing a reject's clothes. The table's sources are "
                 "cited in the comment beside it: elite baseball pitching reaches ~34 m/s at the "
                 "hand and ~9 m/s at the shoulder (Fleisig et al., kinematic chain studies), "
                 "competitive boxing 6-9 m/s at the fist, sprinting ~2x ground speed at the foot "
                 "in swing. Each ceiling sits at or above the published peak for its landmark. "
                 "NOTHING here is fitted: the synthetic fixture demonstrates that the oracle "
                 "fires zero rejections on clean input and that a frozen arm fails, and it "
                 "chooses none of the numbers.",
        remedy="none open. Marker data (lane H) would let these be measured on owned capture "
               "rather than cited; until then they are literature and are labelled as such.",
    ),
    "REACHABILITY_SLACK_M": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-05. The constant term "
                 "of the reachability envelope (slack + ceiling * elapsed_seconds), which exists "
                 "so a stationary landmark's own jitter cannot trip a rule about motion. MEASURED "
                 "on the synthetic fixture as the p99 of the frame-to-frame jitter of the "
                 "triangulation's own error on well-supported cells, rounded up to the "
                 "centimetre, by `tools/compare/d8_occlusion_synthetic.py`. It is NOT taken from "
                 "the error score: at the anatomical speed ceilings a wrist may move 1.13 m in "
                 "one frame, so any slack between 0.02 and 0.40 m is a rounding error on the "
                 "envelope and the score sweep is flat inside 0.06 mm across that whole range. "
                 "Both the flat sweep and the measurement are in the report.",
        selected_against="synthetic truth -- the measured displacement noise of our own "
                         "triangulation under our own detector's heavy-tail model",
        remedy="none open.",
    ),
    "MAXIMUM_INTERPOLATED_GAP_FRAMES": dict(
        provenance=ENGINEERING,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-05. Gaps longer than "
                 "this are not interpolated through: the landmark is HELD on its parent and the "
                 "share is reported as `held_joint_fraction` beside "
                 "`interpolated_joint_fraction`. THE CARD SAID THIS WOULD BE SELECTED ON "
                 "SYNTHETIC TRUTH AND IT COULD NOT BE, for the same reason the ray-angle ceiling "
                 "could not: the fixture's motion is smooth by construction, so a straight line "
                 "through a gap is nearly exact there and the score prefers interpolating at "
                 "every candidate -- a monotone curve whose argmin is 'never hold'. "
                 "`tools/compare/d8_occlusion_synthetic.py` reports that sweep in full, on the "
                 "population the rule acts on, and selects nothing from it. The value is a "
                 "closed-form bound instead: a straight chord across a gap of duration T departs "
                 "from a constant-acceleration trajectory by at most a*T^2/8, and setting that "
                 "equal to REACHABILITY_SLACK_M (the measured jitter of our own triangulation) "
                 "gives T = sqrt(8*slack/a) -- 5.7 frames at a limb acceleration of 20 m/s^2, "
                 "3.6 at 50. The shipped value is the 20 m/s^2 end and THAT CHOICE IS DECLARED, "
                 "NOT DERIVED. Recorded in docs/reviews/occlusion-repair-2026-09-05.md.",
        remedy="OPEN. A fixture whose motion is NOT smooth -- or marker data (lane H) -- could "
               "select this properly. Until then the bound is closed-form with a declared "
               "acceleration, and the clause's value is the diagnostic it emits rather than an "
               "error reduction.",
    ),
    "LANDMARK_PARENT": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-05. Which landmark "
                 "carries which when a long gap is held: a wrist follows its elbow, an elbow its "
                 "shoulder, a shoulder the neck. It is the kinematic chain of a human arm and "
                 "leg, not a fitted or measured quantity, and `root` has no parent and is never "
                 "held. It changes nothing unless a gap exceeds "
                 "MAXIMUM_INTERPOLATED_GAP_FRAMES.",
        remedy="none open.",
    ),
    # ------------------------------------------------- D8b, the segment-length reject
    "SEGMENT_LENGTH_CEILING_FRACTION": dict(
        provenance=OWN_CAPTURE,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-06. A captured "
                 "segment whose length departs from THAT PERFORMER'S OWN take median by "
                 "more than this fraction has its child landmark withheld. It is a REAL-TAKE "
                 "SEED and is registered as own-capture with its source stated: on the "
                 "reference take the eight leg segments hold their length to -5.1 %/+6.2 % "
                 "at p5-p95 on both performers, with 0 frames off their own median by more "
                 "than 15 % on 8 of 8 segments (`tools/compare/captured_limb_stability.py` "
                 "-> artifacts/compare/d8b-length/limb-stability-d9.json), so 15 % is the "
                 "smallest ceiling the honest motion in this take demonstrably clears. IT IS "
                 "OUR OWN CAPTURE AND NEVER MAMMA'S: the arrays are the delivery's own "
                 "triangulated landmarks. THE CARD SAID SYNTHETIC TRUTH WOULD CONFIRM IT AND "
                 "IT DID NOT: `tools/compare/d8b_length_synthetic.py` fails both of the "
                 "card's honest-motion clauses at this ceiling and at every ceiling swept up "
                 "to 0.30, because the fixture's own honest triangulation is far noisier "
                 "than the reference take's -- its uncollapsed frames and its legs cross the "
                 "ceiling without any collapse being injected. That report carries the "
                 "fixture's own measured segment spread beside the take's and the truth "
                 "error of every fired cell, so the two failures can be told apart. The "
                 "value was NOT moved to accommodate them. Recorded in "
                 "docs/reviews/segment-length-2026-09-06.md.",
        selected_against="the reference take's own legs (own capture). The synthetic arm "
                         "reports and did not confirm; nothing was selected on it and "
                         "nothing was selected on a MAMMA-referenced arm",
        remedy="OPEN. The ceiling is a seed from ONE take's legs, and the fixture that was "
               "supposed to confirm it cannot: its honest frames are noisier than the "
               "take's. Marker data (lane H), or a fixture whose triangulation noise "
               "matches the detector's on this rig, would settle it. Until then the number "
               "is a measured seed with a stated source and a stated failed confirmation.",
    ),
    "SEGMENT_LENGTH_RULES": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-06. Which segments "
                 "the length rule measures and which landmark a departure is charged to: the "
                 "CHILD, because the parent is fixed by the segment above it -- except the "
                 "shoulder line, which has no parent, so both of its endpoints are charged. "
                 "It is the kinematic chain of a human arm and leg, not a fitted or measured "
                 "quantity. Deliberately NOT derived from RIGID_LIMBS, which is what the "
                 "sequence solve regularises rather than what a captured frame is judged on; "
                 "RIGID_LIMBS also contains root-to-neck, which this rule does not act on. "
                 "D8c, 2026-09-06: the HIP LINE was added as a tenth row and charges BOTH of "
                 "its endpoints. The hips do have a parent (`root`), so a per-hip `root->hip` "
                 "rule was the more precise candidate and it was MEASURED rather than "
                 "dismissed -- on the same honest mask the hip line's own margin is quoted "
                 "on, `root->left_hip` spreads -9.0 %/+25.5 % at p5-p95 on the falling "
                 "performer (right -9.4/+9.2; the other performer +/-9-10) and at the shipped "
                 "ceiling it would fire on 35 and 17 of his frames and 9 and 4 of hers, most "
                 "of them where the hip line is honest. The pelvis landmark is too loose to "
                 "be a length reference, so `root->hip` was not added. The row is still "
                 "ANATOMY: a name and two joint ids, no fitted quantity, and the ceiling and "
                 "the mode it runs at are D8b's and were not re-selected.",
        remedy="the hip-line half of this entry is CLOSED (D8c, 2026-09-06: the row is in the "
               "table; `tools/compare/d8c_hip_synthetic.py` confirms D8b's mode and ceiling "
               "on the hips' own geometry -- an injected 51.0 mm collapse recovers to 3.9 mm "
               "under `demote` against 50.3 for the nine-row build, with zero fires on the "
               "clean clip and zero outside the injected run). WHAT REMAINS OPEN AND IS "
               "REGISTERED HERE: (1) charging both endpoints over-charges the ONE-HIP frames "
               "-- on the reference take frames 84-86 move one hip at a time and a femoral "
               "head that matched the take is withheld with the one that did not, three "
               "frames, and on the synthetic one-hip arm the recovery is the weakest measured "
               "(20.5 -> 17.0 mm, with the delivered root moving 15.0 mm); (2) a two-view "
               "DEPTH stretch -- the take's frames 158-168, A and C at 140 degrees with the "
               "hip line 16-28 degrees off both rays -- is not realised by any fixture in "
               "this lane, so whether `demote` recovers it is measured only on the real "
               "take's own photographs. The CEILING's remedy is its own entry and stays "
               "open.",
    ),
    "SEGMENT_LENGTH_MODES": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py, since 2026-09-06. The three "
                 "things that can happen to a marked slot's RAYS: 'demote' keeps them (D8's "
                 "path, the sequence solve recovers the point from the same rays plus limb "
                 "length and continuity), 'reject' withholds them all (the slot falls to the "
                 "fill and, past MAXIMUM_INTERPOLATED_GAP_FRAMES, to a hold on the parent), "
                 "'best_ray' keeps only the highest-confidence camera's (the single-ray case "
                 "the solve already recovers). SELECTED ON SYNTHETIC TRUTH by "
                 "`tools/compare/d8b_length_synthetic.py` on an injected consistent "
                 "shoulder collapse: 'demote' wins under both readings of the score -- the "
                 "pooled six-landmark median and the card's own per-group table, where the "
                 "injected shoulders go 96.4 -> 30.0 mm against exact truth while 'reject' "
                 "reaches 42.2 and 'best_ray' 57.2. 'best_ray' chooses its camera by the "
                 "DETECTOR'S OWN confidence and therefore adds no constant.",
        selected_against="synthetic truth (SOMASKEL77 clips posed through our own FK, this "
                         "rig, the real seen pattern replayed, our own detector's "
                         "heavy-tail noise, plus an injected consistent 2D collapse)",
        remedy="none open for the mode. The ceiling it runs at is a separate entry.",
    ),
    "PELVIS_FRAME_SOURCE": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py -- 'C_kabsch_pelvis' since 2026-09-04. "
                 "Which construction turns the triangulated pelvis landmarks into `Hips`' world "
                 "rotation. SELECTED ON SYNTHETIC TRUTH ONLY, on the NOISY arm of "
                 "`tools/compare/d7_pelvis_synthetic.py` -> "
                 "artifacts/compare/d7-pelvis-frame/synthetic.json: the CLEAN arm cannot select, "
                 "because a Kabsch fit of the rest offsets of Hips' three rigid children recovers "
                 "the pelvis rotation EXACTLY (4.2e-8 m residual) for every rigid candidate. The "
                 "selection rule was fixed in docs/reviews/pelvis-frame-2026-09-04.md section 0.4 "
                 "before any number existed. The MAMMA arm reported beside it and selected nothing.",
        selected_against="synthetic truth (GEM-X SOMASKEL77 clips posed through our own FK), "
                         "with our own detector's measured heavy-tail noise",
        remedy="none open. If it moves again it must move on synthetic truth, a held-out camera "
               "or anatomy -- never on the MAMMA arm.",
    ),
    "PELVIS_SMOOTHING_FRAMES": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py -- 0 (no rotation smoothing) since "
                 "2026-09-04. Swept over (0, 3, 5, 9, 15, 21, 31) on the same synthetic arm. The "
                 "pre-registered protocol was I8's own -- interior optimum, plus lag and "
                 "attenuation, with an over-smoothed window required to FAIL on the fast clip. "
                 "It could NOT be executed: at this noise the lag estimator reads -2.4 to -8.1 "
                 "frames and the attenuation 1.2-7.3 (amplification) at EVERY window, window 0 "
                 "included, so neither instrument discriminates. The pre-registered fallback -- "
                 "'if no window beats 0 while holding lag <= 1 frame and attenuation >= 0.9, the "
                 "selection is 0' -- therefore selects 0. The p95 improvement with wider windows "
                 "(33.2 -> 17.2 deg) is REPORTED and selects nothing: a window optimises exactly "
                 "the quantity that arm measures.",
        selected_against="synthetic truth; the sweep and its refusal are in "
                         "artifacts/compare/d7-pelvis-frame/synthetic.json",
        remedy="the window sweep needs a lag/attenuation instrument that works at this noise "
               "before any non-zero window may be selected.",
    ),
    "PELVIS_MINIMUM_RESOLVED_FRACTION": dict(
        provenance=ENGINEERING,
        evidence="src/autoanim_gnm/commercial_multiview.py -- 0.5. Below this fraction of frames "
                 "with a resolved Spine1 the WHOLE subject falls back to the torso frame with a "
                 "diagnostics status. Deliberately a whole-subject decision: a per-frame flip "
                 "between a spine-derived and a trunk-derived pelvis moves `R_hips . mid` by tens "
                 "of millimetres in one frame and spikes the root. Nothing is fitted to it; on the "
                 "delivered take both subjects resolve at 1.000.",
        selected_against="nothing -- it is a refusal threshold, not a tuned value",
    ),
    "SOMA77_REST_PELVIS_TEMPLATE_M": dict(
        provenance=THIRD_PARTY,
        evidence="src/autoanim_gnm/commercial_multiview.py -- the rest pelvis the measured one is "
                 "aligned onto. A CONVENTION, and the module comment says so: "
                 "src/autoanim_gnm/data/somaskel77-v1.json carries NO rest geometry (the plan card "
                 "is wrong about that); the rest lives per clip in "
                 ".cache/autoanim_gnm/gem-x/outputs/*/soma_motion.npz and is PER PERFORMER -- five "
                 "full-body clips' pelvis source frames differ by up to 10.03 deg. These are the "
                 "component-wise MEDIAN over those five GEM-X/Kimodo somaskel77 rests, "
                 "re-normalised (artifacts/compare/d7-pelvis-frame/rest-pelvis-constants.json). "
                 "MAMMA-FREE: no MAMMA file, ma_cap output, SMPL-X fit or report computed from one "
                 "enters them. Cost of a residual: a leftover pitch delta moves the root fore/aft "
                 "by |mid| . sin(delta) with |mid| ~ 80 mm, on every frame.",
        selected_against="third-party skeleton geometry; never the take, never MAMMA",
        remedy="lane H's owned marker session can measure a real pelvis and replace the "
               "convention with a measurement.",
    ),
    "SOMA77_REST_PELVIS_UP": dict(
        provenance=THIRD_PARTY,
        evidence="as SOMA77_REST_PELVIS_TEMPLATE_M -- the root->Spine1 rest direction, used by the "
                 "A_root_to_spine1 construction, which is measured and NOT the one that ships.",
        selected_against="third-party skeleton geometry",
    ),
    "SOMA77_REST_HIPMID_TO_SPINE1": dict(
        provenance=THIRD_PARTY,
        evidence="as SOMA77_REST_PELVIS_TEMPLATE_M -- the mid(hips)->Spine1 rest direction, used by "
                 "the B_hipmid_to_spine1 construction, which is measured and NOT the one that ships.",
        selected_against="third-party skeleton geometry",
    ),
    "SOMA77_REST_HIP_ACROSS": dict(
        provenance=THIRD_PARTY,
        evidence="as SOMA77_REST_PELVIS_TEMPLATE_M -- the rest hip line, the secondary axis of the "
                 "A and B constructions.",
        selected_against="third-party skeleton geometry",
    ),
    # ---------------------------------------------------------------- anatomy / literature
    "MAXIMUM_FRAME_TRAVEL_DEG": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/head_orientation.py:50 -- 'A hard physical reject, not "
                 "a tuning knob. A human head peaks around 500-800 deg/s, so at 30 fps "
                 "roughly 27 deg between consecutive frames is already extreme; 60 deg is "
                 "1800 deg/s and is not a neck.' Commit 981e437 records the 140 deg "
                 "single-frame flip it was added to reject.",
    ),
    "CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py -- 'A HARD PHYSICAL REJECT ON "
                 "THE CLAVICLE, NOT A TUNING KNOB, AND NOT A SMOOTHER... A human joint "
                 "peaks near 500-800 deg/s (the same physiology `head_orientation."
                 "MAXIMUM_FRAME_TRAVEL_DEG` is drawn from), so at 30 fps 26.67 deg between "
                 "consecutive frames is already the extreme end. A sternoclavicular joint "
                 "has roughly 45 deg of elevation and 30 deg of protraction in total, so "
                 "this ceiling is generous by a wide margin: it is here to reject the "
                 "impossible, never to shape the possible.' D2c, "
                 "docs/reviews/clavicle-origin-2026-09-02.md section 15, added to reject "
                 "the 139 and 164 deg single-frame clavicle flips D2 and D2b's shorter "
                 "lever produced.",
        selected_against="nothing -- it is a physical ceiling, not a fitted value. Its "
                         "inertness was DEMONSTRATED on synthetic truth (I7's FK fixture, "
                         "true clavicle peak 1.38 deg/frame against this 26.67) and the "
                         "degenerate that must fail it is a ceiling below that true peak. "
                         "No MAMMA figure and no reference fitter enters it.",
        note="ANATOMY and not a category called 'physical': the taxonomy above has no such "
             "value, and the head lane's identical ceiling "
             "(`MAXIMUM_FRAME_TRAVEL_DEG`) is registered as ANATOMY. Adding a taxonomy "
             "value would move `tests/test_provenance_audit.py`, which is the registry "
             "owner's file.",
    ),
    "NECK_ROTATION_SHARE": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py:1250 -- 'This is an ANATOMICAL "
                 "distribution, not a measurement... 0.5 is the standard rigging split and "
                 "is deliberately not tuned: there is nothing on this fixture to tune it "
                 "against, and a value fitted to a reference would be a shipped constant "
                 "calibrated on MAMMA.' Commit f6a4973.",
        note="The composed head world orientation is invariant to this split, so no "
             "measured quantity depends on it.",
    ),
    "FINGER_REST_CURL_DEG": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/commercial_multiview.py:1374 -- 'Per-joint flexion in "
                 "degrees about the rig's curl axis, from a relaxed open hand... So the "
                 "fingers get a POSE, not a solve.' Commit f6a4973 labels it a pose, not a "
                 "measurement. The detector's own finger landmarks were rejected as a "
                 "wrist-conditioned prior (tools/head/region_landmark_quality.py).",
    ),
    "FINGER_REST_CURL_THUMB_SCALE": dict(
        provenance=ANATOMY,
        evidence="Same comment block: 'The thumb is shallower and the distal joints curl "
                 "more than the proximal, which is what a hand at rest does.' No fitted "
                 "value; 0.45 is stated as the pose, not a measurement.",
    ),
    "CANONICAL_HEAD_AXES": dict(
        provenance=ANATOMY,
        evidence="src/autoanim_gnm/head_orientation.py:60 -- \"The head's anatomical axes, "
                 "expressed in the CAPTURE convention.\" Commit 981e437: the zero 'is now "
                 "fixed from the template's own anatomy -- HeadEnd above Head for the long "
                 "axis, the eyes for the lateral one -- and never from the reference, which "
                 "would be a shipped constant fitted on a reference-derived artifact.'",
    ),
    "robust_scale_px": dict(
        provenance=LITERATURE,
        evidence="commit 16f24ec -- 'Residual scaling follows Anipose (Karashchuk et al., "
                 "Cell Reports 2021), reimplemented from the published description -- "
                 "soft-l1 on the reprojection block only.' Value shares the 14.0 px "
                 "denomination with `inlier_threshold_px`.",
    ),
    # ---------------------------------------------------------------- own capture / self-consistency
    "DEFAULT_WEIGHTS": dict(
        provenance=OWN_CAPTURE,
        evidence="src/autoanim_gnm/head_orientation.py:40 -- 'The weight grid the temporal "
                 "prior is selected over, and the rule. Selection consults only our own "
                 "reprojection -- never any reference -- so a gate scored against a "
                 "reference cannot be tuned through it. The grid is swept per take rather "
                 "than hardcoded.' Commit 981e437 says the same.",
        note="This is a GRID, not a value. The shipped weight is chosen per take by "
             "minimum in-frame reprojection, so run-report's temporal_weight = 100.0 is a "
             "measurement on that take, not a constant. The grid's endpoints and spacing "
             "themselves have no stated origin -- see the unknown entry "
             "'DEFAULT_WEIGHTS grid extent'.",
        meta_level_caveat="The rule's EXECUTION is reference-free, and that is what the "
                          "flag rule tests. Its ADOPTION was not: CLAUDE.md lists 'the "
                          "weight-selection rule' among the instrument repairs that fixed "
                          "the head, and the head was scored on a gate whose reference is "
                          "MAMMA's head. So an algorithm -- not a constant -- was kept "
                          "because it improved a MAMMA-referenced gate. Not a leak by the "
                          "stated rule, which flags constants; recorded here because it is "
                          "the same class of thing one level up, and because the manifest "
                          "should not let a reader conclude the head solve owes nothing to "
                          "the reference.",
    ),
    "REFERENCE_DETECTOR_WIDTH_PX": dict(
        provenance=OWN_CAPTURE,
        evidence="commit 6579460 -- 'Apple Vision saturates below 1280 and carries ~26 mm "
                 "of 2D error at the subject at any input width. The apparent 2.7x win at "
                 "3840 was survivorship bias from a fixed pixel gate.' The constant exists "
                 "so pixel gates scale with detector width; measured a no-op at 1280, "
                 "'bit-identical on all seven reported metrics'.",
    ),
    "SMOOTHING_WINDOW_FRAMES": dict(
        provenance=SYNTHETIC,
        evidence="src/autoanim_gnm/commercial_multiview.py:1193 -- 'Measured cost: with "
                 "*noiseless* 2D the smoother alone contributes 0.65 mm of median error at "
                 "the reference fixture's joint speeds and 4.42 mm at six times those "
                 "speeds.' Commit b6ff3cd measured it on the synthetic truth fixture's "
                 "zero-noise control across --stride.",
        note="MEASURED on synthetic truth, but not SELECTED there: b6ff3cd says 'longer is "
             "better for MPJPE at both speeds, so the shipped 9 is not wrong on the metric' "
             "and names 9 as the pre-existing value it did not change. Its own origin is "
             "'a fixed 300 ms window at 30 fps' -- a round number, not a fit. Treated as "
             "measured-and-declared rather than selected; it is not MAMMA-derived either "
             "way. Re-selection belongs with I7.",
    ),
    # ---------------------------------------------------------------- engineering limits
    "MAXIMUM_EXHAUSTIVE_ASSOCIATION_CANDIDATES": dict(
        provenance=ENGINEERING,
        evidence="A compute budget, not an accuracy constant. Commit 9eb0393: the "
                 "exhaustive search is '(subjects!)^cameras: 16 assignments at two "
                 "subjects and four cameras, 1,296 at three subjects, 110 billion at four "
                 "subjects and eight cameras.' Exceeding it raises rather than degrades.",
    ),
    "MAXIMUM_SURPLUS_COMPONENTS": dict(
        provenance=ENGINEERING,
        evidence="Caps how many connected components beyond `subject_count` are considered "
                 "(commercial_multiview.py:807). A search bound; no reference is consulted.",
    ),
    "MINIMUM_LIMB_SAMPLES": dict(
        provenance=ENGINEERING,
        evidence="commercial_multiview.py:965 -- the minimum number of directly-"
                 "triangulated frames before a limb length is taken as a per-shot median. "
                 "A sufficiency floor for an estimator, not a tuned accuracy parameter.",
    ),
    "MINIMUM_LANDMARKS_PER_VIEW": dict(
        provenance=ENGINEERING,
        evidence="head_orientation.py:48. Three non-collinear points is the minimum a "
                 "rigid orientation can be resolved from in one view; a geometric "
                 "sufficiency condition, not a fit.",
    ),
    "MINIMUM_SOLVED_FRACTION": dict(
        provenance=ENGINEERING,
        evidence="head_orientation.py:49. A refusal threshold -- below it the solve raises "
                 "and the caller falls back and reports (commit 981e437: 'Callers fall "
                 "back and report; they never guess'). It gates whether an answer is "
                 "returned, not what the answer is. Its VALUE (0.5) has no stated origin.",
    ),
    "minimum_confidence": dict(
        provenance=ENGINEERING,
        evidence="commercial_multiview.py:1888 -- 'The confidence floor is both a gate "
                 "and, through sqrt(confidence), the only channel an observation has to "
                 "declare how much it should be trusted. Fixed at 0.25 it confines usable "
                 "confidences to [0.26, 1.0]... which is why this is now reachable. "
                 "Default unchanged.' The comment documents the value's LIMITS, not its "
                 "origin.",
        note="borderline: documented and reachable, but 0.25 itself is not derived "
             "anywhere. Listed as engineering-limit because the comment argues it is a "
             "floor rather than a fit; if a reviewer disagrees it belongs in `unknown`.",
    ),
    "maximum_evaluations": dict(
        provenance=ENGINEERING,
        evidence="solve_sequence_positions:1003 -- an optimiser iteration cap.",
    ),
    "sample_rate_hz": dict(
        provenance=SCHEMA,
        evidence="The capture frame rate of the fixture; carried into BodyTrack. Not a "
                 "tuned quantity.",
    ),
    "subject_count": dict(
        provenance=SCHEMA,
        evidence="How many performers the caller asks for. Data, not calibration.",
    ),
    "pixel_scale": dict(
        provenance=SCHEMA,
        evidence="1.0 means 'observations are already at REFERENCE_DETECTOR_WIDTH_PX'; the "
                 "delivery path always passes the measured ratio (commercial_multiview.py"
                 ":1946). Not a constant in the delivery build.",
    ),
    # ---------------------------------------------------------------- third-party
    "CROP": dict(provenance=THIRD_PARTY,
                 evidence="soma77_pose.py:52 -- the NVIDIA GEM-X SOMA-77 model's own input "
                          "geometry. Vendor-fixed; changing it changes the model contract."),
    "MODEL_WIDTH": dict(provenance=THIRD_PARTY,
                        evidence="soma77_pose.py:53 -- 'The model sees only the centre "
                                 "MODEL_WIDTH of the CROP columns' (:99). Vendor-fixed."),
    "_MEAN": dict(provenance=THIRD_PARTY,
                  evidence="soma77_pose.py:54 -- ImageNet channel means (0.485, 0.456, "
                           "0.406), the normalisation the vendor's checkpoint was trained "
                           "with. Vendor-fixed."),
    "_STD": dict(provenance=THIRD_PARTY,
                 evidence="soma77_pose.py:55 -- ImageNet channel standard deviations "
                          "(0.229, 0.224, 0.225). Vendor-fixed."),
    # ---------------------------------------------------------------- schema / identity
    "SCHEMA_VERSION": dict(provenance=SCHEMA, evidence="Output contract version string."),
    "OBSERVATION_SCHEMA_VERSION": dict(provenance=SCHEMA, evidence="Input contract version."),
    "BODY_OBSERVATION_SCHEMA_VERSION": dict(provenance=SCHEMA, evidence="Input contract version."),
    "LEGACY_OBSERVATION_DETECTOR": dict(provenance=SCHEMA, evidence="Detector name string."),
    "DETECTOR": dict(provenance=SCHEMA, evidence="Detector name string."),
    "PROVIDER_ID": dict(provenance=SCHEMA, evidence="Provider identity string."),
    "JOINT_NAMES": dict(provenance=SCHEMA, evidence="The 19-joint observation contract."),
    "JOINT_INDEX": dict(provenance=SCHEMA, evidence="Derived from JOINT_NAMES."),
    "RIGID_LIMBS": dict(provenance=ANATOMY,
                        evidence="The bone pairs treated as fixed length. Skeletal "
                                 "topology, not a fitted set."),
    "CORE_ASSOCIATION_JOINTS": dict(provenance=SCHEMA,
                                    evidence="Which of the 19 joints the associator scores "
                                             "on. A choice of columns, no value fitted."),
    "_ROOT_DEPENDENT_JOINTS": dict(
        provenance=SCHEMA,
        evidence="D9b, 2026-09-07. WHICH joints the hoist re-aim rewrites after "
                 "`project_generated_foot_contacts` moves the root: the trunk chain, the "
                 "neck and head, the eyes, the two clavicles, the four arm bones and the "
                 "two hands. It is read off the SKELETON'S TOPOLOGY -- every joint whose "
                 "forward-kinematic origin depends on the root translation and is not the "
                 "projection's own output -- and no number is fitted, selected or tuned. "
                 "The complement is stated in the same comment: `Root`, `Hips`, the legs, "
                 "the feet, the toes and the fingers.",
    ),
    "SOMA77_TO_AUTOANIM": dict(
        provenance=SCHEMA,
        evidence="soma77_pose.py:61 -- the index map from SOMA-77's 77 joints to the "
                 "19-joint contract. CLAUDE.md records that it maps 17 and drops every "
                 "finger, toe and HeadEnd; docs/HEAD_FEET_HANDS_PLAN.md owns the "
                 "consequence. A convention, not a calibration.",
    ),
    "MAXIMUM_PEOPLE": dict(provenance=SCHEMA, evidence="mediapipe_pose.py detector cap; "
                                                       "not on the SOMA-77 delivery path."),
    # ---------------------------------------------------------------- soma_motion schema
    "SOMA_MOTION_SCHEMA_VERSION": dict(provenance=SCHEMA, evidence="Output contract version."),
    "SOMASKEL77_SCHEMA_VERSION": dict(provenance=SCHEMA, evidence="Skeleton contract version."),
    "SOMASKEL77_PATH": dict(provenance=ASSET,
        evidence="src/autoanim_gnm/data/somaskel77-v1.json -- the vendor skeleton contract "
                 "checked into the repo, pinned by SOMASKEL77_SEMANTIC_SHA256."),
    "SOMASKEL77_NAMES": dict(provenance=ASSET, evidence="Read from SOMASKEL77_PATH."),
    "SOMASKEL77_PARENTS": dict(provenance=ASSET, evidence="Read from SOMASKEL77_PATH."),
    "SOMASKEL77_PROVENANCE_SHA256": dict(provenance=SCHEMA,
        evidence="A digest OF the upstream block, computed at import. Provenance machinery, "
                 "not a calibration."),
    "SOMASKEL77_SEMANTIC_SHA256": dict(provenance=SCHEMA,
        evidence="Pins the vendor skeleton contract; a change to the file is refused rather "
                 "than absorbed. Provenance machinery."),
    "_SKELETON": dict(provenance=ASSET, evidence="The parsed SOMASKEL77_PATH contract."),
    "ALLOWED_PROVIDER_COMMITS": dict(provenance=THIRD_PARTY,
        evidence="Upstream NVIDIA GEM-X and Kimodo commit hashes the provider is pinned to. "
                 "Vendor identity, not a fitted value."),
    "GEM_X_CONTACT_SCHEMA_ID": dict(provenance=SCHEMA, evidence="Contact channel contract id."),
    "GEM_X_CONTACT_NAMES": dict(provenance=SCHEMA,
        evidence="The vendor's own contact channel names."),
    "KIMODO_CONTACT_SCHEMA_ID": dict(provenance=SCHEMA, evidence="Contact channel contract id."),
    "KIMODO_CONTACT_NAMES": dict(provenance=SCHEMA, evidence="The vendor's channel names."),
    "CONTACT_SCHEMAS": dict(provenance=SCHEMA, evidence="Derived from the two above."),
    "_DELTA_MAPPING": dict(provenance=SCHEMA,
        evidence="soma_motion.py:627 -- the joint-name correspondence between SOMA and the "
                 "AutoAnim rig, with the left/right anatomical swap documented in the "
                 "comment at :636: 'SOMA anatomical left is positive source X, while "
                 "AutoAnim\'s reviewed canonical skeleton defines Left on negative X.' A "
                 "convention, not a calibration."),
    "_DETAILED_DELTA_MAPPING": dict(provenance=SCHEMA, evidence="Derived from _DELTA_MAPPING."),
    "MAX_SOMA_FRAMES": dict(provenance=ENGINEERING,
        evidence="30 fps * 60 s * 120 min + 1 -- a two-hour input ceiling. A resource "
                 "bound written as arithmetic, not a tuned value."),
    "MAX_SOMA_DURATION_TICKS": dict(provenance=ENGINEERING,
        evidence="30 minutes in ticks. A resource bound."),
    "FK_TOLERANCE_M": dict(provenance=ENGINEERING,
        evidence="soma_motion.py:623 -- a VALIDATION tolerance: FK positions must agree "
                 "with the stored joint positions to 1 mm or the load is refused. It gates "
                 "consistency of an input, and does not enter any delivered value."),
    "QUATERNION_TOLERANCE": dict(provenance=ENGINEERING,
        evidence="A unit-norm validation tolerance on input quaternions. Same character as "
                 "FK_TOLERANCE_M: it refuses bad input, it does not shape output."),
    "BOX_PADDING": dict(provenance=UNKNOWN,
        evidence="soma77_pose.py:58 -- the person box is expanded by 1.25 before the "
                 "SOMA-77 crop (:106). Introduced whole in commit 2eca1f3 with no stated "
                 "origin. Not MAMMA-derived: it is a detector-side crop parameter and no "
                 "MAMMA figure existed when it landed. It is, however, a real accuracy "
                 "knob -- it sets how much context the keypoint model sees."),
    # ---------------------------------------------------------------- unknown
    "inlier_threshold_px": dict(
        provenance=UNKNOWN,
        evidence="commit 6579460 states what 14 px MEANS ('~78 mm at the subject at 1280 "
                 "but ~26 mm at 3840') and makes it scale with detector width. Nothing in "
                 "the code, docs or history states where 14 came from. Not MAMMA-derived: "
                 "it predates every MAMMA comparison in this repo.",
    ),
    "maximum_epipolar_px": dict(
        provenance=UNKNOWN,
        evidence="Introduced whole in commit 9eb0393 with the graph associator. That "
                 "commit validates the associator against the exhaustive path ('0 of 150 "
                 "frames' differ) but never states where 60 px came from.",
    ),
    "minimum_shared_joints": dict(
        provenance=UNKNOWN,
        evidence="Introduced whole in commit 9eb0393, same silence as maximum_epipolar_px. "
                 "4 of the 19-joint contract must be shared before a cross-view pair is "
                 "scored at all.",
    ),
    "ambiguity_ratio": dict(
        provenance=UNKNOWN,
        evidence="Commit 9b69d8b introduced both ambiguity gates and validated the RESULT "
                 "on synthetic noise sweeps ('a 4-angle x 3-separation x 5-seed sweep at "
                 "0.5 px noise differs from exhaustive on 0/60'), which is a check that "
                 "the gate does not break the answer, not a selection of 0.7.",
        note="A sweep on synthetic noise exists and could select these; it did not, and "
             "the commit does not claim it did.",
    ),
    "ambiguity_margin_px": dict(
        provenance=UNKNOWN,
        evidence="Introduced in commit 9b69d8b alongside ambiguity_ratio, with the same "
                 "silence: the commit validates the resulting associator, never the value.",
    ),
    "smooth_weight": dict(
        provenance=UNKNOWN,
        evidence="Introduced in commit 16f24ec with solve_sequence_positions. The commit "
                 "cites Anipose for the residual SCALING and the percentage limb error, "
                 "and states two deviations from it, but does not say where the weight "
                 "2.0 came from. LADDER_EXECUTION_PLAN.md §4 lists 'selecting "
                 "smooth_weight toward MAMMA's amplitude' as a leak path to avoid -- as a "
                 "future risk, not as something already done.",
    ),
    "length_weight": dict(
        provenance=UNKNOWN,
        evidence="Introduced in commit 16f24ec alongside smooth_weight, with the same "
                 "silence: the commit cites Anipose for the percentage limb error but not "
                 "for this weight.",
    ),
    "neck_sigma_m": dict(
        provenance=UNKNOWN,
        evidence="head_orientation.py:443. Commit 981e437 explains WHY the prior lives at "
                 "the neck ('which is where anatomy says the smoothness lives') but not "
                 "where 10 mm came from. Anatomically plausible as a neck-position "
                 "tolerance; nothing states it.",
    ),
    "template_prior": dict(
        provenance=UNKNOWN,
        evidence="head_orientation.py:445. No stated origin in code, docs or 981e437.",
    ),
    "DEFAULT_WEIGHTS grid extent": dict(
        provenance=UNKNOWN,
        evidence="The rule that picks from the grid is documented and reference-free; the "
                 "grid's endpoints (0 to 1e5) and half-decade spacing are not. A grid too "
                 "narrow would silently clamp the per-take selection.",
        synthetic=True,
    ),
    "Hips rest height 0.98 m": dict(
        provenance=ASSET,
        evidence="src/autoanim_gnm/body.py:119, CANONICAL_HUMANOID's Hips offset -- the "
                 "shipped rig asset's own geometry. commercial_multiview.py:1526 now reads "
                 "it from the skeleton instead of repeating the literal, and the comment "
                 "there records why. A property of the asset, not a calibration.",
        synthetic=True,
    ),
    "foot contact envelope (capture path)": dict(
        provenance=UNKNOWN,
        evidence="commercial_multiview.py:1665-1669 overrides project_generated_foot_"
                 "contacts' defaults for captured data: velocity 0.30 m/s, ground band "
                 "0.08 m, 2 frames, root correction 0.08 m, against the generated-track "
                 "defaults 0.10 / 0.035 / 3 / 0.05. The comment states the REASON "
                 "('Calibrated capture contains noisier frame-to-frame ankle estimates "
                 "than a generated track') but not where the four numbers came from. "
                 "Neither set is MAMMA-derived: LADDER_EXECUTION_PLAN.md §1 records that "
                 "the delivered foot has been solved from SOMA-77's ToeBase since f6a4973, "
                 "and §2/I4 records that MAMMA's feet are still UNSCORED -- so there was "
                 "no MAMMA foot figure to fit against.",
        synthetic=True,
    ),
    "velocity_threshold_m_per_s": dict(provenance=UNKNOWN, evidence="See 'foot contact "
                                       "envelope (capture path)'. Generated-track default."),
    "ground_band_m": dict(provenance=UNKNOWN, evidence="See 'foot contact envelope "
                          "(capture path)'. Generated-track default."),
    "minimum_contact_frames": dict(provenance=UNKNOWN, evidence="See 'foot contact "
                                   "envelope (capture path)'. Generated-track default."),
    "maximum_root_correction_m": dict(provenance=UNKNOWN, evidence="See 'foot contact "
                                      "envelope (capture path)'. Generated-track default."),
}

# --------------------------------------------------------------------------------------
# Data the delivery build consumes: calibration files, weights, assets, training inputs.
# These do not appear in any source scan, and one of them is the declared MAMMA item.
# --------------------------------------------------------------------------------------
DATA_INPUTS = (
    dict(name="camera rig calibration",
         path="artifacts/commercial-multiview-soma77/camera-rig.json",
         provenance=DECLARED_MAMMA,
         evidence="SUBSTITUTION_LADDER.md §3 rung 0: 'nothing owned; camera-rig.json IS "
                  "MAMMA's ma_cap'. Status 'not owned'. Byte-identical (sha256 80d28a41...) "
                  "to artifacts/soma77-full/camera-rig.json. LADDER_EXECUTION_PLAN.md lane "
                  "H retires it with an owned fixture; until then EVERY metric-scale figure "
                  "in this lane rests on it.",
         declared=True,
         retired_by="lane H, the owned multicam fixture"),
    dict(name="test footage",
         path="MAMMA example take pushing_and_lifting_from_ground, frames [60, 210)",
         provenance=DECLARED_MAMMA,
         evidence="run-report.json: 'test_fixture_license_scope: MAMMA example "
                  "footage/calibration: research comparison only'; "
                  "'production_claim: false'. The footage is the licence-scoped input, not "
                  "a constant, but it is the population every band in this lane is quoted "
                  "on.",
         declared=True,
         retired_by="lane H"),
    dict(name="SOMA-77 keypoint model (NVIDIA GEM-X)",
         path=".cache/autoanim_gnm/gem-x -- ONNX, passed to soma77_pose.py as MODEL.onnx",
         provenance=THIRD_PARTY,
         evidence="run-report.json runtime_dependencies.detector: 'Apple Vision "
                  "VNDetectHumanBodyPoseRequest (person boxes) -> NVIDIA GEM-X SOMA-77 "
                  "(keypoints)'; runtime_dependencies.mamma_weights: false. Vendor weights; "
                  "their training data is NVIDIA's and is not enumerable here. No MAMMA "
                  "output has ever entered a training run in this repo -- "
                  "LADDER_EXECUTION_PLAN.md §2 defers a pseudo-label detector until after "
                  "lane H precisely because 'labels triangulated on MAMMA's calibration put "
                  "MAMMA-derived data into weights'.",
         declared=False),
    dict(name="person detector (Apple Vision)",
         path="VNDetectHumanBodyPoseRequest, OS-provided",
         provenance=THIRD_PARTY,
         evidence="Same run-report line. Apple's weights; training data not ours.",
         declared=False),
    dict(name="delivered body rig and mesh",
         path="AutoAnim detailed-hands asset (CANONICAL_HUMANOID + fingers, MPFB mesh)",
         provenance=ASSET,
         evidence="run-report.json runtime_dependencies.body_asset: 'existing AutoAnim "
                  "detailed-hands asset'; runtime_dependencies.smplx_model: false. The rig's "
                  "rest offsets (body.py:116ff) are the asset's own geometry. Ladder rung 6 "
                  "records that delivery carries ONE fixed rig with canonical proportions, "
                  "so no fitted shape -- and therefore no fitted shape from MAMMA -- is in "
                  "the build.",
         declared=False),
    dict(name="trained weights fitted in this repo",
         path="(none)",
         provenance=SCHEMA,
         evidence="run-report.json: mamma=false, mamma_outputs=false, mamma_weights=false, "
                  "smplx_model=false. Nothing in the delivery path is trained here, so "
                  "there is no training-example manifest to audit. This is the strongest "
                  "single fact in the audit: a leak into weights is currently impossible "
                  "because there are no weights of ours.",
         declared=False),
)


# ------------------------------------------------------------------------- the scanner

def module_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def called_names(node: ast.AST) -> set[str]:
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
    return out


def reachable(functions: dict[str, tuple[str, ast.FunctionDef]]) -> set[str]:
    """BFS over the delivery call graph from ENTRY_POINTS."""
    seen, queue = set(), [n for n in ENTRY_POINTS if n in functions]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        for call in called_names(functions[name][1]):
            if call in functions and call not in seen:
                queue.append(call)
    return seen


def literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def render(value) -> str:
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def scan() -> list[dict]:
    """Every module-level constant and every reachable numeric keyword default."""
    functions: dict[str, tuple[str, ast.FunctionDef]] = {}
    trees: dict[str, ast.AST] = {}
    for relative in DELIVERY_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        trees[relative] = tree
        for name, node in module_functions(tree).items():
            functions.setdefault(name, (relative, node))
    live = reachable(functions)

    items: list[dict] = []
    for relative, tree in trees.items():
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            for name in targets:
                if name == "__all__":
                    continue
                if not name.isupper() and not name.startswith("_"):
                    continue
                value = literal(node.value)
                items.append({"name": name, "kind": "module constant",
                              "file": f"{relative}:{node.lineno}",
                              "value": render(value) if value is not None else "<computed>"})
        for name, node in module_functions(tree).items():
            if name not in live:
                continue
            positional = node.args.args[len(node.args.args) - len(node.args.defaults):]
            pairs = list(zip(positional, node.args.defaults))
            pairs += [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                      if d is not None]
            for argument, default in pairs:
                value = literal(default)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                items.append({"name": argument.arg, "kind": f"keyword default of {name}()",
                              "file": f"{relative}:{default.lineno}", "value": render(value)})
    return items, sorted(live)


def classify(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        curated = CURATED.get(item["name"])
        record = dict(item)
        if curated is None:
            record.update(provenance=UNKNOWN,
                          evidence="not curated: nothing in the code, docs or history was "
                                   "found stating an origin for this value")
        else:
            record.update({k: v for k, v in curated.items() if k != "synthetic"})
        out.append(record)
    # Curated entries that describe an inline literal or a data-shaped item rather than a
    # scannable symbol are added by hand; they are marked so the report is honest about
    # which entries the scanner found and which a human put there.
    for name, curated in CURATED.items():
        if curated.get("synthetic"):
            out.append({"name": name, "kind": "inline literal (curated by hand, not scanned)",
                        "file": "see evidence", "value": "see evidence",
                        **{k: v for k, v in curated.items() if k != "synthetic"}})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/compare/provenance.json")
    args = parser.parse_args()

    scanned, live = scan()
    manifest = classify(scanned)
    for entry in DATA_INPUTS:
        manifest.append({"kind": "data input", "file": entry["path"], "value": "-", **entry})

    seen: set[tuple] = set()
    unique = []
    for entry in manifest:
        key = (entry["name"], entry["file"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    unique.sort(key=lambda e: (e["provenance"] != MAMMA_DERIVED,
                               e["provenance"] != DECLARED_MAMMA,
                               e["provenance"] != UNKNOWN, e["name"]))

    leaks = [e for e in unique if e["provenance"] == MAMMA_DERIVED]
    declared = [e for e in unique if e["provenance"] == DECLARED_MAMMA]
    unknown = [e for e in unique if e["provenance"] == UNKNOWN]

    report = {
        "schema_version": "autoanim.provenance-manifest/1.0",
        "step": "I8",
        "rule": "any constant whose SELECTION cites artifacts/mamma, ma_cap, SMPL-X "
                "outputs, pred_joints/pred_vertices, or a report computed from them is "
                "MAMMA-DERIVED and is a leak",
        "entry_points": list(ENTRY_POINTS),
        "modules_scanned": list(DELIVERY_MODULES),
        "functions_reachable": live,
        "counts": {"manifest": len(unique), "leaks": len(leaks),
                   "declared": len(declared), "unknown": len(unknown),
                   "clean": len([e for e in unique if e["provenance"] in CLEAN])},
        "leaks": leaks,
        "declared": declared,
        "unknown": unknown,
        "manifest": unique,
        "blind_to": [
            "inline numeric literals inside function bodies: curated by hand from a read "
            "of the delivery functions, NOT exhaustively enumerated. An inline magic "
            "number nobody noticed will not appear here.",
            "constants that enter through a data file rather than through source -- which "
            "is exactly how the one declared MAMMA dependency, the camera rig, enters.",
            "provenance the history never recorded. `git log -S` finds the commit that "
            "introduced a value; it cannot recover a reason nobody wrote down. Every such "
            "case is `unknown`, never a guess.",
            "the third-party detectors' training data, which is NVIDIA's and Apple's and "
            "is not enumerable from here.",
            "anything reached only through a caller outside the scanned modules.",
            "an asymmetry in the scan itself: module-level constants are collected from "
            "every scanned module regardless of reachability, while keyword defaults are "
            "filtered to the reachable call graph. So a module constant that no delivery "
            "function reads still appears (conservative), and a keyword default reached "
            "only from outside these modules does not (a real gap).",
            "the ADOPTION of an algorithm, as opposed to the selection of a constant. The "
            "flag rule tests where a NUMBER came from. A solver rule kept because it "
            "improved a MAMMA-referenced gate passes this audit -- see the meta_level_"
            "caveat on DEFAULT_WEIGHTS.",
        ],
        "verdict": "LEAK" if leaks else "CLEAN",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"manifest {report['counts']['manifest']} items from "
          f"{len(live)} reachable delivery functions")
    print(f"  leaks    {len(leaks)}")
    for entry in leaks:
        print(f"    LEAK      {entry['name']}  {entry['file']}  = {entry['value']}")
    print(f"  declared {len(declared)}")
    for entry in declared:
        print(f"    declared  {entry['name']}  {entry['file']}")
    print(f"  unknown  {len(unknown)}")
    for entry in unknown:
        print(f"    unknown   {entry['name']}  {entry['file']}  = {entry['value']}")
    print(f"  clean    {report['counts']['clean']}")
    print(f"\nwrote {args.out}  ->  {report['verdict']}")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
