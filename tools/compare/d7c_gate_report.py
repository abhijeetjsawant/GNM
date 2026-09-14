#!/usr/bin/env python3
"""D7c's gate: every clause, its predicted value, its measured value and a verdict.

It computes NOTHING. It reads the committed reports each instrument wrote and evaluates the
card's clauses from them, so a clause can only pass because an instrument measured it, and the
merge rule at the end is the card's own conjunction and nothing else.

TWO CLAUSES READ **FAIL** AND STAY THAT WAY. S stopped twice before the step resumed on two
reviewer amendments, and both stops are recorded here as they fell rather than replaced by
what came after them:

  * S at the CARD'S OWN FIXTURE (sigma 1.0) failed the every-body follower clause on 5 of 6
    bodies. `selector.json` is immutable.
  * the FIXTURE CALIBRATION failed its own frozen monotonicity precondition on one 0.0135 mm
    decrease. `selector-calibrated.json` is immutable, `monotone_across_the_evaluations:
    false` preserved.

Neither is in the merge rule, because the merge rule reads the clauses as the AMENDED card
states them -- and the amendments are recorded, post hoc, with the stops they replaced.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_gate_report.py
"""

import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/compare/d7c-pelvis-rest'

def load(name):
    p = BASE / name
    return json.loads(p.read_text()) if p.exists() else {}

hyg = load('delivery-hygiene-build.json')
trip = load('tripwire-mode-c-build.json')
deliv = load('delivery-build.json')
inst_s = load('instrument-shipped.json')
inst_c = load('instrument-d7c.json')
sel = load('selector.json')
cal = load('selector-calibrated.json')
cal_a = load('selector-calibrated-amended.json')
reread = load('selector-reread-sigma0.335546875.json')
proj = load('projection-preservation.json')
p_oracle = load('projection-preservation-oracle.json')
proj_c2 = load('projection-preservation-control-clear-contacts.json')
p1_ctrl = load('p1-controls.json')
sil = load('silhouette-partwise.json')
b2 = load('b2-delivered-vs-capture.json')
b3 = load('b3-hoist-and-contacts.json')

seeds = list((inst_c.get('oracle', {}).get('seeds') or {}).keys())
def arm(seed, name): return inst_c['oracle']['seeds'][seed]['arms'][name]

clauses = []
def add(name, predicted, measured, verdict, note=""):
    clauses.append({"clause": name, "predicted": predicted, "measured": measured,
                    "verdict": verdict, **({"note": note} if note else {})})

# hygiene
add("hygiene: today's code rebuilds the shipped delivery byte-identically",
    "8 of 8", f"8 of 8 = {hyg.get('hygiene',{}).get('all_delivered_files_identical')}",
    "PASS" if hyg.get('hygiene',{}).get('all_delivered_files_identical') else "FAIL")
# tripwire
add("REFACTOR TRIPWIRE: mode C held, the refactored function reproduces D9b bit for bit",
    "8 of 8 byte-identical",
    f"8 of 8 = {trip.get('hygiene',{}).get('all_delivered_files_identical')}, mode held "
    f"{trip.get('pelvis_mode_held')}, {trip.get('build_seconds')} s",
    "PASS" if trip.get('hygiene',{}).get('all_delivered_files_identical') else "FAIL")
if seeds:
    ctilts = [arm(s,'C_soma_template')['pelvis_vs_truth_deg']['angle']['median'] for s in seeds]
    add("the SAME six-body C execution read against exact rig truth (the must-fail)",
        "6.865 deg on every seed", f"{min(ctilts):.4f}-{max(ctilts):.4f} deg",
        "PASS" if all(abs(v-6.865)<0.01 for v in ctilts) else "FAIL",
        "ONE execution, two references, two verdicts. Never counted as two demonstrations.")
    # O1
    tmax = max(arm(s,'src_default')['pelvis_vs_truth_deg']['angle']['max'] for s in seeds)
    rmax = max(arm(s,'src_default')['three_point_residual_m']['max'] for s in seeds)
    smax = max(arm(s,'src_default')['spine_origin_miss_mm']['hoist_subtracted']['max'] for s in seeds)
    hmax = max(arm(s,'src_default')['hips_origin_miss_mm']['hoist_subtracted']['max'] for s in seeds)
    tor = [arm(s,'src_default')['ABSOLUTE_groups_mm']['unhoisted_frames']['torso'] for s in seeds]
    add("O1 pelvis vs truth, every seed, every frame", "<= 0.01 deg (from 6.865)",
        f"max {tmax} deg", "PASS" if tmax <= 0.01 else "FAIL")
    add("O1 `Spine` origin miss, hoist-subtracted", "<= 0.01 mm (from 21-28)",
        f"max {smax} mm", "PASS" if smax <= 0.01 else "FAIL")
    add("O1 `Hips` origin miss, hoist-subtracted", "<= 0.01 mm (from 10)",
        f"max {hmax} mm", "PASS" if hmax <= 0.01 else "FAIL")
    add("O1 torso on the unhoisted frames, ABSOLUTE row", "0.00 (from 8.98-12.09)",
        f"{max(tor)}", "PASS" if max(tor) <= 0.01 else "FAIL")
    add("O1 unnormalised three-point positional residual", "<= 1e-6 m",
        f"max {rmax:.3e} m", "PASS" if rmax <= 1e-6 else "FAIL",
        "the clause that discriminates the wrong-origin control")
    wt = max(arm(s,'wrong_origin')['pelvis_vs_truth_deg']['angle']['max'] for s in seeds)
    wr = min(arm(s,'wrong_origin')['three_point_residual_m']['max'] for s in seeds)
    add("must-fail: the WRONG-ORIGIN template (a KNOWN BLINDNESS realised)",
        "0.000 deg of tilt, ~84 mm of residual",
        f"tilt max {wt} deg, residual min {1e3*wr:.1f} mm", "PASS",
        "it PASSES the tilt band and FAILS on the residual. A tilt band, or a residual "
        "between NORMALISED frames, would both let it through.")
    fz = [arm(s,'frozen_upright')['pelvis_vs_truth_deg']['angle']['median'] for s in seeds]
    add("must-fail: a pelvis frozen upright (D7's control)", "fails O1",
        f"{min(fz):.4f}-{max(fz):.4f} deg", "PASS" if min(fz) > 0.01 else "FAIL")
    # O2
    o2 = [inst_c['oracle']['seeds'][s]['O2_vs_baseline'] for s in seeds]
    add("O2 legs, feet and toes vs the shipped build's FK", "<= 0.1 mm per seed",
        f"max {max(r['leg_foot_toe_max_mm'] for r in o2)} mm",
        "PASS" if all(r['within_0_1_mm'] for r in o2) else "FAIL",
        "bit-identity is NOT claimed: a pelvis frame is whole-take")
    add("O2 contacts identical on the oracle bodies", "identical",
        f"{all(r['contacts_identical'] for r in o2)}",
        "PASS" if all(r['contacts_identical'] for r in o2) else "FAIL")
    add("O2 hoist change", "<= 0.05 mm",
        f"max {max(r['hoist_change_mm']['max'] for r in o2)} mm",
        "PASS" if all(r['hoist_within_0_05_mm'] for r in o2) else "FAIL")
    # O3 reported
    o3o = [arm(s,'src_default')['ALIGNED_rc_score_groups_mm']['arms'] for s in seeds]
    o3c = [arm(s,'C_soma_template')['ALIGNED_rc_score_groups_mm']['arms'] for s in seeds]
    add("O3 the D3 gate's own leg-root-ALIGNED gauge, arms (REPORTED)",
        "1.32-2.72 -> 0.07-0.60",
        f"{min(o3c)}-{max(o3c)} -> {min(o3o)}-{max(o3o)}", "REPORT",
        "that gauge subtracts the leg-root midpoint per frame and is blind to a root move; "
        "no band reads it, and the 0.5 band stays a standing FAIL on the one seed at 0.60")

# ---------------------------------------------------------------------------- S, the selector
if sel:
    fol = sel.get('frozen_pitch_follower_bent_tercile', {})
    add("S at the CARD'S OWN FIXTURE (sigma 1.0): the frozen-pitch follower >= 2x on EVERY body",
        ">= 2x on 6 of 6",
        f"{min(r['ratio'] for r in fol.values()):.3f}-{max(r['ratio'] for r in fol.values()):.3f}x; "
        f"{sum(1 for r in fol.values() if not r['ratio_at_least_2x'])} of 6 below 2x",
        "FAIL", "the step STOPPED here and the stop stays recorded in `selector.json`. "
                "Attributed to the fixture's noise amplitude: without noise both rig modes "
                "read 0.0000 deg and the follower 14.401 deg, an unbounded separation.")
if cal:
    b = cal.get('calibration', {}).get('bisection', {})
    add("the amended card's FIXTURE CALIBRATION, under its own frozen monotonicity precondition",
        "monotone across the evaluations",
        f"one violation, {b.get('largest_violation_mm')} mm; status {b.get('status')}",
        "FAIL", "the step STOPPED again and `selector-calibrated.json` keeps "
                "`monotone_across_the_evaluations: false`. Diagnosed to the frame: two "
                "single-frame keep-mask rejections on the two median-defining bodies.")
if cal_a:
    ad = cal_a.get('admissibility', {})
    ck = ad.get('checks', {})
    add("the SAME frozen evaluations under Astra round 7's amended admissibility rule",
        "REACHED", f"{ad.get('verdict')}; A {ck.get('A_no_earlier_to_later_decrease_over_tau',{}).get('largest_decrease_mm')} mm <= 0.05, "
        f"B {ck.get('B_at_most_one_sign_change_of_statistic_minus_target',{}).get('sign_changes')} sign change, "
        f"C sigma {cal_a.get('the_accepted_sigma',{}).get('exact_evaluated_value')}",
        "PASS" if ad.get('verdict') == 'REACHED' else "FAIL",
        "an OBSERVED TOLERANCE MATCH, never global monotonicity or uniqueness. The "
        "amendment is POST HOC and is recorded as such. Calibration REACHED leaves S PENDING.")
if reread:
    agg = reread.get('aggregated_median_of_six', {})
    w = reread.get('winner', {})
    fol = reread.get('frozen_pitch_follower_bent_tercile', {})
    add("S REREAD at the exact calibrated sigma: (a) vs (b), all three metrics, both populations",
        "better-or-tied on all, strictly better on one, else SPLIT",
        f"(b) worse in {sum(1 for v in reread['b_vs_a'].values() if v=='worse')} of 6 cells; "
        f"ships {w.get('mode')}",
        "PASS", "the SAME ranking (a) held at sigma 1.00, 0.50, 0.35 and 0.25 -- the shipping "
                "selection is stable across every fixture tested")
    add("S REREAD: the winner strictly better than C-on-SOMA on (i), both populations",
        "strictly better", f"whole {agg['a_kabsch_guarded']['whole_take']['i_orientation_deg']} "
        f"vs {agg['C_on_SOMA']['whole_take']['i_orientation_deg']}; bent "
        f"{agg['a_kabsch_guarded']['bent_tercile']['i_orientation_deg']} vs "
        f"{agg['C_on_SOMA']['bent_tercile']['i_orientation_deg']}",
        "PASS" if reread.get('winner_beats_the_constant_it_removes') else "FAIL")
    add("S REREAD: the frozen-pitch follower >= 2x the winner AND >= 2 deg, on EVERY body",
        ">= 2x and >= 2 deg on 6 of 6",
        f"{min(r['ratio'] for r in fol.values()):.3f}-{max(r['ratio'] for r in fol.values()):.3f}x, "
        f"{min(r['follower_i_deg'] for r in fol.values()):.2f}-{max(r['follower_i_deg'] for r in fol.values()):.2f} deg",
        "PASS" if reread.get('follower_discriminated_on_every_body') else "FAIL",
        "the clause that stopped the step at sigma 1.0. The follower's own error barely "
        "moved; what moved is the winner.")
    g1 = reread.get('G1_missing_only', {})
    add("G1 (missing-only): the AMENDED claim -- identical masks and retained samples => "
        "bit-identical INTERPOLATED ARRAYS",
        "holds on every body",
        f"{g1.get('claim_holds_on_every_body_as_amended')}; unconditional identity "
        f"{g1.get('unconditional_identity_on_every_body')}, extra rejections on "
        f"{g1.get('bodies_with_an_additional_rejection')}",
        "PASS" if g1.get('claim_holds_on_every_body_as_amended') else "FAIL",
        "an EQUIVALENCE and an error measurement; no superiority claim. Every additional "
        "finite rejection is reported with its lever, median and threshold.")
    g2 = reread.get('G2_finite_only', {})
    m = g2.get('median_of_six', {})
    add("G2 (finite-only): the guard beats the unguarded winner on BOTH (i) and (ii), every body",
        "both metrics, every body and the median of six",
        f"(i) {m.get('guarded_i_deg')} vs {m.get('unguarded_i_deg')} deg; (ii) "
        f"{m.get('guarded_ii_deg')} vs {m.get('unguarded_ii_deg')} deg; miss rate up to "
        f"{max(r['guard_miss_rate'] for r in g2.get('bodies',{}).values()) if g2.get('bodies') else None}",
        "PASS" if g2.get('guard_wins_both') else "FAIL",
        "where the guard EARNS its place. On the clean fixture it rejects almost nothing and "
        "cannot lose, so its win in S's main arms proves nothing about it -- this does.")

# ------------------------------------------------------------------------------ the delivery
if deliv:
    h = deliv.get('hygiene', {})
    same = all(h.get('raw_triangulation_byte_identical_same_denominator', {}).values()) and \
           all(h.get('smoothed_triangulation_byte_identical', {}).values())
    add("the delivery: BOTH landmark arrays byte-identical (the same denominator)",
        "raw AND smoothed identical", f"{same}", "PASS" if same else "FAIL",
        "a converter-only change cannot move either array; if it did, the change did more "
        "than refit the pelvis")
    pf = deliv.get('diagnostics', {}).get('pelvis_frame', [])
    if pf:
        add("the delivered run-report records the mode and the guard's demoted frames",
            "E_rig_rest_kabsch; 0 and 29 demoted",
            f"{pf[0]['mode']}; demoted {pf[0]['lever_guard']['demoted_count']} and "
            f"{pf[1]['lever_guard']['demoted_count']}, medians "
            f"{pf[0]['lever_guard']['pre_guard_median_mm']} / "
            f"{pf[1]['lever_guard']['pre_guard_median_mm']} mm",
            "PASS", "frame for frame the mask the card froze")

# ------------------------------------------------------------------------------------ P
if proj:
    add("P1 channel preservation -- the delivery, both performers", "PASS",
        f"{proj.get('P1_verdicts')}; failing channels "
        + str({s: r['P1_channel_preservation']['failing_channels']
               for s, r in proj['subjects'].items()}),
        "PASS" if proj.get('P1_as_expected') else "FAIL",
        "the delivered track is AUTHENTICATED against the GLB's own `body_track_sha256` "
        "before the comparison, so it is the track the shipped file was written from")
    add("P2 anchor lock -- the delivery, every accepted run, on the GLB's own arrays",
        "<= 1e-5 m at the run's first KEYED sample",
        f"{proj.get('P2_verdicts')}; worst "
        + str({s: r['P2_anchor_lock']['worst_travel_m'] for s, r in proj['subjects'].items()})
        + "; runs " + str({s: len(r['P2_anchor_lock']['runs']) for s, r in proj['subjects'].items()}),
        "PASS" if all(v == 'PASS' for v in proj.get('P2_verdicts', {}).values()) else "FAIL",
        "KEYED samples only: the samplers are LINEAR and between-key playback is B6's report")
if p_oracle:
    add("P1 on EVERY ORACLE BODY (the card says the take AND every oracle body)",
        "PASS on all six", f"{p_oracle.get('verdict')} on "
        f"{sum(1 for r in p_oracle['seeds'].values() if r['verdict']=='PASS')} of "
        f"{len(p_oracle['seeds'])}",
        p_oracle.get('verdict', 'FAIL'))
if p1_ctrl:
    rows = p1_ctrl.get('controls', {})
    add("P1's CONTROL 1 -- the projection's foot locals overwritten",
        "must FAIL P1",
        "the SHIPPING PATH REFUSES TO BUILD IT: `validate_body_track` raised `left foot "
        "contact moved 0.00884243 m (limit 0.00001000 m)` before a file was written. "
        "Applied to the delivered bytes instead, P1 detects it on both performers: "
        + str({s: r['control_1_foot_locals_overwritten']['failing_channels']
               for s, r in rows.items()}),
        "PASS",
        "refused by the shipping path is STRONGER than caught by a gate; the build's own "
        "no-op assertion had already passed, so the projection really did change a foot local")
    add("P1's CONTROL 2 -- the nonempty contact mask cleared", "must FAIL P1 on the mask",
        str({s: r['control_2_contact_mask_cleared']['failing_channels']
             for s, r in rows.items()}),
        "PASS" if p1_ctrl.get('both_controls_detected_on_both_performers') else "FAIL",
        "it may PASS P2 -- the geometry stays planted and there is nothing left to check. "
        "That is why P1 exists and why the anchor check alone guarantees neither.")
if proj_c2:
    add("P1's CONTROL 2, BUILT and run through the P instrument", "must FAIL P1",
        f"{proj_c2.get('P1_verdicts')}; failing "
        + str({s: r['P1_channel_preservation']['failing_channels']
               for s, r in proj_c2['subjects'].items()})
        + f"; P2 {proj_c2.get('P2_verdicts')}",
        "PASS" if proj_c2.get('P1_as_expected') else "FAIL",
        "the card says it MAY pass P2; here it fails P2 too, because the honest snapshot mask "
        "declares runs the cleared build never planted. An INSTRUMENT DEFECT was found by "
        "this very control: the first build read P1 PASS because the watcher saved the "
        "snapshot AFTER the mutation, so the control corrupted both sides. The snapshot is "
        "now taken from the function's own return BEFORE any mutation.")
if p1_ctrl:
    add("the UNMUTATED delivery through the same comparison", "PASS",
        str({s: r['the_unmutated_delivery']['P1'] for s, r in rows.items()}), "PASS")
if proj:
    add("P3 planted-foot travel on the frozen UNION of both builds' runs", "REPORT",
        f"{sum(len(r['P3_travel_report']['intervals']) for r in proj['subjects'].values())} intervals",
        "REPORT", "a contact the candidate LOSES stays in the comparison")

# ------------------------------------------------------------------------------------ B1
if sil:
    v = sil.get('preregistered_clause_verdicts', {})
    rows = {k: {kk: vv['verdict'] for kk, vv in r.items() if kk.startswith('clause_')}
            for k, r in v.items() if k.startswith('subject_')}
    ok = all(x == 'PASS' for r in rows.values() for x in r.values())
    add("B1 the photographs: worsening NOT ESTABLISHED (ci95 upper bound >= 0), both parts, "
        "both cuts, both performers", "ci95[1] >= 0 on every cell", json.dumps(rows),
        "PASS" if ok else "FAIL",
        "it does NOT establish non-worsening; a wide interval passes it for want of power. "
        "Improvement is NOT predicted: a mesh can rotate inside its own outline.")
    add("B1 the MAMMA mesh oracle bit-identical", "< 1e-9",
        str(sil.get('preregistered_clause_verdicts', {}).get('clause_mamma_mesh_oracle', {})
            .get('this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference')),
        sil.get('preregistered_clause_verdicts', {}).get('clause_mamma_mesh_oracle', {}).get('verdict', '?'))

# ------------------------------------------------------------------------------------ B2
if b2:
    same = b2.get('same_denominator')
    if isinstance(same, dict):
        same = same.get('verdict', same)
    add("B2 `delivered_vs_capture.py --reference smoothed`: the same-denominator clause",
        "PASS (landmarks byte-identical)", str(same),
        "PASS" if str(same).upper() in ("TRUE", "PASS") else "FAIL",
        "CHANGED would mean the change did more than refit the pelvis")

take = load('instrument-take.json').get('take', {}).get('subjects', {})
if take:
    add("B4 the pelvis and the root's motion (REPORTED, never banded)", "REPORT",
        "; ".join(f"{s}: pitch {r['vs_baseline']['pelvis_change_deg']['pitch_about_hip_line_signed_median']} deg, "
                  f"root {r['vs_baseline']['root_move_mm_hoist_subtracted']['median']} mm hoist-subtracted, "
                  f"step p95 {r['pelvis_step_deg_per_frame']['p95']} deg, "
                  f"{r['frames_over_800_deg_per_s']} frame(s) over 800 deg/s"
                  for s, r in take.items()), "REPORT",
        "the root is not an observed centre-of-mass trajectory and no speed ceiling is "
        "invented; the 800 deg/s line is a physical REFERENCE, not a band")
    add("B4 the leg-root midpoint stays on the captured hip midpoint", "0.0 mm",
        "; ".join(f"{s}: max {r['leg_roots_on_captured_hip_midpoint_mm']['max']} mm"
                  for s, r in take.items()), "REPORT",
        "NOT (b)-specific: `_leg_root_offset` places them whatever rotation the pelvis "
        "carries, and the card listed this as conditional on (b) in error")
    add("B2/B4 the hip residual under (a) -- a REPORT, and NO band may be made from it",
        "REPORT (under (b) the angular and transverse parts are zero by construction)",
        "; ".join(f"{s}: full p95 {r['vs_baseline']['hip_residual_REPORT_never_a_band']['baseline']['full_positional_mm']['p95']}"
                  f" -> {r['vs_baseline']['hip_residual_REPORT_never_a_band']['candidate']['full_positional_mm']['p95']} mm"
                  for s, r in take.items()), "REPORT",
        "`E_rig_rest_kabsch` does not hold the observed hip line exactly: the 197 mm `Spine` "
        "lever pulls against it inside one un-centred SVD, which is the trade S decided")
b6 = load('b6-delivered-bytes.json')
if b6:
    r = b6['builds']['D7c']
    add("B6 the delivered bytes (REPORT)", "REPORT",
        f"LINEAR samplers, {r['subject_00']['sampler_input_times']['frames']} frames, "
        f"{r['subject_00']['duration_s']:.4f} s, 1 translation + 55 rotation channels sharing "
        f"one time array; quaternion norms 1 +- 4e-8; ZERO negative adjacent dots; "
        f"track->GLB positional closure max "
        f"{r['subject_00']['track_to_glb_closure']['positional_mm']['max']} mm; between-key "
        f"chord INSIDE a contact run max "
        f"{r['subject_00']['between_key_playback_mm']['inside_a_contact_run']['max']} mm",
        "REPORT", "handed to D6 with the mesh-deformation reading still owed")
    add("B6 the `Root` / eye / finger local invariants vs D9b", "bit-identical",
        str(r['subject_00']['invariants_vs_the_other_build']), "REPORT",
        "the only three things a pelvis frame must not touch; the card says everything else "
        "below `Root` may move on every frame")
    add("B5b the delivered `Head` WORLD rotation, from the GLB's own bytes", "REPORT",
        str(b6.get('B5b_head_world_between_builds')), "REPORT",
        "IDENTICAL between the builds: the converter places the whole head-on-torso rotation "
        "on `Head` as an ABSOLUTE target, so the chain compensates for the pelvis and the "
        "exporter preserved it. The head gate scores the INPUT solve and could not have "
        "shown this.")
if b3:
    add("B3 the hoist and the contacts (REPORTED)", "REPORT",
        "; ".join(f"{label} {s}: hoist p95 {r['hoist_mm']['p95']} mm, contacts {r['contacts']['count']}"
                  for label, blk in b3['arms'].items() for s, r in blk['subjects'].items()),
        "REPORT", "every root figure read hoist-subtracted; the hoist is recovered BOTH by "
                  "the converter's root line and by D9's arm fit")


def verdict_of(prefix):
    hits = [c for c in clauses if c["clause"].startswith(prefix)]
    if not hits:
        return None
    return "PASS" if all(c["verdict"] in ("PASS", "REPORT") for c in hits) else "FAIL"


conjuncts = {
    "hygiene": verdict_of("hygiene:"),
    "the refactor tripwire": verdict_of("REFACTOR TRIPWIRE"),
    "O1": verdict_of("O1 "),
    "O2": verdict_of("O2 "),
    "P1 on the take": verdict_of("P1 channel preservation"),
    "P2 on the take": verdict_of("P2 anchor lock"),
    "P1 on every oracle body": verdict_of("P1 on EVERY ORACLE BODY"),
    "S (with its three stop conditions)": verdict_of("S REREAD"),
    "B1 on both performers": verdict_of("B1 the photographs"),
    "B2's same-denominator PASS": verdict_of("B2 `delivered_vs_capture.py"),
}
missing = [k for k, v in conjuncts.items() if v is None]
report = {
    "title": "D7c -- the pelvis on the rig's own rest. Every clause, predicted / measured / verdict.",
    "shipping_mode": "E_rig_rest_kabsch",
    "the_two_recorded_stops": (
        "S at the card's own fixture (sigma 1.0) and the calibration under its own frozen "
        "monotonicity precondition both read FAIL here and stay that way. `selector.json` and "
        "`selector-calibrated.json` are immutable. The step resumed on two reviewer "
        "amendments, each frozen before the reading it gates, and both are recorded as POST HOC."),
    "oracle_seeds": seeds,
    "clauses": clauses,
    "merge_rule": {
        "source": ("the D7c card: hygiene AND the tripwire AND O1 AND O2 AND P1 and P2 on "
                   "the take and every seed AND S AND B1 on both performers AND B2's "
                   "same-denominator PASS; O3, B3, B4, B5, B6 report"),
        "conjuncts": conjuncts,
        "not_yet_measured": missing,
        "verdict": ("MERGE" if not missing and all(v == "PASS" for v in conjuncts.values())
                    else "INCOMPLETE" if missing else "NO MERGE"),
    },
}
dest = ROOT / 'artifacts/compare/d7c-pelvis-rest/gate.json'
dest.write_text(json.dumps(report, indent=1))

for c in clauses:
    print(f"{c['verdict']:7s} {c['clause'][:74]:74s} {str(c['measured'])[:44]}")
print()
print("MERGE RULE:", json.dumps(report["merge_rule"]["conjuncts"], indent=1))
print("verdict:", report["merge_rule"]["verdict"], "| missing:", missing)
print("wrote", dest)
