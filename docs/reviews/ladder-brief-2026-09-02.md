# Brief: how to tackle each rung of the substitution ladder

Context. AutoAnim is building a commercially clean, in-house markerless multi-camera body
capture. MAMMA (MPI, research-licensed, never ships) is the benchmark to reach and where
possible pass. It is also the only reference we have; there is no ground truth anywhere in
the project. The lane's standing rules: replace ONE part at a time with MAMMA's retained
output supplying the rest; no gate a constant can pass (every band ships with a degenerate
solution that fails it) and no gate no oracle can pass; same denominator; numbers from
different references never share an axis; a figure counts only if a script writes it as a
JSON report under artifacts/. Nothing MAMMA or SMPL-X produces may enter a delivered
artifact, trained weights or a shipped constant. Body model for delivery is Meta MHR
(Apache-2.0) with momentum (MIT) as the solver candidate.

Fixture: one take, 4 cameras at 3840x2160, 150 frames, two performers in contact.
MAMMA's retained run is on disk: ma_cap calibration, ma_masks, ma_2d (512 dense landmarks
+ visibility + contact per subject), ma_3d (triangulated_3d_pts, smplx_betas[16],
smplx_pose[165], pred_joints[127], pred_vertices[10475]).

## Where the ladder stands today (all figures from on-disk reports, 2026-09-02)

| rung | MAMMA | ours | status | headline (reference beside it) |
|---|---|---|---|---|
| 0 calibration | ma_cap | none; we use MAMMA's rig | not owned | - |
| 1 masks | YOLO+SAM2 tracklets, cross-view id | none; identity done later on keypoints | not built | - |
| 2 2D landmarks | MammaNet 512 dense | SOMA-77, 77 joints @1280w, 17 body + 5 head + toes mapped | measured, NO control arm | 2.91 px p50 self-reprojection (own triangulation). Cross-detector figures (2.2x bulk, ~18.6 px coherent bias) exist only in docs |
| 3 association | mvpose-style temporal re-ID inside ma_3d | associate_frame_graph | measured | 0/300 switches on MAMMA-labelled 2D, holds at +-1 frame; margin p05 in contact +204 px; 38 contact frames |
| 4 triangulation | triangulate_batch, 512 pts | triangulate_point | at instrument floor | 11.1 / 9.2 mm vs MAMMA's exact 512 landmarks at full coverage |
| 5 temporal | inside the fit: angular_acc + pts3d_temp losses in runs 3-4 | solve_sequence_positions + fill + Savitzky-Golay | dark, both references banned | exposure counts only |
| 6 shape | 16 betas free in run 2 | none in delivery (one fixed rig, 540 mm shoulders). Instrument: SMPL-X 10-beta fit to our own limb lengths | measured (instrument), absent (delivery) | 19.3 / 14.1 mm mean abs limb error vs rig's 39.5 / 39.1; control: head 724/671 mm above pelvis |
| 7 pose | 4 Adam runs, reprojection of 512 + L2 to triangulated pts | positions_to_body_track (analytic, no optimiser). Instrument: least-squares SMPL-X fit to our 15 joints | measured | raw capture 36.1 / 41.3 mm; capture->SMPL-X 27.7 / 40.1 mm vs MAMMA pred_joints (15 joints); control 14.7 / 11.4 vs our own capture |
| 8 hands | hand pose in same fit | delivered: rest-curl constant; prototype MHR hand-chain fit | prototype, doc-only | 35 mm held-out reprojection; 43-88 mm fingertips vs MAMMA; open thrashing defect |
| 9 head | locked-head SMPL-X | rigid 5-landmark fit, neck-anchored | measured | 6.39 / 8.25 deg P1 median vs MAMMA head in our thorax frame; oracle 3.62 / 3.87; constant 15.4 / 16.2 fails |
| 10 feet | contact channels + intersection loss | Toes constant, foot welded to torso | measured vs ourselves only | MAMMA's feet never scored |
| 11 delivered | pred_vertices/pred_joints | rig retarget + MPFB mesh (+180 deg facing defect open) | measured | 151.6 / 137.8 mm as delivered; 136.6 / 113.0 sized; same reference as rung 7 |

Reading: the front end reaches the reference; the delivery rig throws ~100 mm away; the
missing part is a body model in the delivery path. Hands, feet, head each have their own
axis and gap. There is NO surface/mesh instrument at all (delivered rung scores 15 joints).

## Draft plan, per rung, in proposed execution order

Roles: Fable (me) plans, reviews every gate, supervises. Sol / Sonnet do research where a
rung needs it before code. Opus agents execute. Every step lands as: instrument first
(writes JSON, registered in tools/compare/ladder.py with control + oracle arms), then the
build, then the ladder rerun and board update in the same pass.

### Step A. Shape into the delivery path (rung 6) — the largest measured loss
- Build: MHR + momentum per-performer body fit from our own limb lengths (the
  tools/fitter prototype), wired into the delivery path so the retarget targets a body the
  performer's size. Not SMPL-X (licence).
- Instrument: rung 7's scoreboard "sized" arm on the same 15 joints vs MAMMA pred_joints,
  which shares a reference with the SMPL-X instrument arm (27.7 / 40.1). Band: the MHR-driven
  arm must land within X mm of the SMPL-X instrument arm on both performers (X to be set; I
  propose 10 mm) and beat today's sized rig (136.6 / 113.0) by more than the bootstrap CI.
  Degenerate control: the canonical rig round-trip (must still be 0.00 on canonical input)
  and the unfitted MHR mean body (must be worse than the fitted one). Oracle: SMPL-X fit
  itself (research, instrument-only).
- Blindness: surface; hands/feet/head; accuracy.
- Research first (Sol/Sonnet): momentum's identity-parameter regulariser for 68 raw scale
  channels without SOMA's PCA (licence); bone-length vs locator-offset non-identifiability
  (FITTER_PLAN §7) — what to pin.
- Risk: the 180 deg facing defect and the clavicle-origin converter defect (36-47 mm on
  arms) sit in the same stage; fix order matters for attribution.

### Step B. Pose rung other direction (rung 7) — MAMMA's fitted joints into our converter
- Build: pred_joints mapped via scoreboard PAIRS into positions_to_body_track; isolates the
  retarget on MAMMA's input. Cheap; from disk. Instrument-only.
- Band: none needed; it is a pricing rung. Reports converter cost with clean input.

### Step C. A discriminating detector instrument (rung 2)
- Build: JSON output for mamma_residuals.py and sam3d_ladder.py; register bulk ratio,
  coherent bias, tail; add a control (shuffled camera-offset already exists in common_mode).
- Then the pseudo-label campaign decision (train on our own triangulated labels) is made
  on tracked figures. Licence gates (NVIDIA model licence, performer releases) are open.

### Step D. Score MAMMA's feet (rung 10) — the bar first
- Build: ankle/toe joints from pred_joints through subject_map; foot axis + toe angle
  distributions per performer; then our ToeBase solve scored against it with two controls
  (constant axis, today's delivered foot).

### Step E. Hands as a report (rung 8)
- Build: the held-out-camera protocol in hand_fit writes JSON; register 35 mm and the
  thrashing figure (angle sd per DoF vs MAMMA's 5.6 deg); fix the temporal-term unit
  mismatch (radians vs pixels) as the first candidate; band: held-out must not worsen while
  sd drops toward MAMMA's.

### Step F. Our detector into MAMMA's fitter (rung 2, reverse direction) — decide after C
- Needs a mapping of our 17 joints onto SMPL-X joint targets and a Modal run of
  run_ma_3d.py with a modified loss. Instrument-only. Expensive; only if C says the gap is
  bulk not bias.

### Step G. Calibration: the owned multicam fixture (rung 0) — rank 0 in BODY_LANE_PLAN
- Not engineering this lane can finish alone; hardware. Retires "instrument-only" on
  every rung below.

Parked deliberately: masks (nothing consumes them), temporal (no valid reference until
single-ray fixtures or markers).

## Questions for the reviewer

1. Rank the steps. Where does my order go wrong, and why?
2. The strongest case AGAINST Step A being next.
3. For each step: is the instrument right, what is it blind to, is the band pre-registrable,
   and what degenerate solution must it reject that I have not named?
4. What is missing from the decomposition — in particular a surface/mesh instrument
   (the user wants "skeleton with mesh"), and whether temporal (rung 5) is truly unmeasurable.
5. What should be researched before any code is written, per step, and what can run in
   parallel vs must be sequential for attribution to hold.
6. Anything here that would fit a shipped constant on a MAMMA-derived artifact by accident.
