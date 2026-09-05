# Ladder execution plan — how each rung gets tackled, and by whom

**Current state lives in `docs/LADDER_STATUS.md`** (generated from `docs/ladder-status.json`
by `tools/compare/status.py`; never hand-edit it). This document is hand-edited and holds the plan.

**Snapshot: 2026-09-02.** Companion to `docs/SUBSTITUTION_LADDER.md` (what is measured
and how) — this is *what gets built, in what order, gated by what*. It supersedes
§4 of the ladder doc. Read §1 (verdicts) before the steps: three of the draft's
premises were wrong and the order changed because of them.

**How it was made.** Fable drafted a per-rung plan and put the *same* brief and six
questions to two reviewers who did not see each other's answers: **Codex** (GPT, xhigh
reasoning, repository access, 1.7 M tokens) and **Sol** (the advisor). Three Sonnet
research passes ran in parallel on the parts both reviewers were likely to need
(MHR/momentum shape parameters, the perfect-2D oracle injection path, the silhouette
instrument). Where the reviewers disagreed, the choice is noted inline and the dissent
is kept in §5 rather than smoothed away. Every code-level claim a reviewer made was
checked against the source before it entered this plan.

---

## 1. Verdicts on the draft — what changed and why

**The draft's Step A ("shape into the delivery path", MHR fit, next) is demoted.** Both
reviewers, independently, for compounding reasons:

- *It is not one part.* Sol: rung 2's 27.7 / 40.1 mm changed the body model **and** the
  pose recovery (analytic converter → least-squares fit); the ladder cannot yet split
  them. Codex: Step A conflates six components — landmark-to-joint convention,
  per-performer skeletal scale, per-frame pose, rest-skeleton serialisation and
  propagation, mesh binding, and surface identity (which 19 joints cannot observe at
  all: MHR's 45 identity coefficients move vertices up to 9.65 cm and joint centres by
  **0.000000 mm**, verified by the fitter prototype).
- *The sized arm already on disk says proportions are the smaller lever.* Canonical →
  sized moves 151.6 → 136.6 and 137.8 → 113.0 mm: 15–25 of the ~100. The retarget
  decomposition in `BODY_LANE_PLAN.md` says the same — performer bone lengths recover
  19–35 % of the retarget gap; the converter (clavicle-origin defect, 36–47 mm on the
  arms; "no rig joint origin sits where its landmark does") is the rest. Building a
  shape fitter through today's converter lands near 136 / 113, not near 28 / 40.
- *The band was not preregisterable.* "Within 10 mm of the SMPL-X arm" scores
  convention convergence: SMPL-X shares its joint regressor with MAMMA's `pred_joints`;
  MHR does not (its hand root sits 2.5–3 cm from SMPL-X's wrist). The canonical
  round-trip control was misstated as 0.00 — it is 36–47 mm on the arms today. And the
  SMPL-X arm beats raw capture by 8.4 mm on performer 0 but only 1.2 mm on performer 1
  (40.1 vs 41.3), inside any block-bootstrap interval on 150 correlated frames — so a
  "must beat raw capture" band is one the oracle itself does not robustly pass.
- *Two delivery defects sit in the same stage* (the 180° facing yaw and the clavicle
  origin), so any body-model result measured before they are split can be
  compensation.

**Accepted from Codex, verified in source:**
- **A shipped constant was selected on a MAMMA-referenced arm.** `THORAX_SMOOTHING_FRAMES
  = 15` in `src/autoanim_gnm/commercial_multiview.py:1679` carries the comment "chosen
  by the oracle arm alone", and its caller is the shipped head solve. That is a
  MAMMA-derived shipped constant. It is not grandfathered: step I8 re-selects it from
  synthetic or owned data, or justifies it from anatomy, before the next delivery build.
- **The feet premise was stale.** Since commit f6a4973 (2026-09-01) the delivered foot
  is solved from SOMA-77's `ToeBase`, not welded to the torso. The ladder's
  `delivered-foot.json` predated that commit; re-scored today the delivered foot
  agrees with the triangulated `ToeBase` direction to 1.6–3.2° — **because it is now
  solved from that very direction**. The instrument scores the solver against its own
  input and is retired as a gate. The bar that remains is MAMMA's feet (I4).
- **The hand band in the draft was invalid.** SMPL-X axis-angle spread and MHR
  Euler-channel spread are incomparable (`BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md`
  §"metric"). The repo already selected parameterisation-invariant metrics: held-out
  camera reprojection and wrist-local fingertip amplitude / jitter. Use those (I5).
- **The 2.2× detector figure is not same-denominator** (both reviewers): ours is
  scored against raw triangulation, MAMMA's against a body-regularised fit that
  consumed the landmarks. Retire it as a headline; register the reference-free
  figures (I3).
- **Step F is not a one-part substitution** as drafted; it needs a matched control —
  MAMMA's own 2D reduced to the identical sparse targets through the identical
  modified loss — or the sparsity and the loss change get charged to our detector.
- **"The front end reaches the reference" is too broad.** Triangulation is pinned
  given MAMMA's 2D; association is pinned given a perfect, count-correct detector;
  our own detector still has no discriminating report. Say "geometry", not "front end".

**Accepted from Sol, mapped by Sonnet:**
- **A whole-pipeline oracle on perfect 2D.** Project `pred_joints` into the four
  cameras and run our entire pipeline on it. `scripts/build_synthetic_truth_fixture.py`'s
  `observations_for` + `gate_g1` is already this design for an FK skeleton; swap the
  joint map for the scoreboard's `PAIRS` and feed `pred_joints`. Second arm: inject
  MAMMA's own measured 2D residual distribution as noise, ≥5 seeds. (I2)
- **The surface instrument exists on disk and nobody consumes it.** MAMMA's SAM2 masks
  (`ma_masks`, 3840×2160 binary PNGs, 150 frames × 2 persons × 4 cameras) are the one
  reference that is *not* model-mediated. Silhouette precision / recall / IoU of our
  rendered mesh, with MAMMA's mesh through the same rasteriser as the oracle and a
  dilated blob as the degenerate control. Needs a third id resolution first (mask
  tracklet id ↔ `body_id` ↔ our subject are three id spaces). (I6)
- **Temporal is not unmeasurable.** Held-out-camera reprojection of the smoothed 3D
  scores transverse lag (Sol); synthetic exact trajectories with injected single-ray
  slots score recovery and smoothing separately (Codex). Neither scores depth. (I7)

**Where the reviewers split, and the choice** — full register in §5:
- *Fixture rank.* Codex: rank 1, start now. Sol: last in the engineering list, because
  it is not engineering. **Choice:** it is a parallel lane owned by the user (lane H),
  started now; it gates accuracy claims and nothing in lanes I or D waits for it.
- *What comes right after the instruments.* Sol: MHR mean body + momentum IK as the
  pose-solver rung, then shape. Codex: facing → converter → skeleton schema first, and
  only then any MHR work, because a fitted skeleton that is not propagated "silently
  serialises canonical names and downstream consumers reconstruct the canonical body".
  **Choice:** Codex's prerequisites (D1–D3) then Sol's solver step (D4) then scaling
  (D5). One part per step; each with its instrument frozen first.
- *Naming.* Codex: "A is skeletal scaling, not a body/mesh fit." **Accepted**; D5 is
  "skeletal scaling with pinned locator offsets, interim until markers".

---

## 2. The plan — three lanes

**Lane H — hardware, owned by the user, starts now, runs in parallel.** The owned
multicam fixture and marker/scan protocol (`BODY_LANE_PLAN.md` §0, rank 0 since
2026-08-30). Codex's validation additions, adopted: held-out calibration views, a
moving-wand reconstruction as the independent check, drift measurement, distortion
residual by image region, handedness and metric-scale checks; and performer releases
covering ML training use. Preregister its bands from the motion/error budget, never
from MAMMA. It retires "not owned" on rung 0 and the instrument-only status of every
configuration below it; **it does not make any comparison containing MAMMA output
non-instrumental.**

**Lane I — instruments, read-only against retained files, parallel.** Each is one Opus
agent in its own worktree, writing one JSON report and one extractor; Fable owns the
`ladder.py` registry (Codex: one registry owner, or the registrations collide). Every
report ships with its control and, where one exists, its oracle arm, or the ladder
flags it.

| step | instrument | reference | must reject | oracle | research done |
|---|---|---|---|---|---|
| **I1 retarget split** | `retarget_cost.py` writes JSON; add the *re-solved* sized arm (`positions_to_body_track` on the sized skeleton, not FK of canonical rotations — the scoreboard's sized arm is a lower bound); add **B**: MAMMA `pred_joints` through `PAIRS` into the converter (15→19 adapter: the 4 unmapped joints NaN, verified the function tolerates it) | our capture (root-relative) and MAMMA `pred_joints` (15 joints) — reported separately, never one axis | wrong joint permutation; L/R swap; input positions copied without valid rotations; root alignment hiding facing | canonical round-trip: 0.00 on the legs, the arms' 36–47 mm is the *known* converter cost and must reproduce | Sonnet: `positions_to_body_track` :1493 rest offsets, :1526 Hips literal, :1565/1568 clavicle origin |
| **I2 perfect-2D oracle** | `gate_g1` pattern fed by `pred_joints`; whole pipeline; score 15 joints; log `ReconstructionDiagnostics` per arm | MAMMA `pred_joints` | — (it *is* the oracle arm for rungs 2–7) | second arm: MAMMA's measured 2D residual as noise (`mamma_residuals.py` deciles), ≥5 seeds | Sonnet: observation schema, projection convention (Z-up, `camera.project`, no flip), head/toe kwargs optional |
| **I3 detector reports** | three separate reports: cross-view self-agreement (`sam3d_ladder.py`), per-camera static common-mode with split-half fit/score (`common_mode.py`), residual spread vs MAMMA projections; retire 2.2× | reference-free; MAMMA projections (deflated, say so) | offsets fitted and scored on the same frames; frozen-skeleton projections; joint-semantic mismatch | — | none needed; define the decision rule for the next detector experiment *before* writing JSON |
| **I4 feet bar** | MAMMA's ankle (7/8) and foot (10/11) joints from `pred_joints` (ankle→foot 140 mm, verified) → foot direction in the **shin frame**, frame-paired against ours on the same valid frames; separate foot orientation, toe articulation, contact | MAMMA `pred_joints` (parity, never truth) | constant axis; foot welded to shin; **time-shuffled MAMMA** (matches the distribution, not the tracking); L/R swap; mirrored forward axis | — | index semantics verified; angle periodicity and local frame definition to be written down before code |
| **I5 hands report** | `hand_fit.py` held-out-camera protocol writes JSON; wrist-local fingertip amplitude / jitter / roughness and translation-only jitter; the thrash figure by the gauge-invariant metric | held-out camera; MAMMA fingertips (agreement) | frozen rest hand (sd 0); strong low-pass / lag; duplicated frames; prior-dominated solver that never leaves initialisation | — | the temporal-term unit mismatch (radians vs pixels, 1.9 % of objective) is the first candidate fix; change one prior/initialisation rule per run; **weight selection defined without MAMMA before coding** |
| **I6 surface** | silhouette precision / recall / IoU per camera per frame per subject vs SAM2 masks, rasterised with `cv2.fillConvexPoly` (no z-buffer needed; pyrender absent) at a documented resolution; plus symmetric point-to-surface distance vs `pred_vertices` as an *agreement* figure only | SAM2 masks (not model-mediated); `pred_vertices` (agreement) | dilated blob (precision must fall as recall rises); mean-body MHR; frozen pose; camera-facing billboard; shuffled subject mesh | MAMMA's mesh through the identical rasteriser against the same masks | Sonnet: mask format, 300 PNGs/camera, tracklet id stable within camera; `camera_overlay.py` renders the real GLB through the real rig; three id spaces must be resolved by pelvis agreement + same-frame 2D, never by IoU itself |
| **I7 temporal** | 5a recovery: synthetic exact trajectories with injected single-ray slots and correlated outages; 5b smoothing: spikes, gaps, fast events → 3D error, phase lag, peak attenuation, recovery latency; plus held-out-camera reprojection lag on real footage | synthetic truth; held-out camera | identity / no smoothing; frozen trajectory; over-smoothed; time-shifted truth | exact-truth oracle | the current synthetic fixture never enters the sequence solve (`BODY_LANE_PLAN.md` ~1334) — fixture must be extended first |
| **I8 leak audit** | provenance manifest for every runtime constant, weight, calibration and training example; reject any delivery build whose manifest names `artifacts/mamma`, `ma_cap`, SMPL-X outputs or reports computed from them; re-select `THORAX_SMOOTHING_FRAMES` | — | — | — | Codex's list in §4 |

**Lane D — delivery, strictly sequential, one part per step.** Each step: instrument
frozen and reported (lane I) → change → ladder rerun → history line → board card.

| step | change | rung | gate (instrument · band · degenerate) |
|---|---|---|---|
| **D1 facing** | fix the rig's mirrored left/right naming — a relabel (negate rest X, correct `DEFAULT_MPFB_JOINT_MAP`, permute the asset's bone names, flip `_frame_alignment`'s secondary, restate `CANONICAL_HEAD_AXES`, drop `soma_motion`'s compensating `_DELTA_MAPPING`), never a vertex reflection (winding and normals invert and no joint gate can see it). Five sites, located 2026-09-02 | 11 | **gate as corrected after the fix was built (2026-09-02):** forward-dot against the footage (repaired `camera_overlay.py`, and `facing_location.py`) on every camera: Hips, chest, Neck median > +0.9 and p05 > 0 against both references; **Head and the mesh nose within the block-bootstrap CI of MAMMA's own head through the same frames (+0.84)** — the draft's +0.9 on the head was a band no oracle passes, and "≥ oracle" is a band a welded head passes, so neither is used · handedness triple product agrees with capture and MAMMA read both ways (from the feet and from the torso) · feet unmoved · I1 round trip, rungs 7/9/11 unchanged to four decimals · the pre-repair build, a sagittal mirror and 180°/90° yaws must fail · **I6 IoU is NOT a band here**: with the limbs held still the silhouette cannot see facing on this take (5 of 8 cells moved within their spread, signs mixed) — the fix is the corrected control I6 could not run · before merge: hands rendered before/after with a computable curl-direction check, the asset regenerated through Blender/MPFB with the corrected joint map and its request hash matching, one handedness check on a real `soma_motion` export |
| **D2 converter** | clavicle direction about the rig shoulder origin (:1565/1568); Hips rest height from the skeleton, not the 0.98 literal (:1526); ground projection root-normalised or bypassed in the instrument | 7 | I1 canonical round-trip on the arms drops from 36–47 mm toward the legs' 0.00 · I1-B (MAMMA joints in) drops in step · legs must not regress (the collarbone attempt of 2026-08-31 broke them 46–67 mm and was reverted) |
| **D3 skeleton schema** | per-performer rest skeleton serialised and propagated: `positions_to_body_track` reads `rest` from it, FK, skin matrices, projection, export, sockets and validation consume it | 6 | synthetic exact-skeleton oracle: an MHR subject with independently perturbed scale channels round-trips through export and FK to the same joints · a canonical body must still round-trip 0.00 · a fitted skeleton that arrives canonical downstream fails |
| **D4 pose solver** | MHR **mean** body + momentum IK on our triangulated joints; shape held at mean; one part changed: the solver | 7 | scoreboard arm vs MAMMA `pred_joints`, **bias and spread reported separately** (BATTLE1_BODY_PROFILE method), band on spread, against the SMPL-X arm on identical frames with the moving-block bootstrap (lag-1 autocorrelation 0.99) · vs-our-own-capture control between 0 and the raw distance · constant bone lengths required · an "IK" that places joints on the points scores 36.1 / 41.3 and must not pass |
| **D5 skeletal scaling** | fitted MHR scale channels (68 raw momentum channels, project-owned regulariser, **never** the SAM-licensed 28-dim PCA) with locator offsets pinned to the detector's *documented* convention offsets — interim until markers make offset and length separately identifiable | 6 | **primary: exact synthetic scale recovery** on MHR subjects (truth exists) · secondary: MAMMA agreement, reported, never selected on · reject global-scale-only, shoulder-only, per-frame scale, offsets absorbing proportions (report fitted segment lengths, not only residuals), mean-body control must score worse |
| **D6 mesh binding** | re-skin the delivered mesh to the fitted skeleton | 11 | I6 silhouette vs SAM2 with the MAMMA-mesh oracle · skinning correctness: finite vertices, bind-pose preservation, render/FK consistency · point-to-surface vs `pred_vertices` as agreement only |
| **D7 pelvis frame** | give `Hips` its own frame from the pelvis's own landmarks, so the trunk is no longer one rigid block and the root offset no longer rides the lean (D2b isolated the root's silhouette share at −0.022 / −0.013 IoU, tilt-dependent, ~0 upright). SOMA-77 `Hips` (index 0) is **already mapped as `root`** and unused for any frame; `Spine1` (1) and `Spine2` (2) are unmapped. Map them through the toe/head landmark feed (`_toe_landmarks` → `_toe_world_for_subject`, `triangulate_point` under the pipeline's own association); in `positions_to_body_track` the Hips primary axis comes from the pelvis, the secondary stays the hip line, Spine/Chest/UpperChest keep the thorax frame (Spine's local becomes pelvis⁻¹·thorax through `_set_world`), the root formula is unchanged. Candidates, **selected on synthetic truth only**: root→Spine1 (rigid, 43 mm lever), mid(hips)→Spine1 (rigid, 108 mm), a Kabsch fit of the rest pelvis to {root, Spine1, LeftLeg, RightLeg} (the head's lesson: fit, don't triangulate); any constant rest-pitch offset between a candidate and the rig's Hips +Y is a **convention** fixed before any take is measured, never the take (**corrected 2026-09-04:** `somaskel77-v1.json` carries names, parents and a coordinate system and NO rest geometry; the rest lives per clip in `.cache/autoanim_gnm/gem-x/outputs/*/soma_motion.npz` and is per performer — the five full-body clips' rests differ by up to 222 mm and their pelvis source frames by 1.75–4.42° median / up to 10° pairwise, `docs/reviews/pelvis-frame-2026-09-04.md` §0.2 — so the shipped rest-pitch is the component-wise median over those five rests, registered as third-party provenance with its cost stated). root→Spine2 and a line through root/Spine1/Spine2 carry Spine1's flexion (14.5° / 20.9° median / p95 on the squat clip) and are *lumbar* frames, reported as such. A pelvis temporal window like `THORAX_SMOOTHING_FRAMES`, selected on synthetic truth, registered in provenance. **No per-frame definition switching**: gaps interpolated like the body joints, or the whole subject falls back with a diagnostics status. Card written 2026-09-04 (Sol-reviewed) | 7 | **effect size, measured 2026-09-04 on the fixture's own clips:** the rigid pelvis departs from the trunk line by 26° / 33° median / p95 on the squat clip's bent frames (tilt > 20°), correlation +0.93 with tilt; 4–9° on upright clips · **pre-registration error, found by the gate (2026-09-04):** that is pelvis-vs-TRUNK-LINE, which proves today's code wrong; the world-vertical must-fail needed pelvis-vs-VERTICAL, which nobody measured before the band and which reads only 1.1–12.7° on every one of the five clips — so a frozen upright pelvis is near-truth on this motion source and the degenerate was refuted on clean data (5.25° against a ≥ 20° band) · **SYNTHETIC TRUTH (selector and band):** clean arm first per candidate against the posed SOMASKEL77 Hips world rotation, then I7's measured heavy-tail noise; band: orientation error on bent frames below the thorax-as-pelvis control (today's code) *and* below a world-vertical pelvis (the plausible shortcut) *and* below the best lumbar frame; oracle: exact input at rest-geometry precision; an over-smoothed window must fail on the fast clip; frames over 800°/s at Hips reported, never banded · **RIGIDITY on the real take:** root–Spine1, mid(hips)–Spine1 and root–Spine2 length stability over the take against the body controls (the instrument that killed `HeadEnd`), reported before the frame is trusted — cross-view self-agreement is not rigidity · **ROUND TRIP canonical:** legs 0.00, torso 0.00, arms 0.55 / 0.08 unchanged (pre-registered; if the orthogonalised hip line moves the legs, find the line, do not excuse it); the D3 closure band rerun on the rebuilt GLB read from its own bytes · **SILHOUETTE (I6 partwise, the photographs — the one band the candidate cannot optimise):** pre-registered: torso+legs rises on the bent tercile; arms within their CI of D3 (the thorax frame is untouched); the upright tercile within its CI; the MAMMA mesh oracle bit-identical; the rigid-translation arm reported as an *over-attribution*, not an isolation (D7 leaves the leg roots on the captured hips, so translating D3's mesh moves legs D7 does not) · **REPORTED, never banded:** the Hips *joint's* horizontal offset from the captured hip midpoint vs trunk tilt, beside the world-vertical degenerate (D2b's rig-hip-mid instrument is zero by construction since D2b, so this replaces it) — it must move and a constant can zero it; the hoist (predict: moves by no more than the hip-joint shift); rung 11 vs MAMMA; MAMMA's pelvis as an oracle arm (`smplx_pose[:, :3]` pelvis pitch minus spine3, body joint 9, vs ours through `subject_map`, agreement only); facing (Hips forward-dot median > 0.9, p05 > 0, all handedness signs unchanged); the head gate unchanged exactly (it reads `_thorax_frames`, not the converter) |
| **D7b trunk re-solve** | after the pelvis frame, aim the thorax chain from its OWN origin. Measured 2026-09-05 from the delivered GLB's own bytes (numbers reproduced 2026-09-05 by a committed instrument, `tools/compare/delivered_vs_capture.py`, D7b's first deliverable): D7 moved the delivered `Neck` off the captured neck landmark, 14.3 → 58.9 mm median on performer 0 (131 on the bent tercile), 28.6 → 44.0 on performer 1, while Hips went 19/17 → 5/5. Mechanism, confirmed in `positions_to_body_track`: `Hips` takes the pelvis frame, `Spine`/`Chest`/`UpperChest` take `torso_world` aimed along `neck − hip_mid`, and the `Spine` origin now sits 197 mm up the PELVIS axis from the leg-root midpoint (117 mm above `Hips`), so a pelvis pitched δ from the trunk line displaces it 56 / 116 mm (whole take / bent tercile, performer 0) and the straight rigid chain above, aimed from a displaced origin, misses the neck by that. The rule 'after replacing a parent, RE-SOLVE the chains below' was not applied to the spine. Fix, one change inside the loop and gated behind `pelvis_world is not None` so the legacy path stays bit-identical (`tests/test_pelvis_frame.py`): compute `torso_world` AFTER `Root` and `Hips` are set, from `neck − _joint_origin(..., "Spine")` with the shoulder line as secondary; the three rests above `Spine` are collinear +Y so the neck lands on its landmark along the aim and the residual is trunk LENGTH only. The arms follow (D2 aims the clavicle from the FK'd `Shoulder` origin); the feet read `torso_world` on the frames that fall back to it and are reported, not banded. No new constant, no window, no provenance entry. Card written 2026-09-05 (Sol-reviewed) | 7b | **floor, measured 2026-09-05 before the band:** the length residual ‖L_rest − ‖neck − Spine_origin‖‖ under the D7 pelvis is 20.9 / 41.7 mm median whole-take / bent tercile on performer 0 and 18.3 / 11.9 on performer 1 (L_rest 448 / 401 mm) — the trunk chord shortens when the spine flexes and a rigid straight chain cannot follow; that share is handed to **D5** (spine ratios and a flexible chain), with the part a distributed flexion would recover computed and reported here · **DELIVERED vs CAPTURE (new instrument, the figure that saw the defect):** every mapped joint from each delivered GLB's own bytes against the pipeline's own triangulated landmarks in absolute world, D3 archive / D7 / D7b rescored on identical draws (block 15, 2000), per joint median / p95 / bent tercile; `retarget_cost.py` stays as it is and is labelled BLIND to this class (it re-solves with no spine landmark) · **B1 (the placement claim; the candidate optimises it, so it is paired with the floor and the photographs):** Neck-from-file median within 5 mm of the length floor on both performers, whole take and bent tercile, and below D7 with the CI clear (predict 21 / 42 and 18 / 12; against D3 whole-take performer 0 is predicted slightly WORSE, 14 → 21, because D3's chain lay on the trunk line by construction, and better on every bent cut, 105 → 42) · **must-fail:** the shipped D7 (59 / 44) and a root-translation degenerate that zeroes the neck by moving the hips · **B2 (untouchable):** root translation, `Hips`, `UpperLeg`, `LowerLeg` locals bit-identical to D7; hips / knees / ankles from the file unchanged; feet locals reported · **B3:** shoulders / elbows / wrists from the file not worse than D7 with CI, predicted to improve on bent frames · **B4 (the photographs):** part-wise silhouette on the tilt terciles, torso AND arms not worse than D7 on EITHER performer with the CI clear, improvement predicted on performer 0's bent tercile torso; MAMMA mesh bit-identical · **B5:** `Head` WORLD orientation unchanged to 1e-9 on every frame (the locals of `Neck`/`Head` change with `UpperChest`, so the head gate is RERUN and its figures reported, byte-equality not claimed); D3 closure on the rebuilt GLB from bytes ≤ 1e-6 m; canonical round trip legs 0.00, torso and arms reported before / after, not banded (the round trip rebuilds its torso from upper-arm origins) · **B6 reported:** rung 11 vs MAMMA per joint; facing dots; all 16 handedness signs; frames where the torso frame turns > 60°/frame (predict 0) · **B7 synthetic (SOMASKEL77 posed clips, I7 noise):** neck placement error of aim-from-Spine-origin vs aim-from-hip-mid under the Kabsch pelvis; clean arm must reach the length floor to 1e-6 · **merge rule, fixed before numbers:** B1 on both performers AND B2 exact AND B4 on both performers AND B5; B3, B6, B7 report; any failed clause stated in the review |
| later | hands and feet solves after their bars (I4, I5); **F** (our detector into MAMMA's fitter) only after I3 and only with the sparse-MAMMA matched control; a pseudo-label detector only after lane H, because labels triangulated on MAMMA's calibration put MAMMA-derived data into weights | 8, 10, 2 | — |

---

## 3. Roles and protocol

- **Fable** — plans, owns the ladder registry, reviews every report and every gate
  before it enters the ladder, supervises lane D one step at a time, republishes the
  ladder and the board in the same pass as each measurement.
- **Sol / Sonnet** — research before code, per step, where §2 says "research done" is
  incomplete: I4's local-frame and periodicity definitions, I7's fixture extension, D5's
  momentum regulariser (the `RefineConfig.regularizer` pull toward a supplied vector is
  the one usable prior; `CalibrationConfig` only gates whether scale moves).
- **Opus agents** — execute one step each in an isolated worktree; the deliverable is
  a script that writes a JSON report plus an extractor stub, never a registry edit;
  lane D steps additionally land tests and a rerun of the delivery build.
- **Every step ends the same way:** report JSON under `artifacts/` → extractor →
  `ladder.py` rerun (history line only if a headline changed) → the rung's chart in `VISUALS`
  (ours beside MAMMA's, lower/higher is better in words) → board card → commit together.
  Console must not print `NO CONTROL ARM` or `NO VISUAL` for the step's rung.
- **Selectors.** Any constant, weight, threshold or prior is selected by a held-out
  camera, the vs-our-own-capture control, synthetic truth or anatomy. The MAMMA arm
  **reports; it never selects.** MAMMA-derived outputs (I1-B, I2, offset tables) live
  under `artifacts/compare`, never under `artifacts/commercial-multiview-*`.
- **Statistics.** One take, two performers, 150 correlated frames: block bootstrap
  only, candidate and oracle on identical draws, and no generalisation claim beyond
  this take until lane H delivers a second.

---

## 4. Leak paths, listed so they are not walked into

Codex's list, adopted whole: tuning MHR regularisation, locator limits, scale bounds,
pose weights or stopping criteria until the MAMMA score improves; picking a band after
seeing the SMPL-X arm and iterating against it; shipping per-camera offsets fitted
against MAMMA projections; choosing facing, clavicle-anchor, foot-axis, contact or hand
smoothing constants from MAMMA agreement; selecting `smooth_weight` toward MAMMA's
amplitude; the SAM 3D scale PCA; training a detector on labels triangulated through
MAMMA's calibration; letting F's experiments set shipping loss weights; copying
MAMMA-derived foot ranges, hand statistics or betas into MHR defaults. Containment:
freeze the delivery commit and configuration hash before any MAMMA evaluation; keep
reference configuration physically separate from delivery configuration; provenance on
every runtime constant. **Already found: `THORAX_SMOOTHING_FRAMES`.**

---

## 5. Dissent register

| question | Codex | Sol | choice and reason |
|---|---|---|---|
| fixture rank | rank 1, start immediately | last; not engineering | parallel lane H, started now; nothing in I or D blocks on it |
| first delivery step | facing → converter → schema, then MHR | MHR mean body + momentum IK, then shape | Codex's order: without D3 a fitted skeleton is reconstructed canonical downstream; Sol's solver step becomes D4 |
| what A is | skeletal scaling, not a body fit | shape, blind to surface | renamed; surface goes through I6/D6 |
| surface instrument | three: synthetic skeleton truth, skinning correctness, held-out silhouette on owned masks; `pred_vertices` never a shipping gate | SAM2 masks + point-to-surface to `pred_vertices` | all of them; `pred_vertices` is an agreement figure only |
| temporal | synthetic 5a/5b, exact truth | held-out-camera lag on real footage | both; neither scores depth |
| oracle on perfect 2D | not raised | central | adopted; Sonnet found the template already in the repo |
| detector step C | separate reports, split-half offsets | reference-free figures, retire 2.2× | same conclusion by different routes |
| hands band | angle SD invalid; amplitude/jitter + held-out | angle SD is a quantity the solver optimises | same conclusion |

---

## 6. Where this stands (end of 2026-09-02) — state lives in `LADDER_STATUS.md`

The go was given on 2026-09-02 and **lane I is complete**: I1, I2, I4, I6, I8, then I3, I5,
then I7, each an Opus agent in a worktree with the ladder owner wiring its report and its
chart. Every rung now carries a chart, ours beside MAMMA's, lower/higher-is-better in words.
**D1 shipped the same day**: the facing defect was a left/right naming mirror at five sites,
not a yaw (§2's D1 row carries the corrected gate); repaired by relabelling, delivery rebuilt.
`THORAX_SMOOTHING_FRAMES` moved 15 → 9 on synthetic truth, the audit is clean, and the head
gate's fixed 20° band is recorded as near-miscalibrated for the narrower frame (not moved).
**D2 shipped on 2026-09-03 as three changes** (the clavicle aimed from its own pivot, the root on the captured hips, a physical reachability reject at the clavicle; `docs/reviews/clavicle-origin-2026-09-02.md`), with its costs stated: the silhouette against the photographs fell while every joint instrument improved, and the round trip can no longer score a temporal step. **D3 shipped on 2026-09-03** (`docs/reviews/performer-skeleton-2026-09-03.md`): orientation found the delivery carrying TWO skeletons (the code rig for the solve and every instrument, the MPFB asset's own rest in the exported file, 81–195 mm apart); one rest skeleton per performer now travels from the converter into the delivered file, proved by closure (GLB = the track's own FK to 5e-7 m, the old exporter 96/93 mm off) and an exact-skeleton oracle (legs exact, arms 0.43–0.69 mm against a 0.5 band that FAILS on one of six bodies and stays failed). The silhouette that fell 8 of 8 under D2b rose 8 of 8 (0.521 → 0.627); rung 11 71/68 → 47/46. Handoffs: the shoulder construction, the thigh vertical and the spine ratios to **D5**; the binding to **D6**; the pelvis frame (SOMA-77's pelvis root is mapped as `root` and unused for any frame; its two lower-spine joints are unmapped) as its own pose step, **D7**, whose card was written on 2026-09-04 and which ran before D4. **D7 shipped on 2026-09-05** (`docs/reviews/pelvis-frame-2026-09-04.md`) with its selector band FAILED and stated: on synthetic truth with the detector's noise a pelvis frozen upright (7.3°) beats the rigid pelvis fit (10.2°), which beats the old trunk-line pelvis (27.6°), because the fixture's true pelvis sits 1–13° from vertical on every clip; the merge fired on a second pre-registered rule, the frozen-upright control rendered as a delivery and scored through the part-wise silhouette on identical draws (worse than D7 by 0.218 IoU on performer 1's bent tercile, level on performer 0), and what that caught is the control's root placement, not the frame's orientation. The window is 0 by refusal (a converter step with no new dependency, closing a measured cost; D4's card needs restating after D3 — which rest the solver's output lands on — and pymomentum now installs, `/tmp/momenv`, MHR assets under `.cache/mhr/assets`). The MHR subject in D3's card was substituted by independently perturbed per-bone channels (pymomentum unavailable). Lane H is
the user's, and I7 handed it a candidate one-frame sync offset on D001. Nothing in this plan
fits a constant on a MAMMA-derived arm at any point; the MAMMA arm reports and never selects.
