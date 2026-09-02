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

## 6. What starts now, if the go is given

In parallel, each an Opus agent in a worktree, Fable reviewing: **I1, I2, I4, I6 (id
resolution first), I8.** I3 and I5 follow when the first four have reports. Lane D
starts with D1 only after I6 has a report and the forward-dot gate is written down.
Lane H is the user's: the fixture and the releases. Nothing in this plan ships anything
until D1; nothing in it fits a constant on a MAMMA-derived arm at any point.
