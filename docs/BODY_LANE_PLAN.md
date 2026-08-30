# Body lane — plan of record

**Snapshot: 2026-08-30.** Written to be picked up cold by a session that does not have
the conversation this came from. Read sections 0–2 before doing anything.

**Section 3 (MEASURED) will rot.** Every entry carries the instrument that produced
it and that instrument's blindness, inline, because cross-references die when the
reader is grepping. If a new measurement contradicts a MEASURED entry, **stop and
re-verify the instrument before proceeding** — that is a tripwire, not advice.

---

## 0. First action, before any engineering

**Ask the user whether the Battle 2 marker session is booked.** If it is not, it
remains the standing blocker and both advisors' rank 0.

Nothing in this lane is ground-truth-verified. Every number below is relative to
MAMMA's retained output, or to synthetic truth that contains none of calibration
error, lens distortion, sync error, soft-tissue artefact or joint-definition error —
all five first-order on real footage. **The marker session is the only path to an
accuracy claim.** Checklist: `docs/BATTLE2_ACTION_CHECKLIST.md`. The three items with
real lead time are the marker reference (weeks), genlock (days–weeks) and performer
releases covering ML training use.

---

## 1. Standing rules — instructions, not history

**No gate a constant can pass.** Every acceptance band ships with a demonstration
that a degenerate solution fails it. Four constructs in this lane could not fail: a
bone-length gate reading 0.00% because bone lengths are constants in that estimator;
a jitter gate reading better-than-MAMMA on a frozen rest-pose hand; a control
comparing against a *constant* σ, which is an algebraic identity; and a `both` arm
byte-identical to the arm it was meant to be compared against.

**Same denominator.** Score both arms of a comparison on the same population. A
composition shift has masqueraded as an effect twice — most recently a visibility
channel "worth 1.3 mm" that was worth 0.16 mm once both arms were scored on identical
landmarks.

**The recurring defect is a correct measurement carrying a claim it does not
support.** Seven instances so far. The measurement is usually right; the sentence
attached to it is where the error lives. Before writing a conclusion, ask what the
instrument is structurally blind to.

**Confirm a suspected defect with a second, independent measurement before acting.**

---

## 2. What this lane is trying to do

A commercially clean, in-house markerless multi-camera body capture matching MAMMA:
~13.5 mm MPJPE single-person, ~20 mm two-person, against Vicon. MAMMA and SMPL-X are
non-commercial; hence the clean-room rebuild. Body model is Meta MHR (Apache-2.0,
weights included). Full context: `docs/OWNED_BODY_CAPTURE_PLAN.md` and
`docs/OWNED_BODY_CAPTURE_RESEARCH.md`.

---

## 3. MEASURED — with instrument and blindness inline

**The gap is the detector, not the geometry.**
Our `triangulate_point`, unchanged, fed MAMMA's 2D lands **8.9 mm** from MAMMA's own
512 landmarks (exact `verts_512` regressor), **10.2 mm** at full coverage with no
visibility gating. Fed *our* detector's 2D, the same geometry is out by 24.7 mm of
spread plus 33.9 mm of per-joint bias.
*Instrument:* substitution, `tools/swap-harness/swap_true.py`.
*Blindness:* **saturates at ~10 mm** — MAMMA's own fit sits 9.3–11.7 mm from its own
triangulation, so nothing below ~10 mm against this reference is readable. Stops at
triangulation; never exercises `solve_sequence_positions`.

**The detector's error tail is the story: ours is 7.5× MAMMA's.**
MAMMA's 2D residual (visibility ≥ 0.5, native 3840): 1.44 / 2.50 / **4.29** / 6.99 /
10.37 / **13.08** px. Ours scaled to 3840: 1.59 / 4.11 / **9.33** / 18.10 / 34.50 /
**98.40**.
*Instrument:* `tools/swap-harness/mamma_residuals.py`.
*Blindness:* different statistics — MAMMA's is a fit residual, deflated because its
fit consumed the landmarks it is scored against. **The median ratio is unfair to us
by an unknown amount; the tail ratio is too large to be entirely artefact.**

**Triangulation has little headroom left.**
Single-frame Cramér-Rao bound on this rig is 33.2 mm at four clean views; we measure
16.4 mm — half the memoryless bound, because temporal and limb priors aggregate
across frames.
*Instrument:* `tools/swap-harness/crlb.py`. *Blindness:* single-frame; says nothing
about the sequence solve.

**Resolution is not a lever.** 1280 → 3840 gives 14.5 → 14.8 px spread. At 1280 the
person crop is already 286 px against the model's 256 px input, so it is downsampled
either way.
*Instrument:* `tools/swap-harness/res_ablation.py`, spread only, native-pixel
normalised. *Blindness:* spread only; bias not comparable. **Would still matter for
hands, at ~20 px.**

**σ and visibility cannot be ranked.** Corrected pricing: baseline 11.62 mm
(mixture) / 14.38 (lognormal); a σ head buys +0.95 / **+2.15**; a visibility head
buys **+2.75** / +1.19; a *wrong* channel costs −3.6 under both. **The ordering
reverses between two noise models that both fit our data.** On real data a visibility
channel used as a weight is worth **+0.16 mm** on identical landmarks.
*Instrument:* synthetic fixture + the paired real-data comparison.
*Consequence:* **Battle 4 keeps MammaNet's full μ/σ/visibility triple.** The
reordering that briefly appeared in `OWNED_BODY_CAPTURE_PLAN.md` is withdrawn.

**SAM 3D Body is integrated, licence-clean, and useful only as an orientation
source.**
Runs at ~45.8 s/person-frame on CPU. Emits `pred_joint_coords` on **exactly our
127-joint MHR rig** (checked), `pred_global_rots`, `hand_pose_params`, and accepts our
calibrated intrinsics.
- *Fusing four per-view fits:* **do not build.** 118.2 mm cross-view disagreement,
  67% along the viewing ray but **42.8 mm transverse**, which fusion cannot fix;
  averaging four leaves ~21 mm against our existing ~10 mm — and **~21 mm is a floor**,
since that division by √4 assumes independent errors while monocular models share
appearance biases across views. "Do not build" holds a fortiori.
  (`tools/swap-harness/sam3d_fusion_conditioning.py`)
- *Bone direction:* raw 6.9° median / 72.1° p90; after cross-view majority at 15°,
  **3.2° median / 6.9° p90 — measured *inside* the 74% of bone-frames that keep ≥3
  views.** Always quote the denominator with the degrees; without it this is the
  survivorship shape again. 74% is n=160 bone-frames, a **±8-point binomial band**.
  The 15° gate was tuned on MAMMA footage and is **provisional until re-derived on
  owned data.** (`tools/swap-harness/sam3d_orientation_agreement.py`)
- *…but for long segments this channel is redundant.* Two endpoints at our own ~10 mm
  give a direction uncertainty of √2·10/L: **thigh 2.0°, upper arm 2.9°, forearm
  3.2°, head 3.5°** — at or better than SAM 3D's 3.2°. The channel only pays where the
  segment is short: **hands past the wrist, 9.0° from positions against 3.2°**, and
  marginally feet at 4.5°. **So the "orientation channel" is in substance a hand
  channel**, and an evaluation that includes limbs would "pass" by re-measuring what
  triangulation already had.
- *Twist:* **not usable.** 10.7° median, 139.7° p90, forearms 22–26° — the textbook
  monocular blind spot. **The orientation source is scoped to direction only.**
*On the world transform:* verified only to **gross error** — 0.28 m rules out an axis
swap and nothing finer. That is sufficient here for one specific reason: **directions
are translation-invariant, so the 88.4 mm along-ray disagreement — 67% of the total
that killed fusion — cannot contaminate the orientation result at all.**
*Blindness of both orientation results:* reference-free. Four monocular fits can agree
and be wrong together through shared appearance bias. **Agreement is a necessary
condition, never an accuracy claim.**

---

## 4. WITHDRAWN — do not re-derive these

Each was published and then refuted **by our own follow-up measurement**. They are
listed so a fresh session does not rediscover them as findings.

| withdrawn claim | why |
|---|---|
| "SAM 3D's 2D is 1.83× worse, 2.70× in the tail" | compared its 10-frame ladder against SOMA's hard-coded **full-take** row. On identical frames the median gap is 1.42×. **The tail is not readable for either detector** — SOMA's same-frames tail is ~5× its full-take tail, so the harness association contaminates both, and reading it was itself the error being corrected |
| "the epipolar ladder can rank detector families" | it sees only the **incoherent** component. SOMA's error is 18.6 px coherent bias + 16.6 px spread — bias is the larger term and passes through triangulation untouched |
| "the temporal smoother helps on real footage" | scored against MAMMA's *fitted* mesh, which is 2.6–2.9× smoother than its own raw triangulation. The test rewards smoothing. **G10's re-reading survives; the number does not** |
| "μ 10.4 mm / visibility 4.5 / σ 0.4" | noise fitted to `_epipolar_distance_px`, which returns the **symmetric** distance, as if one-sided — 2.00× overstatement |
| "visibility is worth 1.3 mm" | cross-denominator. 0.16 mm on identical landmarks |
| "SOMA-77 carries a 61 mm body-fixed offset" | true in the synthetic domain only; **does not transfer** to real footage (cosine +0.039) |
| "SAM 3D is promptable with our 2D" | `process_one_image` has **no keypoint-prompt parameter**; the internal sampler uses the model's *own* predictions |
| "the estimator has little headroom" | scoped to **triangulation**; the sequence solve was never in the swap |

---

## 5. OPEN — not measured, do not assume

- **Absolute accuracy of anything.** Gated on Battle 2.
- **Detector bias against truth on real footage.** The synthetic fixture cannot
  measure it — flat-shaded renders do not transfer.
- **Whether bone directions actually improve the reconstruction.** Consuming them
  means adding an **orientation term to a position-only solver** — a real estimator
  change with its own increment discipline, not a data hookup.
- **Forearm twist**, which nothing available estimates well.
- **Association under contact.** MAMMA's pre-associated 2D is a truth-grade reference
  for it and has not been used.
- **`pred_global_rots`' basis convention, per joint.** Column 0 aligns with the bone
  at |cos| 0.87–0.99 — and 0.87 is **30° of slack**, so the convention must be pinned
  *per joint* against `src/autoanim_gnm/data/mhr-skeleton-v1.json` rest directions,
  not assumed globally. Every orientation result so far was measured convention-free
  from positions precisely to avoid this; using the matrices directly requires it
  first.
- **Whether the 74% acceptance holds on owned footage**, or is a property of this
  clip. The 15° gate is provisional for the same reason.

---

## 6. Build sequence

Co-authored with both advisors; where they differed the choice is noted inline.
Each step names what it changes, what measures it, the pre-registered band, and the
degenerate solution that band must reject. **Do not skip the degenerate check.**

### Step 0 — book the marker session *(calendar action, today, parallel to everything)*

Both advisors' rank 0. And this week's failures turn into **hard acquisition
requirements**, because each names a term no owned footage can currently exercise:

- **a turning sequence, ≥90° of yaw.** No owned clip turns more than 17°, which is
  exactly why body-fixed and world-fixed offsets could not be separated on real
  footage. Without it that question stays unanswerable;
- **fast acting, p95 joint speed ≥ 1.9 m/s**, so the temporal smoother can be scored
  against truth rather than against a smoothed reference;
- **checkerboard, a distortion model, and a measured sync residual** — the terms the
  synthetic fixture lists as absent by construction and which therefore have never
  been measured at all;
- **two subjects in physical contact**, the hard case for association.

*Until this lands, every band below is a consistency band, not an accuracy band.*

### Step 1 — association under contact *(measure; build only if it fails)*

MAMMA's retained `ma_2d` is **pre-associated by subject** — truth-grade labels for the
one stage nothing else can check, on a contact clip. Strip the labels, pool 2×512 per
camera, run `associate_frame_graph` (commits `9eb0393`, `7750c98`), score against
MAMMA's split. Add a ±1-frame stream-offset arm to price sync sensitivity.

*Band:* zero switches over the take, **with the inter-subject distance and
epipolar-affinity margin distributions published beside the count.**
*Degenerate solution:* "zero switches" is passable by a constant when bodies are far
apart. **Without the margin distribution, report nothing.**
*Why first:* needed under every candidate architecture, cheapest rung with a
truth-grade reference, and it **fixes a defect in our own harness** — the 160 px
root-proximity association currently admits cross-person pairs and contaminated the
epipolar tail for *both* detectors (SOMA's same-frames tail 57.58/93.83 against its
full-take 11.49/32.79).

### Step 2 — the orientation channel, scoped to short segments only

**Rescoped after fable's redundancy check, and the rescoping nearly eliminates it.**
Our own positions already give 2.0–3.5° of direction on limbs and spine — at or
better than SAM 3D's 3.2°. The channel pays only where the segment is short: **hands
past the wrist (9.0° from positions against 3.2°)** and marginally feet. So this is
**a hand channel**, and it merges with Step 3.

*What it changes:* `solve_sequence_positions` is position-only; consuming directions
means an orientation residual with its own weight and Jacobian sparsity — **a real
estimator change, not a data hookup.**
*Band:* marginal value **over position-derived directions**, on the scoped segments
only; held-out reprojection must not regress; acceptance quota per segment on the
**full denominator**.
*Degenerate solutions:* a zero-weight term reproduces the current solver and passes
any "does not regress" test — **the band requires measured improvement, not absence
of harm**; a channel computed from triangulated positions passes any limb band, which
is why limbs are out of scope; and a frozen direction passes on low-rotation
dialogue, so the eval set must include a turning sequence or the yaw-90/180 renders.

### Step 3 — the hands question, where the orientation channel actually lives

SAM 3D's `hand_pose_params` come from a learned prior into our own rig, against the
27-DoF chain that finished at **33.9 mm held-out**. Never compared. The resolution
finding cuts the *other* way here: at a ~20 px hand crop, resolution should matter.

*Band:* leave-one-camera-out on the same four folds and four hands as increment 6.
*Degenerate solution:* a frozen mean hand scores ~98 mm; the band must reject it.

### Step 4 — pseudo-label campaign for Battle 4 *(build, gated)*

**Fable's, and it is the strongest idea in this plan.** Labels = our triangulated
positions on **owned footage**, filtered by our own gates. Training a per-view
detector on **our own convention** kills the ~18.6 px coherent bias *by construction*
— the largest measured term in the budget — because there is no convention to cross.

*Licence preconditions, pre-registered as gates rather than footnotes:* archive the
NVIDIA OML text bound to our revision, and establish the exact attribution string —
**do not invent one.** Label provenance recorded per clip.
*Degenerate solution:* the detector learns the smoother's lag. **Train on unsmoothed
triangulation; test stratified by speed.**

### Explicitly not built

MHR parameter fusion (gate failed). The reverse swap (firewall impossible on a
one-person project; their fitter cannot consume 19 joints; question already
answered). Separate visibility and σ heads as *requirements* (+0.15/+0.16 mm, three
independent confirmations — they stay in the triple, they do not drive the design).
Resolution work (refuted).

## 7. What of MAMMA stays in the loop, and until when

Its licence bars producing artifacts for commercial purposes. **The line is: MAMMA is
a measuring instrument, never in the shipping path.** Nothing MAMMA produces may enter
a delivered artifact, a trained model's weights, or a calibration constant that ships.

| retained | what it is for | what replaces it | retirement trigger |
|---|---|---|---|
| `ma_2d` landmarks + visibility | the substitution slot: the only way to hold our geometry fixed and vary the detector | our own detector, once one exists | Battle 4 delivers a detector, **or** Battle 2 delivers markers — whichever first |
| fitted meshes / `pred_joints` | the per-joint bias and spread reference | marker reference | Battle 2 delivers |
| `triangulated_3d_pts` | shows what *their* fitter does to *their* input, so our geometry can be scored against a like stage | nothing needed after Battle 2 | Battle 2 delivers |
| `verts_512.pkl` regressor | exact landmark correspondence, so comparisons are not nearest-vertex approximations | — | with the above |
| `ma_masks` | **not used.** We consume no masks | — | already retired |
| `gt_vertices` / `gt_joints` | **nothing — byte-copies of `pred_*`, verified. A decoy.** | — | never use |
| the fixture's camera rig + footage | the fixture itself | owned calibration | **MPI-derived and barred from shipping**; retires at the first owned multicam fixture, which can precede Battle 2 |

**No shipped constant, threshold, weight or calibration may be fitted or tuned on a
MAMMA-derived artifact.** Structural go/no-go decisions may cite them; numbers that
ship must be re-derived on owned data. **First casualty: Step 2's 15° majority gate,
provisional until re-derived.**

**And one dependency is now dual-provenance:** `mhr_model.pt` is sha256-identical
between NVIDIA GEM-X and Meta SAM 3D Body, so MHR can be re-homed under the SAM
License if the NVIDIA OML ever becomes a problem.

**A calibration constant fitted on MAMMA outputs cannot ship** — that is a licence
trap, not a technical one, and it is why the arm (i) offset work would have been
unusable even had it transferred.

---

## 8. Instrument table — what can produce a readable number

| instrument | readable to | blind to | **banned uses** |
|---|---|---|---|
| 3D vs MAMMA's fit | **~10 mm floor** — its fit sits 9.3–11.7 mm from its own triangulation | anything below that | any sub-10 mm claim; **scoring the smoother** — the reference is 2.6–2.9× temporally smoothed |
| 3D vs `triangulated_3d_pts` | ~9–12 mm self-noise | — | scoring the smoother — its noise is *correlated* with the 2D being smoothed, the opposite bias to the fit |
| 2D vs fit projections, **spread** | ~4–5 px per camera | bias appears as convention offset | quoting bias as detector error |
| cross-view epipolar ladder | the **incoherent** component only | **the entire coherent axis** — SOMA's *larger* term | **cross-family detector verdicts**; any comparison not same-frames-same-harness; **reading the tail without an association audit** |
| cross-view consistency of monocular fits | mm, reference-free | shared appearance bias | quoting "fits agree to X mm" as accuracy |
| majority-vote orientation residual | ~1° | survivorship by construction | quoting the residual **without per-segment acceptance on the full denominator** |
| synthetic fixture, corrected noise | sub-mm, truth-referenced | calibration, distortion, sync, soft tissue, joint definition | any "our MPJPE" claim; **detector pricing** |
| association vs MAMMA's subject split | truth-grade labels | this take only | generalisation beyond the contact case |
| reprojection onto our own detections | — | — | **banned as a gate entirely** |
| marker session | **the only absolute accuracy instrument** | — | not yet booked |

---

## 9. Traps that cost real time

- **`_epipolar_distance_px` returns the SYMMETRIC distance** — the sum of two
  one-sided distances. **Halve it before fitting a per-view σ.** Measured ratio 1.962.
- **MPS is impossible for SAM 3D.** float64 inside the TorchScript MHR module, out of
  reach of any external shim. CPU only, ~45.8 s/person-frame.
- **SAM 3D hardcodes `.cuda()`** in at least three places. Shims live in
  `workers/commercial_multiview/sam3d_body_pose.py`, never in the vendored checkout —
  its licence forbids reverse engineering.
- **`cam_int` must be a torch tensor of shape (B,3,3)**, not numpy, not (3,3).
- **The system Python 3.13 has a broken cert store.** Set `SSL_CERT_FILE` to certifi's
  bundle or `torch.hub` reports "no internet connection".
- **The upstream prints to stdout**, mixing with JSONL. Use `AUTOANIM_SAM3D_OUT`.
- **`gt_joints` / `gt_vertices` in MAMMA's output are byte-copies of `pred_*`.**
  There is **no ground truth on disk.** Never score against a variable named `gt_`.
- **The Bash working directory persists between calls.** A relative path after a `cd`
  put a venv inside a vendored source tree.

---

## 9a. Was SAM 3D Body worth it? — the scoping verdict

Asked directly, and recorded because a future session will ask it again.

**What it bought, measured:** three doors closed — fusing per-view MHR fits (118 mm
disagreement, ~21 mm floor against our existing ~10), monocular HMR as a 3D source,
and twist (forearms 22–26°). One licence family verified as commercially clean. One
integration banked and working. And **one candidate role surviving: hands.**

**What it cost:** about a day — a 2.8 GB gated download, an isolated venv, four
shims, ~80 CPU inferences, four measurement rungs, and a published negative that had
to be withdrawn.

**The honest ledger, and it is not "diversion".** All three of those doors were live
candidates 48 hours ago; fusion was a numbered build step in this plan. Closing them
by *building* one and discovering the same answer would have cost weeks each. A day
for three confirmed dead ends is an acceptable scouting trade, and recording it as
pure waste would teach the wrong lesson — that scouting is waste. What made it *feel*
like a diversion is the reversal pattern, not the information yield.

**But the mission path does not route through it, and that is the answer.** The
measured gap is the detector's tail — 7.5× MAMMA's at p95 — and SAM 3D does not touch
it: its own 2D is worse in the bulk and its tail is unreadable through the only
instrument available. Note too that **the strongest idea to come out of the whole
detour, the Step 4 pseudo-label campaign, does not use SAM 3D at all.** The plan's own
ranking — Step 0 Battle 2, Step 1 association, Step 4 pseudo-labels — survives the
question unchanged. That is the weight it deserves.

**Standing decision, so this is not relitigated:** SAM 3D gets **exactly one more
pass**, the hands test already specified as Step 3 — leave-one-camera-out on the same
four folds and hands as increment 6, against arm D's 33.9 mm, with the frozen-hand
degenerate check. **If it does not beat 33.9 mm on that first pass, close it in this
document and stop.** Iterating after a first-pass loss is the sunk-cost failure, and
it is pre-committed against here rather than argued about later.

**One untested use exists and is deliberately not being run.** SAM 3D carries a
full-body prior, so its behaviour on *occluded* landmarks specifically — the tail —
is genuinely unmeasured, because the ladder pooled everything. It is not being tested
because **no readable instrument exists for it**: per-landmark occlusion labels would
have to come from MAMMA's visibility channel, re-entering MAMMA as reference, and the
epipolar ladder is already banned for cross-family verdicts. Anyone proposing this
test must **name the instrument and its blindness first.** That is this document's own
discipline, and this is the first place a future session will be tempted to skip it.

## 10. Where we stand against MAMMA — the honest paragraph

Our geometry is not the problem: given MAMMA's own 2D, our unchanged triangulator
lands within 9–10 mm of MAMMA's own landmarks, which is inside the range MAMMA's
fitter itself moves from its own triangulation — so on that axis we are at the
instrument's floor and cannot be distinguished from it. What separates us is the
detector, and specifically its **tail**: at p95 our 2D error is roughly 7.5× MAMMA's,
98 px against 13 at native resolution. That single ratio is the most durable finding
in this lane and it survived every correction. **But no number here is
ground-truth-verified.** Everything is relative to MAMMA's own output — an estimate,
not truth — or to synthetic data missing five first-order error terms. We do not know
whether our body track is at 20 mm or 40 mm against markers, and we will not know
until Battle 2 delivers. The honest position is that we have strong evidence about
*where* our error is, no measurement of *how much* of it there is, and a detector
programme that is now correctly aimed but not begun.
