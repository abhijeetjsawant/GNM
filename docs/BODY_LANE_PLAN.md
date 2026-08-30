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

> **Asked 2026-08-30. The answer was "not sure", from the person who would book
> it.** Treat that as *not booked, nothing in motion* — and note that it is the
> second time this question has been the rank 0 item without moving. **Both
> advisors independently ranked the booking above every piece of engineering
> available this session**, and Fable's tie-break was explicit: if only one thing
> gets done, it is the booking, because no engineering result moves a parity
> *claim* one millimetre without it.
>
> Fable's recommendation, recorded because it changes the ask: **build the owned
> multicam fixture and the physical-constraint kit as one build-out**, which turns
> the booking from *"rent a stranger's stage"* into *"bring markers to our
> stage"* — and delivers scale, sync and distortion as measured, owned,
> commercially clean numbers in days rather than weeks.

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

A commercially clean, in-house markerless multi-camera body capture matching MAMMA.
MAMMA and SMPL-X are non-commercial; hence the clean-room rebuild. Body model is
Meta MHR (Apache-2.0, weights included).

**⚠ State the target as three numbers, never one — corrected 2026-08-30.** This
section previously read *"~13.5 mm MPJPE single-person, ~20 mm two-person, against
Vicon"*, which silently welded an MPJPE-against-a-fitted-reference figure onto a
marker axis. `BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md:1046` already forbids exactly that:
the two *"must never appear in the same row or the same table."* On the same data
MAMMA scores:

| metric | MAMMA | reference |
|---|---:|---|
| MPJPE, vs a fitted reference | **12.96–13.5 mm** | Vicon + MoSh++ pseudo-GT |
| PVE | **17.18 mm** | — |
| **held-out marker error — the actual Vicon-parity claim** | **22.48 mm** | Vicon + MoSh++ scores **21.62 mm** on the same task |

**The parity claim is +0.86 mm against a marker pipeline that is itself ~21.6 mm from
its own held-out markers** (repeated on the Dance set: 27.59 vs 26.15, gap 1.44 mm).
So the honest target is **held-out marker error within ~2 mm of a marker-based
pipeline**, i.e. **~20–22 mm** — not 13.5.

**And the consequence is sharper than "the target moved": marker ground truth cannot
discriminate below about 20 mm, so a 13.5 mm gate scored against markers is
*unfalsifiable*.** Battle 2 can tell us whether we are at 25 mm or 45 mm. It cannot
tell us whether we are at 13 mm or 18 mm, and no marker session ever will. If a
sub-20 mm claim is ever needed it requires a different instrument — a body scan, or
synthetic truth — and the plan must say which.

*Two provenance notes on the 13.5 itself:* it is **read off Fig. 12 of the MAMMA
paper**, not a published table — order 13–15 mm with chart-read error — and its
ground truth is **Vicon markers passed through a MoSh++ SMPL-X fit**, i.e.
model-mediated pseudo-GT. Cite it that way or not at all.
*This is not new to the repo* — `OWNED_BODY_CAPTURE_PLAN.md:18–23` states it
correctly. It had simply never propagated into this plan's §2, and from there into
everything that quoted it.

Full context: `docs/OWNED_BODY_CAPTURE_PLAN.md` and
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

**Where the pipeline's mass actually is — and `valid_joint_fraction` is not a
quality signal.** On the real SOMA-77 run (`artifacts/commercial-multiview-soma77`,
150 frames × 19 joints × 2 subjects = 5,700 slots): **5,080 triangulate directly**,
**620 are filled**, and **the sequence solve recovers 6.**

**But 600 of those 620 are the two ears, missing in every frame of both subjects,
because SOMA-77 emits no ear landmarks at all.** Genuine missing evidence is
**20 slots** — subject 0: right elbow 4, right wrist 4; subject 1: left wrist 10,
left ankle 2. So:

- **`valid_joint_fraction = 0.891` sits against a structural ceiling of 0.895**
  (17 of 19 joints). We are at **99.6% of the maximum achievable** with this
  detector. **It is a constant wearing a metric's clothes** — it cannot move
  without changing the detector's joint set, and reading it as coverage quality is
  the eighth instance of this lane's defect. *It nearly became the ninth: this
  entry first said "614 interpolated" as though occlusion caused it.*
- **The solve recovers 6 of ~20 eligible slots, not 6 of 5,700.** Ears carry
  **zero rays**, so `candidate = (~direct) & (support >= 1)` excludes them
  entirely. Its recovery rate on actual candidates is ~30%, and
  `constraint_recovered_joint_fraction = 0.001` is quoted on the wrong
  denominator. **The solve is under-exercised, not broken.**
- **Real occlusion on this fixture is 0.35% of slots.** Any instrument built here
  to study occlusion has almost no events to study.
Then `_fill_and_smooth_positions` runs over everything and **moves
already-triangulated joints by a median of 5.6 / 7.9 mm** (subject 0 / 1), p95
48.9 / 76.5 mm, max 898 / 1,355 mm — on joints **clear of any gap**, so this is
not a window artefact at the edges of filled runs.
**The large moves are outlier repair, not vandalism:** joints moved >100 mm had a
median raw temporal acceleration of **440 / 635 mm·frame⁻²** against **13** for
joints moved <10 mm — a 34–46× ratio. The Savitzky-Golay stage is functioning as
the pipeline's **de facto outlier filter.**
*Instrument:* direct comparison of `raw_triangulated_world_positions_z_up_m`
against `triangulated_world_positions_z_up_m` in the retained body tracks.
*Blindness:* **these are displacements, not errors.** Nothing here says whether a
move was toward truth. Both candidate references are **banned for scoring the
smoother** — MAMMA's fitted mesh is 2.6–2.9× smoother than its own triangulation,
and `triangulated_3d_pts`' noise is correlated with the 2D being smoothed.
*Consequence, and it is a prediction Step 0a can test:* **if the smoother already
absorbs the detector tail, the oracle-veto ceiling will be small.** The 5.6–7.9 mm
median move on well-behaved joints is the lag half of the same trade, and it is
the same order as our entire triangulation budget.

**Triangulation has little headroom left.**
Single-frame Cramér-Rao bound on this rig is 33.2 mm at four clean views; we measure
16.4 mm — half the memoryless bound, because temporal and limb priors aggregate
across frames.
*Instrument:* `tools/swap-harness/crlb.py`. *Blindness:* single-frame; says nothing
about the sequence solve.

**Most of the "coherent joint-definition bias" is not per-joint at all — it is ONE
2D translation per camera, and removing four numbers buys ~7 mm.** Hypothesis C2 from
the external research pass, tested the same day.

Decomposing each (subject, frame, camera) 2D residual against MAMMA's fit projections
into a common translation plus joint-specific residuals:

| arm | 3D median | p95 |
|---|---:|---:|
| as measured | 45.11 mm | 126.86 mm |
| **static per-camera offset removed** | **37.92 mm** | **94.58 mm** |
| full per-frame common mode removed *(oracle)* | 37.16 mm | 90.15 mm |
| **shuffled common mode *(control)*** | **44.22 mm** | **147.08 mm** |

**The common mode is static, not per-frame:** 3.65 px static against 1.94 px
time-varying (A001 3.65 · B001 2.87 · C001 3.71 · D001 4.77). **So ~90% of the
available gain comes from four constants**, and the expensive per-frame half is worth
0.76 mm more. In 2D it is **25% of the median residual** — 6.36 px → 4.75 px.
*Control:* subtracting a common mode fitted to a *different frame* gives 44.22 mm and
makes p95 **worse** (147.08). The gain is not an artefact of subtracting some offset.
*Instrument:* `tools/swap-harness/common_mode.py`.
*Blindness, and it is the whole caveat:* the offsets were fitted **against MAMMA's
fit**, so this is a ceiling, and **§7 forbids shipping a constant fitted on a
MAMMA-derived artifact.** It must be re-derived on owned data. Unlike the oracle veto,
that is straightforward — a static per-camera 2D offset is a calibration constant,
estimable once per rig without any reference pose system.
*What it is NOT yet:* the cause is unresolved. A static per-camera 2D translation is
equally the signature of **small calibration error** (a term the plan lists as absent
by construction and never measured on our rig), a **crop/box convention**, or the
mean across joints of a genuine convention offset. Correlation with the person's 2D
extent is weak (r = −0.17), which argues against crop *scale*.
*Consequence:* **the bias may be ~4 numbers rather than 19 per-joint conventions**,
which would make it far cheaper to kill than the pseudo-label campaign assumes — and
Step 4's "kills the bias by construction" claim should be re-examined against that.

**Camera count: the geometry bound improves as 1/√N and the real system does not
move at all. Do not buy cameras for accuracy.** *(This entry is a correction of its
own first draft. The measurement is right; the conclusion I drew from it was wrong,
and the repo had already answered the question.)*

Synthetic views on the ring fitted to the real rig, corrected noise, single-frame
CRLB: **4 views 16.60 mm · 6 views 13.58 (−18%) · 8 views 11.76 (−29%) · 12 views
9.60 (−42%)** — textbook 1/√N (16.60·√(4/8) = 11.74 against 11.76 measured).
*Control:* a synthetic 4-camera ring reproduces the real rig within **0%**.
*Instrument:* `tools/swap-harness/camera_count.py`.

**⚠ But that improvement is unrealisable, and `OWNED_BODY_CAPTURE_RESEARCH.md`
already said so.** Its budget table carries the row *"Camera count, 4 vs 16 |
+0.5 mm single / +3 mm two-person | **smallest term on the list**"*, and MAMMA's own
Fig. 12 ablation **saturates**: 2 → 4 → 8 → 12-16 cameras gives ~24 / **13.5** /
**13.5** / ~13 mm single-person. **Four to eight buys ~0 mm on a real system.**
The CRLB is a *geometry* bound that assumes detector noise is the only limit; the
empirical curve is flat because **the detector dominates**, which is this lane's own
central finding. So the bound is correct and does not transfer.
*What I got wrong:* "no document treats camera count as a design parameter" —
**false**, the research doc prices it and demotes it. I re-derived a closed question
and briefly promoted it into the rig spec on a number that cannot be realised. Tenth
instance of a correct measurement carrying a claim it does not support, and this one
is mine.
*Where extra cameras may still pay, unpriced:* **two-person** (+3 mm, the larger
term), the **35.3% two-eligible fraction**, **hands** (8→2 cameras costs 7.5 mm on
hands against 2.7 mm on body), and — per Simon et al., CVPR 2017, multiview
bootstrapping — **pseudo-label purity for Battle 4**, which has an analytic
view-count-to-precision relation and is the one plausibly superlinear return.

**⚠ One caveat on the saturation evidence, and it stops this being fully closed.**
Every ablation showing a flat curve is from a **model-fitted** system. MAMMA fits
SMPL-X; *Better Together* (arXiv:2503.09293) reports 8→4 cameras costing only
0.2–0.9 mm — **but its own keypoint-only arm scores 51.72 against 32.61 W-MPJPE with
photometric refinement**, and the refinement is what flattens the curve. **Our
architecture is keypoint triangulation with an analytic retarget — the arm that
degrades faster.** So "camera count saturates" is established for fitted systems and
**not established for ours.** Do not re-open it as a rig change; do record that the
evidence does not cleanly transfer, and that acquiring a body fit would also be what
makes 4 cameras safe.
*Blindness, and it is the more interesting half:* the redundancy argument — that more
cameras rescue the **35.3% of slots with only 2 eligible views** — **cannot be
extrapolated, because the model fails its own control.** Measured per-camera
eligibility is p = 0.890 over 19,499 joint-camera slots; assuming independence that
predicts **6.2%** of slots at ≤2 eligible on four cameras, against a **measured
35.3% — a 5.7× miss on the one case we can check.** So occlusion is strongly
*correlated* across views, and how much more cameras buy in redundancy is
**unmeasurable until we own multi-camera footage.**
*Consequence:* put 6–8 cameras in the owned rig spec for the 18–29% of information
it demonstrably buys, and **do not** justify it by starved slots.

**Resolution is not a lever.** 1280 → 3840 gives 14.5 → 14.8 px spread. At 1280 the
person crop is already 286 px against the model's 256 px input, so it is downsampled
either way.
*Instrument:* `tools/swap-harness/res_ablation.py`, spread only, native-pixel
normalised. *Blindness:* spread only; bias not comparable. **Would still matter for
hands, at ~20 px.**

**σ and visibility were never comparable — a confidence channel is a *gate*, not a
weight.** *(Corrected 2026-08-30. Full working: `docs/CONFIDENCE_GATE_NOT_WEIGHT.md`.)*
The two published arms differ by **which side of `minimum_confidence = 0.25` they
put the bad observations on**, not by which channel they are: visibility set them
to 0.125 and they were **dropped from `triangulate_point`**; σ clipped them to
0.250001 and they were merely re-weighted. Same information, two mechanisms.
Holding the information fixed and moving only the gate, on 5 seeds and 11,424
observations per arm: baseline 11.92 mm; **dropping buys +3.09 mm, re-weighting
buys +0.90** — and the re-weighting arm lands on the published σ arm to the
decimal, because it *is* that arm. **⚠ That +3.09 mm is synthetic and has no
real-footage counterpart: the oracle veto entry below measures the same operation
at ~0 on real data.** What survives here is *which mechanism is live*, not its size. The shuffled control drops the same 794
observations at random and **costs −3.35 mm**, so the value is in knowing *which*
to drop.
*Why the weighting path is so weak:* `solve_sequence_positions` returns
triangulation on every slot that triangulated (`output[recovered] = solved[recovered]`),
and early-returns entirely when nothing failed — so the robust-loss weight is
**verified inert** on this fixture (`weight_before_loss` flipped gives
bit-identical output). What σ measured is `triangulate_point`'s `sqrt(conf)` DLT
row weighting.
*Instrument:* `tools/swap-harness/gate_vs_weight.py`, synthetic fixture.
*Blindness:* **this fixture cannot exercise the sequence solve at all** — not
because nothing fails to triangulate (168 of 1,596 slots per subject do) but
because those failures carry **zero retained rays**, and `candidate` requires
`support >= 1`. **What exercises the solve is a single-ray slot, not occlusion as
such.** It prices a mechanism, never an accuracy, and contains none of the five
first-order real-footage terms.
*Consequence, unchanged and now better supported:* **Battle 4 keeps MammaNet's
full μ/σ/visibility triple.** The largest available win is deciding **which
observations to drop**, and a visibility head is what produces that decision. The
reordering that briefly appeared in `OWNED_BODY_CAPTURE_PLAN.md` stays withdrawn.
On real data a visibility channel used as a *weight* is worth **+0.16 mm** on
identical landmarks — consistent with this, and always was: weight ≈ dead, gate ≈
alive.

**A perfect oracle veto of the detector's tail is worth ~0 mm, because
`triangulate_point`'s inlier gate already rejects it.** This is the number §9a said
might "close SAM 3D entirely for free", and it does — but it was a live question,
not a formality: the synthetic oracle *gate* was worth +3.09 mm, so the prior had
moved against this result before it was measured.

| threshold | flagged | oracle Δp50 | Δp95 | shuffled Δp50 | Δp95 | **flagged obs the gate actually used** |
|---:|---:|---:|---:|---:|---:|---:|
| 20 px | 4.6% | +0.15 mm | +2.09 | −0.53 | −2.65 | **9.7%** |
| 30 px | 4.0% | +0.12 mm | +2.22 | −0.45 | −3.75 | **2.8%** |
| 45 px | 3.5% | +0.06 mm | +3.43 | −0.45 | −1.10 | **1.8%** |
| 60 px | 3.3% | +0.05 mm | +1.47 | −0.34 | −1.01 | **1.9%** |

**The last column is the mechanism.** At the 60 px operating point only **1.9%** of
flagged observations were consumed by the inlier subset — the exhaustive
enumeration over 2..N views with a 14 px inlier threshold has already thrown them
away. A learned veto would be re-doing work the geometry does for free.
*Median Δ is under 2 mm at every threshold; p95 Δ runs +0.7 to +3.4 mm and is
non-monotonic, so it is noisy and never approaches 5 mm.* Read as: **below the
"build it" line everywhere, below the "close it" line on the median, ambiguous at
p95.**
*Control that can fail, and does:* a shuffled veto dropping the same count at
random costs **−0.34 to −1.32 mm median, −1.01 to −4.41 mm p95.** The instrument
can see an effect; there is not one to see.
*Instrument:* `tools/swap-harness/veto_ceiling.py`, 4,166 baseline-triangulated
slots, both arms scored on the **baseline's** denominator with attrition printed
beside the millimetres (1 lost slot at 60 px).
*Instrument validation, three ways:* the debiased residual reads p50 4.2 / p90 11.6
/ p95 18.3 / **p99 194.0 px** at 1280. The epipolar ladder's independent p95 is
32.8 px **symmetric**, which halves to **16.7 px** one-sided against our measured
**18.3** — within 10%. And p50 4.2 px matches §8's "2D vs fit projections, spread
~4–5 px per camera" exactly. The p95→p99 jump is the catastrophic tail, seen
directly for the first time.
*And it explains the synthetic disagreement.* The real tail is **bimodal**: 92.9%
under 14 px, **3.6% in the 14–45 px band where the inlier gate is weakest, 3.5% at
≥45 px** where it discards them for free. The synthetic mixture put 7% at σ=36 px —
engineered into exactly that gap, which is why it priced an oracle gate at +3.09 mm.
*Blindness:* MAMMA's fit is the reference, so **no absolute number here is readable
below ~10 mm** — only the paired difference is, because both arms share the
reference exactly. Subject assignment excluded **72 of 600 camera-frames** as
near-ties. **And the ~0 is a property of this rig's redundancy, not a law.**
**35.3% of slots have only two *eligible* views** — the bound on the gate's power,
since at two eligible views it cannot reject anything. (36.5% *used* two views;
"used" can be two because the gate trimmed a bad third, which is the mechanism
working.) On occlusion-heavy owned footage the two-eligible fraction grows and the
veto's value grows with it. **The verdict is scoped to the ≥3-eligible-view
regime.**
*Consequence:* **no veto of any provenance is worth building on this rig — SAM 3D's
occlusion prior included. SAM 3D closes.** Reopening it would need new evidence from
a rig that does not exist yet, which is what "closed" means here. *And the prediction recorded earlier today was
right in direction, wrong in mechanism:* the tail is absorbed by
`triangulate_point`'s inlier gate, **not** by the Savitzky-Golay smoother.

**Association passes on a genuine contact clip — and contact halves the margin
without ever inverting it.** MAMMA's `ma_2d` labels stripped, 2×19 landmarks pooled
per camera in shuffled order, `associate_frame_graph` run with **no temporal
history** (each frame judged on its own evidence, which is the harder test):
**zero switches over 300 slot-frames and 1,200 camera-slot decisions**, and
unchanged under a **±1-frame stream offset** on A001 (2 boundary-frame
unassignments, no switches).
*The clip is genuinely hard, and the centroids lie about it:* centroid separation
never drops below 388 mm, but the closest **surface** distance reaches **2 mm**,
with **38 of 150 frames in contact** under 50 mm.
*The margins, published beside the count because the band is void without them —
symmetric px at 3840:* the correct pairing was cheaper in **900 of 900** camera-pair
decisions. Split on the same denominator, **in contact** min +156.6 / p05 +204.0 /
median +575.4; **apart** min +183.6 / p05 +309.8 / median +1098.0. **Contact halves
the median margin and never inverts the decision.**
*Instrument:* `tools/swap-harness/association_swap.py`.
*Blindness, and it is the whole scope of the result:* MAMMA's landmarks are
**projections of a fitted mesh**, so this scores our association given a
*geometrically perfect* detector. It says nothing about association on our own 2D —
where the real run already defers **32 of 150 frames** to the exhaustive path. The
19 points are farthest-point-sampled, not our semantic joints, so the margin
magnitudes are not comparable to production. **And the input is always
count-correct** — exactly two detections per camera, every frame — so
`frame_is_ambiguous` never fires and **the ambiguous and ghost-detection paths are
entirely untested**, including the docstring's own worry, "a person assembled from
two performers."
*Consequence, stated to exactly what the instrument supports:* **given a
count-correct, geometrically perfect detector, the matching logic is not the
problem.** That is a positive attribution: if association fails on our footage, the
fault is upstream in the detections, not in this stage. Do not build association
work on this evidence.

**The retarget is the largest displacement between what we reconstruct and what we
deliver — of the same order as the entire reconstruction error budget.** *(Stated
that way deliberately: it is measured root-relative, delivered-against-captured,
over 13 landmarks, and the detector gap is measured against MAMMA's landmarks in
pixels and millimetres of spread. **Different references and different
populations — do not subtract them.**)*
`positions_to_body_track` rebuilds the body on **DETAILED_HUMANOID's canonical bone
lengths**, consuming only bone *directions*; the performer's limb lengths are
discarded. Measured root-relative on the reference run, median over 13 landmarks:
**79.0 mm (subject 0) / 75.4 mm (subject 1)**, p95 **239 / 250 mm**. It decomposes:

| | subject 0 | subject 1 | what it is |
|---|---:|---:|---|
| **A** round trip on a canonical body | 0.00 mm legs · 36–46 mm arms | 0.00 · 46–47 | **converter defect** — loss on a body the rig represents exactly |
| **B** the real performer | 79.0 mm | 75.4 mm | converter + proportions |
| **B − A** | +19 to +152 mm | +19 to +185 mm | **the cost of never fitting a body** |

Performer bones against canonical: upper arm **+16 to +27 mm**, forearm **+15 to
+29**, thigh **−26 to −30**, shin **−15 to −29**; torso span pelvis→neck **+6 to
+65**. Arms too long, legs too short, on both performers.
*The converter defect is separately actionable:* the clavicle chain measures its
direction from a synthetic anchor, `pelvis + 0.72·torso_up`, then applies it as a
rotation about the rig's **own** shoulder origin. Different origins, so the
delivered shoulder direction differs from the requested one by **14.9° / 19.1°
median** and sits 1.75× too far out. Downstream bones take directions from
measured differences, so relative arm *pose* survives; the arm *root* is displaced.
*Instrument:* `tools/swap-harness/retarget_cost.py`.
*Control that can fail, and did twice:* a canonical-proportioned body must round
trip. It reads **0.00 mm** on legs, hips and neck — so the harness is sound and the
arm residual is real. The first two designs failed it, once on a wrong pelvis
definition and once because the stage **also ground-projects the character**
(`project_generated_foot_contacts`, ~90 mm of vertical shift that is *not* error).
Everything above is therefore scored root-relative.
*Blindness:* this measures the delivered rig against the *captured* positions. A
generic-proportioned character is a legitimate product choice; what this bounds is
**any MPJPE claim scored on the delivered rig**, not the animation's quality.
*Not new to the repo, new to this plan.* `docs/COMMERCIAL_MULTIVIEW_BODY.md`
already states the neutral body puts "joint endpoints … hundreds of millimetres
from the reconstructed performer" and defers per-character shaping to **P2/P3**.
That never propagated here, and this plan has been reasoning as though *"the
detector is the programme"* were the whole story.

**We already measure each performer's bone lengths and throw them away — but
putting them back is a schema change, not a code edit.** *(Two of this entry's
first-draft claims were refuted by Fable within the hour; both corrections are
folded in, and the refutations are the more useful half.)*

`estimate_limb_lengths_m` (`:937`) returns *"one set per subject per take"*, and its
**only** caller is `solve_sequence_positions:1036`, which uses them as a solver
prior. `positions_to_body_track` then rebuilds on `DETAILED_HUMANOID`'s canonical
rest offsets and the measurement is discarded. **Confirmed by grep — one call site.**

Substituting them into a scratch skeleton, root-relative, **pooled mean** across 13
landmarks (see the statistic warning below):

| arm | subject 0 | subject 1 |
|---|---:|---:|
| A — canonical body round trip *(control)* | 19.1 mm | 21.4 mm |
| B — canonical rig on the performer | 105.4 mm | 120.8 mm |
| D — own **limb** lengths | 93.7 (14%) | 112.0 (9%) |
| **C — own limb lengths + torso span** | **89.0 (19%)** | **86.2 (35%)** |

Per region, C against B: **legs 39–63%** of landmark error removed (ankles
53.0→26.1, 45.6→24.6), **shoulders 6–23%** (211.6→162.9 on subject 1), **neck up to
64%** (73.8→26.2).

**⚠ Report the mean, never the pooled median.** The landmark population is bimodal —
legs ~25 mm, arms ~180 mm — so a pooled median jumps the gap on one subject and sits
pinned on the other. The first draft of this entry read **"44%"** off that median;
the same statistic gives **2%** on subject 1. **A rank statistic over a bimodal
population is not a summary**, and this is the ninth instance of a correct
measurement carrying a claim it does not support.

*Two further corrections, both Fable's:* the first draft said torso rescaling was
**harmful** — that was a bug in the instrument, not a finding. The chain sum measures
Hips→Neck while the scoring origin hangs 80 mm below Hips, so `span/canon` forced a
guaranteed **+80 mm** overshoot; its signature was arm C's neck median reading
*exactly* 80.00 mm on both subjects. Corrected, **torso scaling is the largest single
proportion win on subject 1**, and the claim that the upper body is *"gated behind
Step 0c"* is **withdrawn as unsupported.** Both canonical torsos are too *long* for
these performers, not too short.

*Instrument:* `tools/swap-harness/retarget_cost.py`, arms A/B/C/D.
*Control:* arm A must round-trip a canonical body — **0.00 mm** on legs, hips, neck.
*Blindness:* delivered-vs-captured, not comparable to any detector number; lengths
only, nothing about dense **shape**.

*Consequence — and it is not the cheap win the first draft claimed.* **The delivered
track is bit-identical with and without the substitution** (max |Δq| 2.8×10⁻¹⁶,
Δroot 0.000 mm), because `_from_to` normalises and a scaled rest offset carries the
same direction. **The entire benefit lives in the skeleton used for FK and export,
and `BodyTrack` serialises no rest offsets.** So (a) is a **schema change** —
serialised rest offsets or a per-take skeleton asset with a version bump — plus
threading it through `body_export`, `unified_gltf` (skin matrices, bind alignment),
`body_binding` (GNM socket geometry) and `body_projection` (which builds
`provider_rest` *from* canonical offsets), plus a decision about re-binding the
delivered character mesh, which has proportions of its own.
**And the silent-failure mode is the dangerous part:** a performer-proportioned track
carries canonical joint *names*, so it validates and round-trips as canonical and is
indistinguishable once serialised. `skeleton_for_joint_names` resolves by name to one
of two canonical skeletons; any consumer that keeps doing so will silently FK
canonical and reproduce arm B exactly, **with no gate to catch it.** That gate has to
be designed before the schema work, not after.
*(b) dense shape* stays deferred to the Battle 2 body scan.

**And there is a published, fully commercially-clean system doing exactly (a) —
`Pose2Sim`.** BSD-3-Clause, RTMPose (Apache-2.0) detector, OpenSim (Apache-2.0) IK,
**no SMPL anywhere**, actively maintained. Its architecture is ours — detector →
triangulate → **scale a skeleton to the subject** → IK. Two things it gives us:
- **The pattern for the schema problem.** It carries per-subject scaling as a
  first-class model, rather than a track that silently validates as canonical.
- **A fix for our coherent bias, from the same layer.** Pose2Sim reports a
  systematic **~15° hip offset** in running — a *label-convention* error, the same
  failure mode as our ~18.6 px coherent bias — and its **scaled-skeleton IK layer
  absorbs it.** If that transfers, the body fit and the detector's convention bias
  are one build, not two, which reorders the plan's economics.
*Caveat:* its published validation is in **degrees of joint angle** (3.0–4.1° MAE),
not millimetres of joint centre — it validates the architecture, not our target.
**And do not inherit its synchronisation.** Verified in source, not documentation:
`Pose2Sim/synchronization.py` takes `argmax` over integer lags and casts to `int` —
**integer-frame only, no sub-sample peak interpolation**. OpenCap's `utilsSync.py` is
the same (`np.correlate` + `argmax`; its own docstring says *"lag: in terms of the
index"*), and the OpenCap paper publishes **no sync accuracy anywhere**. Quantised to
±0.5 frame = **±16.7 ms at 30 fps = 16.7 mm at 1 m/s and 83 mm at 5 m/s.** The sync
module imports only numpy/pandas/cv2/scipy/anytree, so it is cleanly reusable in
isolation — but add **quadratic or spline** sub-sample interpolation of the
correlation peak, never linear, which carries a documented chord-vs-arc bias.

**A visibility head is bounded by its false-positive rate, not its recall.**
Sweeping head quality on the synthetic fixture, 3 seeds: at zero false positives,
recall 0.25 / 0.50 / 0.75 / 1.00 buys +0.96 / +1.50 / +2.23 / **+2.75 mm** —
roughly linear, and cheap. (The recall-1.00 row is the same oracle-gate arm as the
+3.09 mm above, at three seeds rather than five; one arm, two sample sizes.) At full recall, a false-positive rate of 0.02 / 0.05 /
0.10 / 0.20 buys +2.34 / +1.54 / **+0.44** / **−4.27 mm**. **Break-even is between
10% and 20% false positives**, and the variance explodes with it (sd 0.52 → 2.09).
*Instrument:* `tools/swap-harness/gate_quality_curve.py`.
*Degenerate check:* recall 0.00 at zero false positives lands on the baseline to
the decimal, 11.62 against 11.62.
*Blindness:* four cameras — a joint has little redundancy to spare, which is why a
false positive costs so much; the break-even moves with camera count and this
curve does not say where. **The false positives are iid, the friendliest possible
structure**; a real head fails correlated in time and across views, taking out
several views of one joint at once, so **~10% is a ceiling under the kindest
noise, not a target.** Synthetic: "bad" is a known label here and is the very
thing being estimated on real footage.
**⚠ And this curve prices a win that does not appear on real footage.** A visibility
head used as a gate *is* a veto, and the oracle veto measured on real data is worth
**~0** (see the veto entry above). The synthetic mixture parks 7% of observations at
**σ = 36 px**, inside the 14–45 px band where the 14 px inlier gate is weakest; the
real residual is **bimodal** — 92.9% under 14 px, 3.6% in that band, 3.5% at ≥45 px
where the gate discards them for free. **Use this curve for the *shape* of the
constraint, never for the size of the prize.**
*Consequence:* **Battle 4's visibility head is bounded by a precision constraint,
not a recall target.** A head tuned the usual way, maximising recall of occluded
landmarks, walks into the harmful half of this curve. **The axis is the finding;
the ~10% is provisional until re-derived on the owned rig** — this fixture's rig
is MPI-derived and barred from shipping, so this is the second casualty of §7's
rule after the 15° majority gate.

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
| "σ and visibility cannot be ranked — the ordering reverses between two noise models" | the two arms straddled `minimum_confidence`, so one **gated** and one **weighted**. A mechanism comparison wearing a channel comparison's clothes. Same denominator, violated at the level of the estimator path. `docs/CONFIDENCE_GATE_NOT_WEIGHT.md` |

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

## 5a. The substitution ladder — what is pinned, and what is dark

**The method, restated by the user 2026-08-30:** *"rather than changing everything
at once we can use parts of MAMMA so it gives us a way to measure — if we have to
measure A and all B, C and D are new, then we'd not know where the problem is."*
That is the discipline this lane runs on and `tools/swap-harness/README.md`
credits it correctly.

**The ladder has pinned the front of the pipeline and nothing after triangulation.**
Three states, and the third is the one that matters — some darkness is a day of
work, some is a missing instrument, and confusing the two invites an
uninformative swap whose null gets read as a result.

| # | stage | what pins it | state |
|---|---|---|---|
| 1 | 2D detection | MAMMA's `ma_2d` landmarks + visibilities | **PINNED.** Its 2D into our triangulation lands 8.9 mm; ours is 7.5× its tail at p95. *This is the measured gap.* |
| 2 | subject association | `ma_2d` is **pre-associated by subject** — truth-grade labels | **PINNED 2026-08-30.** Zero switches over 300 slot-frames on a real contact clip (surfaces to 2 mm), robust to ±1-frame offset; correct pairing cheaper in 900/900 pairs. **Given a perfect detector.** Says nothing about association on *our* 2D, where 32/150 frames already defer to the exhaustive path |
| 3 | triangulation | scored against MAMMA's exact 512 landmarks via `verts_512` | **PINNED.** At the instrument's ~10 mm floor; cannot be distinguished from MAMMA's own fit |
| 4 | sequence solve | `triangulated_3d_pts` as the `world` argument | **DARK — no instrument exists.** It needs **single-ray slots**, and this fixture has **~20 of them in 5,700**. It recovers 6 of those ~20 (~30%), so it is **under-exercised, not broken.** Injecting MAMMA's clean 3D exercises nothing at all |
| 5 | fill + Savitzky-Golay smooth | — | **DARK — both references banned.** Touches every joint, moves triangulated ones 5.6–7.9 mm median with a tail past 0.9 m, and is the de facto outlier filter. MAMMA's fitted mesh is itself 2.6–2.9× smoothed; `triangulated_3d_pts` is correlated with the 2D being smoothed |
| 6 | positions → rotations + ground projection (`positions_to_body_track`, :1296) | round trip on a canonical-proportioned body | **MEASURED 2026-08-30.** 79.0 / 75.4 mm root-relative median, p95 239 / 250 mm — the largest displacement between what we reconstruct and what we deliver, **not comparable to the detector gap** (different reference, different population). Analytic and closed-form; **no IK solver, no optimiser.** Of that, **36–47 mm on the arms is a converter defect**, not proportions — the clavicle direction is applied about the wrong origin, 14.9–19.1° off |
| 6b | **parametric body-model fit** | MAMMA fits a **dense SMPL-X shape per subject** | **DOES NOT EXIST**, and it is the majority of row 6. Nothing in the delivery path fits MHR, SMPL-X or any shape model — we drive a **generic mesh, never fitted to the performer**: arms +16 to +29 mm too long, legs 15–30 mm too short. Already scoped as **P2/P3** in `COMMERCIAL_MULTIVIEW_BODY.md`. SAM 3D Body is the repo's only MHR fitter and is **off-path with no caller** |
| 7 | calibration, sync, lens distortion | MAMMA's rig — **MPI-derived, barred from shipping** | **NOT OWNED.** Half a frame at 30 Hz costs 29–48 mm at our joint speeds, and MAMMA's own genlocked rig still slipped two frames. Never measured on anything of ours |

**Why the end-to-end number and the pinned number must not be subtracted.** Our
pipeline sits **60.83 mm median / 184.43 mm p95** against MAMMA on subject 0
(79.83 / 221.15 on subject 1), while our triangulation given MAMMA's 2D sits at
**8.9 mm**. Two warnings before anyone uses those figures. **They are scored on the
*sparse reconstructed joints*, not on the delivered rig** — row 6 adds another
75–79 mm root-relative on top. And **the script that produced
`evaluation-vs-mamma.json` is not in the repo**: grep finds no producer of
`all_median_mm`, so the artifact cannot be regenerated, against this project's own
rule that everything under `artifacts/` must be. Treat it as a historical note
until a generator exists. They are also **not the same denominator**: the first compares 19 joints of *our* convention through IK
against MAMMA's fitted joints, the second compares MAMMA's own 512 landmarks to
themselves. The hips read 86–96 mm in that comparison, which is the signature of
**joint-definition offset, not error**. The difference between the two numbers is
a *sum* of detector tail, convention offset, **un-fitted body proportions**,
smoothing lag and sync — **and separating that sum is exactly what rows 2, 5, 6,
6b and 7 are for.** Quoting "10 mm versus 61 mm" as a gap would be this lane's
signature defect.

**And row 6b reframes the lane's own thesis.** The plan says *"the detector is the
programme"*, which is right about the *detector's* contribution. But MAMMA fits a
dense shape to each performer and **we never fit one at all** — so a share of that
60.83 mm is a generic mesh worn by a specific body, and **no amount of detector
work can reach it.** How large a share is unmeasured. That question is cheap to
ask — row 6 is deterministic, so MAMMA's own joint positions can be pushed through
our converter and the residual read directly — and it has never been asked.

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

**Synchronisation — settled by research 2026-08-30, in priority order.**

1. **Wire a GPIO hardware trigger, and if the cameras accept one, stop there.**
   Basler measures **<100 ns** inherent jitter; the Stanford multi-camera array held
   **200 ns** across the whole rig. Use **GPIO (<0.5 µs line delay), not
   opto-isolated (4.5–28 µs)**, and drive a fast edge — a slow edge converts
   amplitude noise into timing jitter. This beats every software method by three to
   four orders of magnitude and makes the rest of this list unnecessary.
2. **If not, flash sync — and it inverts the rolling-shutter finding.** A 1–5 ms
   flash illuminates only a *band of scanlines* on a rolling sensor, and the band's
   row index encodes sub-frame phase directly. Šmíd & Matas (VISAPP 2017) measure
   **0.3–0.5 ms** std dev, **MIT licence**
   (`github.com/smidm/flashvideosynchronization`), no custom hardware — 2.7× better
   than RocSync and **10× inside our ≤5 ms budget**.
   **So rolling shutter is not only the liability; it is also the cheapest instrument
   for verifying sync — and global shutter removes both.** That is a real tension in
   the sensor decision: resolve it by shooting a flash calibration during a
   rolling-shutter pilot to measure the row-time term first, then choose the
   production sensor. RocSync (MPL-2.0, 1.34 ms) is the fallback for global-shutter
   or mixed rigs.
3. **Lock exposure manually.** Auto-exposure re-times the shutter and desynchronises
   a nominally synced rig — Azure Kinect's own documentation warns it *"quickly
   pushes the cameras out of sync"*, and it commits to 100 µs of centre-of-exposure
   only with exposure locked.
4. **If footage-based sync is ever built, build the STBA family, not pairwise
   epipolar.** Our ~90°-separated four-camera ring is precisely where the pairwise
   methods fail: Meyer et al. hold frame accuracy only to **45°** separation, and
   Elhayek et al. got 0.24–0.35 frames on most cameras but **2.61 frames** on the one
   with a distinctly different viewpoint. STBA fuses all views through a motion prior
   instead, and its own divide-and-conquer *"becomes stable after four cameras"* —
   our exact count. Its honest accuracy is **±0.12 frames** (4.0 ms at 30 Hz), and
   **no source code has ever been released**, so there is no licence question.
5. **Two free validation instruments, both of which fail loudly** — and both satisfy
   the no-gate-a-constant-can-pass rule: Meyer's **loop-closure check** (offsets
   summed around every camera pair must be zero, which a constant offset cannot
   satisfy), and OpenCap's **reprojection-peak selection** among the three
   correlation peaks nearest zero lag, which disambiguates periodic motion.

*Ranked below these:* audio sync is limited by acoustics, not electronics — 1 m of
mic separation is **2.915 ms**, and audio and video are encoded separately so
microsecond audio alignment can still sit a frame out in video. Timecode is the worst
measured option. Head-to-head on reprojection error: LED+interpolation **1.02 px**,
flash **3.09**, audio **11.34–12.90**, timecode **54.60**.

**Two further rig-spec additions, 2026-08-30.**

- **Camera count: withdrawn as a rig-spec change.** Briefly promoted here on a CRLB
  improvement that does not transfer — MAMMA's own ablation is flat from 4 to 8
  cameras, and the research doc already rates it *"the smallest term on the list"*
  (§3). If more cameras are bought, buy them for **two-person** (+3 mm), **hands**,
  or **pseudo-label purity**, and never for single-person bulk accuracy.
- **Pre-register how markers map to our 19 joints, before the session, in writing.**
  Raw marker trajectories are *surface* positions; joint centres come from a
  regression protocol carrying its own error and a **third** joint convention. The
  checklist correctly asks for raw trajectories and says nothing about the mapping.
  **Without this the session discovers joint-definition offset as "error"** — this
  lane's signature defect, waiting at the finish line, on the one instrument meant to
  settle everything. One page now is the whole fix.

*Until this lands, every band below is a consistency band, not an accuracy band.*

### Step 0a — **veto ceiling DONE 2026-08-30; the census is what remains**

> **The veto half is answered: ~0 mm, and SAM 3D closes.** Result, controls and
> blindness in §3. The mechanism — the inlier gate already discards 98% of flagged
> observations — is the durable part; the millimetres are the evidence for it.
>
> **The census half was not run, and should now be re-ranked below Step 4.** Its
> stated purpose was to tell Battle 3 what to synthesise and Battle 4 what to be
> robust to — but the same pass showed the tail's 3D effect is already absorbed by
> the inlier gate, so **the detector gap that remains is the bulk and the coherent
> bias** (§10). Classifying tail events characterises the half that no longer
> drives the number. Step 4's pseudo-labels attack the half that does. **Do not run
> the census because it is written here**; the four redesigns below bind if it ever
> does run.
>
> One census-shaped finding fell out for free: the debiased residual jumps from
> **p95 18.3 px to p99 194.0 px** at 1280. Whatever the tail is, it is a small
> population of very large failures, not a heavy shoulder.

*Original specification, retained — the redesigns bind if the census is run:*

Both advisors ranked this second, behind only the booking, and **both refused it
as written in §9a.** It is the direct continuation of the gate/weight correction:
gating is the one live mechanism, and this prices perfect gating on *real* data.

**The prior has moved against §9a's own guess.** §9a expected the veto ceiling to
be "plausibly under ~2 mm", which would close SAM 3D for free. The synthetic
oracle drop is worth **+3.09 mm**, so the question is now genuinely live rather
than confirmatory. Do not pre-write the conclusion.

*Four redesigns, pre-registered, all mandatory:*

1. **Debias before thresholding.** A 30 px threshold against fit projections
   censors by direction: with ~18.6 px of coherent bias, an event aligned with it
   enters the census at ~11 px of incoherent error while an opposed one needs
   ~49 px. Subtract the **per-joint, per-camera median offset** first, and/or
   raise the census threshold to ≥3× bias (~60 px). **Say plainly that this is
   first-order only** — convention offsets project pose-dependently.
2. **Correspondence limits what may be published.** Our SOMA-77 names against
   MAMMA's 512 landmarks will not support a per-joint *magnitude* table. Publish
   per-joint **event counts**; classification needs only gross geometry —
   occlusion from `ma_masks`, left/right swap at ≈ shoulder-width, person
   confusion by other-subject proximity, box failure.
3. **Assign detections to subjects with `ma_masks` or MAMMA's `ma_2d` labels —
   never by root proximity.** The 160 px root-proximity rule is in WITHDRAWN as
   the recorded contaminant of the epipolar tail; a census that re-imports it
   re-imports the defect. `tools/swap-harness/association_swap.py` already carries
   the label-matching machinery for this and can be reused directly.
4. **The veto denominator is the baseline's, not the veto's.** Vetoing changes
   which points triangulate, so scoring survivors is the composition shift that
   has fooled this lane twice. Score the **slots triangulable in the baseline
   arm**, wherever they land in the veto arm (triangulated, recovered, or
   interpolated — post-fill positions), and **publish the newly-untriangulable
   count beside the millimetres.**

*Bands, pre-registered before the measurement:*
ceiling **< 2 mm** ⇒ no veto of any provenance gets built, SAM 3D's occlusion
prior included, and SAM 3D closes. **≥ 5 mm** ⇒ veto/visibility work becomes a
first-class Battle 4 requirement and the owned fixture's shoot list goes
occlusion-heavy. The census must place **≥90% of >60 px events into named
classes**; if "unclassified" dominates, the reference cannot read the tail and
**nothing is reported.**

*Degenerate solutions the bands must reject:* a **shuffled veto** of equal count
must not reproduce the oracle's delta — without that arm, report nothing; and a
survivor-only denominator shows improvement by attrition, which is why the
denominator is fixed above.

### Step 0b — land the weighting diff and its harness together *(small, and HEAD is broken)*

`scripts/measure_detector_outputs.py` is committed while the `weight_before_loss`
parameter it passes is not, so **a clean `HEAD` checkout raises `TypeError`** —
verified, no `**kwargs` catchall in either signature. Commit them together.

Rewrite the diff's comment before it lands: *"Measured on the synthetic fixture
before changing the default"* describes a measurement this fixture **structurally
cannot make**. **Do not flip the default.** And add the test the correction
implies — a constructed **single-ray** slot where the two orderings measurably
differ — so the flag stops being dead code guarded by an unfalsifiable comment.

### Step 0c — **ATTEMPTED 2026-08-30, band failed, reverted. The defect is one level up.**

> **What was tried.** Measure every chain's direction from the joint's own
> forward-kinematic origin instead of from a synthetic anchor — the principled fix,
> since the child lands at `origin + rotated_offset` and the direction must therefore
> be measured *from that origin*.
>
> **What happened.** The canonical round trip went **0.00 mm → 46–67 mm on the legs**
> and 36–47 → 61–78 mm on the arms. Reverted. **The band existed to catch exactly
> this, and it did — on the first run, before any of it reached a conclusion.**
>
> **Why, and this is the finding.** *No rig joint origin coincides with its captured
> landmark.* `root_translation = pelvis − (0, 0.98, 0)` places **Hips at the captured
> pelvis**, while the upper legs hang **80 mm below and 90 mm to each side** of Hips
> by their rest offsets. So a leg direction measured from the rig's own hip origin
> starts 80 mm displaced. (The rig's hip separation is 180 mm against a **measured
> 207 mm** on this performer — the same mismatch from the other side.)
>
> **So "fix the clavicle anchor" was the wrong target.** The clavicle's synthetic
> `pelvis + 0.72·torso_up` anchor is a *symptom* of a rig whose joint origins are not
> placed to match the capture; fixing the symptom while the origins are wrong made it
> worse. **The real Step 0c is the root/hip placement convention** — place the rig so
> its hip joints land on the captured hips, then the origin-based directions become
> correct for every chain at once.
>
> *That change moves `root_translation`, which the ground projection and everything
> downstream consume, so it is no longer the small self-contained fix this step was
> written as.* It also now overlaps the per-performer skeleton work: the 180-vs-207 mm
> hip separation is a **bone length we already measure and discard**, and
> `RIGID_LIMBS` already contains `("left_hip", "right_hip")`.
> **Re-scope it together with the skeleton-scaling build rather than ahead of it.**

*Original specification, retained — the band stands and should be reused:*

The only part of the 75–79 mm retarget cost that is **not** the missing body fit.
`positions_to_body_track` measures the shoulder direction from a synthetic anchor,
`pelvis + 0.72·torso_up`, then applies it as a rotation about the rig's **own**
shoulder origin — different origins, so the delivered shoulder direction is
**14.9° / 19.1° off** and lands 36–47 mm from the requested landmark **even on a
body with canonical proportions.**

*Band, pinned before anyone writes the fix:* arm A of
`tools/swap-harness/retarget_cost.py` — the canonical round trip — must reach
**≤ 5 mm median on all four arm landmarks, on both subjects**, with legs, hips and
neck **unchanged at 0.00 mm**. Not "improved": the legs prove an exact fixed point
is reachable whenever the direction is measured between the actual landmark pair,
so 36 → 30 mm is a failure, not progress.
*Degenerate solutions:* tuning 0.72 until subject 0 improves — it must hold on both
performers, and the round trip is subject-independent, so a constant that fits one
body is visible immediately. And a fix that merely *moves* error into the elbow or
wrist passes a shoulder-only band, which is why the band names all four landmarks.
*Blindness:* this closes the converter defect and **not one millimetre of the
proportion mismatch**, which is P2/P3. Do not let a good result here be read as
progress on row 6b.

### Step 1 — association under contact — **DONE 2026-08-30, band met, do not rebuild**

> Result and its blindness are in §3. **Zero switches, margins published, contact
> halves the median margin without inverting a single decision.** The band said
> "build only if it fails"; it did not fail, so **nothing is built here.**
>
> **What the result does not license.** It was scored on MAMMA's landmarks, which
> are projections of a fitted mesh — a *perfect* detector. Association on our own
> 2D is untouched and is the case that actually defers 32 of 150 frames to the
> exhaustive path. If association work is ever revisited, that is the target, and
> it needs no MAMMA asset at all.
>
> **And it did not fix the harness defect** the original "why first" claimed it
> would. The 160 px root-proximity rule is still live wherever it is used; this
> measurement simply did not go near it. That obligation moved to **Step 0a
> redesign #4**, which now has this script's label-matching machinery to reuse.

*Original specification, retained for provenance:*

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
provisional until re-derived. Second: the visibility head's ~10% false-positive
break-even** — measured on this fixture's MPI-derived rig, so the axis ships and
the number does not.

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
| synthetic fixture, corrected noise | sub-mm, truth-referenced; the **eligibility gate** and **DLT row weighting** inside `triangulate_point` | calibration, distortion, sync, soft tissue, joint definition — **and `solve_sequence_positions` entirely**: its failed slots carry zero retained rays, so `candidate` is empty, the solve early-returns and its objective is never exercised. **Exercising it needs single-ray slots, not occlusion** | any "our MPJPE" claim; **detector pricing**; **pricing anything that acts through the sequence solve**, the robust-loss weight ordering included |
| any confidence-channel arm | the mechanism its values actually trigger | the gate/weight distinction, unless the gate position is held fixed across arms | **comparing two arms that straddle `minimum_confidence`** |
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
- **OpenPose is a hard licence block, and a common web summary is wrong about it.**
  Its LICENSE reads *"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE
  ONLY"* and — worse — *"all and any such derivatives and modifications will be owned
  by Licensor."* It does not merely bar commercial use; it claims ownership of what
  we build on it. Summaries calling it Apache-2.0 are incorrect. Pin detector
  provenance the same way MHR's is pinned: ViTPose/RTMPose are Apache-2.0, and
  academic-benchmark weights (Human3.6M, InterHand) are poisoned for shipping.
- **Every number this lane owns was taken on a global-shutter, hardware-synced rig
  we do not own.** So our results inherit *none* of the sync or rolling-shutter error
  — and an owned rig will be worse than the fixture unless it matches that
  discipline. This is not a caveat on the measurements; it is a floor on the owned
  build.
- **Surface-visibility semantics must never be fed to a consumer that assumes
  skeletal-joint semantics.** Feeding MAMMA's `visibilities` — a *surface* mask,
  where a landmark facing away reads ~0 — as the confidence leaves only **2.4% of
  camera pairs scorable** (22 of 900) and aborts association outright. The
  arithmetic says why, and says it is **not** a result about visibility channels:
  the mask passes 45.9% of points per view, so a camera pair expects
  0.459² × 19 = **4.0** shared points against `minimum_shared_joints=4` — exactly
  on the threshold, and opposing views are anti-correlated. A **per-joint**
  occlusion head at the measured 7% bad fraction would leave 0.93² × 19 ≈ **16**.
  Do not read this as "an occlusion channel breaks association."
- **Every confidence channel must be checked against all three gates before it
  ships**, because they fire independently: `triangulate_point` eligibility
  (`:319`), the epipolar shared-joint mask (`:621`), and the sequence-solve `seen`
  mask (`:1022`). Today the gate/weight correction was `:319`, and the association
  abort was `:621`. A channel priced through one of them has not been priced.
- **`positions_to_body_track` does not only retarget — it ground-projects.** It
  ends in `project_generated_foot_contacts`, which plants the character on the
  floor and shifted the root ~90 mm vertically on the reference run. **Score it
  root-relative**, or that shift reads as error. It cost two harness designs.
- **`artifacts/.../evaluation-vs-mamma.json` has no generator, and never had one.**
  Nothing in the working tree produces `all_median_mm`, and
  `git log --all -S all_median_mm` returns **nothing** — so it was not deleted, it
  was **never committed.** The 60.83 / 79.83 mm figures quoted in
  `COMMERCIAL_MULTIVIEW_BODY.md` came from an uncommitted script: the method is
  unknown and unrecoverable. Do not build on them; re-derive if the number matters.
- **`valid_joint_fraction` and `interpolated_joint_fraction` are dominated by a
  constant.** SOMA-77 emits **no ear landmarks**, so 2 of our 19 joints are missing
  in every frame of every subject: 600 of the 620 filled slots on the reference
  run. `valid_joint_fraction = 0.891` against a structural ceiling of **0.895**.
  **Neither number is a quality signal**, and `constraint_recovered_joint_fraction`
  is quoted on the same poisoned denominator — the solve recovers 6 of ~20 eligible
  slots, not 6 of 5,700. Score coverage on the joints the detector actually emits.
- **The detectors disagree definitionally, not just noisily.** MediaPipe *derives*
  neck and root as midpoints where Apple Vision predicts them; SOMA-77 predicts
  interior joint **centres** where the others predict **surface** landmarks, and
  has no ears and no true nose (it substitutes Head). Any cross-detector comparison
  inherits these as **bias**, not variance.
- **A confidence below `minimum_confidence` is a different mechanism from one just
  above it.** Below, the observation is *dropped* from `triangulate_point`; above,
  it is merely re-weighted, and re-weighting is close to inert. Two arms that
  straddle the threshold are not two channels. **Hold the gate fixed, or you are
  measuring the gate.**
- **`solve_sequence_positions` is a recovery pass, not a re-estimator.** It returns
  triangulation on every slot that triangulated, and early-returns before
  `least_squares` when nothing failed. Any change to its objective is invisible on
  a fully-visible fixture — verified bit-identical. Score it only where slots
  actually fail to triangulate.
- **`scripts/measure_detector_outputs.py` is committed; the `weight_before_loss`
  parameter it passes is not.** A clean `HEAD` checkout raises `TypeError` — no
  `**kwargs` catchall in either signature. Land them together.

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

**Reconciled verdict: neither helping nor diverting — it was mandatory de-risking,
and it is now complete.** The counterfactual was never "skip it". It was an untested
*"shouldn't we be using SAM 3D?"* haunting every architecture decision from here to
Battle 4. One session bought its permanent removal.

**And one item was on the wrong side of the ledger.** The withdrawn negative is not a
SAM cost. Evaluating SAM **exposed a defect in the ladder harness** — association
contamination and unequal frames — that would have silently mismeasured the *next*
cross-family comparison too, including any Battle 4 candidate ranked against SOMA-77.
Finding that on a low-stakes comparison is a benefit, and I had it filed as a cost.

**The honest ledger.** All three of those doors were live
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

**Standing decision — and it is stricter than the one I first wrote.** I had
pre-committed to "one more pass" on hands as a guard against iterating. Fable pointed
out that running it *at all, now,* is itself the sunk-cost move in the opposite
direction: **hands are Battle 5 by this plan's own ordering**, and testing them today
would be retroactive justification for the detour. Accepted.

**So: zero days on SAM 3D hands now.** Record the pointer and walk away —
*"SAM 3D hand-root direction 9.0° from positions against 3.2°; test against the
27-DoF chain that reached 33.9 mm held-out"* becomes **Battle 5's first
pre-registered rung**, and nothing more.

**And the direction channel narrows once more.** With head at 3.5° from our own
positions, head drops out alongside the limbs. Hands defer to Battle 5. **The
channel's entire near-term content is feet.** That is the honest residue.

> **Superseded 2026-08-30 by Step 0a.** Both advisors accepted the *aim* of the
> two paragraphs below and **rejected their design** — the 30 px threshold is
> direction-censored by the coherent bias, the per-joint magnitude table is not
> supported by the joint correspondence, and the veto measurement is
> survivorship-biased by construction. Step 0a carries the redesign and the
> pre-registered bands. **The guess below that the veto ceiling is "plausibly
> under ~2 mm" is no longer the prior**: a perfect oracle gate is worth +3.09 mm
> on the synthetic fixture, so the question is live, not confirmatory.

**What to do instead — and it acts on the one number that is the gap.** The 98-vs-13
px p95 is currently a *ratio*, not a design spec. **Classify the tail events** on the
existing fixture — occlusion, left/right swap, person confusion inside the crop, box
failure — using `ma_masks` and the fit projections as reference. That converts "7.5×
at p95" into *what Battle 3's synthetic corpus must contain and what Battle 4's
training must be robust to*. Solo-executable today.

**And measure the ceiling first, because it may close SAM 3D entirely for free.**
Flag our observations with >30 px residual against the fit projection, check whether
`triangulate_point`'s subset enumeration actually *used* them, and compare the 3D
error of tail-contaminated against clean triangulations. That number is the value of
a **perfect oracle veto**. If it is under ~2 mm — plausible, since the gate already
rejects single-view outliers when four views exist, and the +0.16 mm visibility result
points the same way — **then no veto of any provenance is worth building, SAM's
occlusion prior included, and SAM 3D closes without spending the day.** It falls out
of the same pass as the attribution study.

> **ANSWERED 2026-08-30: yes, and the mechanism guessed above is exactly right.**
> Oracle veto worth **+0.05 mm median / +1.47 mm p95** at a 60 px threshold, with a
> shuffled control failing at −0.34 / −1.01 mm. **Only 1.9% of flagged observations
> were ever consumed by the inlier subset** — the gate had already discarded them.
> Full result, controls and blindness in §3.
>
> **So SAM 3D closes, on the criterion this section set for itself.** Its one
> surviving candidate role was hands, deferred to Battle 5; the occlusion-prior
> role is now measured at ~0 and is withdrawn outright. Nothing further is owed to
> this integration.
>
> Credit where due, stated precisely. The *reasoning* above — "the gate already
> rejects single-view outliers when four views exist" — was right, while the
> synthetic fixture had meanwhile priced an oracle **gate** at +3.09 mm and made it
> look wrong. **The fixture was not blind to the gate** — it runs the same
> `triangulate_point`. Its *noise model* parked 7% of observations at σ=36 px,
> inside the 14–45 px band where that gate is weakest, while the real tail is
> bimodal and mostly outside it. **A synthetic result is only as good as the noise
> model's agreement with the real error's *shape*, not just its quantiles** — the
> mixture matched our deciles and still put the mass in the wrong place. That is the
> lesson, and it is a new one; §8's fixture blindness is about the sequence solve
> and is a different failure.

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

**Amended twice on 2026-08-30. First, the tail is the wrong half of the detector
gap.** *"What separates us is the detector, and specifically its tail"* — the tail
part no longer survives. A **perfect** veto of every tail observation is worth
**+0.05 mm median** on real footage, because `triangulate_point`'s inlier gate
already discards 98% of them. The substitution gap has not moved: MAMMA's 2D into
our triangulation still lands at 8.9 mm against our own detector's 24.7 mm spread
plus 33.9 mm bias. So what actually separates us is **the bulk and the coherent
bias** — p50 9.33 px against 4.29, about 2.2×, plus ~18.6 px of convention offset —
**not the tail.** The tail is the most visible difference in 2D and the least
consequential in 3D. It still matters for hands, for coverage, and in the
two-eligible-view regime (35.3% of slots).
**This raises Step 4.** Pseudo-labels on owned footage kill the coherent bias *by
construction* and attack the bulk directly — they are aimed at what remains, and
nothing else in the plan is.

**Second, and separately: that paragraph describes the *reconstruction*. It does
not describe what we ship.** Between the reconstructed
joints and the delivered character sits a retarget onto a **generic body that is
never fitted to the performer**, and it costs **75–79 mm root-relative median,
p95 239–250 mm** — larger than the detector gap the whole lane is organised
around. Roughly 36–47 mm of that, on the arms, is a converter defect and fixable
now; the rest is the absent body-model fit, which is P2/P3 work and which **no
detector improvement can touch.** So the sentence *"the detector is the
programme"* is true of the reconstruction and **silent about the delivered
character** — not false, silent. Retargeting onto a differently-proportioned body
*is* what retargeting does, and positional mismatch with the performer is partly
correct behaviour rather than error. Whether row 6 matters to the product depends
on what the product promises: **performer-faithful positions** (contact, props,
two-character staging) or only **faithful motion**. That question has never been
asked here, and it decides whether P2/P3 is a nice-to-have or a requirement.

**And it does not move the parity timeline.** Battle 2 scores markers against the
**reconstruction**, which is where MAMMA's 13.5 mm lives too. Row 6 sits between
the reconstruction and the shipped character. It is a product-side gap with a
planned fix, not evidence that parity is further away than believed.
