# Research pass — the next builds, 2026-08-30

**Why now, and it is not "we have accumulated results".** The lane's bottleneck
changed *kind* today. Until now it was measurement-limited and the substitution
ladder could answer questions. That ladder is close to exhausted: the three
pinnable stages are pinned, the two dark stages have **no instrument** and need a
fixture that does not exist, and the two missing stages need **building**. Every
remaining move costs engineering weeks, which is exactly when a week of research is
cheap. And **the aim moved today** — the detector's *tail* is not the driver, the
bulk and the coherent bias are — so every research document in this repo was
written against a thesis that no longer holds.

**The standing risk, named up front:** a research pass can become another SAM 3D —
scouting that closes doors without advancing the mission. Each question below names
the decision it would change. Anything that would change no decision is not here.

**Claim tags, so verification is real rather than ceremonial.**
**[T]** derived from evidence in this repo or measured today — attack it.
**[R]** recalled, not verified — treat as a coin flip until checked.
**[D]** a document that must be *read*, never recalled.

---

## 1. The body fit decomposes, and half of it is free — MEASURED TODAY

**[T] The pipeline already measures the performer's own bone lengths and then
throws them away.** `estimate_limb_lengths_m` (`commercial_multiview.py:937`)
returns *"one set per subject per take"* — and its **only** caller is
`solve_sequence_positions:1036`, which uses them as a solver prior.
`positions_to_body_track` then rebuilds the body on `DETAILED_HUMANOID`'s canonical
rest offsets and the measurement is discarded. Verified by grep: there is no other
call site.

**Measured consequence — CORRECTED after Fable's verification. The first draft of
this section got two things wrong, and the corrections matter more than the
original claims.**

Pooled **mean** across 13 landmarks, root-relative, 150 frames, per performer:

| arm | subject 0 | subject 1 |
|---|---:|---:|
| A — canonical body round trip *(control)* | 19.1 mm | 21.4 mm |
| B — canonical rig on the performer | 105.4 mm | 120.8 mm |
| D — own **limb** lengths | 93.7 (14%) | 112.0 (9%) |
| **C — own limb lengths + torso span** | **89.0 (19%)** | **86.2 (35%)** |

Per region, C against B: **legs 39–63%**, **shoulders 6–23%**, **neck up to 64%**.

> **CORRECTION 1 — the "~44%" headline was a statistic artifact.** It was the pooled
> *median*, and the landmark population is bimodal (legs ~25 mm, arms ~180 mm), so the
> median jumps the gap on subject 0 and sits pinned on subject 1 — where the same
> statistic reads **2%**. Report the mean. *(Also: the first draft's "53.0 / 61.1"
> pairs were **left/right of subject 0**, presented as if they were subject 0 /
> subject 1.)*
>
> **CORRECTION 2 — "torso rescaling is actively harmful" was a bug in the
> instrument.** The chain sum measures Hips→Neck while the scoring origin hangs 80 mm
> below Hips, so `span/canon` forced a guaranteed **+80 mm** overshoot — its signature
> was arm C's neck median reading *exactly* 80.00 mm on both subjects. Corrected,
> **torso scaling is the largest single proportion win on subject 1** (neck
> 73.8 → 26.2 mm), and the claim that the upper body is *"gated behind Step 0c"* is
> **withdrawn as unsupported.** Both canonical torsos are too *long* for these
> performers.
>
> **Fable also under-credited me in one direction:** hip width is already in the same
> lengths dict (`RIGID_LIMBS` includes `("left_hip","right_hip")`), and substituting
> it takes hips 18.9 → 9.5 mm with knees and ankles a further 2–6 mm. The measurement
> was under-used, not over-read.

*Instrument:* `tools/swap-harness/retarget_cost.py`, arms A/B/C/D.
*Control:* arm A, a canonical-proportioned body, must round-trip — reads **0.00 mm**
on legs, hips and neck.
*Blindness:* delivered-vs-captured, a different reference and population from any
detector number. And it says nothing about dense **shape** — only lengths.

**What this changes.** "P2/P3 parametric body fit" was carried as one indivisible
build. It is two: **(a) skeleton scaling from lengths we already compute — owned
code, no new data, no fixture, ~44% of the gap**, gated behind Step 0c for the
upper body; and **(b) dense shape**, which stays deferred.

**[T] Dense shape: defer to the Battle 2 body scan, already in the rig spec.**
Estimating shape from images re-opens the monocular inference the SAM 3D verdict
closed. Do not.

**Blast radius — CHECKED, and it is a variant, not an edit.** Proof by bit-identity:
the delivered track is unchanged by the substitution (max |Δq| 2.8×10⁻¹⁶, Δroot
0.000 mm), because `_from_to` normalises and a scaled rest offset keeps its
direction. **The whole benefit lives in the skeleton used for FK and export, and
`BodyTrack` serialises no rest offsets** (`body.py:446`).

So §1(a) is **not** "owned code, no new data". It is:

1. a **BodyTrack schema change** — serialised rest offsets or a per-take skeleton
   asset, with a version bump;
2. threading it through `body_export.py:436` (plus canonical-rest alignment against
   the MPFB armature), `unified_gltf.py:169,461` (skin matrices, bind alignment),
   `body_binding.py:461` (GNM socket geometry), `body_projection.py:377–390` (builds
   `provider_rest` *from* canonical offsets);
3. a decision about **re-binding the delivered character mesh**, which carries
   proportions of its own — unscoped, a dimension rather than a number.

**And the silent-failure mode is the dangerous part.** A performer-proportioned track
carries canonical joint *names*, so it validates and round-trips as canonical and is
indistinguishable once serialised. `skeleton_for_joint_names` (`body.py:331`) resolves
by name to one of exactly two canonical skeletons, and `validate_body_track`
(`body.py:745`) checks against canonical. Any consumer that keeps resolving by name
silently FKs canonical and **reproduces arm B exactly, with no gate to catch it** —
while generated lanes (`speech_motion.py:523`, `soma_motion.py:759,797`) keep emitting
canonical tracks, so composition would blend two proportion systems and raise nothing.
**Design that gate before the schema work, not after.**

**One ambiguity to resolve first:** §5a says the vs-MAMMA figures are "scored on the
sparse reconstructed joints, not the delivered rig" *and* that they compare "19 joints
of our convention **through IK**". Those disagree, and **whether this build moves the
parity number at all depends on which is true.**

---

## 2. The pseudo-label campaign — design

Top engineering priority, not begun, and now the only thing aimed at what remains
of the detector gap.

- **[R] Labels by leave-one-camera-out reprojection** — triangulate from N−1 views,
  supervise the held-out view. Breaks self-confirmation for view-specific error.
  **Residual risk it cannot touch:** view-consistent appearance bias, this plan's
  own "agree and be wrong together". Only data diversity attacks that.
- **[T] Label gating, from today's numbers:** require **≥3 *used* views** — 35.3% of
  slots are 2-eligible and can reject nothing, so labels there are unfiltered;
  filter on `used_camera_indices` / `reprojection_errors_px`; reuse
  `association_swap.py`'s margin machinery to gate assignment; train on
  **unsmoothed** triangulation (already pre-registered); stratify by speed.
- **[D] Fine-tune SOMA-77 versus train fresh is licence-gated, not technically
  gated.** Do not decide it from recall. Read `NVIDIA_BODY_DEPENDENCIES.md` and the
  archived OML text bound to our revision. **If it is not archived, that gate is
  unmet and that is the finding.** The checklist's backbone question — whether a
  DINOv3 initialisation carries its own terms — is still unanswered. If derivative
  commercial weights are blocked, Step 4 couples to Battle 3's corpus.
- **[T] Evaluation without truth, pre-registered now:** the epipolar ladder reads
  **spread only**, which is acceptable *here* because the bias is killed by
  construction and that claim is structural rather than numerical; LOO reprojection
  on held-out cameras; the synthetic fixture only with a **shape-matched** noise
  model. Markers remain the only accuracy read.
- **[R] Failure modes beyond the recorded traps:** easy-pose imbalance, distribution
  narrowing to one rig/wardrobe/lighting, catastrophic forgetting under fine-tuning,
  teacher–student collapse.
- **Data volume: no number is offered.** Diversity beats volume, and the corpus
  licence emails in `BATTLE2_ACTION_CHECKLIST.md` are the parallel path. **Do not
  invent a figure.**

---

## 3. Is the substitution ladder exhausted?

**[T] For decisions, yes. Two evaluation-only slots remain, both unranked:**

- MAMMA's **visibility channel** can *evaluate* a Battle 4 visibility head, never
  train it — §7 bars anything MAMMA-derived from shipped weights.
- **`contacts` / `floor_contacts`** (in both `ma_2d` and `smplx_params`) could score
  `project_generated_foot_contacts`, whose ~90 mm root corrections have never been
  checked against anything. No imminent decision hangs on it.

**[T] The next real instruments must be owned.** A fixture with deliberate
**single-ray slots** is the only thing that can ever measure the sequence solve, and
known-length objects are the only calibration check that does not overfit its own
recording. **That makes the owned fixture load-bearing for three separate
questions** — the sequence solve, calibration, and Step 4's training data — not just
for Battle 2. It is the highest-value build that does not need the marker session.

---

## 4. What the aim shift invalidates

- **[T] Battle 3's synthetic corpus re-aims** from occlusion-heavy to pose and
  appearance diversity, and gains a pre-registerable gate: **any synthetic corpus
  used to price detector work must reproduce the real error's *shape*, not just its
  deciles.** Today's mixture matched our deciles and still put the mass in the wrong
  place.
- **[T] The tail keeps exactly two reservations:** hands, at ~20 px crops, and the
  2-eligible-view regime.
- **[D] To be checked:** grep `OWNED_BODY_CAPTURE_PLAN.md` and
  `SHELLS_STAGE0_PLAN.md` for bands or costings priced on tail metrics. Those are
  the entries the shift silently invalidates.

---

## 5. Verification — what survived Fable's check

**§1 discard claim — CONFIRMED.** One call site, verified by grep.
**§1 blast radius — CONFIRMED as a variant**, by bit-identity. See §1.
**§1 headline "~44%" — REFUTED.** Statistic artifact; see Correction 1.
**§1 "torso harmful" — REFUTED.** Instrument bug; see Correction 2.

**[D] The NVIDIA OML gate is UNMET — exactly the branch that was pre-registered.**
`NVIDIA_BODY_DEPENDENCIES.md:188–206`: *"No local copy of that agreement was found
tied to a GEM-X or Kimodo model revision"*; register entry `N0-LIC-003` at `:305`.
Nothing quotable exists in-repo — only headline declarations that commercial use and
derivative-model rights are *subject to conditions*. SOMA-77 is GEM-X's bundled
`Dinov3_ViTPose_huge_metrosim_256x192` under the OML
(`OWNED_BODY_CAPTURE_PLAN.md:122–124`), and the DINOv3-initialisation question is
confirmed open — the enquiry to NVIDIA is unsent or unanswered
(`BATTLE1_INCREMENT4_SOMA77_DETECTOR.md:105–107, :206`). **Step 4's
fine-tune-vs-train-fresh decision cannot be made from repo evidence today.**

**[D] Tail-priced entries — fewer than assumed.**
`OWNED_BODY_CAPTURE_PLAN.md:95` — **partially** invalidated: the p95 framing falls,
the ~25 mm median share survives. `:211` (Battle 3, *"Occlusion: co-locate clips…
as MammaSyn did"*) — **the one concrete corpus-design entry the aim shift re-aims.**
`:190` (Battle 2 shoot-list occlusion block) — fine as *evaluation* coverage,
affected only as training design. `BATTLE0_DETECTOR_WIDTH_FINDINGS.md:142` still
carries the stale framing. **`SHELLS_STAGE0_PLAN.md`: zero invalidated entries** —
its p90/p95 gates guard face-lane identity fitting, untouched by any body finding.

**[R] LOO-reprojection pseudo-labels — VERIFIED as standard practice**, with two
dissents that change the design:
- **"Only data diversity attacks view-consistent appearance bias" is too strong.**
  Weak/strong augmentation on the supervised held-out view (FixMatch lineage)
  attacks it cheaply. Adopt it.
- **Replay — the standard anti-forgetting fix — is impossible here**, because
  SOMA-77's training data is NVIDIA-internal synthetic. Mitigation is limited to
  adapters, a frozen backbone, or a very low learning rate — **which couples the
  technical fine-tuning question to the licence gate above.**

## 6. What this pass did not ask — and one of them is a legal gate

1. **Performer releases covering ML training use.** The memo gated Step 4 on
   *NVIDIA's* licence and never on **our own performers'**. It is a named lead-time
   item in the Battle 2 checklist and it gates the legality of training on owned
   footage at all. **Decision-relevant inside a month.**
2. **The delivery-schema question should have been a design decision**, not
   something discovered by a blast-radius check.
3. **The `≥3 used views` label gate discards exactly the hard 35.3%** — it
   manufactures the easy-pose imbalance the same section warns about, and
   stratifying by *speed* does not stratify by *visibility*. Re-design the gate.
4. **Standing above all of it: the marker session is still not booked.** This pass
   planned the next builds without once naming plan §0's rank-0 blocker.
