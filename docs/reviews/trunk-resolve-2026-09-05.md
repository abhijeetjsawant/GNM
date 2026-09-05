# D7b — re-solve the trunk chain onto the captured neck after the pelvis frame

**2026-09-05, branch `ladder/D7b`.** Report: `artifacts/compare/d7b-trunk/gate.json`.
Instruments: `tools/compare/delivered_vs_capture.py` (new), `d7b_trunk_delivery.py`,
`d7b_silhouette_partwise.py`, `d7b_trunk_gate.py`. Extractor stub:
`tools/compare/extractors/d7b_trunk.py`. Tests: `tests/test_trunk_resolve.py`.

**Merge rule's mechanical outcome: DO NOT MERGE.** B1, B2 and B4 pass on both performers.
B5 FAILS *as written* — and the reason is that it was pre-registered in a unit no float32
delivery can satisfy. Section 7 states both readings and recommends neither; the call is
the coordinator's.

---

## 0. The pre-registration, restated verbatim

From `docs/LADDER_EXECUTION_PLAN.md` §2, the **D7b card**, written 2026-09-05 and frozen
before any number in this document existed. It is carried into
`artifacts/compare/d7b-trunk/gate.json` under `preregistration` and quoted here in full.

> **floor, measured 2026-09-05 before the band:** the length residual
> ‖L_rest − ‖neck − Spine_origin‖‖ under the D7 pelvis is 20.9 / 41.7 mm median whole-take /
> bent tercile on performer 0 and 18.3 / 11.9 on performer 1 (L_rest 448 / 401 mm) — the
> trunk chord shortens when the spine flexes and a rigid straight chain cannot follow; that
> share is handed to **D5** (spine ratios and a flexible chain), with the part a distributed
> flexion would recover computed and reported here
>
> **DELIVERED vs CAPTURE (new instrument, the figure that saw the defect):** every mapped
> joint from each delivered GLB's own bytes against the pipeline's own triangulated
> landmarks in absolute world, D3 archive / D7 / D7b rescored on identical draws (block 15,
> 2000), per joint median / p95 / bent tercile; `retarget_cost.py` stays as it is and is
> labelled BLIND to this class (it re-solves with no spine landmark)
>
> **B1 (the placement claim; the candidate optimises it, so it is paired with the floor and
> the photographs):** Neck-from-file median within 5 mm of the length floor on both
> performers, whole take and bent tercile, and below D7 with the CI clear (predict 21 / 42
> and 18 / 12; against D3 whole-take performer 0 is predicted slightly WORSE, 14 → 21,
> because D3's chain lay on the trunk line by construction, and better on every bent cut,
> 105 → 42)
>
> **must-fail:** the shipped D7 (59 / 44) and a root-translation degenerate that zeroes the
> neck by moving the hips
>
> **B2 (untouchable):** root translation, `Hips`, `UpperLeg`, `LowerLeg` locals bit-identical
> to D7; hips / knees / ankles from the file unchanged; feet locals reported
>
> **B3:** shoulders / elbows / wrists from the file not worse than D7 with CI, predicted to
> improve on bent frames
>
> **B4 (the photographs):** part-wise silhouette on the tilt terciles, torso AND arms not
> worse than D7 on EITHER performer with the CI clear, improvement predicted on performer 0's
> bent tercile torso; MAMMA mesh bit-identical
>
> **B5:** `Head` WORLD orientation unchanged to 1e-9 on every frame (the locals of
> `Neck`/`Head` change with `UpperChest`, so the head gate is RERUN and its figures reported,
> byte-equality not claimed); D3 closure on the rebuilt GLB from bytes ≤ 1e-6 m; canonical
> round trip legs 0.00, torso and arms reported before / after, not banded (the round trip
> rebuilds its torso from upper-arm origins)
>
> **B6 reported:** rung 11 vs MAMMA per joint; facing dots; all 16 handedness signs; frames
> where the torso frame turns > 60°/frame (predict 0)
>
> **B7 synthetic (SOMASKEL77 posed clips, I7 noise):** neck placement error of
> aim-from-Spine-origin vs aim-from-hip-mid under the Kabsch pelvis; clean arm must reach the
> length floor to 1e-6
>
> **merge rule, fixed before numbers:** B1 on both performers AND B2 exact AND B4 on both
> performers AND B5; B3, B6, B7 report; any failed clause stated in the review

---

## 1. The defect, and the one line that closes it

D7 gave `Hips` its own frame from the pelvis's own landmarks and left `Spine`, `Chest` and
`UpperChest` aimed along `neck − hip_mid`. `Spine` hangs `rest["Spine"]` off `Hips` **in
the pelvis frame** — about 197 mm up the pelvis axis from the leg-root midpoint — so a
pelvis pitched away from the trunk line carries that origin off the line, and a straight
rigid chain aimed from a displaced origin misses the captured neck by the displacement.
The standing rule *after replacing a parent, RE-SOLVE the chains below it* had not been
applied to the spine.

The change is one block inside `positions_to_body_track`'s loop, after `Root` and `Hips`
are set and gated on `pelvis_world is not None`:

```python
if pelvis_world is not None:
    torso_world = _frame_alignment(
        (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
        neck - _joint_origin(world, frame, root_translation, rest, "Spine"),
        shoulder_across)
```

Nothing else in `src` moves. No constant, no window, no provenance entry. `_joint_origin`
reads `world[frame, Hips]`, which is why the block must sit after the two `_set_world`
calls; and the gate means a legacy caller's track is bit-identical, which
`tests/test_pelvis_frame.py::test_the_legacy_hips_channel_is_still_the_trunk_line_bit_for_bit`
and four tests in `tests/test_trunk_resolve.py` assert.

`Spine`, `Chest` and `UpperChest` share this one world rotation and their rests are
collinear +Y with zero X and Z on both performers, so `Neck`'s origin lands **on** the ray
to the captured neck and what remains is the trunk's LENGTH error alone. That is the floor,
and it belongs to D5.

---

## 2. The instrument that sees it, and what it is blind to

`tools/compare/delivered_vs_capture.py`, committed **before** the src change and
reproducing the coordinator's figures to 0.1 mm on the first run. It reads each delivered
GLB with `d3_skeleton_gate.glb_joint_positions` — the file's own bytes, its own `children`
hierarchy, its own animation channels — converts rig Y-up back to capture Z-up as
`(x, −z, y)`, and scores every mapped joint against
`triangulated_world_positions_z_up_m` from the same delivery directory. Nothing is
re-detected and nothing is re-solved.

Which rig joint carries which landmark's claim is the converter's own construction: it aims
each bone until its CHILD's origin reaches the landmark, so `LeftUpperLeg` is scored against
`left_hip`, `LeftLowerLeg` against `left_knee`, and so on. `Hips` is scored twice — the
`Hips` joint against the hip-landmark midpoint, and the LEG-ROOT midpoint against it,
because D2b's root formula puts the leg roots there.

**Blind to.** (a) *Orientation*: a joint whose origin is on its landmark can still be
rotated about it; every finger, toe, eye and the whole head-on-neck rotation is invisible
here. (b) *The mesh*: this scores the SKELETON the file carries; the skin bound to it can
balloon or tear without moving a number — the silhouette is for that, and it has its own
blindness (§5). (c) *The landmarks*: they are our own triangulation, so a common-mode
detector error is inside the reference and cannot appear as an error. (d) It cannot say
whether a change is right in the world, only whether the delivered skeleton sits closer to
the points that delivery was solved onto — which is why B1 is paired with the floor, with
B2 and with the photographs.

`tools/swap-harness/retarget_cost.py` is left exactly as it is and **labelled BLIND to this
class**: it re-solves the track through the converter with no spine landmark, so it reads
D3's torso figure (14.46 / 26.21 mm) on the D7 delivery unchanged. It is not wrong; it
answers a different question.

---

## 3. Every band, with its number and its verdict

| band | what it asked | measured | verdict |
|---|---|---|---|
| **floor** | reproduce 20.9 / 41.7 and 18.3 / 11.9 mm, L_rest 448 / 401 | 20.922 / 41.706 and 18.321 / 11.859; L_rest 448.08 / 401.45 | **PASS** |
| **B1** | Neck within 5 mm of the floor, both performers, whole and bent; below D7 with the CI clear | gap **0.62 / 0.38** mm (p0) and **0.00 / 0.03** mm (p1); D7b−D7 −41.0 mm [−59.2, −24.1] and −23.0 [−53.3, −15.4] | **PASS** |
| **B1 must-fail a** | the shipped D7 must fail the 5 mm band | D7 is 38.0 / 25.7 mm from its own floor | **FAILS as required** |
| **B1 must-fail b** | a root-translation degenerate that zeroes the neck | neck **0.0002 mm**; hips 21.1 / 22.1 mm (p0) and 22.6 / 20.5 (p1), knee 20.7 / 20.8, ankle 24.2 / 21.4 | **B1 passes it, B2 rejects it** |
| **B2** | root, `Hips`, `UpperLeg`, `LowerLeg` bit-identical; hips/knees/ankles from the file unchanged | every banded array bit-identical on both performers; all eight from-file medians identical to 1e-9 | **PASS** |
| **B3** | shoulders/elbows/wrists not worse than D7 with CI; improve on bent frames | not worse on 12 / 12 cells; every bent-tercile point estimate improves (12 / 12), 6 of 12 with the CI clear of zero | **PASS (reported)** |
| **B4** | torso AND arms not worse than D7 on either performer; improvement predicted on p0's bent torso; MAMMA mesh bit-identical | torso +0.0093 [−0.0053, 0.0172] (p0) and +0.0104 [−0.0026, 0.0140] (p1); arms +0.0044 and −0.0039, both spanning zero; **p0 bent torso +0.0294 [0.0020, 0.0355]**; oracle bit-identical, split-vs-unsplit 0.0 | **PASS** |
| **B5 head** | `Head` WORLD orientation unchanged to 1e-9 per frame | worst **5.12e-6° / 4.52e-6°** = 8.9e-8 / 7.9e-8 rad — 0.75 / 0.66 of one float32 ULP | **FAIL as written** (§7.1) |
| **B5 closure** | D3 closure on the rebuilt GLB from its own bytes ≤ 1e-6 m | 4.16e-7 / 4.45e-7 m | **PASS** |
| **B5 round trip** | legs 0.00, torso and arms reported | legs 0.00 / 0.00, torso 0.00 / 0.00, arms 0.55 / 0.08 mm | **PASS (reported)** |
| **B6** | rung 11, facing, 16 signs, 0 frames over 60°/frame | 0 frames over the ceiling on both arms and both performers; 16 of 16 signs unchanged; one forward-dot moved > 0.02 | **REPORTED** |
| **B7** | clean arm reaches the length floor to 1e-6 | worst gap **2.16e-7 m** | **PASS (reported)** |
| **hygiene** | the D7 code on the same inputs byte-identical to the shipped delivery | all 8 delivered files byte-identical | **PASS** |

### B1 in full, from the delivered files' own bytes

| | performer 0 | performer 1 |
|---|---|---|
| D3 (before the pelvis frame) | 14.3 mm | 28.6 mm |
| D7 (the pelvis frame) | 58.9 | 44.0 |
| **D7b** | **21.5** | **18.3** |
| the length floor | 20.9 | 18.3 |
| D3, bent tercile | 105.2 | 14.3 |
| D7, bent tercile | 131.3 | 33.1 |
| **D7b, bent tercile** | **42.1** | **11.9** |
| the floor, bent tercile | 41.7 | 11.9 |

The prediction was 21 / 42 and 18 / 12. It is 21.5 / 42.1 and 18.3 / 11.9.

### B2 from the file, every joint this step must not touch

`LeftUpperLeg` 4.607, `RightUpperLeg` 4.748, `LeftLowerLeg` 9.788, `RightLowerLeg` 7.390,
`LeftFoot` 14.041, `RightFoot` 11.280 mm on performer 0 — identical to D7 to 1e-9 on both
performers, and identical again on the leg-root midpoint (0.000 mm) and the `Hips` joint
(80.0 mm, which is the rig's own hip drop and is expected).

**The feet, reported and never banded.** Pass C falls a foot back to `torso_world` when its
toe landmark is missing, and this step turns `torso_world`. Performer 0: **0 of 150** frames
changed on either foot. Performer 1: `LeftFoot` changed on **5 of 150** frames (worst
12.20°), `RightFoot` on 0 — the toe solve resolves 98.3 % of that performer's frames. The
root translation and the foot contacts stayed bit-identical regardless, because the ground
projection reads the feet and those five frames did not move a contact run.

---

## 4. The degenerates, and which band catches which

* **The shipped D7 build** is B1's first must-fail and it fails: its delivered `Neck` sits
  38.0 mm (p0) and 25.7 mm (p1) from its own length floor, against a 5 mm band.
* **A root-translation degenerate** — the root translated per frame by
  `neck_landmark − Neck_origin` — scores a **perfect** neck (0.0000 mm) and B1 alone waves
  it through. B2 rejects it outright: its root translation is not bit-identical by
  construction, and its hips, knees and ankles move to **20.5–24.2 mm** from their
  landmarks against D7b's 4.6–15.6 — the left hip alone goes 4.607 → 21.076 on performer 0
  and 5.392 → 22.612 on performer 1, four times worse, because the root moved by the whole
  neck error (21.5 / 18.3 mm median) and every joint below the hips moved with it. This is
  the pair the card asked for, and it is the reason B1 is never quoted alone.
* **The legacy path** is its own control: with no spine landmark the new aim never executes
  (`tests/test_trunk_resolve.py::test_the_legacy_path_never_reaches_the_new_aim` watches
  `_joint_origin` and asserts it is never asked for `Spine`), and the track is bit-identical.
* **The D7 aim itself** is reachable as a control through the converter's own substitution
  point rather than a re-implementation: `_joint_origin` is module level and called by bare
  name, so returning the hip midpoint for `Spine` and the real origin for every other joint
  reproduces D7's line at the identical call site. Both the unit tests and B7 use it.

---

## 5. What each instrument here is blind to

* **`delivered_vs_capture.py`** — §2. Orientation, the mesh, common-mode detector error,
  and it cannot tell "closer to the points we solved onto" from "right".
* **The part-wise silhouette** — depth; a left/right mirror of a fore-aft symmetric pose;
  and, within each part, *where inside the outline* a limb sits. Every figure is read
  against a mesh bound to an asset whose proportions are not the performer's (the oracle
  reaches 0.71–0.78 where we reach 0.54–0.62), so it scores a CHANGE against a large
  standing shape mismatch. It also cannot see the skin distortion the post-D7 review found
  — ballooning hips on lying frames, a tearing thigh in the squat. Precision and recall are
  reported beside every IoU for that reason, and here **both** rise on performer 0
  (precision 0.839 → 0.859, recall 0.619 → 0.622), so this is not the dilated-blob pattern.
  D6 owns the mesh, with a mesh-distortion instrument first.
* **The D3 closure** sees only that the exported file agrees with forward kinematics of the
  track it carries. It cannot see whether the track is right. (Its
  `glb_rest_equals_track_rest_max_m` reads 1.9585 here and reads 1.9585 on the D7 build too:
  a glTF node translation lives in its parent's aligned asset rest frame, so lengths are
  preserved and directions are not. Not banded; the closure figure is the band.)
* **The canonical round trip** has no spine landmark, so it runs the legacy trunk-line path
  and is blind to D7 and D7b alike. It is here only to show the legacy path still closes.
* **The head oracle** proves the head's *world* orientation is unchanged; it says nothing
  about whether the head solve is right — that is the head gate's job, and it was rerun.
* **The MAMMA arm** reports agreement with a research fitter's convention and nothing more.

---

## 6. The reported arms

**Rung 11 vs MAMMA**, `tools/compare/mamma_scoreboard.py`, delivered arm, subject map from
`tools/head/subject_map.py` (MAMMA's `body_id-00` is our subject 1). Median over the
15 joints: **48.3 → 33.1 mm** on performer 0 and **49.6 → 42.4** on performer 1. Of the
eight joints this step can move, seven improve on performer 0 and five on performer 1. The
seven untouched joints are identical, as B2 requires.

One row needs naming before it is misread. **`neck` vs MAMMA gets worse** — 91.3 → 96.4 mm
on performer 0 and 89.0 → 97.7 on performer 1 — while against *our own* neck landmark it
went 58.9 → 21.5 and 44.0 → 18.3. MAMMA's neck joint is not the SOMA-77 `Neck2` landmark;
they are different points on the body, and the D7 review recorded the same convention
disagreement in the other direction. `nose` moves the other way on the same build
(131.8 → 74.3 and 102.8 → 69.1). This is a convention disagreement reported, not the MAMMA
arm dissenting, and **nothing here selects anything**.

**Facing and handedness.** All **16** triple-product signs unchanged. One forward-dot median
moved by more than 0.02: `delivered_torso_Chest` against MAMMA's forward on performer 0,
0.9858 → 0.9636 — still far above the 0.9 the D1 gate banded.

**The torso frame's own travel.** 0 frames over 60°/frame on either build and either
performer, as predicted. Medians 2.22 → 2.56 (p0) and 2.56 → 3.05 °/frame (p1); worst frame
20.8 and 19.3.

**The head gate, rerun** (`tools/head/head_gate.py`, scoring
`artifacts/head-lane/head-solve-shipped.npz`). Its figures are unchanged, as they must be:
it reads the head solve and the triangulated landmarks, both byte-identical across these
builds. `ORACLE_mamma_head_our_thorax` PASS on both subjects; `candidate_multiview_fit`
FAIL on P1 with 7.27 / 7.47° median. Rerun and quoted, not assumed.

**B7, synthetic truth.** Five posed SOMASKEL77 clips, the Kabsch pelvis, and I7's own
measured heavy-tail pixel noise recovered through the real triangulator. Three seeds at stride 3 (24 frames per clip),
against D7's synthetic pass at five seeds and stride 1 — B7 is report-only and this is
enough to separate the two aims by three-fold on the one clip that bends. The clean arm
reaches the length floor to **2.16e-7 m**, comfortably inside the 1e-6 band. Two sentences
are needed to read the absolute numbers: **(a)** four of the five clips are near upright
(tilt 1.6–16.6° median), where the two aims coincide by construction, so only the squat clip
(26.3°) separates them — and there the aim from the `Spine` origin reads **23.2 mm** against
the aim from the hip midpoint's **71.0**, and on that clip's bent frames 22.0 against 93.0.
**(b)** the absolute figures on the other four clips (89–140 mm before the ground
projection) are large because B7 runs the canonical `DETAILED_HUMANOID` on SOMASKEL77
bodies with no sizing — a trunk of the wrong length against a body of the wrong
proportions. They are a property of the fixture, not of the delivery, and both arms carry
them equally. Under noise the squat clip reads 23.0 against 80.7.

**The distributed-flexion figure, handed to D5.** Splitting the pelvis-to-thorax angle
equally over the two interior hinges (`Chest` and `UpperChest`) and taking the polyline's
chord, a curved chain of the same three segments would recover **32.3 mm of performer 0's
41.7 mm bent-tercile floor** — about three quarters of it — and 0.2 mm of performer 1's
11.9 mm. Whole-take the same construction recovers **−1.9 mm** (p0) and **−0.4 mm** (p1):
negative, because on a near-straight frame there is nothing to bend and an equal split
shortens a chord that was already short. So the prize is concentrated exactly where the
floor is, on bent frames, and on the performer who bends. The pelvis-to-thorax angle it is
computed from reads 23.5° median and 53.9° on performer 0's bent tercile.

The equal split is an assumption and is stated as one: real lumbar and thoracic flexion is
not equal, and only spine ratios (D5) can say what it is. Nothing here is fitted to it and
nothing is built.

---

## 7. What the coordinator should know

### 7.1 B5 failed as written, and the band was pre-registered in an unreachable unit

The card asked for the `Head` **WORLD** orientation "unchanged to 1e-9 on every frame,
computed from the two tracks". Measured, D7 against D7b: **5.12e-6° / 4.52e-6°** worst
frame, i.e. 8.9e-8 / 7.9e-8 radians. Against a band of 1e-9 that is a factor of ~80–90, and
the band is FAILED. It is not moved. (The 4.68e-6° that appears in evidence item 3 below is
a different pairing — D7b against the head-solve oracle — and the two must not be
confused.)

Why it cannot be met. The delivered track stores every local rotation as **float32**.
`Head`'s parent is `Neck`, whose world is a slerp involving `torso_world` — the very
quantity this step turns — so *both* the `Neck` and `Head` locals are re-derived and recast,
and the world is then recomposed down a six-link chain. Recomposing an unchanged world
through a different float32 decomposition cannot be exact.

Three pieces of evidence, in increasing strength:

1. **The input is byte-identical.** The two run reports' `head_orientation` diagnostics are
   equal under sorted-key serialisation. The head solve did not run differently.
2. **A float32 storage-floor control.** D7's own `Head` world — unchanged by construction —
   re-expressed under D7b's delivered `Neck` world, cast to float32 exactly as
   `_set_world` and the track cast it, and recomposed: **2.41e-6°**, one float32 ULP. The
   measured departure is 0.75 and 0.66 of one ULP.
3. **An oracle arm.** `artifacts/head-lane/head-solve-shipped.npz` carries the head solve's
   own **float64** world rotations. Put through the converter's change of basis they are the
   head *both* builds were handed. D7 sits 4.52e-6° / 5.26e-6° from it; D7b sits
   5.12e-6° / 4.68e-6° — **on performer 1 D7b is closer** — and the medians are identical to
   1.708e-6° on all four arms. The head did not move.

So there are two readings and the choice is not the agent's:

* **As written**, B5 fails, and the merge rule's mechanical outcome is **DO NOT MERGE**.
  That is what `gate.json` says and what this review reports.
* **Read as "the head's world orientation is unchanged to the delivery's float32 storage
  floor"**, B5 passes on all three clauses and the rule's outcome is MERGE.

This is a pre-registration error of the same class as D7's two: a band written in a unit
nobody checked against the artifact's own precision. It belongs beside them in the log.
The agent recommends neither reading and has moved no band.

### 7.2 The brief named the wrong `work/` directory

The brief said to rebuild "from `artifacts/commercial-multiview-b2/work/` COPIED". That
directory cannot serve the hygiene clause: its `run-report.json` carries no `detector` key,
its four `*-observations.jsonl` differ from the shipped ones by SHA, and it has no
`*-soma77-observations.jsonl` at all. `artifacts/commercial-multiview-soma77/work` was used
instead — which is what `d7_world_vertical_delivery.py` copied — and the substitution is
recorded in `d7b_trunk_delivery.py`'s docstring and in the gate's `hygiene` block.

The hygiene arm was then run **first, with `src` unchanged**, into
`artifacts/compare/d7b-trunk/delivery-hygiene/`, and reproduced all eight delivered files of
`artifacts/commercial-multiview-soma77` byte for byte:

```
subject-00.glb              8a87ae6f8a53df3e…  identical
subject-00.body-track.json  1c449fa801ac87b0…  identical
subject-00.body-track.npz   f83d41893c745de2…  identical
subject-00.mapping.npz      d6430a7bc3e74d54…  identical
subject-01.glb              6498d10c332d4c22…  identical
subject-01.body-track.json  96d48b05af192f49…  identical
subject-01.body-track.npz   b5e7a1e3b3823a9c…  identical
subject-01.mapping.npz      b3fbae503b29b0d8…  identical
```

Every difference in the D7b arm therefore belongs to the one-line src change. The D7b
build's own triangulated landmarks are byte-identical to the shipped ones on both
performers, so B1–B4 share one denominator. **Nothing was written under
`artifacts/commercial-multiview-soma77/`** by any instrument in this step.

### 7.3 Refuted predictions

* **B5's 1e-9**, above. The only refuted prediction in the card.
* **B1: none refuted.** Every figure landed on its prediction, including the deliberately
  unflattering one — against D3, performer 0's whole-take neck was predicted to get slightly
  *worse*, 14 → 21, and it is 14.3 → 21.5, with the D7b−D3 interval spanning zero
  ([−24.5, +9.7]). On every bent cut it improves as predicted, 105 → 42.
* **B3's "improve on bent frames"** is confirmed on 12 of 12 joint-performer cells by point
  estimate (performer 0's bent tercile: upper arms 51.5 → 17.6 and 61.1 → 20.0, lower arms
  48.5 → 18.9 and 53.6 → 20.3, hands 58.8 → 22.0 and 56.7 → 29.4 mm). Only 6 of those 12
  carry an interval clear of zero, because a bent tercile is ~50 frames and a block-15
  bootstrap has three blocks to work with. Reported as such.
* **B6's "0 frames over 60°/frame"** confirmed on both builds and both performers.
* **B4's predicted improvement** on performer 0's bent-tercile torso landed with the interval
  clear of zero (+0.0294 [0.0020, 0.0355]).

### 7.4 Instrument repairs made during the pass (not refutations)

* **B7's clean arm is measured BEFORE the ground projection.** The projection translates the
  root *after* the aim is taken, so on a projected track the identity
  `neck error == length floor` is displaced by the hoist — 143.6 mm on one fixture clip. The
  claim is about the converter, so D3's watcher pattern captures the unprojected track and
  **both** are reported per clip. Measured after the projection the worst gap is 20.2 mm;
  before it, 2.16e-7 m.
* **The unit test's angle metric.** `arccos` of a dot product is ill-conditioned near 1: a
  float32 quaternion's norm is 1 ± 2e-7 and that alone reads as 0.036° of "error". The
  parallelism test uses `arcsin` of the cross-product magnitude instead.
* **`delivered_vs_capture.py`'s CI keys** are `ci95_mm` / `bent_tercile_ci95_mm`; the gate
  was written against `ci95` and corrected.

### 7.5 Things the coordinator must do, and one that must not be done here

* **The extractor is a stub and `ladder.py` was not edited**, per the brief. To register it:
  `from extractors.d7b_trunk import x_trunk_resolve`, routing the `trunk_` keys to rung 7
  and the `silhouette_` keys to rung 1. `VISUALS` is declared in the extractor and every one
  of its bar keys resolves (`python3 tools/compare/extractors/d7b_trunk.py` self-checks
  this and prints `NONE` missing). `visuals.py` has **no reference-line support**, so the
  length floor is carried as an `alt` bar labelled "Floor: what a rigid straight spine
  cannot beat" — same unit, same reference and the same frames as the bars beside it, which
  is why it may share the axis.
* **`ladder.py` and `status.py` were not run**, per the brief.
* **The delivery under `artifacts/compare/d7b-trunk/delivery/` is not the shipped one.** If
  D7b is merged the delivery has to be rebuilt in place and checked byte-identical to this
  branch's build, exactly as D7's merge did.
* **`docs/parity-board.html` was not touched** — the board is updated in the same pass as
  the plan, and neither is this agent's to edit on a branch.

### 7.6 Test suite

`tests/test_trunk_resolve.py` (9 test functions, 10 cases with the parametrised one; new
file) and `tests/test_pelvis_frame.py` (14 tests, unedited) are green: 24 passed. The wider suite: `pytest tests/ -q --ignore=…modal…` →
**4 failed, 1138 passed, 16 skipped** in 618 s. All four failures are **pre-existing**, and
that is shown rather than asserted: the whole `src/` tree at the branch point `c92a9be` was
extracted to a scratch directory and each failing file run against it, where every one
fails identically.

| failing test | reproduces at `c92a9be` | what it is |
|---|---|---|
| `test_body_compositor.py::test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock` | yes | asserts a JS string in generated preview HTML; `body_compositor.py` imports `BodyTrack` and never `commercial_multiview` |
| `test_body_export.py::test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound` | yes | line 145 — the **recorded miss** already in `LADDER_STATUS.md`: it asserts the pre-D3 exporter's root |
| `test_phase4_app.py::test_home_and_health` | yes | face lane, unrelated |
| `test_retarget_cost.py::test_converter_rotations_depend_on_bone_lengths_only_through_the_clavicle` | yes | the swap harness's own fixture; it has no spine feed, so it runs the legacy path this step cannot reach |

The only `src` change in this branch is 33 lines inside `positions_to_body_track`
(`git diff --stat c92a9be..HEAD -- src/` shows one file). `tests/test_retarget_cost.py` is
worth one extra sentence because it is body-lane: the harness supplies no spine landmark,
so `pelvis_world` is `None` and the new block never executes — its failure is the same one
it had before this branch.
