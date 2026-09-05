# D8b — captured segments that break the performer's own bone length

**2026-09-06, branch `ladder/D8b`.** Report: `artifacts/compare/d8b-length/gate.json`.
Instruments: `tools/compare/captured_limb_stability.py` (D8's, extended and committed
first), `d8b_length_synthetic.py`, `d8b_length_delivery.py`, `d8b_length_silhouette.py`,
`d8b_length_gate.py`. Extractor stub: `tools/compare/extractors/d8b_length.py`.
Tests: `tests/test_segment_length_reject.py` (new; no existing test edited).
Every instrument's full output is in `artifacts/compare/d8b-length/logs/`.

**Merge rule's mechanical outcome: DO NOT MERGE on the fixture this step first ran on
(v1); MERGE on the repaired fixture (v2). Both are on the record; see §6.** The reviewer
did not merge on an override: he read §4 and §7.1, agreed that both failed clauses were
measured defects in the instrument rather than evidence about the rule, and sent the
fixture back to be repaired and the SAME pre-registered clauses rerun on it. §8 is that
repair — what was wrong, how each defect was measured, what was changed, and the reruns.
**No band was moved, no ceiling was changed, and `src/` is byte-identical between the two
runs.**

---

## 0. The pre-registration, restated verbatim

From `docs/LADDER_EXECUTION_PLAN.md` §2, the **D8b card**, written 2026-09-06 and committed
at `17e0dbc` before this branch's first instrument ran. Carried into
`artifacts/compare/d8b-length/gate.json` under `preregistration` and quoted there in full.

> **instrument first:** `captured_limb_stability.py` on the shipped delivery must reproduce
> the figures above (16 / 4 / 4 / 0 frames off; 122–274 mm; A–C–D agreement to 1–11 px on
> frames 110–122) before any src change
>
> **SYNTHETIC (selector and bands):** I7's fixture with the real seen pattern replayed PLUS
> an injected CONSISTENT collapse — both shoulders moved toward the neck by a factor
> (0.35–0.75) in every view that sees them for a run of 8–15 frames, the detector-bias mode
> measured here — scoring 3D error of the shoulders, elbows and wrists in the run for
> today's code, (a), (b), both ceilings; oracle: clean input → zero rejects and
> bit-identical output, AND on the un-collapsed frames of the collapsed clip zero rejects
> (the ceiling must not fire on honest motion); must-fail: a ceiling so tight it fires on
> the legs (any leg reject on the clean fixture = FAIL) and a whole-take hold
>
> **REAL TAKE, bands the candidate cannot optimise:** (B1) part-wise silhouette, arms AND
> torso, window and whole take, not worse on either performer with the CI clear on the
> D7b/D8 predicate, improvement predicted on performer 1's window; (B2) raw array
> byte-identical; leg rejects REPORTED, predicted 0; (B3) `delivered_vs_capture --reference
> raw` and vs the repaired points, D9 vs D8b, identical draws, hands and elbows; (B4)
> captured shoulder line frames off >15 %: 16 → ≤ 4 (performer 1), 4 → ≤ 2 (performer 0),
> forearm 4 → 0 — the candidate optimises this directly so it is paired with the
> photographs and the oracle; (B5) MAMMA's joints on the window through `subject_map`,
> report only
>
> **merge rule, fixed before numbers:** synthetic selector holds for the shipped ceiling
> AND both oracle clauses AND B1 on both performers AND B2; the rest reports

**The coordinator's amendment**, received 2026-09-06 while the instrument was running and
**before any synthetic number existed**, and applied in full:

> add a THIRD candidate (c) and record why (b) is expected to reduce to the must-fail.
> (b) REJECT-the-rays leaves the slot with no ray, so it cannot be recovered by
> `solve_sequence_positions`; it falls to `_fill_and_smooth_positions`, and an 8–15 frame
> collapse exceeds the 6-frame gap clause, so the landmark is HELD on the parent — that is
> the frozen-arm must-fail under another name. (c) KEEP THE ONE BEST RAY: keep only the
> highest-confidence camera's ray for the child landmark(s) and drop the others, so the
> existing single-ray recovery places the point. If (c) is selected, register nothing new
> beyond the ceiling: the choice of ray is by the detector's own confidence.

---

## 1. The defect, and why no existing gate could see it

On frames 110–122 the falling performer's captured shoulder line reads **122–274 mm on the
nine frames it is collapsed**, against that performer's own take median of **364 mm** (raw)
/ 361 mm (smoothed). His captured upper arm swings **428 → 280 → 333 mm** against a 277 mm
bone. And the cameras all agree with it:

| | A001 | B001 | C001 | D001 |
|---|---|---|---|---|
| frames supporting both shoulders, of 13 | **13** | 4 | **13** | **12** |
| confidence, on the frames all three support | 0.85–0.96 | sub-floor 0.13–0.19 | 0.88–0.99 | **0.40**–0.91 |
| reprojection residual of the raw point | \<7 px | — | \<5 px | \<7 px |

Over the whole range 110–122 the three agree with the raw point to **0.49–11.26 px**; on
the nine collapsed frames alone, to **0.5–6.9 px**. Three supporting views, so D8's
conditioning gate is silent by construction, and it is *right* to be: the depth is
determined. The point it is determined at is simply wrong, because the 2D detector places
both shoulders inward in **every** view as the performer twists and bends.

**This is the one failure class every geometric gate in the pipeline is structurally blind
to.** Epipolar distance, reprojection residual and ray-pair angle all measure agreement
*between* views. The views agree. The performer's own bone lengths are outside that family
and are the only evidence that can see it: a man whose shoulder line is 364 mm for 137
frames of a 150-frame take does not have a 122 mm shoulder line on frame 110.

---

## 2. The instrument, and the one card figure that does not reproduce

`captured_limb_stability.py` is D8's instrument. Its **default path is untouched, and that
was verified rather than asserted** (`logs/18-limb-stability-d8-default-path.log`):
`--reproduce d8` emits the SAME ten clauses with the SAME `card_says` values as D8's own
committed run (`artifacts/compare/d8-occlusion/limb-stability.json`). Five of them now read
FAIL, and they are exactly the five smoothed post-repair COUNTS that D8 itself moved — its
review records 27 -> 18 and 18 -> 4 — while every raw and observation clause still passes.
The code path did not change; the build under it did, which is the whole reason
`--reproduce d8b` exists. Two things were
added: `--reproduce d8b`, which checks the D8b card's figures on a post-D8 build instead
(D8's clauses describe the *pre-D8* defect and asserting them on a repaired build would be
asking the repair to leave the defect in place), and a **per-camera classification table**
over frames 110–122 — support, confidence and the raw point's reprojection residual, per
camera, per landmark, per frame. The table above is that block, not a chat message.

It was run and committed **before any change to `src/`**, and **9 of the card's 10 clauses
reproduce on the first run**: the 122–274 mm range (raw), the 122 mm at frame 110 (raw), the
428 → 280 → 333 upper arm against a 277 mm bone (smoothed), performer 1's forearm on 4
frames, performer 0's shoulder line on 4, the legs on 0, and all three per-camera clauses.

**The one that does not.** The card says performer 1's shoulder line is off by more than
15 % on **16** frames after D8. Measured on the shipped D9 build: **18**. D9 is
converter-only and moves no landmark, and **D8's own committed review**
(`docs/reviews/occlusion-repair-2026-09-05.md` §5) records this figure as *27 → 18*. So 18
is what both builds carry and the card's 16 reproduces on neither. Recorded as a card
figure that does not reproduce; **no band was moved** — B4's band is on the *after* value.

**Two readings the card compresses, both now in the report.** The card's "122–274 mm" is
the range over the *collapsed* frames (nine of the thirteen are off by more than 15 % and
read 122.2–274.4; the other four read 346–357, near the median). Its "confidence 0.4–0.95"
reproduces its lower figure exactly (0.3975, D001 at frame 110) on the frames all three
cameras support, and its upper figure is a rounding of 0.9889. Its "1–11 px" is the maximum
over the whole range (11.26 px, D001 at frame 119) and a rounding of the minimum (0.49 px).
The card also, like D8's, quotes **two arrays** without saying so: the shoulder line's range
is RAW and the upper arm's sequence is SMOOTHED.

**What this instrument is blind to** is unchanged from D8 and is the reason it is never a
band on its own: *truth* (a limb welded to its own median scores perfectly), *common-mode
error*, *direction* (a length invariant cannot score direction — CLAUDE.md), and *the
delivered file*.

---

## 3. What ships, and where the rule sits

`_reject_inconsistent_segments` computes, per performer per segment, the **median length
over that take** from the triangulated points, and withholds any frame whose length departs
from it by more than `SEGMENT_LENGTH_CEILING_FRACTION`. The departure is charged to the
segment's **CHILD** — both endpoints for the shoulder line, which has no parent. Nine
segments: the shoulder line, both upper arms, both forearms, both thighs, both shins.

**Where it sits is a band, not a detail.** It runs inside `_repair_occluded_slots`
**between** D8's conditioning gate and D8's reachability reject. A slot the conditioning
gate has already withheld is NaN, so its segments are not measurable and the length rule
cannot fire on the very cells whose triangulation D8 already judged unusable. Run *first*
instead, it would fire on every ill-conditioned two-view slot and no clean-input oracle
could pass. Measured on the real take, the raw shoulder line is off performer 1's own median
by more than 15 % on 28 frames and the length rule fires on **13** of them: the other 15 are
slots D8's conditioning gate had already withheld, so the length rule never saw them.

**Three modes** for what happens to the marked slot's rays, all reachable from
`reconstruct_multiview` so one instrument runs every arm through one code path:

* **`demote`** — the point is withheld, the rays are kept (D8's path). The sequence solve
  sees a slot to recover and resolves it from those same rays plus the performer's own limb
  lengths and continuity.
* **`reject`** — the point and the rays are withheld. With no ray the slot is not a
  candidate for `solve_sequence_positions` at all, so it falls to the fill and, past
  `MAXIMUM_INTERPOLATED_GAP_FRAMES`, to a hold on the parent.
* **`best_ray`** — every ray but the highest-confidence camera's is withheld, so the slot
  becomes exactly the single-ray case the solve already recovers. The camera is chosen by
  the **detector's own confidence**, so this mode adds no constant of any kind.

**`demote` ships**, selected on synthetic truth (§4). `retained_observations` itself is
never written to — a copy is made for the solve — and `world` is never written to, so
`raw_triangulated_world_positions_z_up_m` is the fixed reference for the whole step.

---

## 4. The synthetic fixture: what it selected, and the two things it refuted

The fixture is D8's, imported and not re-implemented, with **one** addition: a consistent 2D
collapse, both shoulders moved toward that camera's own neck detection by a factor, in every
view, before the noise and before the keep mask. The run is chosen where both shoulders keep
**three or more** supporting views for at least twelve consecutive frames, so D8's
conditioning gate is silent by construction. Frames 7–18; factor 0.5; the injected fault
measures **98.9 mm median / 115.9 p95** on the raw triangulated shoulders against exact
truth, before any rule runs.

### 4.1 The mode is selected, and it is `demote`

3D error against exact truth on the injected cells, by landmark group. Lower is better.

| landmark group | today (D9) | **(a) demote** | (b) reject | (c) best_ray |
|---|---|---|---|---|
| **the collapsed shoulders** | **96.4 mm** | **30.0 mm** | 42.2 | 57.2 |
| the elbows below them | 15.3 | 18.7 | 20.4 | 31.8 |
| the wrists below those | 15.6 | 14.2 | 14.9 | 14.5 |
| pooled over all six | 20.3 | 21.7 | 22.9 | 29.3 |

`demote` wins under **both** readings — the pooled median and the card's own per-group table
— and the ordering survives the card's factor range (at 0.35: 22.6 / 22.9 / 27.2; at 0.75:
18.5 / 20.3 / 21.1). The shoulders-only figure is flat at 30.0–31.8 mm across ceilings 0.05
through 0.20 and degrades above that, so the shipped 0.15 is not at an edge of the useful
range.

**The coordinator's prediction about (b) holds.** `held_joint_fraction` rises from 0.0038 to
0.021–0.033 under `reject` and stays at 0.0038 under `demote` and `best_ray`: the rejected
slots fall through the solve to the gap clause and are held rigidly on the parent, which is
the frozen arm under another name, and (b) scores 42.2 against (a)'s 30.0.

### 4.2 The pooled median is a gate a constant can pass, and is therefore refuted

The frozen-arm control — every arm landmark held at its own first frame for the whole take —
scores **16.4 mm pooled**, better than today (20.3) and better than every candidate. Under
the lane's standing rule that ends the matter: **no claim may rest on the pooled median
here.**

Two mechanisms, both measured rather than asserted:

* **The population is bimodal.** Only the two shoulders are injected; the four joints below
  them move only as far as the solve carries the fault. So 24 of the 72 cells sit at the
  size of the fault (~99 mm) and 48 at the size of the fixture's own noise (~15 mm), and a
  median of a mixture sits at the boundary between the two modes. Repairing the injected
  cells moves the median *into* the noise mode, so the number can rise while every cell it
  is made of improves or holds. The pooled figure does not even move with the injection
  factor (20.258 at 0.35, 0.5 and 0.75), which is the same fact from another angle.
* **The frozen control is weak on this fixture.** The six landmarks travel only **10.2 mm
  median / 19.2 mm max** over the injected run — at stride 1 these clips barely move — so a
  frozen arm's error is bounded at 19 mm by construction, against a 99 mm fault. The control
  cannot lose, and that is a property of the fixture and not of the candidate.

The mode selection is identical under both readings, so nothing about *what ships* turns on
this. What it does mean is that the merge rule's selector clause, read as coded on the
pooled median, reads **false** — and that reading is recorded rather than swapped after the
fact (§6).

### 4.3 The fixture cannot host the card's two honest-motion clauses, and that is provable

Both remaining failures are the same fact. The card asks for **zero** length rejects on the
un-collapsed frames of the collapsed clip, and **zero** leg rejects on the clean fixture.
Measured at the shipped 0.15: 73 rejects outside the run, and **14 leg rejects** — on a clip
with nothing injected.

The reason is not the rule. It is that **the fixture's honest triangulation is far noisier
than the reference take's** — about **9x** on the p95 side of the leg spread, and 6x on the
worst single honest deviation — and the ceiling is seeded on the take:

| | reference take, raw legs | fixture, raw legs (noisy, uncollapsed) |
|---|---|---|
| p5–p95 spread about own median | −5.1 % / +6.2 % | **−13.2 % / +58.2 %** |
| frames off own median by >15 % | 0–1 per segment | **1–9 per segment** |
| worst single deviation | ~23 % | **135 %** |

And the table settles it as a *band*, not an opinion: **no swept ceiling gives zero leg
fires on this fixture** — 123 at 0.05, 29 at 0.10, 14 at 0.15, 7 at 0.20, 4 at 0.25, **2 at
0.30**. A ceiling loose enough to pass the leg must-fail there would have to exceed the
worst honest leg deviation, 1.35, at which point it could not catch a 50 % collapse. The
clause and the fixture are incompatible at every ceiling, and that is arithmetic.

**Which failure it is, measured with truth.** For every cell the rule fired on outside the
injected run, the raw triangulated position of the charged landmark was scored against exact
truth, beside the same figure for the cells the rule left alone:

| | fired cells | cells left alone | ratio |
|---|---|---|---|
| oracle 2 (uncollapsed frames) | 37.9 mm median | 26.8 mm | **1.41×** |
| the leg must-fail (clean arm) | 37.9 mm | 26.5 mm | **1.43×** |

So the rule fired on triangulations that really were worse — but only 1.4× worse, because
the fixture's *honest* cells already sit at 27 mm of error. The card's clause assumed a
fixture whose honest frames are well triangulated. This one is not. **The band is not moved
and the clause reads FAIL.**

### 4.4 The oracle that does pass

Clean, fully-seen, noise-free input: **zero** length rejects, and the smoothed *and* raw
arrays bit-identical to the same run with the rule off. And on the real take, a run with
`segment_length_ceiling_fraction=None` is byte-identical to the shipped D9 delivery on both
arrays — the src change is exactly inert when off.

---

## 5. Every band, with its number and its verdict

| band | what it asked | measured | verdict |
|---|---|---|---|
| **instrument first** | reproduce the card's figures before any src change | **9 of 10** clauses, first run; the miss is the card's 16 against a measured 18, contradicted by D8's own committed review | **FAIL as coded**, §2 |
| **synthetic selector** | a candidate beats today's code | **v1:** demote wins among the three under both readings; on the collapsed shoulders 96.4 → 30.0 mm. On the POOLED median no candidate beats today (20.3 → 21.7) and a frozen arm beats them all — that metric is refuted. **v2:** on the collapsed-shoulder cell the card names, **114.2 → 8.5 mm** | **v1 FAIL as coded** (§4.2) · **v2 PASS** (§8.3) |
| **oracle 1: clean input** | zero rejects, bit-identical | 0 rejects; smoothed AND raw bit-identical, on both fixtures | **PASS on both** |
| **oracle 2: honest frames** | zero rejects outside the injected run | **v1: 73 rejects** (the fired cells 1.41× worse than the honest ones against truth). **v2: 0 rejects**, both performers | **v1 FAIL** (§4.3) · **v2 PASS** (§8.3) |
| **must-fail: legs** | any leg reject on the clean fixture = FAIL | **v1: 14** at the shipped ceiling, no swept ceiling giving 0. **v2: 0** at the shipped ceiling, with 0.05 and 0.075 still firing — the degenerate is still demonstrated | **v1 FAIL** (§4.3) · **v2 PASS** (§8.3) |
| **must-fail: frozen arm** | a whole-take hold must score worse | **v1:** 16.4 mm pooled / 18.5 on the shoulders against the candidate's 21.7 / 30.0 — the control WINS, because the fixture's arms travel only 10.2 mm over the run. **v2:** **158.7 mm** against the candidate's 8.7 | **v1 UNINFORMATIVE** (§4.2) · **v2 FAILS as required** (§8.3) |
| **B1 photographs, arms AND torso, both cuts** | not worse on either performer, CI upper bound ≥ 0 | 8 of 8 cells PASS. p0 arms +0.00006 / +0.00009, torso +0.00109 / −0.00010; p1 arms −0.00013 / −0.00193, torso +0.00030 / −0.00050 | **PASS on both** |
| **B1 prediction** | a rise predicted on performer 1's window | arms **−0.0019** and torso **−0.0005**, both point estimates DOWN, both intervals spanning zero | **REFUTED**, §7.3 |
| **B2 raw array** | byte-identical between the D9 and D8b builds | **identical on both performers**; observations identical; smoothed differs, as it must | **PASS** |
| **B2 legs (reported)** | leg rejects, predicted 0 | **NOT 0** — performer 1 `right_knee` 1, `right_ankle` 1; performer 0 none. Both are CORRECT fires: the raw right thigh reads 325 mm against a 406 median and the raw right shin 496 against 404 | **prediction REFUTED**, §7.2 |
| **B3 placement vs RAW** | delivered hands and elbows against the raw finite points | every joint moves by ≤ 1.5 mm at the median. p1 `RightUpperArm` **−0.90 mm** [−0.98, −0.40] and `LeftUpperArm` −0.58 mm; `LeftHand` +0.80 mm. p0 `LeftHand` −0.15 mm | **REPORTED**, and biased against the repair by construction, §7.5 |
| **B4 captured frames off** | p1 shoulder 16 → ≤ 4; p0 4 → ≤ 2; forearm 4 → 0 | p1 shoulder **18 → 0**; p0 **4 → 1**; p1 forearm **4 → 5** on its own median and **4 → 4** on a fixed one | **2 of 3 PASS, forearm FAILS either way**, §7.4 |
| **B5 MAMMA, reported** | predicted closer on the shoulders and elbows | performer 1: left shoulder **43.7 → 41.6**, left elbow **60.2 → 57.3**, left wrist **29.1 → 26.7**, right wrist 27.5 → 26.4. Performer 0 unchanged to 0.2 mm | **prediction HELD** |
| **B5 D3 closure** | the delivered GLB agrees with FK of the track it carries, ≤ 1e-6 m | D8b **4.55e-7 / 5.08e-7 m**; D9 4.86e-7 / 4.73e-7 | **PASS** |
| **B5 same denominator** | expected to report CHANGED | smoothed landmarks changed on both performers | **CHANGED as expected** |
| **B5 head gate** | rerun and reported | every figure byte-equal before and after — and structurally blind, §7.6 | **REPORTED** |
| **hygiene** | today's code on the same inputs byte-identical to the shipped delivery | **all 8 delivered files identical**, run BEFORE any src change | **PASS** |
| **provenance** | no MAMMA-derived constant | 0 leaks, CLEAN; three new entries curated | **PASS** |

### The capture itself, before and after

Frames off that performer's own take median by more than 15 %, smoothed array. Lower is
better.

| | performer 0 | performer 1 |
|---|---|---|
| shoulder line | **4 → 1** | **18 → 0** |
| shoulder line, min/max mm | 267/378 → **295/378** | 155/395 → **334/395** |
| upper arm L | 0 → 0 | **7 → 3** |
| forearm L | 0 → 0 | **4 → 5** |
| every leg segment | 0 → 0 | 0 → 0 |
| **hip line** | 0 → 0 | **23 → 23** (not in the rule's segment list) |

The 122 mm shoulder line is gone: on performer 1 the delivered shoulder line never leaves
±15 % of his own width again. The forearm is the one segment that gets worse, §7.4.

### The hygiene arm, run first with `src` unchanged

All eight delivered files byte-identical to `artifacts/commercial-multiview-soma77`, and the
observation caches byte-identical before and after the build. Every difference in the D8b
arm therefore belongs to the src change and to nothing else. **Nothing was written under
`artifacts/commercial-multiview-soma77/`** by any instrument in this step.

---

## 6. The merge rule, applied mechanically

> **merge rule, fixed before numbers:** synthetic selector holds for the shipped ceiling
> AND both oracle clauses AND B1 on both performers AND B2; the rest reports

| clause | holds |
|---|---|
| the synthetic selector holds for the shipped ceiling | **no** — on the pooled median no candidate beats today (§4.2) |
| oracle 1, clean fully-seen input | **yes** — 0 rejects, bit-identical |
| oracle 2, the collapsed clip's honest frames | **no** — 73 rejects (§4.3) |
| B1 on both performers | **yes** — 8 of 8 cells, CI upper bound ≥ 0 |
| B2, the raw array byte-identical | **yes** |

**Outcome on the v1 fixture: DO NOT MERGE.** No band was moved to change it.

### The same rule, on the repaired fixture (§8)

| clause | v1 | v2 |
|---|---|---|
| the synthetic selector holds for the shipped ceiling | **no** | **yes** — 114.2 → 8.5 mm on the collapsed-shoulder cell |
| oracle 1, clean fully-seen input | yes | yes |
| oracle 2, the collapsed clip's honest frames | **no** — 73 rejects | **yes** — 0, both performers |
| B1 on both performers | yes | yes (unchanged; it does not read the fixture) |
| B2, the raw array byte-identical | yes | yes (unchanged) |

**Outcome on the v2 fixture: MERGE.** The clauses are the card's, unchanged. What changed
between the two rows is the fixture and nothing else.

### What the failures are and are not

Neither failure is on the delivery. Both are on the synthetic fixture, and §4 gives each one
a measurement that says which kind of failure it is:

* **The selector clause** fails on a metric that a frozen arm also beats — so the metric
  does not discriminate here, and it is refuted by the lane's own standing rule rather than
  by the candidate. On the card's own per-group wording the candidate takes the collapsed
  shoulders from 96.4 mm to 30.0 mm against exact truth. The agent does not swap the
  reading: the clause as coded reads false, both readings are stated, and the coordinator
  decides.
* **Oracle 2 and the leg must-fail** fail because the fixture's honest legs swing
  −13.2 %/+58.2 % at p5–p95 where the reference take's swing −5.1 %/+6.2 %, and no ceiling
  in the swept range gives zero leg fires there. This is D8 §7.1's shape one step on: *a
  fixture that cannot reproduce the honest half of the phenomenon cannot adjudicate a rule
  about it.*

**On the real take** — where the ceiling's seed was measured — the same rule fires on exactly **2**
leg slots in the whole take — both on the same frame, both correct fires on genuinely broken
captures (§7.2) — and every leg segment reads 0 frames off by more than 15 % after the
repair, as before it.

---

## 7. What the coordinator should know

### 7.1 The card's synthetic clauses and this fixture are incompatible, and it is provable

This is the finding of the step. The ceiling is a **real-take seed** — the reference take's
own legs, 0 frames off by more than 15 % on 8 of 8 segments — and the card asked the fixture
to confirm it. The fixture cannot: its honest legs spread to +58.2 % at p95 where the
take's reach +6.2 %, and the leg must-fail has no solution at any ceiling that would still
catch the fault. That is not an argument, it is the swept table in §4.3.

Three ways forward, none of them a moved band:

1. **Reduce the fixture's own noise** so its honest frames resemble the take's. The
   heavy-tail draw is applied to all seventeen mapped landmarks here (I7 applies it to six);
   the replayed mask leaves many two-view leg slots. Either could be the cause and neither
   was investigated, because doing so would be tuning the fixture until the band passes.
2. **Restate the clause on the population it was written for.** "Zero rejects on honest
   motion" is a claim about a *well-triangulated* honest frame. The fired cells here are
   1.4× worse than the un-fired ones, so the rule is discriminating; it is discriminating on
   a body whose honest triangulation is already 27 mm out.
3. **Marker data (lane H)**, which would settle both the seed and the confirmation.

**And the reading that points the other way, which is equally consistent with these
numbers.** The ceiling is a seed from ONE take's legs. A fixture whose honest triangulation
is worse than that take's is also a preview of what this rule does on a *worse* take — where
it would withhold far more than 2 leg slots, and where the sequence solve would be recovering
from a prior rather than from evidence. Both readings fit the measurements; the coordinator
picks, and this branch does not.

### 7.2 B2's leg prediction is refuted, and both fires are correct

The card predicted 0 leg rejects on the real take; there are 2, both on performer 1.
Both are on the SAME frame, array index 120 (frame id 180). `right_thigh`: raw length
**325.0 mm** against his own 406.2 median, 20.0 % off. `right_shin`: raw **496.3 mm**
against 403.8, 22.9 % off. A thigh does not
shorten by 81 mm and a shin does not lengthen by 92 mm; these are captures that were wrong
and are now withheld. B2's *band* is the raw array's byte-identity, which passes; the leg
counts are the card's reported column, so this is a refuted prediction and never a failed
band. Every leg segment reads 0 frames off by more than 15 % after the repair, as before it.

### 7.3 B1's predicted direction is refuted, and the photographs cannot resolve this step

The card predicted a rise on performer 1's window. Both point estimates **fall** — arms
−0.0019 [−0.0051, +0.0038], torso −0.0005 [−0.0044, +0.0021] — and both intervals span zero.
Every one of the eight banded cells passes the "not worse" predicate, and every one of them
is a difference of 1e-4 to 2e-3 IoU. **This instrument cannot see this change**: the
delivered shoulder moves by well under a millimetre at the median (§7.5) and an arm can move
that far inside its own outline without moving a pixel. The band passes; it is not evidence
for the step, and it is not evidence against it.

### 7.4 The forearm gets worse, and that is a real cost

B4's third clause asked for performer 1's forearm to go from 4 frames off to 0. It goes to
**5** — and the coordinator asked which frame, and whether a demoted elbow recovered to a
place that changed the forearm length. Both are now answered, and the answer is that **the
extra frame is an artefact of the band's own denominator.**

**The frame is id 83.** D9's off-frames are ids 84, 85, 86, 87; D8b's are 83, 84, 85, 86, 87.

**The elbow on frame 83 was never withheld.** The length rule fired on performer 1's shoulder
line at ids 84–86 and on his upper arm at id 85; frame 83 drew nothing. The elbow there moved
**5.51 mm** anyway, because its neighbours were withheld and the sequence solve and the
9-frame Savitzky–Golay window carry that in. The forearm lengthened by 1.85 mm, 291.35 →
293.20 mm.

**What crossed the line is the median, not the elbow.** Frame 83 was already **14.10 %** off
in D9 — 0.9 % under the reporting cut. The rule withheld 7 forearm frames elsewhere in the
take, which changed the population the instrument's median is computed over, and the median
fell 255.36 → 253.10 mm. Decomposed on the two moving parts:

| | against D9's median (255.36) | against D8b's own median (253.10) |
|---|---|---|
| D9's length, 291.35 mm | 14.10 % — under | 15.11 % — **over** |
| D8b's length, 293.20 mm | 14.82 % — under | 15.84 % — **over** |

The median's 2.26 mm move is sufficient on its own; the elbow's 1.85 mm is not.

**So B4 is the one band in this step whose denominator moves with the candidate.**
`captured_limb_stability.py` recomputes each build's median from that build's own array, so
a rule that withholds frames moves the reference the count is taken against — the lane's
*same denominator* rule, broken by an instrument rather than by an arm. Recomputed with D9's
medians held fixed for both builds (`artifacts/compare/d8b-length/b4-fixed-denominator.json`)
the forearm reads **4 → 4**, and **every other cell in the table is identical under the two
readings**. The clause still FAILS — the band asks for 0 — but the *regression* is not real,
and B4's other two clauses are unaffected because their counts do not straddle the cut.

### 7.5 B3 is structurally biased against the repair, and the numbers are small either way

Every delivered joint moves by **≤ 1.5 mm at the median** against the raw captured points.
Performer 1's shoulder joints move *closer* with the interval clear (`RightUpperArm`
−0.90 mm [−0.98, −0.40]); his `LeftHand` moves 0.80 mm further. But the reference is the raw
triangulated point, and **on the frames the step acts on that is precisely the point the
step judged unreliable** — so agreeing with it scores well and disagreeing scores badly,
whatever the truth is. D8 recorded this blindness (`occlusion-repair-2026-09-05.md` §7.8)
and it is why the card lists this band as reported. The `--reference smoothed` arm is in the
report too, and its `same_denominator` is false by construction: each arm is scored against
its own repaired points, so it says how far each delivery sits from what it was solved onto
and is never a comparison between the two.

### 7.6 What was rerun, what is blind, and what was not touched

* **`tools/head/head_gate.py` was rerun** and every figure is byte-equal before and after.
  That is not reassurance: it scores `artifacts/head-lane/head-solve-shipped.npz` and reads
  its torso frame from the *shipped* delivery, neither of which this build writes, so it is
  **structurally blind** to D8b's effect on the delivered head. `_solve_head_for_subject`
  and `_thorax_frames` both read the SMOOTHED positions, so on a D8b build the head *solve*
  itself changes. The figure that would see it — the delivered head's world orientation from
  the two GLBs' own bytes — is not in the card.
* **`tools/compare/d3_skeleton_gate.py` was not rerun as a whole.** Its closure clause was
  recomputed inline in `d8b_length_gate.py` on both builds from the same two sources the
  gate reads. Running the whole gate rebuilds a third delivery into a committed instrument's
  own output directory.
* **`ladder.py` was not edited and `ladder.py` / `status.py` were not run**, per the brief.
  To register the extractor: `from extractors.d8b_length import x_segment_length_reject`,
  routing the `capture_` and `synthetic_` keys to rung 4, the `placement_` keys to rung 7 and
  the `silhouette_` keys to rung 1. `python3 tools/compare/extractors/d8b_length.py`
  self-checks that every `VISUALS` bar key resolves — it does, 11 figures and 20 controls.
* **`docs/parity-board.html` was not touched** and **`docs/LADDER_EXECUTION_PLAN.md` was not
  edited** — the coordinator amends the card.
* **The fixture repair is in §8**, and the reviewer's instruction that produced it is
  quoted there. `artifacts/compare/d8b-length/synthetic.json` was NOT rewritten; v2 is a
  separate report and a separate `synthetic_v2` block in the gate.
* **`artifacts/compare/provenance.json` was overwritten once** by a first run before the
  `--out` flag was used; the step's own copy is at
  `artifacts/compare/d8b-length/provenance.json`. Both are regenerable and gitignored.
* **The branch was not pushed and not merged.**

### 7.7 Two findings beyond the bands

**(a) The hip line collapses too, and is not in the card's segment list.** On performer 1
the captured hip line is off its own median by more than 15 % on **23 frames** smoothed and
30 raw, reading as low as 139.7 mm against a 214.4 mm median — the same class of failure as
the shoulder line, on a segment the shipped rule does not act on because the card's list
does not name it. It is unchanged by this step (23 → 23) and it is registered as an open
remedy on `SEGMENT_LENGTH_RULES` in `provenance.py`. Adding it is a one-line change and a
new band; it was not made here.

**(b) `tests/test_occlusion_repair.py`'s "off" arm is no longer pre-D8 code.** Its
`clean_runs` fixture turns off D8's three rules but does not pass
`segment_length_ceiling_fraction=None`, so both of its arms now carry D8b's rule. Nothing
broke — the rule fires zero times on that clean fixture, which is why the oracle still
passes — but the test no longer means what its docstring says. It was not edited (the brief
forbids editing existing tests); the coordinator may want to add the keyword.

### 7.8 The rule's own boundaries, asserted as tests rather than described

Three are in `tests/test_segment_length_reject.py` because they are the ways this rule fails
rather than the ways it works:

* **A take-long collapse is invisible.** The reference is the performer's own median, so a
  segment that is wrong on every frame moves the median with it and nothing fires.
* **The fault must be a MINORITY of the take.** Found while building the pipeline-level mode
  test: at an exactly 50/50 split the median lands between the honest and the collapsed
  value and *every* frame reads off by more than the ceiling, so the rule withholds the
  whole take — a worse failure than the one it is fixing.
* **The child cascade is counted, not hidden.** A pure shoulder collapse makes the upper arm
  read long, so both elbows are withheld with the shoulders: 16 cells on an 8-frame
  collapse, asserted exactly. On the real take it is 6 cells on performer 1 and 0 on
  performer 0. *(The first draft of this counter was a tautology — it counted every fire —
  and the test that caught it asserted `> 0`. The counter and the test are both fixed; the
  synthetic and the delivery were rebuilt afterwards so no report carries the wrong number.)*

### 7.9 The test suite

`tests/test_segment_length_reject.py` is new and all **18** of its tests pass. The full
suite reads **1181 passed, 16 skipped, 4 failed** (`logs/15-pytest-full.log`), and **none of
the four failures is reachable from this change** — not one of them calls
`reconstruct_multiview`, and each is a pre-existing item the ladder status log already
records:

| test | why it is red |
|---|---|
| `test_retarget_cost.py::test_converter_rotations_depend_on_bone_lengths_only_through_the_clavicle` | red on main and on the D7/D8/D9 code: its last clause asserts the root translation is independent of sizing, which stopped being true when D3 put the per-performer rest into the root formula. A pre-D3 contract |
| `test_body_export.py::test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound` | asserts the old exporter's root — the 0.8 offset the track root no longer carries. Listed in the status log as an uncommitted-test item |
| `test_body_compositor.py::test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock` | the compositor's HTML, nothing to do with capture |
| `test_phase4_app.py::test_home_and_health` | the app's health endpoint reports `degraded` in this environment |

`tests/test_occlusion_repair.py`, `tests/test_pelvis_frame.py`, `tests/test_trunk_resolve.py`
and `tests/test_arm_origin.py` — the four that guard the steps D8b sits between — are all
green.

### 7.10 For the next agent

* The shipped mode is `demote` and it is the *only* one of the three that keeps evidence in
  play; `best_ray` was the coordinator's own suggestion and it scores worst on the fault
  (57.2 mm against 30.0), because one ray plus a limb-length prior is weaker than three rays
  plus the same prior even when the three rays are biased.
* The step's whole value on the real take is the shoulder line: **18 frames off → 0** on the
  falling performer, with MAMMA — which reports and never selects — putting his left
  shoulder, left elbow and both wrists 1.1–2.9 mm closer than before. Everything else moves
  by less than a millimetre or not at all.
* If the coordinator intends to merge over the two synthetic failures, the honest form of
  that decision is: *the band was written for a fixture that turned out not to exist, the
  real-take evidence is the seed and the shoulder-line count, and the fixture is being
  fixed in a later step.* That is a decision for the coordinator and this branch does not
  take it.

---

## 8. The fixture repair, and the same clauses rerun on it

**Added 2026-09-06 after the reviewer's decision.** D8b was **not** merged on an override.
The reviewer read §4 and §7.1, agreed that both failed clauses were **measured defects in
the instrument** rather than evidence about the rule, and returned the step with an
instruction: repair the fixture and rerun the same pre-registered clauses on it.

**Nothing about the candidate moved.** `src/autoanim_gnm/commercial_multiview.py` is
byte-identical between the two runs, `SEGMENT_LENGTH_CEILING_FRACTION` is still 0.15, and
every band is the card's. `artifacts/compare/d8b-length/synthetic.json` is left exactly as
it was; v2 is a separate report and a separate `synthetic_v2` block in the gate.

### 8.1 Defect 1 — the honest frames were not honest

**The defect, measured on v1.** The fixture's own *uncollapsed* legs spread −13.2 %/+58.2 %
at p5–p95 where the reference take's spread −5.1 %/+6.2 %. "Zero rejects on un-collapsed
frames" was being asked of a body whose honest frames were already broken.

**The repair.** A scale on the magnitude of I7's heavy-tail draw. The model
(`sweep.heavy_tail_magnitude`) and the sigma (`sweep.NOISE_SIGMA_PX`, our own detector's
measured 3.13/3.26 px) are unchanged and still imported; only the amplitude is scaled. It is
a **fixture parameter** — not in `src/`, not a band, and it selects no shipped constant.

**It is calibrated, not chosen.** `--calibrate` sweeps nine scales, measures the fixture's
honest leg spread at each with the identical statistic
`captured_limb_stability.py` reports on the take, and takes **the largest scale whose honest
legs stay inside the take's own spread** — largest rather than smallest, because a quieter
fixture is an easier one and the rule should face as much honest noise as the take has. The
sweep is committed at `artifacts/compare/d8b-length/synthetic-v2-noise-calibration.json`:

| scale | legs p5 / p95 | frames off >15 % |
|---|---|---|
| 0.00 | **−0.0000 / +0.0000** | 0 |
| 0.10 | −0.019 / +0.027 | 0 |
| 0.15 | −0.029 / +0.040 | 0 |
| **0.20 — selected** | **−0.038 / +0.055** | **0** |
| 0.25 | −0.049 / +0.069 | 2 |
| 0.35 | −0.064 / +0.118 | 4 |
| 0.50 | −0.098 / +0.191 | 11 |
| 1.00 (v1) | −0.131 / +0.190 | 11 |
| *the reference take* | *−0.051 / +0.062* | *0* |

**And the sweep diagnosed v1's defect exactly.** At scale **0.00 the honest legs spread
exactly zero** — the fixture's geometry, its replayed mask and its two-view leg slots
contribute *nothing*. Every bit of v1's spread was the noise amplitude: I7 applies this draw
to six thorax joints and v1 applied it at full sigma to all seventeen mapped landmarks. That
is the whole of defect 1, and it means the repair is a correction rather than a tuning.

On v2 the honest legs read **−3.8 %/+5.5 % with 0 frames off by more than 15 % on all eight
leg segments**, against the take's −5.1 %/+6.2 % and 0 frames off.

*Order of discovery, disclosed:* 0.25 sat in the file as a placeholder while the calibration
mode was being written; the sweep replaced it with the measured 0.20.

### 8.2 Defect 2 — the collapse ran on a nearly static clip

**The defect, measured on v1.** The six scored landmarks travelled **10.2 mm median /
19.2 max** over the injected run, so a frozen arm's error was bounded at 19 mm against a
99 mm fault: the must-fail could not lose, and the pooled selector was bimodal by
construction. **The take's own figure** on frames 110–122 (performer 1, smoothed) is
**188.1 mm median** at **40.7 mm/frame** of shoulder travel — 51.3 mm/frame on the raw array.

**The repair, and its honest limit.** The collapse is now injected on the run with the *most*
landmark travel, chosen by that quantity rather than by run length, on the fastest clip and
stride the motion source has. **Measured across every full-body clip and every usable
stride, no motion source reaches the take's rate:**

| clip | best 12-frame travel, stride 1 → 2 → 3 | shoulder step |
|---|---|---|
| `autoanim_squat/research-squat-640` | 88.8 → **123.7** mm | **21.3 mm/frame** |
| `autoanim_dialogue/amy-cuddy-dialogue-body` | 31.5 → 44.4 → 45.4 mm | 8.7 mm/frame |
| `autoanim_will_acting/will-stephen-acting-body` | 12.3 → 14.3 → 17.6 mm | 2.6 mm/frame |

v2 therefore runs the **squat clip at stride 2** as the injected performer (35 frames, a
10-frame collapse — the card's range is 8–15, and 10 keeps the fault well inside the minority
the median needs, §7.8). The chosen run travels **105.7 mm median / 201.8 max**: **10× v1**
and **56 % of the take's 188.1**. The shortfall is stated and not closed — closing it would
mean inventing motion the fixture does not have, and that is the same error as tuning a
fixture until a band passes.

### 8.3 The same clauses, rerun

The injected fault on v2 is **113.4 mm median** on the raw shoulders against exact truth.
3D error on the injected cells, by landmark group; lower is better.

| landmark group | today (D9) | **(a) demote** | (b) reject | (c) best_ray | frozen control |
|---|---|---|---|---|---|
| **the collapsed shoulders** | **114.2** | **8.5** | 8.5 | 55.2 | **160.8** |
| the elbows below them | 3.6 | 11.0 | 33.8 | 32.0 | 172.9 |
| the wrists below those | 5.3 | 5.3 | 5.3 | 5.3 | 148.5 |
| pooled over all six | 6.1 | 8.7 | 9.3 | 30.2 | **158.7** |

| clause | v1 | v2 |
|---|---|---|
| **selector**, on the collapsed-shoulder cell the card names | 96.4 → 30.0 (pooled reading FAILED) | **114.2 → 8.5 mm — PASS** |
| **oracle 1**, clean fully-seen input | PASS | **PASS** — 0 rejects, smoothed and raw bit-identical |
| **oracle 2**, honest frames of the collapsed clip | FAIL — 73 rejects | **PASS — 0 rejects, both performers** |
| **must-fail: legs** | FAIL — 14, and no ceiling gives 0 | **PASS — 0** at 0.15, while 0.05 (7) and 0.075 (1) still fire |
| **must-fail: frozen arm** | UNINFORMATIVE — the control wins | **FAILS as required — 158.7 mm against 8.7** |
| **raw array untouched** | PASS | **PASS** |

Three things are worth saying precisely.

**The mode selection did not change, and on v2 the shoulder cell alone does not separate (a)
from (b).** Both withhold the same points, and on a 10-frame run both reach 8.474 mm on the
shoulders — a tie broken by iteration order, which would be no selection at all. What
separates them is the **elbows** (11.0 against 33.8) and the pooled figure (8.7 against 9.3),
and both prefer `demote`; the moving-block bootstrap on identical draws puts `reject` behind
`demote` at P = 0.000 and `best_ray` at P = 0.000. So the selection is the same as on v1 and
is now carried by a different cell, which is stated rather than smoothed over.

**The pooled median is still diluted, and the metric refutation recorded on v1 stands
unchanged.** On v2 the pooled figure still reads *worse* than today (6.09 → 8.67), for
exactly the reason §4.2 gives: only two of the six landmarks are injected, and repairing them
moves the median into the mode made of the other four. What has changed is that the pooled
metric is **no longer a gate a constant can pass** — the frozen arm scores 158.7 mm against
the candidate's 8.7 — so the refutation is preserved as a finding and no longer decides
anything. This is why v2's selector reads the collapsed-shoulder cell, which is what the card
names ("scoring 3D error of the shoulders, elbows and wrists in the run"), with the pooled
figure reported beside it.

**The elbow cascade is now visible as a cost.** On a quiet fixture the elbows go 3.6 → 11.0
mm: withholding a shoulder withholds the elbow under the card's child rule, and the solve
puts it back 7 mm worse than the triangulation had it. On v1 that cost was invisible inside
15 mm of fixture noise. It is the synthetic counterpart of the real take's forearm going
4 → 5 (§7.4), and it is the clearest open question this step leaves behind: **the child rule
discards good points along with bad ones, and now there is a fixture that can measure it.**

### 8.4 What the repair does and does not license

* Both v1 failures were **instrument defects, and both are now measured rather than argued**:
  one was entirely the noise amplitude (proved by the scale-0 row), the other entirely the
  clip's motion (proved by the cross-clip travel table).
* The merge rule's outcome is **MERGE on v2 and DO NOT MERGE on v1**, and both are in
  `gate.json` under `merge_rule` and `merge_rule.on_fixture_v1`. The coordinator is not asked
  to take the second on trust.
* **v2 is still not the take.** Its landmarks travel 56 % as far, and its bones are exactly
  rigid where a real body's segment lengths are not. The reading in §7.1 that points the
  other way survives the repair: the ceiling remains a seed from one take's legs, and a
  worse take would make this rule withhold more.
* Nothing here changes what ships. `demote` at 0.15 was selected before the repair and is
  selected after it.
