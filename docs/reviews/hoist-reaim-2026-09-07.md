# D9b — the foot-contact hoist re-aim

**Branch** `ladder/D9b`. **Card** `docs/LADDER_EXECUTION_PLAN.md` §2, committed at `a58a6b4`
before any measurement on a candidate build existed. **Pre-dispatch review** Grok 4.6
(Cursor) in Sol's place, `docs/reviews/hoist-reaim-grok-review-2026-09-07.md`.
**Reports** `artifacts/compare/d9b-hoist/{instrument,gate,silhouette-partwise}.json`,
`delivery-hygiene-build.json`, `oracle-tripwire.json`, every log under `logs/`.

**MECHANICAL OUTCOME: MERGE.** Every conjunct of the pre-registered rule passes. Two
existing tests fail on structural pins this step legitimately moved and both re-pins are
carried in a new file; they are §9.4 and are the coordinator's call, not a band.

---

## 0. The pre-dispatch review, and what each of its eight findings changed

| # | Grok's finding | what it changed, and whether it mattered |
|---|---|---|
| 1 | never re-run pass C — it rewrites the legs, feet and toes and would overwrite the projection's `candidate_local` foot lock | the src is a DEDICATED re-solve of the named chain. It mattered: a pass-C re-run would have shipped a translated root with an unplanted foot and B4 would have caught it only after the fact |
| 2 | the global quaternion hemisphere walk must not be re-run; it can flip the bits of an unhoisted frame that follows a hoisted one | each rewritten local is signed against the DELIVERED local of its own frame. It mattered twice over: the walk had ALREADY run before the projection, so a re-solve that wrote raw `_set_world` output would have undone its sign flips and failed the tripwire outright |
| 3 | `groups_mm` is a MEDIAN over concatenated joint-frames and the hoisted frames are a minority, so O3's "worse" is not guaranteed | O3 was restated as the statistic it is. Half held: the hoisted-frame median and p95 moved as predicted (3.6–7.8 and 3.6–6.6 mm), the whole-take median did NOT hold — it moved 0.5–1.5 mm (§6.4) |
| 4 | B1 and O1 are CLOSURE quantities the candidate sets directly, not bands it cannot optimise | relabelled; they enter the merge rule only in conjunction with B2, B4 and the tripwire, which fail every cheat that zeroes them |
| 5 | a pass-B accepted-set change is a HOLD, not a pre-scoped B2 exception | the set was measured identical on all eight calls, so the hold did not fire |
| 6 | on the D3 oracle no head solve is passed, so `Head`, `Neck` and the eyes inherit the trunk and MUST move | they are in the declared set. It mattered: freezing `Head` there would have left O1's trunk ray and O4 fighting |
| 7 | hands, fingers and eyes must be listed as allowed to move or an agent leaves a stale wrist world | hands are in the declared set and move; fingers and eyes are in it and do NOT move, because their locals are constants and an exact identity in float32 |
| 8 | "hoist forced to zero" means the re-solve still RUNS on the pre-projection root with the locks in place | that is what both tripwire arms do — and building it surfaced a result of its own (§5.2) |

---

## 1. The pre-registration, restated verbatim

> **merge rule, fixed before numbers:** hygiene AND the tripwire AND B1 (excess → 0 on every
> frame, same denominator PASS) AND B2 AND B3 on both performers AND B4 AND O1 AND O2.
> O3 and B5 report.

Nothing below moves a band, re-selects a constant or overrides a clause.
`maximum_root_correction_m` (0.08), `ground_band_m` (0.08),
`velocity_threshold_m_per_s` (0.30) and `minimum_contact_frames` (2) are untouched, and no
new number is introduced: `_ROOT_DEPENDENT_JOINTS` is a list of joint NAMES read off the
skeleton's topology, curated in `tools/compare/provenance.py` as `schema` (§9.5).

---

## 2. The defect

`positions_to_body_track` aims every bone and then calls
`project_generated_foot_contacts`, which does three things AFTER the aiming: a per-run root
translation that plants a slow, low foot on its first frame, a foot LOCAL-rotation lock on
those runs, and a whole-take vertical lift by the deepest penetration. **A translation
leaves every rotation alone**, so each root-dependent bone still points along
`target − origin_BEFORE_the_hoist` while its origin has moved. It is D7b's defect on the
neck and D9's on the arms, one operation later, and it has been in the delivery since D2b.

Measured on the shipped D8c delivery, from the file's own bytes, **before any src change**
(`instrument.json`):

* the hoist, recovered TWICE by methods that share no line of code — D8c's converter-line
  subtraction `root − (pelvis − rest[Hips] − _leg_root_offset)` and D9's over-determined
  least-squares fit of one translation to the four arm bones' rays — **agrees to 0.0003 mm
  on every frame of both performers**, fit residual 0.0001 mm. Median 0.0002 / 0.0001 mm,
  p95 **12.54 / 8.72**, max 19.71 / 9.53; above 0.5 mm on **67 / 22 of 150** frames and
  below 0.001 mm on 83 / 128.
* **the whole-take lift is 0.0**: the delivered feet bottom out **16.66 / 12.96 mm above**
  the floor, so on this take every millimetre of hoist is the per-run plant. Contacts
  (38, 51) / (11, 18) in 9+13 / 5+2 runs.
* from the PRE-hoist origin the seven root-dependent bones sit on their rays to **0.0002 mm
  over the whole take** — they were aimed there. From the DELIVERED origin they miss, on
  the hoisted frames, by median 5.25 / 3.73 and 4.72 / 6.13 (elbows, L/R, performer 0 / 1),
  5.29 / 4.38 and 5.66 / 6.01 (wrists), 6.07 / 5.02 and 4.36 / 6.32 (shoulders),
  4.79 / 5.46 (neck), p95 6.6–14.4, and by 0.0003 mm on every unhoisted frame.
* the four LEG bones satisfy the ray constraint from NEITHER origin (1.20–2.09 mm at the
  knees, 2.74–4.41 at the ankles, from the pre-hoist origin) because they are aimed
  landmark-to-landmark. **That is D9-legs, not this step**, and their larger miss on the
  hoisted frames (4.58–9.22 mm) goes forward with the number.

---

## 3. What ships

One block at the end of `positions_to_body_track`. The projection runs ONCE, on today's
feet — it can, because the feet are root-INDEPENDENT (`Hips` from the pelvis landmarks, the
legs landmark-to-landmark, the foot frame from the toes), so the contact detector reads
bit-identical feet and returns bit-identical runs, correction, locks and contacts — and then
every root-dependent chain is re-solved from the hoisted root in the converter's own order:
the D7b trunk frame from the `Spine` origin, the `Neck` slerp, `Head` and the eyes, the two
clavicles, **pass B's sequence rule rerun over the whole take**, the four arm bones, the
hands.

Pass A's tail and pass C's arm half are lifted VERBATIM into
`_aim_trunk_neck_and_clavicles` and `_aim_arms_and_hands`, so the passes and the re-solve
run the same lines on the same arrays. **This was the choice the card left open** — factor a
helper, or re-run the passes under a root override — and it is pre-registered here as
chosen because the tripwire can only be meaningful if the two callers cannot drift apart.

READ-ONLY after the projection: the root, the contacts, and the foot and toe LOCAL
rotations. The projection is never run a second time (planted feet read zero speed on the
runs they were planted on, so a second pass is not a fixed point). The re-solve runs on
EVERY frame — 0.5 mm is a report cut and never control flow.

`hoist = float64(projected.root) − float64(track.root)` and the re-solve aims from
`root_translation + hoist`: **the converter's own float64 root plus the float64 hoist, and
not the float32 root read back off the track.** At zero hoist that is `root_translation` bit
for bit, which is what lets the tripwire see a reordered float operation rather than a
rounding. The gap to the delivered float32 root is ~3 × 10⁻⁸ m; the gate measures the
consequence and it is inside the 0.0003 mm floor.

---

## 4. The instrument, committed before any `src/` change

`tools/compare/d9b_hoist_gate.py`, run on the shipped build at `48dbb47`. It reproduced
every figure the card quotes (§2) and produced the tripwire's reference material. Two
readings CORRECT the card, both in the direction of caution:

* the oracle's unhoisted-frame ray-miss floor is **0.0004–0.4488 mm max**, not the card's
  "0.00–0.03 mm" — that was a median. "Unhoisted" is the 0.5 mm REPORT cut, so a frame
  hoisted by 0.44 mm sits inside it. **O1 was therefore also pre-registered in the stricter
  absolute form it should have had: ≤ 0.01 mm on every frame**, the float32 FK floor.
* seed 20260906 carries a nonzero whole-take penetration lift (0.0262 mm) on every frame, so
  its floor is that constant. The real take's lift is exactly 0.

### 4.1 The oracle audit, and why the D3 gate cannot see this defect

The D3 gate's six exact-skeleton bodies were rebuilt **through `d3_skeleton_gate.oracle_block`
itself**, with wrappers recording `positions_to_body_track`,
`project_generated_foot_contacts` and `_reachable_clavicle_sequence`. Nothing is
re-implemented — the 9–19 mm lesson.

Contacts FIRE on all six: (35, 52), (41, 8), (25, 57), (44, 0), (0, 57), (38, 51), hoisting
34–67 of 150 frames by p95 9.89–14.44 mm, max 14.58–24.38. And
**`retarget_cost.score` reads IDENTICAL on the pre-projection track, the delivered track and
the hoist-removed track, on every seed** (arms 0.80–1.17, legs 0.05–0.07, torso 8.23–11.47).
It subtracts each frame's leg-root midpoint from the rig and each frame's hip midpoint from
the reference — a per-frame TRANSLATION — so it is structurally blind to a root move and
must read the correct fix as worse. The D3 gate's own reading of D7c's residue (arms
0.8–1.2, torso 8–11) is therefore **not** contaminated by the hoist.

---

## 5. The refactor tripwire

### 5.1 Why an end-to-end old-src/new-src pair, and not an in-process check

Pass A and the re-solve share a helper. A refactor that changed that helper's arithmetic
would move BOTH, and an in-process "the re-solve changed nothing" assertion would pass on it.
The reference has to come from the previous src, and it does:

* **the real take.** `--mode zero-hoist` (the projection runs in full; contacts and the foot
  LOCAL lock stand; only the returned root is the one that went in) built under the D8c src
  at `48dbb47` and under this one. **All eight delivered files byte-identical** —
  `subject-00.glb` `5b28551d…`, `subject-01.glb` `a2917235…`, and the six others.
* **the six oracle bodies (O4).** Every rotation, root and contact bit-identical on all six
  seeds, against the `post_rotations` / `pre_root` this instrument recorded on the D8c run —
  which IS what the previous src delivered on a zero-hoist arm, because it had no code after
  the projection.

The unit test `tests/test_hoist_reaim.py` carries the same clause in miniature on its own
travelling fixture.

### 5.2 A result the tripwire produced on its way

The card's **lock-without-correction degenerate cannot be built**. Keeping the foot lock and
the contacts while dropping `roots += correction` raises

    BodyValidationError: left foot contact moved 0.01245879 m (limit 0.00001000 m)

out of `BodyTrack.__post_init__` before a file is written. The shipping path refuses it,
which is stronger than B2 and B4 catching it afterwards. The tripwire arm clears
`foot_contacts` for that reason alone, and its own root and contacts are therefore not
comparable with the delivery's — only its rotations are, which is what the clause reads.

---

## 6. The real take

The delivery (`artifacts/compare/d9b-hoist/delivery`, 323.5 s, the real build script, the
shipped delivery's own cached detections copied in) saw bit-identical feet and got
bit-identical answers back: contacts (38, 51) / (11, 18), the same runs, hoist p95
12.54 / 8.72 over the same 67 / 22 frames, `ground_penetration_before_m` 0.0.

### 6.1 B1 — the closure, and the same denominator

The banded quantity is the **per-frame excess over each build's own aim floor**: the floor is
the bone LENGTH error no aim can remove (`arm_aim_floor_per_frame` for the elbows and
wrists, `trunk_length_floor` for the neck), and the excess over it is exactly the ray miss.

| worst frame of 150, mm | D8c | D9b |
|---|---|---|
| performer 0 — `LeftHand` (wrist) | **18.07** | **0.0001** |
| performer 0 — `RightHand` | 12.77 | 0.0001 |
| performer 0 — `LeftLowerArm` (elbow) | 9.90 | 0.0000 |
| performer 0 — `RightLowerArm` | 7.92 | 0.0001 |
| performer 0 — `Neck` | 5.46 | 0.0001 |
| performer 1 — `RightLowerArm` | 6.13 | 0.0000 |
| performer 1 — `RightHand` | 5.53 | 0.0002 |
| performer 1 — `LeftLowerArm` / `LeftHand` | 4.89 / 4.81 | 0.0001 / 0.0003 |
| performer 1 — `Neck` | 3.34 | 0.0001 |

Zero on **every** frame, max and not median, against the 0.01 mm float32 floor. The ray miss
from the delivered origin says the same thing directly: all seven root-dependent bones,
**0.0003 mm max over the whole take on both performers**.

**Same denominator returns to PASS.** The raw AND the smoothed triangulated arrays are
byte-identical across the arms on both performers — a clause the D8, D8b and D8c steps could
not make, because their rejects sit between the two arrays. It was pre-registered as an
expected pass, and CHANGED would have meant the change did more than re-aim.

### 6.2 What actually moved on the person, and one honest complication

The whole-take paired median difference in placement is 0.0 mm on every joint, because more
than half the frames are byte-identical. **On the hoisted frames** the delivered joints move
CLOSER to their landmarks:

| median, hoisted frames, negative = closer | performer 0 | performer 1 |
|---|---|---|
| `Neck` | **−0.56** | **−0.52** |
| `LeftUpperArm` / `RightUpperArm` | −2.52 / −0.99 | −2.56 / −5.62 |
| `LeftLowerArm` / `RightLowerArm` | −0.90 / +0.00 | −3.47 / −1.39 |
| `LeftHand` / `RightHand` | −1.37 / −0.42 | −4.32 / −0.32 |
| `Head` (position; its world rotation is the absolute solve) | +0.52 | −0.80 |
| every leg, foot and hip joint | **0.00** | **0.00** |

**A minority of hoisted frames read further out** (p95 +2.5 to +7.2 mm on performer 0), and
the mechanism is stated rather than smoothed over: the arm's floor is CHAINED from its
`UpperArm` origin, which the trunk and clavicle re-aim also moves, so the elbow is optimal
for its NEW origin and this comparison is not against a fixed reference. The banded quantity
— the excess over each build's own floor — is what goes to zero everywhere.

### 6.3 B2, B3, B4

**B2.** From the two builds' own bytes: the root, the contacts, both landmark arrays and
every frozen joint (`Root`, `Hips`, both legs, both feet, both toes) identical on every
frame; **every joint identical on all 83 / 128 unhoisted frames**; the 13 joints that moved
are `Spine`, `Chest`, `UpperChest`, `Neck`, `Head`, both `Shoulder`s, both `UpperArm`s, both
`LowerArm`s and both `Hand`s — a subset of the declared set. The two eyes did not move: their
local is an exact identity in float32 either way. **Pass B's accepted set is identical**:
eight calls, zero rejections, one all-accepted mask throughout. The HOLD did not fire.

**B3.** 8 of 8 pre-registered clauses PASS, both parts, both cuts, both performers, every CI
upper bound at or above zero. Every difference is within ±0.001 IoU — level, as
pre-registered. **The unhoisted-frame cut reads exactly 0.0 with a zero-width interval**,
which is B2 confirmed a second time through the pixel path. MAMMA's mesh oracle is
bit-identical (0.0). On the hoisted frames the whole person rises 0.0062 / 0.0008 IoU with
the interval NOT clear of zero: reported, not claimed. Improvement was not predicted and the
reason is arithmetic — 4–6 mm on a mesh rasterised at a quarter of native resolution is
under a pixel on most cameras.

**B4.** Contact count, runs, per-run planted-foot travel (0.0004 / 0.0005 mm) and the lowest
delivered foot (16.66 / 12.96 mm) IDENTICAL by construction.

### 6.4 O3 — the two gauges disagree on the same files, and both are right

| worst of six seeds | D8c | D9b |
|---|---|---|
| ALIGNED arms, whole take (what the D3 gate prints) | 1.17 | **2.70** |
| ALIGNED arms, hoisted frames | 1.18 | **8.88** |
| ALIGNED arms, p95 | 7.87 | **14.06** |
| ALIGNED legs | 0.07 | 0.07 |
| ABSOLUTE arms, whole take | 2.72 | **1.12** |
| ABSOLUTE arms, hoisted frames | 8.92 | **1.38** |
| ABSOLUTE arms, p95 | 14.46 | **7.98** |
| ABSOLUTE torso, hoisted frames | 15.41 | 13.30 (better on every seed) |
| ABSOLUTE legs, hoisted frames | 8.71 | 8.71 (unchanged) |

The gauge removes each frame's root offset, so a bone correctly re-aimed from the hoisted
origin misses the ALIGNED target by the perpendicular part of the hoist. **The band is not
moved** — the D3 arm band (0.5 mm) was already a standing fail at 0.80–1.17 and now reads
2.70 — and re-pinning an aligned score in the same commit as the candidate is the
clavicle-round-trip mistake. The absolute-frame companion row is the answer and it is now in
the D9b gate. The oracle's absolute LEG error, 4.15–8.71 mm on the hoisted frames and
bit-identical before and after, is the CONTACT MODEL's own cost on exact truth: a foot
planted where the truth is still moving. It is handed to the projection's own step.

### 6.5 B5 — the head gate

**Byte-identical across this change**, measured directly rather than assumed: the D8c source
file checked out over this branch produces the same 24 lines. Against the D8c step's own log
it differs by 0.01–0.07° on performer 1 only — and that is **D8c's**, not this step's. That
log was written at 19:46 and D8c's close-out rebuilt
`artifacts/commercial-multiview-soma77` in place at 20:42; the gate reads the delivery's
smoothed landmarks, which the hip-line row moves. The head-gate re-run owed at close-out is
D8c's debt. The delivered `Head` position against `nose` is in §6.2.

---

## 7. The must-fails

| degenerate | passes | fails | measured |
|---|---|---|---|
| the D8c build itself | — | B1 | ray miss 3.7–6.3 mm median on the hoisted frames; excess up to 18.07 mm |
| `maximum_root_correction_m = 0` (a real build) | B1, O1 (ray miss 0.0003 mm) | **B4, B2** | contacts (38, 51) / (11, 18) → **(0, 0)**; root differs by up to 15.5 / 8.5 mm. Penetration is 0 with or without the correction, so penetration could NOT have exposed it |
| keep the lock, drop `roots += correction` | — | refused by the shipping path | `BodyValidationError`, foot moved 12.46 mm against a 0.01 mm limit. Forced through by clearing the contacts it fails B2 (root) and B4 (contacts) |
| re-aim the arms but not the trunk and clavicles | B1's arm rows | **B1's trunk row** | its trunk and clavicle rays ARE the shipped build's, by construction: 4.79 / 5.46 mm (neck) and 4.36–6.32 (shoulders) median on the hoisted frames. **Computed, not built** — the number is exact without a build, and that is stated rather than dressed up as a measurement |
| re-aim, then re-derive the root | — | B1, B2, B4 | excluded by construction: pass A's root line is never re-entered and the projection never runs twice. Iterated to a fixed point it recovers the pre-projection solve and fails B2 and B4 |

---

## 8. Every clause: predicted, measured, verdict

| clause | predicted / band | measured | verdict |
|---|---|---|---|
| **hygiene** | 8 of 8 byte-identical before any src change | 8 of 8; raw and smoothed identical; observations unchanged before and after | **PASS** |
| **instrument first** | reproduce the card's figures on the shipped build | every figure reproduced; two card readings corrected in the cautious direction (§4) | **PASS** |
| **oracle audit** | contacts fire on all six; the aligned gauge reads identical pre / delivered / hoist-removed | (35,52) (41,8) (25,57) (44,0) (0,57) (38,51); identical on every seed | **PASS** |
| **tripwire, real take** | 8 of 8 byte-identical, old src vs new, hoist forced to zero | 8 of 8 | **PASS** |
| **O4 tripwire, oracle** | every rotation, root and contact bit-identical on six seeds | bit-identical on all six | **PASS** |
| **B1** excess → 0 on every frame | 0, banded 0.01 mm | worst frame **0.0003 mm** on both performers | **PASS** |
| **B1** ray miss from the delivered origin | at the float32 floor | 0.0003 mm max, whole take, all seven bones | **PASS** |
| **B1** elbows, wrists, shoulders, neck closer on the hoisted frames | by the predicted amounts | −0.32 to −5.62 mm median; a minority of frames further, attributed to the chained floor (§6.2) | **held, with the complication stated** |
| **B1** hips / knees / ankles IDENTICAL | identical | 0.0000 mm | **PASS** |
| **B1** same denominator | expected PASS | raw and smoothed byte-identical, both performers | **PASS** |
| **B2** root, contacts, `Hips`, legs, feet, toes | identical every frame | identical | **PASS** |
| **B2** every joint on every unhoisted frame | identical | identical, 83 / 128 frames | **PASS** |
| **B2** moved joints only from `Spine` up and out the arms | within the declared set | 13 joints, all declared; eyes and fingers did not move | **PASS** |
| **B2** pass B's accepted set | identical, or the step STOPS | identical on all eight calls, zero rejections | **PASS** |
| **B3** ARMS and TORSO+LEGS, whole take and bent tercile, both performers | not worse, CI upper bound ≥ 0 | 8 of 8 clauses PASS; all within ±0.001 IoU | **PASS** |
| **B3** improvement NOT predicted, whole take level | level | level; hoisted-frame rise 0.0062 / 0.0008 IoU, interval not clear of zero | **held** |
| **B3** MAMMA mesh oracle | bit-identical | 0.0 | **PASS** |
| **B4** contacts, runs, `penetration_before`, lowest foot | identical | (38,51) / (11,18); 0.0; 16.66 / 12.96 mm | **PASS** |
| **B4** planted-foot travel per run | identical | 0.0004 / 0.0005 mm, run for run | **PASS** |
| **O1** absolute ray miss on six exact bodies | ≤ the unhoisted floor; stricter form ≤ 0.01 mm | **0.0005 mm max** on every hoisted frame | **PASS** |
| **O2** legs, root, contacts, hoist per seed | bit-identical | bit-identical on all six | **PASS** |
| **O3** aligned median | "may hold" | did NOT hold: 0.80–1.17 → 1.32–2.70 | **prediction FAILS**, attributed to the alignment (§6.4) |
| **O3** aligned hoisted median and p95 worse by ~3–7 mm | reported | +3.6 to +7.8 (hoisted), +3.6 to +6.6 (p95) | **held** |
| **O3** aligned legs identical | reported | 0.05–0.07, identical | **held** |
| **O3** absolute companion row | new, reported | arms better on every seed and every cut | **reported** |
| **B5** head gate | rerun, expected line-identical | byte-identical across the src change; the 0.01–0.07° drift is D8c's (§6.5) | **reported** |
| **B5** delivered `Head` vs `nose` | reported | +0.52 / −0.80 mm median on the hoisted frames | **reported** |
| **must-fails** | four named | all four demonstrated or excluded by construction (§7) | **PASS** |

**Merge rule conjuncts:** hygiene **PASS**, tripwire **PASS**, B1 **PASS** (with same
denominator **PASS**), B2 **PASS**, B3 both performers **PASS**, B4 **PASS**, O1 **PASS**,
O2 **PASS**. O3 and B5 report.
**MECHANICAL OUTCOME: MERGE.**

---

## 9. Findings beyond the bands

### 9.1 The hemisphere walk is the converter's second whole-take coupling, and it bites backwards
Pass B is the first and was registered. The second — `local[frame, dots < 0] *= −1` walked
from frame 0 — had already run by the time the re-solve starts, so `local` carries flipped
signs while `world` does not, and a re-solve that wrote raw `_set_world` output would have
UNDONE those flips on frames the hoist never touched. Signing each rewritten local against
the delivered local of its own frame fixes it exactly and cannot reach across frames. This
is the D7b seven-normalisations lesson pointed backwards in time, and the tripwire is what
would have caught it.

### 9.2 A translation is invisible to a translation-aligned gauge, and the fix therefore reads worse
§6.4 is the cleanest demonstration in this lane of a correct measurement carrying a claim it
does not support. Two gauges, the same six delivered files, opposite verdicts. Ask of every
gate what it removes before it measures.

### 9.3 The shipping path's own validator refuses one of the card's degenerates
§5.2. `validate_body_track`'s 1 × 10⁻⁵ m contact assertion is a stronger control than the
band written for it. Worth remembering when designing the next must-fail: some degenerates
cannot be built, and that is a better answer than a failing number.

### 9.4 Two existing tests fail on structural pins, and both re-pins are carried in a new file
Neither is a behaviour change and neither is edited (the brief forbids it):

* `test_clavicle_temporal.py::test_a_reachable_take_is_accepted_whole_and_the_track_is_untouched`
  asserts `_reachable_clavicle_sequence` is called exactly TWICE. It is now called four
  times, because reachability is a property of a SEQUENCE and a re-aimed clavicle is a
  different sequence. The property the test guards — nothing rejected on reachable motion,
  the track bit-identical to a pass-through — is asserted in
  `tests/test_hoist_reaim.py::test_pass_B_is_rerun_over_the_whole_take_and_stays_inert`, on
  all four calls, plus the check that the second pass genuinely saw different locals.
* `test_clavicle_origin.py::test_no_constant_arrived_with_the_fix` asserts the literal text
  `_joint_origin(` inside `positions_to_body_track`'s own source slice. The aims that call
  it now live in the two module-level helpers — which is where this codebase puts a function
  an instrument may need to substitute. `test_no_constant_arrived_with_the_hoist_re_aim`
  asserts it against those helpers and asserts the 0.72 anchor gone from the WHOLE module,
  which is stronger than the slice.

Baseline before the change: 1194 passed, 4 failed (`test_body_compositor`,
`test_body_export`, `test_phase4_app` ×2 — pre-existing on this branch and unrelated).
After: the same set minus one flaky `test_phase4_app` case, plus these two.

### 9.5 One name entered the audited surface and is curated, not hidden
`_ROOT_DEPENDENT_JOINTS` is scanned by `tools/compare/provenance.py` because it is a
module-level constant, and the pinned unknown list broke. It is a list of joint NAMES read
off the skeleton's topology, so it is curated as `schema` beside `JOINT_NAMES` and
`CORE_ASSOCIATION_JOINTS` rather than made a local to escape the scan. The pinned list is
unchanged and `tests/test_provenance_audit.py` is green.

### 9.6 A step whose whole-take medians are zero is not a step that did nothing
More than half of each performer's frames are byte-identical by construction, so every
whole-take median difference on this step is 0.0 and every whole-take interval is [0, 0].
The hoisted-frame cut is where the change lives, and every table above that could be
diluted carries it. A reader given only whole-take medians would conclude nothing happened.

---

## 10. What every instrument here is blind to

* **B1 and O1 are CLOSURE quantities.** They say the bone points where it was told to point,
  not that the instruction was right, and a bone on its ray can still be rolled about its
  own axis. They are in the merge rule only with B2, B4 and the tripwire.
* **The ray-miss reference is the delivery's own captured landmarks.** It cannot say the
  landmarks are right. `delivered_vs_capture.py` carries the same blindness and says so.
* **B3 scores the MESH.** A joint can move inside its own outline without moving a pixel,
  and 4–6 mm at a quarter of native resolution is under a pixel on most cameras. It is here
  to catch a regression; it did not and could not show the gain.
* **The D3 gate's oracle gauge cannot see a root move at all** (§6.4), which is why the
  absolute companion row exists and why no band moved for it.
* **This take cannot distinguish a build that skipped pass B's rerun**, because pass B
  rejects nothing here before or after. The accepted-mask dump and the tripwire are the only
  evidence it ran, and the unit test's fixture asserts the second pass saw different input.
* **The oracle's `identical_on_unhoisted_frames` is reported, not banded**, and it is False
  on five of six seeds: "unhoisted" is the 0.5 mm REPORT cut and 1–116 frames per seed carry
  a real sub-cut hoist (seed 20260906 carries a 0.0262 mm whole-take lift on all 150). Those
  frames legitimately move. Seed 20260907, the one with no sub-cut frames, reads True.

---

## 11. What is open after this step

1. **D9-legs.** The four leg bones satisfy no ray constraint from either origin — 1.20–2.09 mm
   at the knees and 2.74–4.41 at the ankles from the pre-hoist origin, and 4.58–9.22 mm on
   the hoisted frames. The excess over the pre-hoist figure is this step's neighbour and the
   whole figure is D9-legs'.
2. **The contact model's own cost on exact truth.** On the six oracle bodies the absolute
   leg error is 4.15–8.71 mm median on 34–67 hoisted frames, p95 10.0–14.8, unchanged by this
   step: the projection plants a foot that the truth is still moving at up to 0.30 m/s. It
   belongs to the projection's own step.
3. **The D3 gate's translation-aligned gauge**, its 0.5 mm arm band and the D2c/D3-frozen
   references its "canonical unchanged" and "round trip" clauses compare against — the
   instrument-debt step, with the re-pin.
4. **The two test re-pins** in §9.4.
5. **D8c's head-gate re-run**, owed at its close-out (§6.5).
