# The feet, measured

Companion to `docs/HEAD_ORIENTATION_MEASURED.md`, opened 2026-09-01 after the head solve
reached the delivered artifact (§6j there). The user accepted the head at 1-of-2 performers
with its documented fixture ceiling and directed the lane to feet; the ordering rule in the
standing goal was overridden deliberately, by them, and that is recorded here rather than
assumed.

The same two rules govern everything below. **No gate a constant can pass**, and **same
denominator**. And the same warning: the recurring defect in this lane is *a correct
measurement carrying a claim it does not support*, so every section says what its
instrument is blind to.

---

## 0. WHAT THE DELIVERED FOOT ACTUALLY IS — and it is not what the plan says

`docs/HEAD_FEET_HANDS_PLAN.md` §2 records *"`l_foot` has **no rotation channel at all**"*.
That is true of the **MHR/momentum fitter** it was measured on. It is **not** true of the
clean-room multiview rig that ships. Read from
`artifacts/commercial-multiview-soma77/subject-*.body-track.npz`:

| delivered channel | subject 0 | subject 1 | |
|---|---:|---:|---|
| `LeftFoot` local rotation, median | 18.29° | 31.65° | **moves** |
| `RightFoot` | 21.14° | 22.09° | **moves** |
| `LeftToes` | 0.000° | 0.000° | **CONSTANT, identity every frame** |
| `RightToes` | 0.000° | 0.000° | **CONSTANT, identity every frame** |

So the feet are the head's situation with one difference. **`LeftToes`/`RightToes` are the
welded constant** — exactly as `Head` was before §6j. But **`LeftFoot` is not a constant,
and that is the more dangerous case**, because a naive *"does it move?"* gate passes it.

**Nothing observed constrains it.** The pipeline's 19 landmark targets end at the ankle —
`left_ankle`, `right_ankle`, and no toe or foot-index anywhere in the contract. The foot's
rotation *about* the ankle is therefore unconstrained by any measurement; whatever those
18–32° are, they are the IK's own behaviour, not an observation of a foot.

### 0a. So is it informed or is it fiction? — measured, not assumed

`tools/feet/delivered_foot_is_fiction.py` scores the delivered foot axis (the rig's
`Foot→Toes` bone, through forward kinematics) against the **triangulated `Foot→ToeBase`
direction** from SOMA-77's retained landmarks — an independent estimate that uses
observations the rig never sees. Mean-removed, because the rig's rest convention and
SOMA-77's skeletal convention need not share a zero; the question is whether the two axes
turn *together*. Beside it, **the control that makes this a gate**: a single fixed axis,
optimally aligned, scored on the same frames.

| | delivered axis vs measured | | a CONSTANT axis vs measured | | |
|---|---:|---:|---:|---:|---|
| | median | p95 | median | p95 | |
| subject 0, left | **13.88°** | 28.76° | 23.42° | 51.16° | **beats the constant** |
| subject 0, right | **14.22°** | 28.32° | 30.98° | 65.08° | **beats the constant** |
| subject 1, left | 43.35° | 96.11° | **24.23°** | 61.29° | **WORSE than a constant** |
| subject 1, right | 27.18° | 80.87° | **26.24°** | 55.72° | **no better than a constant** |

**On subject 0 the delivered foot orientation carries real information. On subject 1 it is
no better than a fixed axis, and on the left foot it is substantially worse.** A single
unmoving axis would beat what we ship, on that performer, on that foot.

> **What this is blind to, and it matters here more than anywhere.** *Both sides are ours.*
> This compares two of our own estimates and neither is ground truth. It is therefore
> **capable of showing that the delivered channel is uninformed, and incapable of showing
> which estimate is wrong** when they disagree. On subject 1 the honest statement is that
> our two independent estimates of the foot axis diverge by more than a constant would, so
> at least one is wrong there — and §1 shows subject 1 is also where the *measured* toe
> direction is weakest. **Do not read the subject 1 row as "the rig is wrong".** Read it as
> "the foot is unresolved on that performer, by two instruments that disagree."

---

## 1. ARE THE TOE LANDMARKS USABLE? — the HeadEnd test, applied before building anything

`HEAD_FEET_HANDS_PLAN.md` calls a toe point *"the difference between unobservable and
observable orientation"* and warns twice that **a better bet is not a result**. It was right
to: fingers failed bone-length stability at 4.4× the body control, and `HeadEnd` — *"the
single most promising head lead on this page"* — failed at **66.5 % / 115.0 %** length
variation against controls at 2.5–4.2 %.

`tools/feet/toe_gate.py` applies that instrument to the toes. Verdict is segment-length
stability against the controls this lane already trusts, on one common frame set.

| segment | s0 sd% | s1 sd% | |
|---|---:|---:|---|
| **`Foot→ToeBase` (ball) left** | **5.7 %** | **9.2 %** | candidate |
| **`Foot→ToeBase` (ball) right** | **5.0 %** | **9.6 %** | candidate |
| `ToeBase→ToeEnd` (toe tip) left | 14.3 % | **63.1 %** | candidate |
| `ToeBase→ToeEnd` (toe tip) right | **144.7 %** | 15.5 % | candidate |
| shin left / right | 2.6 / 3.1 % | 2.5 / 3.6 % | control |
| thigh left / right | 2.6 / 2.3 % | 2.7 / 4.2 % | control |
| forearm left | 9.6 % | 14.6 % | control |
| upper arm left | 4.4 % | 16.9 % | control |

**The two toe joints do not survive or fail together, and that is the finding.**

- **`ToeBase` — the ball of the foot — survives.** At 5.0–9.6 % it is **2.0–2.4× the leg
  controls**, and **better than the arm controls on both performers**. That is nothing like
  HeadEnd's 66–115 % or the fingers' 4.4×. It sits *between* our best and worst trusted
  controls, and the honest phrasing is exactly that — not "as good as the legs", which
  would be cherry-picking the flattering control.
- **`ToeEnd` — the toe tip — fails, in the HeadEnd manner.** 14.3 %, **144.7 %**, 63.1 %,
  15.5 %: unstable, and unstable *differently on each foot of each performer*, which is the
  signature of a point the detector is inventing rather than locating.

### 1a. The ankle angle — an invariant the head had no equivalent of

A standing ankle holds the foot near a right angle to the shin. A toe predicted from a crop
has no reason to respect it, so this is a physical check the length test cannot make.

| | median | sd | frames outside 50–130° |
|---|---:|---:|---|
| subject 0, left / right | 68.1° / 71.2° | 7.3° / 7.4° | **1 / 150** each |
| subject 1, left / right | 67.4° / 68.8° | **20.1° / 21.1°** | **38 / 131**, **46 / 131** |

**Subject 0's ankle is anatomical and quiet** — a real ankle angle, held to 7° across the
take, essentially never leaving the plausible range. **Subject 1's is not**: three times the
spread and roughly a third of frames outside it. Subject 1 is the performer who goes
supine, and this is the same performer whose delivered foot axis §0a found no better than a
constant. **Two independent instruments agree that the foot is unresolved on subject 1**,
which is a stronger statement than either makes alone.

> **Blind to:** *accuracy, entirely.* A stable segment can be stably in the wrong place, and
> `ToeBase` is a skeletal convention, not a measured ball of foot. Also blind to whether the
> detector **measures** the toe or predicts it from the crop — cross-view geometric
> consistency is the only discriminator available without ground truth, which is why every
> invariant above is scored across four independently-viewed cameras. And blind to **ground
> contact**: a foot can be stable in length and still float.

---

## 2. WHAT IS NOT YET MEASURED, AND MUST BE BEFORE ANYTHING IS BUILT

1. **The bar. Score MAMMA's own feet on this footage first.** The standing rule is *measure
   the bar, do not assume it*, and it has not been done for feet. Until it is, no claim
   about surpassing anything is available in either direction.
2. **Epipolar consistency of the toe landmarks**, the check that told the head lane its
   failure was depth and not detection. `ToeBase` passing a length test does not tell us
   whether its 2D is consistent across views.
3. **Whether a `ToeBase`-driven foot solve beats the delivered channel** — and it must be
   scored against **both** controls, the constant axis *and* today's delivered foot, since
   §0a shows today's foot already beats a constant on one performer. **A gate that only
   demonstrates "better than frozen" would pass a solve that is worse than what ships.**

**Not started, deliberately.** The precedent is the whole reason this page exists: `HeadEnd`
was *"the single most promising lead"* and did not survive. `ToeBase` has now survived its
first contact, on one performer convincingly and on the other with a documented question
mark. **That is a licence to measure further, not to build.**
