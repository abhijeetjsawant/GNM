# Head, Feet and Hands — the three regions the body lane does not yet solve

**Status:** diagnosed, not fixed. Nothing here has been repaired; this document exists so
the repair does not start from the wrong premise. Read `docs/BODY_LANE_PLAN.md` §0–2 and
`docs/FITTER_PLAN.md` first. Measured 2026-08-31 on the SOMA-77 four-camera clip.

> **⚠ The head half of this document has been superseded by measurement.
> Read `docs/HEAD_ORIENTATION_MEASURED.md` before acting on §1, §2 or §7.** Four
> things changed, and two of them contradict text still standing below:
>
> 1. **The delivered head is a *constant*, not noise** — welded to the torso frame at
>    `src/autoanim_gnm/commercial_multiview.py:1350`, identity quaternion on every
>    frame. §1c's 43.79° is the *momentum fit's* head, a different arm. See the
>    correction inline at §1c.
> 2. **§2 caution 2 is REFUTED.** The retained observations *do* carry all 77
>    landmarks. No worker re-run is needed. See the correction inline at §2.
> 3. **`HeadEnd` is measured and it does not rescue the head** — 66/115 % length
>    variation, 6–25× its detector's own body controls, the same class as the
>    fingers. §2's "single most promising head lead" no longer holds.
> 4. **The bar §7 demands is measured**, and Apple Vision's ears — already
>    integrated, already run on this footage — are the best-conditioned head input
>    found.

---

## 0. The one-sentence finding

**The rig has everything and the landmarks have nothing.** MHR carries a full head chain,
ball-of-foot joints and forty finger joints. The 17 landmarks we feed it cannot drive any
of them. Every symptom in these three regions is an *input* problem, and no amount of
solver work reaches it.

| Region | MHR joints present | Landmarks driving them | Consequence |
|---|---|---|---|
| head / neck | `c_neck`, `c_head`, `c_jaw`, `l_eye`, `r_eye` | `Head` joint (mislabelled `nose`), 2 eyes, `neck` | orientation is noise — p95 **44°** per frame |
| feet | `l_foot`, `r_foot`, `l_ball`, `r_ball` | ankle only — **one point per foot** | orientation unobservable; `l_foot` has **no rotation channel at all** |
| hands | ~40 finger joints per the MHR skeleton | wrist only | fingers unsolved; route is a constrained MHR hand fit, **not** triangulation — see `FINGER_TRIANGULATION_GATE.md` |

---

## 1. The head, measured

### 1a. `nose` is not a nose

`workers/commercial_multiview/soma77_pose.py:80` maps `nose` to SOMA-77 index **6**, which
is the **`Head` skeletal joint** — a point inside the skull, not a surface landmark. The
adapter comment says so honestly. Downstream did not read it: `tools/fitter/export_fitted.py`
binds `nose → c_head` with a zero locator offset, which is *approximately right for a head
joint* and would be badly wrong for a nose.

Drawn on frame 130, the `nose` dot sits **behind the eyes on both performers** — the
walking man faces right and his `nose` is left of his eyes; the crouching man faces left
and his is right of his. In 2D it sits **2.80 × eye-to-eye** from the eye midline where a
real nose sits at 0.5–0.8, at detector confidence 0.94.

> **Do not "fix" this by moving the point.** The mapping is defensible; the *name* is a
> trap. Anything that infers facing direction from `nose` is already wrong. Rename it, or
> carry the semantics explicitly.

### 1b. The eyes cannot carry orientation at this working distance

| quantity | subject 0 | subject 1 | note |
|---|---|---|---|
| eye-to-eye, triangulated | 59.1 mm median | 59.2 mm median | anatomically right (~63 mm) |
| …its standard deviation | **132.8 mm** | **201.1 mm** | **2–3× the quantity itself** |
| eye axis vs shoulder axis | 12.8° median | 13.6° median | plausible |
| frames where those axes **oppose** | 18/150 (**12 %**) | 8/150 (**5 %**) | a head cannot yaw >90° off the shoulders — these are label swaps or triangulation failures |
| head frame-to-frame rotation | median 8.2°, p95 **138°** | median 9.9°, p95 **126°** | noise, not motion |

At ~5 m the two eyes are about **11 px apart** in a 1280-wide frame. That is at the
detector's spatial precision, so the eye axis direction — and therefore head roll and yaw —
is dominated by noise, and inverts outright on 5–12 % of frames.

SOMA-77 **has no ear joints**. `right_ear`/`left_ear` appear in `markers_mhr_cm.json`
purely as schema and are populated on **0 of 150 frames**. Do not plan around them.

### 1c. It reaches the character

Fitted subject 0, `smoothing = 0.0`, per-frame joint rotation from `fitted_0.gltf`:

| joint | median | p95 | max | landmark support |
|---|---|---|---|---|
| `c_head` | 4.48° | **43.79°** | 78.0° | 59 mm eye baseline |
| `c_neck` | 3.69° | **44.03°** | 89.4° | ditto |
| `r_uparm` | 4.91° | 50.00° | 80.4° | shoulder + elbow |
| `r_lowleg` | 2.39° | **8.07°** | 12.2° | hip, knee **and** ankle |
| `l_foot` | — | — | — | **no rotation channel** |

`r_lowleg` is the control and it is the important row. Same solve, same smoothing, same
footage, same performer — a joint bracketed by landmarks above and below jitters **5.4×
tighter** than the head. The difference is landmark support, not solver quality.

> **Blind to** — this is jitter, not accuracy. A head locked rigid to the chest would score
> 0° here and be perfectly wrong. Jitter is a necessary condition, never a sufficient one.

> **⚠ Corrected 2026-08-31 — this table is about the momentum fit, and the *delivered*
> head is the degenerate solution the note above imagines.** These are **local,
> parent-relative** rotations from `fitted_0.gltf`. The shipped
> `subject-*.body-track.npz` is a different arm entirely: `Chest`, `UpperChest`,
> `Neck`, `Head`, `LeftEye`, `RightEye` are the **identity quaternion on every frame of
> both subjects**, because `positions_to_body_track` assigns them `torso_world`
> verbatim at `commercial_multiview.py:1350`. Head-relative-to-thorax spread on the
> delivered track is **0.000000°**.
>
> And it is worse than "a head locked to the chest would score 0° here". Scored on
> **world** head rotation — the composed statistic, which is what a viewer sees — the
> delivered constant reads **2.56° median / 6.64° p95** on subject 1 against **MAMMA's
> 1.93° / 6.62°**. It does not score zero. **It scores within 0.3% of the reference
> while carrying no head information at all** — and on subject 0 it reads 2.0× MAMMA, so
> the naive gate's verdict depends on which performer was scored. *(MAMMA's
> `body_id-00` is our subject **1**; pair every MAMMA comparison through
> `tools/head/subject_map.py`.)* Full working: `docs/HEAD_ORIENTATION_MEASURED.md` §0.

---

## 2. 60 of 77 joints are discarded — and only part of that is news

`src/autoanim_gnm/data/somaskel77-v1.json` defines 77 joints. The adapter maps **17**.

| discarded range | what it is |
|---|---|
| 15–38 | **left hand** — thumb, index, middle, ring, pinky, full chains |
| 43–66 | **right hand** — same |
| 70–71, 75–76 | **`LeftToeBase`/`LeftToeEnd`, `RightToeBase`/`RightToeEnd`** |
| 4, 7, 8 | **`Neck1`, `HeadEnd`, `Jaw`** |
| 11, 39 | `LeftShoulder`, `RightShoulder` (clavicles) |
| 1, 2, 3 | `Spine1`, `Spine2`, `Chest` |

### Fingers — already settled, do not re-derive

**`docs/FINGER_TRIANGULATION_GATE.md` answers this and its answer is not "extend the
adapter".** That document already records that SOMA-77 emits 15 finger joints per hand and
that the 19-landmark contract discards them. It then measured what happens if you use them:
**independent per-joint triangulation fails at this framing** — 226 % bone-length variation,
phalanges triangulating to 68–196 mm where anatomy says 25–45 mm — because at 4.7 m the rays
to a finger are near-parallel relative to the feature size.

It also records the counter-example that keeps the door open: **MAMMA recovers good fingers
from these exact four videos**, by fitting SMPL-X's constrained hand pose space jointly over
all views and frames. MHR ships a clean equivalent — **27 DoF per hand**, single-axis hinges
on the middle and distal joints, with `parameter_limits` and `linear_joint_range_min/max`,
Apache-2.0 and no MANO. **That is the finger route.** Read that document before writing any
finger code; this one adds nothing to it.

### Toes and the head chain — this part is new

`LeftToeBase`/`LeftToeEnd`/`RightToeBase`/`RightToeEnd` (70, 71, 75, 76) and `HeadEnd`,
`Jaw`, `Neck1` (7, 8, 4) are dropped by the same adapter and are **not** covered by the
finger gate. Two consequences:

- A toe point turns the foot from one landmark into two, which is the difference between
  unobservable and observable orientation. `l_foot` currently has no rotation channel at all.
- `HeadEnd` gives the skull a **long axis** in place of the 59 mm eye baseline whose standard
  deviation is 133 mm. This is the single most promising head lead on this page.

> **Two cautions, both load-bearing.**
> 1. **Predicting is not predicting well, and the finger gate is the precedent.** Fingers
>    looked like "free capability" for exactly this reason and did not survive measurement.
>    Toes are larger and closer to well-tracked joints than phalanges are, so they are a
>    better bet — but *a better bet is not a result*. A constant rest pose looks plausible
>    in any overlay, which is the degenerate solution this project keeps catching.
> 2. ~~**The existing observations cannot answer this.** `artifacts/soma77-full/work/*-observations.jsonl`
>    was written *after* the adapter dropped those joints — it contains 17. Answering the
>    quality question needs a **worker re-run**, not a re-read.~~
>    **REFUTED 2026-08-31 — and this was the session's unblock.** Those files carry
>    `landmarks_soma77`, **all 77 points, every person, every frame, all four cameras**,
>    exactly as `soma77_pose.py`'s docstring says. Proven by **20,043 identity checks
>    with 0 mismatches** — every mapped joint is byte-identical to its
>    `landmarks_soma77[index]`, which is what establishes SOMA-77 *order* rather than
>    merely 77 *entries*. The 2D values are bit-identical (60,129 values, max difference
>    0.0) to the file that produced the shipped tracks, and replaying
>    `reconstruct_multiview` on them reproduces the retained raw triangulation at
>    **0.000000 mm**. **Toes, `Neck1`, `Jaw`, `HeadEnd` and all 30 finger joints are
>    measurable right now, with no detector re-run.** `docs/HEAD_ORIENTATION_MEASURED.md`
>    §3; instrument `tools/head/verify_soma77_retention.py`.
>
> 3. **`HeadEnd` has now been measured and caution 1 was right to warn.** It did not
>    survive: 66.5 % / 115.0 % length variation against body controls at 2.5–4.2 %, and
>    6–25× its detector's own controls on a shared frame set — the same class as the
>    fingers' 4.4×. "A better bet is not a result" was the correct instinct.
>    `docs/HEAD_ORIENTATION_MEASURED.md` §2.

**Before proposing any new detector, enumerate what the two already-integrated adapters
emit** — `workers/commercial_multiview/mediapipe_pose.py` and `apple_vision_pose.swift`.
MediaPipe Pose carries ears, heel and foot-index; its hand model carries 21 points per hand.
Score them on **this footage, same denominator**, against SOMA-77.

> **Done for the head, 2026-08-31 — all three adapters, one frame set.** Apple Vision's
> detections for this window already existed (they are the boxes SOMA-77 was driven
> from); MediaPipe was run for the first time from the cached
> `pose_landmarker_heavy.task`. **Apple Vision's ear axis is the only head axis on this
> footage that stays inside its own detector's body-control range** — 7.6–11.0 % length
> variation, 0.57–0.76× its shoulder control. MediaPipe's ears sit 2.1–3.4× outside its
> controls with a 99° rotation p95; SOMA-77 has no ears and its best head axis sits
> 6–24× outside its controls. **This is not a detector swap** — SOMA-77's body is better
> (shin 2.3–2.7 % against Apple Vision's 5.3–12.1 %) and Apple Vision's ears resolve on
> only 107–127 of 150 frames at 2.35–2.73 cameras. Full table and cautions:
> `docs/HEAD_ORIENTATION_MEASURED.md` §4.
>
> **And do not wire the ears to head yaw off this paragraph.** Everything above scores a
> segment's *length*. Scored on **direction** against MAMMA, the same ear axis has a yaw
> standard deviation of **21–41°** — larger than the entire 13–16° head-on-torso signal
> it would explain — because a differential depth error between an axis's endpoints is
> first order in its direction and only second order in its length. The ears belong in
> the objective of a multi-frame fit, not on a direct path to a rotation channel. See
> `HEAD_ORIENTATION_MEASURED.md` §1's corroboration and §7.4.

---

## 3. A boundary question to settle early, not to answer here

`soma77_pose.py:79` says *"the head is GNM's anyway"*, and a face lane exists
(`src/autoanim_gnm/a2f*.py`, `visual_face_retarget.py`, `visual_track.py`).

That lane owns **expression**. It does not obviously own **where the head points in world
space**, which is rigid pose and is a body-lane output. Decide explicitly who owns head
rigid orientation before building anything, because both lanes can produce it and a silent
disagreement between them is the kind of defect that survives every gate.

---

## 4. Traps banked from the session that produced this

- **A wrong-looking overlay can have a cause nowhere near the geometry.** momentum writes
  glTF times in seconds; Blender converts with the *scene* fps. 30 fps motion in a default
  24 fps scene ran 25 % fast and froze at frame 119.2, and read as a placement error while
  the root measured 32.9 mm and the camera was exact to 0.1 px. See CLAUDE.md.
- **Do not convert pixels to millimetres by feel.** This session called a 10.3 px residual
  "about a centimetre"; the measured scale is **5.92 mm/px**, making it **61 mm** — 6× out.
  Compute `1000·Z/fx` at the working depth, or reproject a known offset.
- **Reprojection cannot score depth.** Error along the camera's own ray is invisible in any
  overlay. Barred as an acceptance gate in this lane.
- **`nose` is the `Head` joint.** See §1a.

---

## 5. Frame indexing — the next session will trip on this

- `artifacts/soma77-full/work/*-observations.jsonl` carries `frame_index` **60 … 209** and
  `image_path` into **`artifacts/commercial-multiview-b2/work/frames/`**.
- `artifacts/commercial-multiview-soma77/source.mp4` is that window already trimmed:
  **149 frames, re-indexed 0 … 148**, 1280×720, 30 fps.
- `markers_mhr_cm.json` index 0 therefore corresponds to **observation frame 60** and
  **source.mp4 frame 0**.
- Camera rigs in `soma77-full/` and `commercial-multiview-soma77/` are **byte-identical**
  for all four cameras; intrinsics are for **3840×2160**, so scale by 1/3 to work at 1280×720.

## 6. Regenerating any of this

Scripts are in `tools/fitter/`; they read `$FITTER_WORK` (default `~/.cache/autoanim-fitter`).

```bash
export FITTER_WORK=~/.cache/autoanim-fitter          # needs: mhr/assets/{lod6.fbx,compact_v6_1.model}
python -m venv $FITTER_WORK/momenv
$FITTER_WORK/momenv/bin/pip install pymomentum-cpu numpy
$FITTER_WORK/momenv/bin/python tools/fitter/export_fitted.py 0   # -> fitted_0.gltf
/Applications/Blender.app/Contents/MacOS/Blender --background \
    --python tools/fitter/render_fitted.py -- $FITTER_WORK/out 5
```

`markers_mhr_cm.json` is triangulated landmarks in **MHR centimetres**, mapping
`capture_m = (x, −z, y) / 100`. A copy of the one measured here is checked in as
`tools/fitter/markers_mhr_cm.EXAMPLE.json` so the numbers above are reproducible.
`render_fitted.py` **sets `scene.render.fps = 30` before import** — do not remove it.

---

## 7. What "surpass MAMMA" has to mean here

No number in this project has ever been checked against ground truth (BODY_LANE_PLAN §1).
Until the marker session, "better than MAMMA" is **not measurable** on head, feet or hands,
and a claim that we beat it is unsupportable in either direction.

Two rules for whatever gets built:

- **Score MAMMA's own hands and feet on this footage first.** The bar is measured, not
  assumed. It may be lower than expected at 5 m, and that changes what "parity" costs.
- **No gate a constant can pass.** A head rigidly locked to the chest passes every position
  gate and scores zero jitter. The head gate must measure orientation *tracking* against an
  independent reference, and must ship with a demonstration that the locked head fails it.
