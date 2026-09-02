# D1 (locate): where the facing defect actually lives

**2026-09-02.** Instruments: `tools/compare/facing_location.py` (report
`artifacts/compare/facing-location.json`), `tools/compare/facing_surface_probe.py`
(Blender, measures the shipped GLB), and the repaired
`tools/swap-harness/camera_overlay.py`. Read-only against the delivery; nothing under
`src/` changed.

## The question

The 2026-08-30 finding was **"every delivered character faces 180° the wrong way"**. On
2026-09-02 the surface instrument (I6) turned our delivered mesh 180° and it got **worse**
on all eight camera×subject cells, decisively on seven — which, read plainly, says the
shipped facing is already correct. Both cannot be true. And the overlay script the original
diagnosis ran through had a timebase bug, so the original evidence was suspect too.

Three readings were on the table: **(i)** an internal yaw already compensated at export,
**(ii)** the 2026-08-30 diagnosis was an artefact of the broken overlay timebase, or
**(iii)** a left/right mirror in the rig's naming.

## The answer: (iii), and the 2026-08-30 diagnosis was right

**Both the shipped MPFB asset and `DETAILED_HUMANOID` name their `Left` bones on the mesh's
anatomical *right*.** Read straight out of the files, no capture and no reference involved:
the mesh's nose (the most anterior head vertex: x ≈ 0, y = 1.518, z = **+0.160**) sits at
rig **+Z**, the eyes sit 74 mm in front of the head joint, the toes 132 mm in front of the
ankle — and the bones named `Left` sit at rig **−0.17** in x, carrying the flesh at −x. A
right-handed body facing +Z with up +Y has its anatomical **left at +X**. The asset is
left-handed with respect to its own face.

`_frame` builds a right-handed basis, so `_frame_alignment` can satisfy *either* "bones
named Left land on the performer's left" *or* "the face points forward" — not both. It
satisfies the first, and the torso frame comes out yawed 180°.

**Everything whose orientation is defined against the rig's rest convention inherits the
mirror** — the torso through `_frame_alignment`, and *the head independently*, through a
second hardcoded statement of the same wrong convention (below). Only the **feet** escape,
because they aim rig +Z straight at a captured world direction and never consult the rest
convention at all.

### Forward-dot against the footage (+1 = with the performer, −1 = exactly opposed)

| part of the delivered character | vs our capture | vs MAMMA |
|---|---:|---:|
| pelvis (`Hips`) | **−1.000** | −0.992 |
| chest (`Spine`/`Chest`/`UpperChest`) | **−0.993** | −0.988 |
| neck | −0.973 | −0.960 |
| head | **−0.916** | −0.909 |
| **the mesh's own nose** (no joint name enters) | **−0.949** | −0.939 |
| feet (solved from the triangulated `ToeBase` since f6a4973) | **+0.939** | +0.920 |
| *ORACLE: MAMMA's forward vs our capture's forward* | **+0.992** | — |

The oracle is the line that makes this decisive. Our capture and MAMMA are two independent
estimates of which way the performers face and they agree at +0.99. The delivered pelvis
reads −1.000 against both. This is not two instruments disagreeing about a few degrees.

Per-subject medians carry moving-block bootstrap intervals (block 15, 2000 resamples,
lag-1 ≈ 0.98); every interval quoted above excludes zero by a wide margin. One take, two
performers, 150 correlated frames — nothing here generalises past this fixture.

### The mirror has a third site, and no skeleton fix reaches it

The head does **not** inherit the torso frame — since 981e437 `positions_to_body_track`
sets it from its own rigid fit as an absolute world rotation — and it is still backwards.
It reaches the same wrong answer by its own route:
`head_orientation.CANONICAL_HEAD_AXES` declares, in the capture convention, that the
subject's **left is rig −X** and **forward is rig −Z**. It is a perfectly good right-handed
frame (`left × up = forward` holds) and it is the mirror image of the geometry it drives:
the asset's nose, eyes and toes are all at rig **+Z**, and the track schema's own
`gaze.direction_body` is `[0, 0, 1]`.

The head *template* is innocent — it is learned from the subject's own landmarks and
`_anatomical_gauge` fixes the zero from that template's own eyes and `HeadEnd`, consulting
no reference. Only the constant it is mapped **onto** is mirrored.

The head's two axes read −0.938 / −0.985 (rig +X vs the performer's left) and −0.890 /
−0.941 (rig +Z vs forward): **both** reversed relative to the mesh's geometry, so the head
is *yawed* like the torso, not reflected — one mechanism, two sites.

### Per camera — "forward-dot against footage on every camera"

Fraction of frames the performer faces each camera, against the fraction our delivered
torso does:

| camera | performer (ours / MAMMA) subj 00 | delivered | disagree | performer subj 01 | delivered | disagree |
|---|---:|---:|---:|---:|---:|---:|
| A001 | 0.34 / 0.25 | 0.80 | **0.86** | 0.73 / 0.80 | 0.31 | **0.95** |
| B001 | **1.00 / 1.00** | **0.00** | **1.00** | 0.19 / 0.22 | 0.80 | **0.99** |
| C001 | 0.71 / 0.78 | 0.24 | **0.91** | 0.65 / 0.59 | 0.39 | **0.93** |
| D001 | 0.00 / 0.00 | **1.00** | **1.00** | **1.00 / 1.00** | **0.00** | **1.00** |

B001 is the cleanest cell: the standing performer faces that camera on **every** frame of
the take (MAMMA agrees on every frame) and our delivered character has its back to it on
every frame. The repaired overlay at frame 75 shows exactly that against take frame 135,
where the performer is walking toward B001 with his face plainly visible.
(`artifacts/compare/d1-facing/overlay-*-f075-*.jpg`.)

### The handedness triple product — the reading no silhouette or by-name score can fake

`sign((across × up) · forward)` with `across = right − left` **by name**. A yaw cannot
change it; a mirror must.

| arm | sign |
|---|---:|
| our triangulated capture | **−1** |
| MAMMA `pred_joints` | **−1** |
| the delivered rig, forward read from its **feet** | **−1** |
| **the same delivered rig, forward read from its torso** | **+1** |
| the delivered **mesh's own skin** | **−1** |
| the rest skeleton in `neutral-body.npz` | **+1** |
| the rest skeleton in `DETAILED_HUMANOID` | **+1** |

**One rig, two answers.** No proper rotation can change this sign, so the posed skeleton is
*not* a rotation of its rest: the torso carries the rest's mirror through as a yaw, and the
independently re-aimed limbs and separately-solved feet do not. That contradiction is the
defect's location, stated as one bit.

The delivered *mesh* reads −1, the same as a real human, while its nose points backwards
and its +X side sits on the performer's right (−0.973). Both flipped means **a pure 180°
yaw of a self-consistent human, not a reflection** — exactly what the 2026-08-30 entry
said.

### Why the other two readings fail

**(i) compensated at export — rejected by direct measurement.** `_compose_rest_and_delta`
builds `animated_world = target_world · alignment · rest_world`, and the inverse bind
matrices cancel `rest_world`, so the net rotation applied to bind geometry is the track's
own world rotation. No compensation exists, and the probe confirms it on the shipped GLB
through the real skinning: vertex order verified against the source asset to **0.002 mm**.

**(ii) an artefact of the broken timebase — rejected.** The bug was real and is fixed, but
it can only show the *wrong frame* of a correct character; it cannot turn one round. Every
figure here is computed with the fps set correctly and the defect is still there. What the
bug *does* explain is why the original write-up's numbers (−0.42 / −0.90) match nothing
measurable today.

## What in the plan turned out to be wrong

1. **The parity board is wrong that the delivered tracks predate the head fix.** They were
   written 2026-09-01 15:15:33 — eighteen hours *after* 981e437 put the head solve on the
   delivery path and thirty seconds *before* f6a4973 was committed. `run-report.json`
   records both the head and toe solves as `solved`, and nothing has touched `src/` since.
   I6 and I4 scored these exact bytes (SHAs verified). All three instruments share one
   build; no rebuild was needed or done.
2. **I6's `control_ours_yaw180_facing` is not the inverse of the defect.** The defect yaws
   only what the torso frame rigidly carries. A whole-body yaw fixes the torso and breaks
   everything else — and the torso is *on the rotation axis*, so it barely moves:

   | | Hips | UpperChest | Head | LeftHand | RightHand | LeftFoot | RightFoot |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | subj 00 displacement (mm) | 84 | 84 | 71 | **826** | **769** | 432 | 702 |
   | subj 01 displacement (mm) | 107 | 107 | 135 | **803** | **900** | 546 | 604 |

   The control moves the hands by ~0.8 m while moving the torso by ~0.1 m. IoU is dominated
   by limb placement, so it falls. **A correct measurement carrying a claim it does not
   support.**
3. **I6's facing-sensitivity demonstration is confounded by the same thing.**
   `control_oracle_yaw180_facing` also moves MAMMA's limbs, so the 0.35–0.62 IoU it loses
   measures sensitivity to a *whole-body yaw*, not to facing with limbs held. **I6 has not
   yet demonstrated that a silhouette can see facing on this take.** The corrected control
   is to yaw only the vertices whose dominant weight is a torso bone, about the torso
   vertical, and rescore.
4. **The D1 gate card is ambiguous, and one of its readings cannot be met.** "A mirrored
   self-consistent human must fail the forward-dot even where IoU passes" has two senses. A
   sagittal-mirrored *pose* leaves facing exactly where it was — measured, not argued
   (control: forward-dot **+0.809**, handedness sign **+1**) — so no forward-dot can reject
   it and the handedness band is what does. A human driven through a *mirrored-naming rig*
   is the other sense, and that is today's build, which fails the forward-dot at −1.000.
   Both are in the proposed gate as separate controls; the card should say which it means.
5. **The −0.42 / −0.90 facing dots have no instrument behind them** in any commit. Strike
   them; quote `forward_dot` instead.
6. **"A pure 180° yaw" is now only true of the torso and the head.** Since f6a4973 the
   feet read +0.94. That change is what made I6's control read backwards.
7. **`camera_overlay.py` never set `scene.render.fps`.** Fixed in commit 46ee1e0 with a
   POSE CHECK so it cannot recur silently. Before: 650.4 mm / 172.9 mm posed-Hips error at
   frame 94. After: 58.4 / 74.1 mm, and *constant* across the take (43–89 mm at frames
   10/50/94/130/149) — a constant residual is the signature of correct timing, and it is
   the known root/hip placement offset, which is D2's territory.
8. **The mirror lives in three places, not one:** `DETAILED_HUMANOID`, the MPFB asset, and
   `head_orientation.CANONICAL_HEAD_AXES`. The plan's D1 row calls it "the rig's mirrored
   left/right naming" and scopes the fix to `_frame_alignment`; both are too narrow, and
   the head site is reached by neither.

## Proposed fix and its gate

**There is no code-only fix.** A left-handed source basis cannot be a proper rotation:
`target @ source.T` would have determinant −1 and `Rotation.from_matrix` will not return
it. Flipping the secondary axis alone was tried on 2026-08-31 and fixed the facing while
destroying the pose, because the limb chains still aim at world directions while the torso
they hang from turns.

1. Negate X throughout the canonical rest skeleton so bones named `Left` sit at **+X**, and
   **mirror the bound mesh with it** — swap each vertex's Left/Right skin weights and
   negate vertex X in the MPFB asset. Skeleton and mesh must move together or the binding
   breaks. This is an asset change with a code change beside it.
2. Then flip the secondary in both `_frame_alignment` calls from `(-1,0,0)` to `(1,0,0)`.
3. **And re-state `CANONICAL_HEAD_AXES`, which neither of the above reaches.** Its columns
   must become left `(1,0,0)`, up `(0,0,1)`, forward `(0,-1,0)` in the capture convention —
   rig +X is capture +X, rig +Y is capture +Z, rig +Z (the face) is capture −Y, and
   `left × up = forward` still holds. Miss this and the body turns round while the head
   stays backwards.

**Gate.**

- **Forward-dot:** median > +0.9 and p05 > 0 for `Hips`, chest, `Neck`, `Head` and the
  mesh nose, on both subjects, against **both** references, bootstrap lower bound above 0.
  The feet must **not** move — their dots stay inside their current intervals.
  **The head band is contingent on step 3**: without it the head fails for a reason that
  is not the torso, and the answer is `CANONICAL_HEAD_AXES`, never a relaxed band.
- **Handedness:** the delivered rig's triple-product sign must agree with our capture *and*
  MAMMA, read **both** ways (forward from the feet, forward from the torso). Those two arms
  agreeing is the fix stated as one bit.
- **Surface:** I6 IoU must not fall on any of the 8 cells, on the front/back-distinguishable
  half already tabled.
- **Degenerates that must fail:** *the current asset* as the mirrored-rest control (if the
  repaired gate passes today's build, the gate is broken); a **sagittal-mirrored human**
  (must fail handedness while passing forward-dot and silhouette); a **180°-yawed** human
  (must fail forward-dot while passing handedness); a **90°-yawed** human; a **constant
  world forward** — but check the take's own yaw range before quoting that one.
- **Blast radius:** `DETAILED_HUMANOID` is shared by the capture retarget, `speech_motion`,
  `soma_motion`, `body_export`, `unified_gltf`'s skin matrices and `body_binding`'s socket
  geometry, and `head_orientation` carries its own copy of the convention. Every lane needs
  a camera overlay before and after — the *repaired* overlay.
- **Sequencing:** cannot ship while the `THORAX_SMOOTHING_FRAMES` leak stands
  (`tools/compare/provenance.py` exits 1 until I8 re-selects it).

## What this instrument is blind to

It is blind to **magnitude**: a dot says which way a part points, not how far off it is, and
−1 is consistent with anything from 160° to 200°. It is blind to everything that is not an
**orientation** — the clavicle-origin error, the bone lengths and the root/hip offset are
untouched here and none are fixed by fixing this. The triple product is a **sign**: it
detects a mirror and says nothing about how good a non-mirrored answer is, and near
axis-coplanarity the sign is noise, which is why the fraction of frames is reported beside
every median (subject 01's capture arm sits at 0.12 on the "wrong" side for that reason).
The reference forward is built from the pelvis and neck of **estimates** on both sides;
this measures *agreement* about facing, and it is decisive only because the disagreement is
a whole reversal.

## Proposed `VISUALS` entry

For the `delivered` rung (n=11); `ladder.py` has one owner, so this is the entry to paste,
not a registration. It ships in `tools/compare/extractors/d1_facing.py` as
`PROPOSED_VISUALS`. One chart, because the finding is a *disagreement between parts of one
character* and the reader needs the pelvis at −1 and the feet at +0.9 on the same axis.

> **Which way the delivered character faces, part by part.** +1 means a part points the same
> way the performer does in the footage; −1 means exactly the opposite. The pelvis, chest,
> head and the mesh's own nose are all reversed; the feet, solved separately from the toes,
> are right. MAMMA's independent reading of the performers agrees with ours at +0.99, so
> this is not two instruments disagreeing about a few degrees — the body is turned round.
> *Higher is better.* Bars: delivered pelvis / chest / head / mesh nose (`ours`), delivered
> feet (`alt`), MAMMA's reading of the performers (`mamma`), our capture turned 180°
> (`control`).

Direction cosines in [−1, +1] are their own axis and share it with nothing else on the page
— not I6's IoU, not I1's millimetres.
