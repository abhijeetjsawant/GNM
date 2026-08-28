# Constrained hand-chain fit — agreed design, not yet built

Status: design agreed 2026-08-29 by both advisors after the finger amendment.
Not started. Prerequisite reading: `docs/FINGER_TRIANGULATION_GATE.md`.

## Both advisors changed position, and converged

Sol originally said "fingers, conditional on a cross-view test". Fable said
"fingers are a trap; the physics forbids it". The cross-view test passed,
the bone-length test failed, and the MAMMA counter-example on our own fixture
then showed the failure was the *estimator*, not the footage.

Both now recommend the same thing: **build the constrained hand-chain fit, and
rank it above the Battle 3→4 synthetic slice.**

Fable's reason for the ranking is the strongest argument on the table and is not
about hands at all: **this is not a detour, it is the fitting stage of Battle 4.**
A sequence-level, uncertainty-weighted, joint-limited model fit across all views
and frames is exactly the second half of MAMMA's architecture. Built now against
SOMA-77's sparse points, it accepts our own dense landmarks later as strictly
better observations. Even if every hand gate fails, the solver is keeper
infrastructure.

## The correction that changes the design

**MAMMA uses no learned pose prior.** Its energy is
`E_ldmks + E_shape + E_temp + E_cont`, where the shape and pose terms are plain
L2, and the paper states outright that *"optimization does not rely on pose
priors or pose initialization"*. Independently corroborated against our own
notes from the paper (research §1, §3).

So MANO's PCA space — the encumbered part — was **never load-bearing**. The
geometric work is done by the kinematic chain with fixed bone lengths plus
temporal coupling, which AniPose independently confirms: limb-length and temporal
regularisation *together* give the largest gain over naive triangulation, and
neither alone does much.

**The encumbered parts of MAMMA are its detector weights and SMPL-X. The fitting
recipe is free.**

## The clean constraint set we already own

MHR parameterises each hand with **27 rotation DoF, not 45** — the middle and
distal finger joints are single-axis hinges, which is what they are — plus
`parameter_limits` and `linear_joint_range_min/max`. Verified by inspecting
`mhr_model_lod1.pt` directly.

Fable flagged MHR's *weights* licence as unconfirmed. It is confirmed: the
release archive carries `assets/LICENSE.txt` = Apache-2.0 verbatim, checked
during increment 4. Code and weights are both Apache-2.0.

Escape hatch if L2 + limits proves too weak: **SAM 3D Body's hand decoder** is a
learned hand regressor predicting MHR parameters, commercial use permitted under
the SAM License — usable as per-view initialisation in place of MANO PCA.

## The risk that will actually bite

**Heatmap confidence is not visibility, and that is the whole gap between "MAMMA
in miniature" and MAMMA.**

MammaNet was *trained* to emit per-landmark uncertainty σ and visibility p, and
the fit downweights by both. SOMA-77 emits heatmap peak confidence, which stays
high when an occluded joint's response is captured by the nearest visible
structure. So during the clasp — the frames the product exists for — the solver
receives *confidently wrong* 2D, and the temporal term then locks and propagates
it: **smooth, stable, wrong, and passing the bone-length gate.**

Mitigation, since we cannot retrain the detector: gate each observation on
**cross-view epipolar agreement** per joint per frame. Views disagreeing beyond
~2× the measured 4.23 px finger residual mean that observation is dropped and the
frame flagged `review_required`. Robust loss on the hand reprojection term.

Three more, in order:

- **Flip-basin hysteresis.** Curl-toward and curl-away are near-degenerate per
  view. With ≤2 usable views during contact the solver can settle in the mirrored
  basin and temporal smoothing makes it sticky. Re-initialise from triangulated
  points whenever ≥3-view support returns; let the temporal term break at
  visibility gaps.
- **Wrong-hand assignment at contact.** Overlapping crops contain up to four
  hands and our cycle-consistent matching is body-level. Gate hand observations
  on wrist-distance sanity before the solve.
- **Fit hierarchically** — palm frame first, fingers second. Wrist orientation
  error swings fingertips 20–35 mm, and a flat solve will trade curl against
  wrist rotation.

Replicate MANO's finger-correlation benefit with animation-standard soft
couplings (DIP ≈ ⅔ PIP, spread limits, L2-to-rest). Budget a week of tuning
there; that is where unanticipated time goes.

## Expected accuracy — calibrate ambition here

MAMMA's own hands-only MPJPE: **16.90 mm** on its singles eval, 26.62 mm RICH,
**47.95 mm on two-person contact**, 78.36 mm Harmony4D — with dense
hands-weighted landmarks, trained uncertainty, and a 512×384 input.

Our deltas: half the linear crop resolution (256×192), 15 sparse points per hand
rather than dense coverage, no trained uncertainty channel. Partially offset by
ViT-Huge against their ViT-Base. Camera count is *not* a delta — 4 vs 16 costs
~2.7 mm.

**Estimated: ~30–60 mm hands-only on free gesture, 60–100 mm during clasp.**

That clears *pose-plausible for dialogue* — open versus closed fist differs by
60–80 mm at the fingertips, so grasp state, pointing, spread and coarse curl
resolve above the noise. It does **not** clear contact precision, which N5.1
already concedes and whose `review_required` rule already covers.

## Acceptance gates

- finger pairwise-distance stability **≤ ~10%**, against MAMMA's 7.3% on the same
  footage and our failed 226%
- articulation variance in a **band** around MAMMA's 0.098 rad/DoF — not
  collapsed, not thrashing. A band, not a pass criterion: variance parity does
  not prove correctness
- side-by-side review against the retained MAMMA take on the same frames
- clasp frames failing the visibility gate stay `review_required`
- **stop rule:** if the fit cannot beat ~15% stability after the tuning week,
  record it and fall back to Battle 3→4. The solver is reused there regardless.

One caution both advisors state: MAMMA's 7.3% stability and 53%-of-body
articulation prove its fingers are *consistent and moving*, not *correct*. There
is no ground truth on this stage — that is Battle 2.

## One cheap experiment worth trying first

MAMMA ran ViTPose-B above its native resolution by interpolating positional
embeddings. The same trick on SOMA-77 (192×256 → 384×512) would double the hand
pixel budget with **no rig change**. Unverified whether the checkpoint tolerates
it; an afternoon to re-run the finger gate, minding the centre-192-columns crop.
