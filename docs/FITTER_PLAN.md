# The fitter — plan of record for the largest structural gap

**Snapshot 2026-08-31.** Written after the user asked why MAMMA's mesh overlays live
action and ours does not. The answer was architectural: **MAMMA fits a body model, we
fit nothing.** This is the plan to close that, with the bands it must pass.

Tags: `[T]` derived from this repo's own measurements · `[R]` recalled, unverified ·
`[D]` a document that must be read.

---

## 0. The reframe: it is a sequence, not a fork

The choice was posed as *build an MHR fitter* **or** *take the HumanIK route*. It is
neither-or; they are stages of one build.

| stage | what it is | status |
|---|---|---|
| 1. schema + gate | mandatory `rest_translations_m` on `BodyTrack`, three-layer gate, the `0.98` hardcode | **designed** (§Step 0d of the plan), days |
| 2. **MVP skeleton fitter** | per-take skeleton + per-frame pose, jointly optimised against landmarks | **the new build** |
| 3. HIK characterisation | `HIKSkeletonGeneratorNode` fed the fitter's skeleton instead of SMPL-X's | consumes stage 2 |
| 4. dense shape | girth, silhouette, mass | waits for the Battle 2 scan |

Stage 1 is the fitter's **delivery mechanism**; stage 3 is its **export target**.
Nothing in the HIK route is throwaway — without a fitter it is a skeleton with no
optimised pose; without stage 1 the fitter has nowhere to put its result.

**And the mirror dissolves.** The fitter's skeleton is built from **measured
positions**, which are correct-handed by construction. `DETAILED_HUMANOID`'s signs are
never touched. After three code-only attempts each failed a different gate today, the
fourth succeeds by not being an attempt.

---

## 1. What 19 landmarks can and cannot fit — the question that decides feasibility

`[T]` **They admit a skeleton, not a shape.** MHR's shape space carries dozens of
identity parameters; 19 joint centres give at most the ~13 inter-landmark distances
`estimate_limb_lengths_m` already computes, aggregated over a take. That is not a
shape estimate — it is a **skeleton** estimate, which is exactly where every defect
chased on 2026-08-30/31 lives (shoulder width 540 vs 355, per-performer sizing, joint
placement).

**MAMMA needs 512 dense surface landmarks *because* it solves dense shape.** We have
neither the landmarks nor — per §10's open product question — a demonstrated need for
dense shape before the body scan exists.

**So the MVP is:** fit the **skeleton-affecting** parameters to take-aggregated
landmark constraints, hold dense shape at the model mean, and solve **pose per frame**
against the same objective.
*What it will fix:* limb lengths, joint placement, stance width, stature, and the
facing — all by optimisation rather than hand-correction.
*What it will not fix:* girth, mass distribution, silhouette. **Sizing fixes
proportions; the scan fixes the body.**

`[R]` **Load-bearing and unverified:** whether MHR's parameterisation cleanly
separates skeleton-proportion parameters from surface-shape parameters. Its
documentation claims a decoupled skeleton/shape rig; if true this MVP is natural, if
false the MVP becomes fitting joint offsets on our own canonical topology directly and
MHR waits.

---

## 2. The band — against truth, never against looks

**Primary instrument: the synthetic fixture.** Its documented blindness is to the
*sequence solve*, which never fires there; **a fitter touches every slot**, so the
fixture prices it against exact 3D truth. Same argument as the constrained
re-estimator entry.

**Pre-registered:** fitted-skeleton FK joint positions against truth must beat the
current pipeline's delivered-vs-captured figure of **105.4 / 120.8 mm** by at least
**~44 mm** — the amount oracle bone-length substitution already recovered. A fitter
that cannot match what handing it the right lengths achieves is not worth keeping.

**Secondary, on real footage:** delivered-vs-point-skeleton median must fall from
**11.7 px** toward the point skeleton's own reprojection spread (~4–5 px per §8).

**Two degenerate solutions the band must reject:**
1. **The mean-body regressor.** A fitter that ignores the observations and returns
   canonical would look entirely plausible. *Test:* feed two synthetic subjects with
   deliberately different proportions; the fitted skeletons must differ **in the right
   direction, by the injected amount**.
2. **The overlay-flatterer.** A fitter optimising reprojection can hide depth error
   along the camera ray. The instrument table bans reprojection as a gate for exactly
   this reason, so **the primary metric stays 3D-against-truth on the fixture — never
   the overlay**, however persuasive the overlay looks.

---

## 3. Three traps, named before writing the objective

1. **Do not fit per-frame bone lengths.** Lengths are per-take constants, solved once
   as the median over directly-triangulated frames — the estimator we already have.
   Per-frame lengths eat detector noise and pulse the skeleton, which is the failure
   `estimate_limb_lengths_m`'s own docstring warns about.
2. **Do not optimise against smoothed positions.** Already pre-registered for Step 4's
   labels, and the same reason applies: the fitter would learn the smoother's lag. Fit
   against **raw** triangulation with the confidence gating already validated.
3. **Do not let the fitter absorb the ground projection.** `project_generated_foot_contacts`
   runs afterwards and shifts the root ~90 mm; if the fitter also solves global
   placement the correction is applied twice. **Decide the ownership boundary — fitter
   owns pose and skeleton, projection owns ground contact — before writing the
   objective.**

---

## 4. To verify before building

- `[R]` MHR's skeleton/shape decoupling and its actual parameter surface.
- `[R]` Whether `HIKSkeletonGeneratorNode` accepts arbitrary proportions without
  HumanIK's own retargeting fighting them.
- `[D]` The MHR repo's licence position on **asset** files — mean mesh, skinning
  weights — as distinct from code and weights. One read of the LICENSE and asset
  manifests; cheap now, expensive after a build.
- `[T]` The ~44 mm band figure, derived from the oracle-substitution numbers. Check
  the derivation rather than the arithmetic.

**If the decoupling or the HIK proportion claim fails**, the MVP becomes fitting our
own canonical-topology skeleton directly — Fable's schema already carries it — and MHR
waits.
