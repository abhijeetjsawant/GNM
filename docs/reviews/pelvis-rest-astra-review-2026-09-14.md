# Astra GPT6 review of the D7c card — 2026-09-14, in Sol's seat (round 1)

Invocation: `codex exec -m gpt-6-astra -c model_reasoning_effort="xhigh" -s read-only`, the brief
`docs/reviews/pelvis-rest-astra-brief-2026-09-14.md` on stdin; 126,668 tokens; Astra reran both oracle arms on the
six bodies in memory and recomputed the guard mask from the retained inputs. Verdict: **do not dispatch as written**.
Every code claim below was verified against the source before adoption (`d7b_silhouette_partwise.py:327` the
`ci95[1] >= 0` predicate; `commercial_multiview.py:2017` `_frame`, `:1537` the length rule's stated blindness to
direction, `:1490` demote keeps the rays, `:2245` the spine's `np.interp`, `:2223` the mode dispatch, `:2882` the root
line; `body.py:41/995` `CONTACT_TOLERANCE_M = 1e-5` checked on Foot and Toes; `body_projection.py:1186` the
`candidate_local` lock; `provenance.py:338` the template registered THIRD_PARTY, MAMMA-free; `d7_pelvis_synthetic.py:260`
every positive-depth camera; `status.py:100` `decide` only adds or removes text). Astra's own numbers reproduced where
this session could check them: the shipped oracle carries 0.033–0.036° yaw and 0.005–0.035° roll beside its 6.865°
pitch (`precard-oracle.json`), so cross-build leg identity (O2) is impossible under an exact hip line; the guarded
pitch-change p95 on performer 1 is 16.5° by the angle (Astra: ~15.9° by the pitch), not the 23.1° the draft quoted
(that was the unguarded figure); the demoted frames on performer 1 are 24–46 (38–46 nine consecutive, interpolated
between 37 and 47, not held) and 140, 141, 144, 145, 147, 148, 149 (142, 143, 146 valid; only 147–149 a terminal hold
from 146).

## What each finding changed in the card

| # | finding | change |
|---|---|---|
| 1 | B1's predicate (`ci95[1] >= 0`) establishes "worsening not shown", not "not worse"; no landmark instrument resolves the pelvis convention; the lying run's depth error is invisible to the silhouettes | B1 reworded to "worsening not established (CI upper bound ≥ 0)"; the convention is declared UNRESOLVED and handed to lane H; performer 1's lying end named as unresolvable by the photographs |
| 2 | a length-honest hip line can be wrong in DIRECTION and the primary-axis construction gives that error full authority (counterexample: 10° hip rotation about the midpoint → 10° / 18 mm under (b), 3.6° / 6.5 mm under Kabsch); window frames 100–102, 104, 106 are D8c's unresolved A–C stretch | the "leg-root error below the alt's BY CONSTRUCTION" clause deleted; (a) vs (b) is a genuine selector question and S decides it; those five frames named and excluded from any directional claim |
| 3 | the guard is a DIFFERENT mechanism from D8b/D8c's demote (discards samples and interpolates world coordinates; demote keeps rays); `_frame_alignment` normalises both axes, so lever length does not weight it; the median must be frozen from the pre-guard input; the draft's 23.1° "guarded" p95 was the unguarded figure; guarded contacts/hoist need the full path | guard described as new and scored in S against synthetic truth WITH its gaps; the lever wording corrected; the median frozen from the unchanged pre-guard input; 23.1 → ~16; guarded delivered figures marked "the agent measures, not pre-registered" |
| 4 | SOMA truth cannot select the rig convention, rig truth cannot establish anatomy; STEP must be the full relative rotation; the missing degenerate is a hip-line follower with frozen/attenuated pitch; both estimators get the same guarded Spine1; the noise path uses all positive-depth cameras, not the A–C support | S rewritten: full-rotation increment error, root-step vector error, the frozen-pitch follower as a must-fail, frozen draws/masks/aggregation, same input to both estimators, the camera-support limitation stated |
| 5 | cross-build planted-foot identity cannot hold; O2 already fails (leg FK 0.054–0.077 mm, hoist 0.008–0.032 mm) because the shipped pelvis carries yaw/roll; replace with a projection-preservation contract | O2 → legs/feet/toes within 0.1 mm and contacts identical per seed; new P: the final GLB preserves the single projection's root, contact mask and foot/toe channels, Foot and Toes at their run anchors within 1e-5 m, with two controls (overwrite `candidate_local`, clear contacts) that must be detected; travel on a fixed population from both builds |
| 6 | one C execution can serve tripwire and must-fail with separate references | stated so; the take's 8-file tripwire stays a separate fixture; not counted twice |
| 7 | report the six aligned arm values; keep the standing FAIL; update the published 2.72 decision text at close-out through `status.py decide` | added to the close-out list |
| 8 | GLB checks the oracle cannot make: sampler times/interpolation, quaternion norms/signs/increments, rest/hierarchy/IBM vs the sized skeleton, rotational closure, delivered Head vs the retained head solve, pelvis/hip/thigh mesh deformation | B6 added (report): the delivered-bytes checks, and a first mesh-deformation reading (inverted/collapsed triangles, edge and area change on the pelvis/thigh region) — D6's "instrument first" |
| 9 | instrument-only constants need containment proof, mode and provenance recorded in the build; move to `tools/compare/` later | a test that deletes the four constants and rebuilds the rig mode bit-identically; `pelvis_frame.mode` already in the run-report; the move handed to instrument debt |
| 10 | B2 checks implementation, not direction ("angular/transverse residual zero"); the unguarded candidate is not a must-fail unless a conjunct fails it; the alt's residual to the observed hip line is not evidence against truth; wrong-origin needs the unnormalised metre residual; a non-discriminating control is a limitation | all five adopted verbatim |
| 11 | the "two held runs" were wrong (see above); holding a world point is not holding pitch; do not switch runs to C | corrected; the gap pattern is injected into S on moving truth |
| 12 | no take-speed ceiling; report vector velocity/acceleration of Root/Hips/Spine from GLB times, separating midpoint motion, rotational compensation and projection; gate root-step VECTOR error vs truth in S | adopted |

---

**Do not dispatch this card as written.** The rest-frame defect is real, but the card contains falsified predictions, an incorrect gap description, and acceptance clauses that do not establish the claims attached to them.

I checked the source, retained inputs and reports, reran both oracle arms on all six bodies in memory, and reproduced the guarded pelvis measurements. No files were changed.

1. **The remaining convention: B1 is a delivery check, not an anatomical referee.**

   The historical predicate really is `ci95[1] >= 0`: it accepts a confidence interval containing substantial worsening. That means **“worsening was not established,” not “not worse.”** Retain that wording if retaining that predicate; establishing non-worsening at zero requires the lower bound to be nonnegative. This distinction is explicit in the [silhouette code](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7b_silhouette_partwise.py:327).

   There is no second absolute-pelvis-convention oracle in these landmarks. A take-fitted rigid template remains ambiguous under a constant change of frame; rigidity, cross-view agreement and held-out landmark prediction can expose inconsistent tracking without resolving that constant. The existing [rigidity instrument acknowledges its anatomical blindness](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7_pelvis_rigidity.py:26).

   Use per-camera photographs and deformation checks to judge the delivered consequences. Keep the anatomical convention unresolved. In particular, the D8c review already identifies a lying run where the available silhouettes cannot resolve the relevant depth error. Passing B1 there does not settle the convention.

2. **Yes: a length-honest hip line can make the primary-axis construction worse.**

   `_frame` preserves the primary direction and removes its component from the secondary. It therefore gives the hip line’s directional error full authority. The length rule explicitly cannot detect same-length rotations. See [_frame](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2017) and [the length-rule blindness](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1537).

   I tested a concrete counterexample using the card’s dimensions: exact Spine1, hips rotated together by 10° while preserving their midpoint and width. The length guard passes exactly. Hipline-primary produces **10° pelvis error and 18.13 mm per-leg-root error**; Kabsch produces **3.576° and 6.49 mm**.

   On this take, name performer 1’s **source frames 160–162, 164 and 166**. They are the five under-ceiling holes in the A–C stretch run, at +9–13% width. The [D8c review](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/hip-line-2026-09-06.md:90) identifies them as unresolved stretch, not independently verified directional truth. In the 60–210 window these are indices **100–102, 104 and 106**. Do not mix those source IDs with the guard’s window indices.

3. **Keep the guard in the evaluated candidate, but stop describing it as an inherited, already-proven remedy.**

   D8b/D8c demotion retains rays for sequence recovery. This guard discards Spine1 samples and interpolates world coordinates. Those are different recovery mechanisms; the shared ceiling does not validate the new one. Compare [D8c’s demotion](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1490) with [pelvis interpolation](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2245). Keep an unguarded ablation and score guarded reconstruction against synthetic truth, including its gaps.

   Also correct the lever explanation: **hipline-primary normalises both source axes**, so its angular jitter is not caused by a 197 mm source weight. Its observed lever is approximately **132/125 mm**. Rest lever length affects Kabsch weighting and downstream placement; it does not weight `_frame_alignment`.

   The subject median is a reasonable operational reference, not established anatomy. Freeze it from the unchanged **pre-guard input**, before interpolation, and use that same value and frame population for all comparisons. Here converter-only inputs are identical, so D8b’s moving-denominator problem is avoidable without changing the estimator. “29 rejected frames” is not independent proof that the median is uncontaminated.

   Finally, **23.1° is performer 1’s unguarded pitch-change p95**. Reproducing the stated guard gives approximately **15.91°**. The guarded JSON does not establish guarded contacts, hoist or delivered-joint movements; those require the full guarded converter/export path.

4. **SOMA truth should not select the rig’s convention. S still needs repair.**

   Excluding SOMA-posed truth from that selection is sound. Conversely, rig-posed truth establishes correctness and noise sensitivity under the rig convention; it cannot establish that convention’s anatomical accuracy.

   **“Leg-root error below the alt on every seed, by construction” is false.** It is true that the candidate follows the *observed* hip direction exactly. It is not true that this direction is closer to *truth*. Question 2 supplies a counterexample.

   The missing degenerate is a **hip-line follower with frozen or strongly attenuated pitch**. Leg-root placement cannot discriminate pitch around the hip line. Aggregate orientation and step scores can also tolerate this when pitch excitation is weak relative to noise. Its failure must be demonstrated on the actual selector; I would not assert without running S that it passes this particular fixture.

   Define STEP as error between full relative rotations, not the difference between scalar step magnitudes: opposite rotations can have identical magnitudes. Freeze the donor motion, masks, noise draws and aggregation before selection. Give both rig estimators the same guarded Spine1 input when comparing their geometry.

   The inherited [noise path](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7_pelvis_synthetic.py:260) uses every positive-depth camera; it does **not** reproduce the measured A–C-only support simply by using the same cameras.

5. **Replace cross-build planted-foot identity with a projection-preservation contract.**

   Capture the candidate immediately after its **single** projection. Require the final converter and exported GLB to preserve that projection’s root, contact mask and protected leg/foot/toe channels. Reconstruct each accepted contact run from the GLB and require both Foot and Toes to remain at their run anchors under the existing **`CONTACT_TOLERANCE_M = 1e-5`**. The [current validator already checks both points](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/body.py:995); export needs its own check.

   Preserve the projection-produced mask so clearing contacts cannot make the lock check vacuous. Report travel on a fixed population derived from both builds as well, so lost contacts do not disappear from the comparison. Explicit controls should overwrite `candidate_local` and clear contact flags; both must be detected. The [projection writes those locks here](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/body_projection.py:1186).

   **O2 itself already fails.** My six-body rerun found different leg locals and different leg FK positions on every seed; maximum FK differences were **0.054–0.077 mm**. Contacts matched. Recovered projection-vector differences reached **0.008–0.032 mm**, so a per-frame 0.01 mm hoist-identity clause also fails on five seeds.

   This is not merely numerical noise: the shipped oracle report contains approximately **0.033° yaw and 0.005° roll** alongside its pitch. Moreover, changed pelvis orientation requires compensating upper-leg locals even if world legs remain fixed. Replace O2’s impossible cross-build identity claim; retain strict preservation **after each candidate’s own projection**.

6. **One execution can legitimately serve both purposes.**

   The refactored C output should equal the old C output and disagree with exact rig truth. Those are different references answering different questions.

   Reuse the C oracle execution, with separate verdicts and explicit references. The take’s eight-file tripwire remains a separate fixture. Do not count the shared oracle execution as two independent demonstrations. The [mode dispatch](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2223) makes this arrangement straightforward.

7. **Update the standing decision record; do not manufacture a new waiver.**

   Report the six values and retain the overall standing FAIL because one seed remains above 0.5 mm. Five improved seeds do not require re-pinning or fresh permission to retain the existing band.

   The published decision currently describes **2.72 mm**, so reporting only in D7c would leave it stale. At close-out, log the new measurement and update the existing decision text through the status tooling, preserving the unresolved gauge/reference debt. [`status.py decide`](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/status.py:100) adds or removes pending decision text; it does not itself approve a gate exception.

8. **Add these checks from the final GLB’s own bytes.**

   Measure actual animation timestamps, duration, channel coverage and interpolation; quaternion norms, adjacent signs and full rotation increments; and samples between keys around gap/contact boundaries. The existing [GLB FK reader](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d3_skeleton_gate.py:169) reads output arrays but ignores sampler input times and interpolation.

   Check the GLB’s rest, hierarchy and inverse bind matrices against the intended sized skeleton, plus track-to-GLB positional **and rotational** closure. Check the delivered Head world rotation against the retained absolute head solve—the head-input gate alone cannot prove the exporter preserved it.

   Measure pelvis/hip/thigh mesh deformation: inverted or collapsed triangles, edge/area changes and local surface distortion. IoU can improve while the skin balloons or tears; the [silhouette instrument explicitly admits this](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7b_silhouette_partwise.py:30).

   Include the contact-anchor checks in question 5 and the expected Root/eye/finger-local invariants. Refusing global bit identity does not remove these narrower contracts.

9. **Not automatically a §4 leak, but “instrument-only” needs containment.**

   These constants are recorded as third-party geometry, explicitly MAMMA-free, in the [provenance registry](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/provenance.py:338). Their presence is not proof of a MAMMA leak.

   A default selector alone is weak containment. Prove the normal delivery path cannot read them, including missing-data paths; record the selected mode and provenance in the build.

   I would ultimately move the legacy estimator and constants to `tools/compare/`, preserving their arithmetic for the tripwire. Establish equivalence first. Production should not import that instrument module. Keeping them temporarily for the refactor is acceptable with demonstrated absence from the rig mode’s data flow.

10. **Several clauses admit constants or directly reward the construction.**

    Hip-line angular residual is zero by construction; half-span norms are rotation-invariant. B2 therefore verifies implementation and placement consistency, not directional accuracy. The card’s “per-hip residual is zero” must say **angular/transverse residual**: the full positional residual still includes width mismatch.

    Lever spread and counts above 800°/s can improve by freezing or replacing observations. Their report-only status is appropriate. But calling the unguarded candidate a **must-fail** is inconsistent when those quantities are absent from the merge predicate. Identify the actual conjunct it fails.

    Likewise, the alt’s greater residual to the observed hip line is expected from its objective. It is not independent evidence that it is worse against truth.

    The wrong-origin control is useful only if O1 measures the **unnormalised three-point positional residual in metres**. A residual between normalised frames would lose the origin discrimination. The [pre-card computes the appropriate positional residual](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/precard/d7c_measure_oracle.py:127).

    Finally, declare a non-discriminating constant control a limitation requiring a discriminating test—not a successful gate.

11. **The “two held runs” are factually wrong.**

    Recomputing the 0.15 mask from the [retained performer-1 inputs](/Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/compare/d7c-pelvis-rest/precard-take-hipline/converter-inputs/call-01.npz) gives:

    **38–46:** nine rejected samples, linearly interpolated between 37 and 47. The interpolated Spine1 moves **12.29 mm per frame**. It is not held.

    **140–149:** rejected indices are **140, 141, 144, 145, 147, 148, 149**. Samples **142, 143 and 146 remain valid**. Only **147–149** form the terminal hold, using **146**, not 139.

    There is no nine-frame smoothing cutoff in [this interpolation path](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2245). CLAUDE.md’s capture-smoother continuity horizon does not change `np.interp` semantics.

    Holding a **world point** is also not holding pelvis pitch: moving hips change `held_spine − hip_mid`. Test that behavior on moving exact truth with the observed gap pattern.

    Do not automatically switch those runs to C. C-on-rig-rest is not today’s short-lever, hips-dominant fit. It shares the rig convention, so this is technically an estimator switch rather than SOMA-to-thorax convention switching; nevertheless it can move the hip line and create boundary discontinuities. D7’s [continuity rationale](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2191) still applies.

12. **Keep absolute take root speed reported; gate temporal reconstruction error.**

    Do not invent a take-speed ceiling from D9b. Root translation includes compensation for pelvis orientation; it is not an independently observed centre-of-mass trajectory. The [root equation](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2882) gives, before projection,
    \[
    \Delta r=\Delta hipmid-(R_t-R_{t-1})\,mid.
    \]
    The 17 mm figure is therefore a possible rotational contribution, not automatically 17 mm of erroneous body travel.

    Report vector velocity and acceleration for Root, Hips and Spine from actual GLB times, separating midpoint motion, rotational compensation and projection. Review them with the frame player.

    For acceptance, extend S to compare **root-step vector error against exact truth**, alongside full pelvis-increment error, with guarded gaps and the frozen-pitch control included. Use the same truth and draws for both estimators. That tests noise amplification and interpolation damage without asserting a new physical-speed constant.

    D9b-derived truth can support that conditional reconstruction test. It cannot establish how much real pelvic motion the performer had. Neither a report row nor a synthetic pass should be presented as resolving that remaining uncertainty.

The required review record could not be saved under `docs/reviews/`: this session’s filesystem permissions are read-only.

---

# Round 2 — 2026-09-14, on the rewritten card. Verdict: not dispatchable yet, four blockers

Verified before adoption: `d7_pelvis_synthetic.py:503` D7 selected on the pooled median orientation error alone; `commercial_multiview.py:2257` legacy C reads the SOMA template explicitly, `:3112–3116` the D9b re-solve preserves exactly Root/Hips/legs/feet/toes/root/contacts, `:2017` `_frame` normalises the secondary (a radial +30 % lever error is invisible to (b)); `body_export.py:648` the GLB carries `body_track_sha256` and no contact mask, `:615` LINEAR samplers; `body.py:1008` the validator anchors each run at its first keyed sample; `delivered_vs_capture.py:524` the same-denominator test is landmark-array equality; the complete 29-frame mask recomputed from `call-01.npz` (median 125.4928 mm) matches Astra's list exactly.

## What each blocker changed

| # | blocker | change |
|---|---|---|
| 1 | S had no executable decision | S now fixes the statistics (geodesic orientation error, full-rotation step error, root-step vector error; pooled medians per body; median of six across bodies), populations (whole take, bent tercile = the 50 largest-truth-tilt frames), ties (0.1° / 0.1 mm), the rule, and three stops |
| 2 | the card let (a) win but delivered (b) | two rig modes (`D_rig_rest_hipline`, `E_rig_rest_kabsch`), the guard on both, `PELVIS_FRAME_SOURCE` set to S's winner, the hip-line predictions conditional on (b) |
| 3 | P conflated three contracts; mask source and protected legs missing | P1 channel preservation against the post-projection snapshot (root, mask, Root/Hips/legs/feet/toes locals; the delivered track authenticated by the GLB's hash), P2 keyed-sample anchor lock, P3 travel on the frozen union; the two controls named against P1; between-key playback to B6 |
| 4 | the rejected-frame list was wrong | the complete mask frozen in the card |
| 3(q) | the frozen-pitch follower undefined | a control law with no constant: observed hip line primary, world +Y orthogonalised as up, consumes no Spine1 |
| 3(q) | a +30 % radial lever error does not corrupt (b) | the guard experiment injects MISSING samples on performer 1's pattern and FINITE wrong-direction samples (another frame's Spine1), scored on transitions, with its own stop |

---

**1. Not dispatchable yet. Four blockers remain, in order.**

1. **S does not yet define an executable decision.** “Better,” “beats,” and “fails on orientation or step” need a specified statistic, comparison reference, aggregation across seeds and whole/bent populations, and treatment of ties. Freezing an aggregation later is not the same as specifying this acceptance rule. The inherited instrument cannot supply it: D7 selected by **pooled median orientation error**, not the new three-metric predicate. [Deciding source.](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7_pelvis_synthetic.py:503) The frozen-pitch control and guard-ablation cases also need the definitions described below.

2. **The card permits (a) to win but specifies delivery as (b).** The mechanism installs `D_rig_rest_hipline`, applies the guard only there, and promises zero transverse hip residual. S can instead select guarded C-on-rest, which has neither that primary-axis guarantee nor a specified production mode. State how the winning estimator reaches production, with the same guarding tested in S; make the hipline-specific predictions conditional on (b). Keeping legacy C untouched for the tripwire is compatible with adding a separate rig-rest Kabsch mode. Legacy C currently reads the SOMA template explicitly. [Deciding source.](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2257)

3. **P needs its mask source, protected channels and time domain fixed.** The GLB does not contain contact flags; the current validator checks anchors at sampled frames; and the rewrite dropped the protected **leg** channels from my recommendation. These determine what P accepts, rather than merely how an agent implements it. Details under question 4.

4. **The full rejected-frame list is still wrong.** Recomputing the stated rule from the [retained input](/Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/compare/d7c-pelvis-rest/precard-take-hipline/converter-inputs/call-01.npz), using the [0.15 ceiling](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1454), gives median **125.4928 mm** and these **29** window indices:

   `24–25, 28–29, 32–33, 38–46, 65, 68, 70, 78–81, 140–141, 144–145, 147–149`.

   “24–46 and [the terminal group]” includes valid frames and omits seven rejected frames between 65 and 81. The stated interpolation between 37 and 47 and terminal hold from 146 remain correct. Freeze the complete mask, or explicitly identify the two long runs as a selected subset.

**2. Walking the merge rule: individual passes are not evidence that the conjunction passes.**

| Conjunct | Constant or degenerate it can still admit |
|---|---|
| Hygiene | Any new candidate: this checks the unchanged shipped execution. |
| Tripwire | Any new rig-mode candidate, including a frozen pelvis, while legacy C remains equivalent. |
| O1 | Rejects a frozen pelvis on sufficiently moving truth. Rejects the wrong-origin control **through the positional residual**, despite its zero tilt. |
| O2 | Can admit a hip-line follower with frozen pitch and compensating root placement/lower-body rotations. Preserving the hip line preserves leg-root placement; it does not establish pitch. |
| P | Can admit an incorrectly estimated pelvis whose own projection is faithfully preserved. P verifies preservation, not pelvis accuracy. |
| S | Cannot yet be adjudicated. The frozen-pitch stop is intended to close the noisy-estimation loophole, but its control and failure predicate remain undefined. |
| B1 | Can admit an anatomically wrong or motion-degenerate result whose silhouette worsening is not established. |
| B2 same-denominator | Any converter-only constant or degenerate that leaves the landmark arrays unchanged. |

The deciding geometry for O2 is the [root-compensation equation](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2101), with legs subsequently aimed from [landmark differences](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2922). B1’s actual acceptance is [the upper-CI test](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/d7b_silhouette_partwise.py:327); B2’s is [landmark-array equality](/Users/abhi_macbook/Projects/apps/AutoAnim/tools/compare/delivered_vs_capture.py:524).

**I have not demonstrated that a listed frozen control passes the entire conjunction.** Conversely, the conjunction can accept a constant **anatomical convention error** shared by the rig truth and candidate. That is the explicitly acknowledged lane-H uncertainty, not an additional dispatch blocker.

**3. The stops are correctly placed; the follower is insufficiently defined.**

A split winner means S has not selected an estimator. A frozen-pitch control that remains competitive means the fixture has not demonstrated the discrimination being claimed. Both should stop the step.

“Frozen / strongly attenuated pitch” still leaves the agent choosing:

- The moving frame in which pitch is measured.
- The initial or reference pitch and where it comes from.
- A freezing law or attenuation factor.
- Missing-data behavior.
- The exact loss comparison that constitutes failure.

Specify one control law before results. A fully frozen control needs no tunable attenuation gain, but still needs its frame and initialization defined. My previous review explicitly said its failure must be demonstrated; it did not define an implementation. [Recorded finding.](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:74)

The ablation also needs clarification. “Every estimator gets guarded Spine1” must exempt the unguarded arm; “C-on-SOMA (today)” needs its input treatment stated.

Furthermore, **a +30% radial lever error alone does not corrupt (b)’s direction**:

\[
s'=m+1.3(s-m)
\]

produces the same hipline-primary frame because [_frame normalizes the secondary direction](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2017). Discarding that point can introduce interpolation error. For genuine missing samples, the unguarded path already [interpolates gaps](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2248). Define finite corruptions versus missing samples, injection order and scored transitions; neither case guarantees that guarding wins. Preserve the stop if the specified experiment does not support the guard.

**4. P’s intended contract is right, but the quoted formulation conflates three contracts.**

**The acceptance mask comes from the frozen post-projection track.** The build writes contacts into the [body-track artifacts](/Users/abhi_macbook/Projects/apps/AutoAnim/scripts/build_commercial_multiview_comparison.py:427). The GLB exports translation/rotation channels and a [body-track hash](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/body_export.py:648), not a recoverable contact mask. Therefore: authenticate the exported track against that hash, preserve its mask against the projection snapshot, and evaluate positions from the GLB’s bytes on those authoritative runs. Do not infer accepted contacts from low GLB velocity.

**Anchor enforcement and union-population reporting should stay separate.** Enforce the candidate’s accepted runs at their projection anchors. Report both builds’ travel on the frozen union of run intervals, retaining each interval’s side and boundaries. Do not demand that the candidate remain planted during a baseline-only run; the card explicitly allows contact selection to change.

**Specify keyed samples for the inherited tolerance.** The validator compares [sampled positions against each run’s first sample](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/body.py:1008). It does not establish continuous playback locks.

I sampled the shipped GLBs using their own timestamps and LINEAR samplers, including quaternion interpolation:

| Performer | Maximum anchor error at keys | Maximum at interval midpoints |
|---|---:|---:|
| 0 | 0.000424 mm | 0.665 mm |
| 1 | 0.000463 mm | 1.311 mm |

The tolerance is **0.01 mm**. Consequently, a continuous-time interpretation already fails the shipped delivery. The [exporter specifies LINEAR interpolation](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/body_export.py:615). Keep between-key measurements in B6 as proposed, or explicitly scope additional work to enforcing continuous locks.

Restore protection of the **Root/Hips and upper-/lower-leg locals**, alongside root translation, contacts and foot/toe locals, after projection. The current converter preserves precisely that ancestry. [Deciding source.](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:3112)

The two controls then fail different assertions:

- **Overwrite the foot lock:** a nontrivial change to projected foot locals fails channel preservation. Foot-position locking alone need not detect it—the foot’s own rotation does not move its origin, and some twists also preserve the toe point. Specify the overwritten values and verify the injection changed a protected channel.
- **Clear contacts:** clearing a nonempty frozen mask fails mask preservation. It does **not** necessarily fail the geometric lock test; the unchanged geometry can remain planted.

Thus both can be guaranteed detectable by the complete P contract, **not by the anchor check alone**.

**5. Two adoption errors remain.**

The full-mask shorthand is incorrect, including in the adoption header; my earlier answer should have supplied the complete mask rather than only correcting the disputed runs. And the rewrite omitted “protected leg” channels from my [projection-preservation recommendation](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:82).

The frozen-pitch requirement was adopted in intent, but naming it did not finish its specification. The other substantive corrections reviewed here reflect round 1 accurately.

Read-only throughout. I recomputed the mask and sampled the shipped GLBs; I did not execute the still-unimplemented S.

---

# Round 3 — 2026-09-14. Verdict: not dispatchable yet, three blockers (specification)

Verified: `commercial_multiview.py:2248` guarded and unguarded arms share the `np.interp` recovery, so a missing-only trial cannot show superiority (Astra checked all six bodies: bit-identical); 78–81 is a four-frame run in the recorded mask.

| # | blocker | change |
|---|---|---|
| 1 | contradictory guard scope | the guard runs under BOTH rig modes, never under A/B/C |
| 2 | no temporal-pair populations | pairs formed on the full sequence, in a population iff both endpoints are; same for transitions |
| 3 | the guard experiment had no executable protocol; missing-only cannot be a strict win | G1 missing-only = bit-identity equivalence + reported recovery error; G2 finite-only with a seeded run/donor law, the superiority stop located there, the guard's miss rate reported |
| — | "one- or two-frame gap" false for 78–81 | corrected |

P1/P2/P3 confirmed as the right split; both controls fail P1 by construction.

---

**Not dispatchable yet. Three blockers remain:**

1. **Contradictory guard scope.** “The guard applies to BOTH rig modes” conflicts with “the guard runs only under `D_rig_rest_hipline`.” Delete the latter restriction; production must use the same guarding S evaluated. This was explicitly [round 2’s requirement](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:185).

2. **S defines frame populations, but not temporal-pair populations.** For the bent tercile, does `(t−1,t)` qualify when **t** is selected, when **both endpoints** are selected, or when either is? The same question applies to corruption/transition scoring. These yield different STEP/root-step medians. Specify the pair masks; never difference successive entries of the filtered 50-frame array. The deciding omission is between “50 frames” and “pooled median over frame pairs” in [the card](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-brief-round3-2026-09-14.md:7).

3. **The guard experiment still lacks an unambiguous executable protocol/verdict.** Specify separate versus combined corruption trials, the finite-run/donor sampling law, and overlap handling if combined. If “beats” applies independently to the missing-only class, it cannot pass: **I checked all six bodies; zero additional finite samples are rejected, and guarded/unguarded interpolated arrays are bit-identical.** Both already use [the same `np.interp` recovery](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:2248). Make missing-only recovery an equivalence/error measurement; explicitly locate the superiority stop on finite corruption, or explicitly define a combined experiment.

**S:** The frozen-pitch law and numeric ties are now executable. The 50-frame orientation population is defined; the temporal populations and corruption experiment need the fixes above.

**P:** Yes—the split is right. Both controls fail **P1 by construction**, provided they mutate the downstream candidate while the snapshot stays immutable. Neither must fail P2; P3 correctly remains reporting. That matches [round 2’s deciding assertions](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:255).

**Round-2 adoption:** The missing-only strict-win interpretation overlooks its [explicit interpolation warning](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:234). Also, the complete mask is correct, but “every other rejected frame is a one- or two-frame gap” remains false: **78–81 is four consecutive frames** ([recorded mask](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-review-2026-09-14.md:191)).