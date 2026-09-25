# Astra GPT6 card review of D4i (2026-09-25), one round at MEDIUM effort. Verdict: NOT DISPATCHABLE AS WRITTEN; two blockers, both adopted

Verified against the source before adoption:
- `d4_glb_closure.py` and `d4_b2_same_denominator.py` write a failed verdict and still exit 0;
- `d4_b3_placement.py` takes the shorter frame count (`min(world.shape[0], capture.shape[0])`);
- `silhouette.py` accepts its mesh cache by modification time;
- the build's `--mhr-python` default points into `/tmp`.

| # | finding | change |
|---|---|---|
| B1 | the migration (`post_merge.sh`, `ladder.py`) happened AFTER the verdict; the close-out outcomes and the rollback scope were not frozen; exit status is not a verdict | the coordinator's edits now happen on the branch BEFORE the merge review, with the final roster run recorded there; `post_merge.sh` takes a delivery-dir argument; the close-out outcomes are frozen; the FULL ROLLBACK is named (revert, restore the archived rig delivery, the old roster, republish, FAIL); every entry's own verdict field is read, never the exit code |
| B2 | the roster conflated observed class, action and post-migration class; the precedence of (b) and (d) was missing; must-fail (i) was unscoped | three fields per instrument; (d) takes precedence over (b); (i) applies in MHR SCOPE to every final class-(a) entry, replacements included |
| debt | the scope claim; exact-row oracle equality; mesh-cache provenance; archive i6 before Phase 1; (iii) is a negative control; positive controls; population completeness; B2 conjoined; no new thresholds for B3 and B4; the spine attribution; the bootstrap records its environment | all adopted into the card's text |

---

**Not dispatchable as written. Two card-contract blockers, in order.** This is a source review; I did not run the instruments or reproduce measurements.

1. **The executable migration happens after the verdict that supposedly accepts it.** The deciding [card line 48](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:48) says “PASS → merge,” then has the coordinator implement `post_merge.sh` and `ladder.py`. That makes substantive, unreviewed implementation part of acceptance after the gate.

   Keep the ownership split, but require the coordinator’s changes on the candidate **before the single merge review**, and run the final roster there. Close-out should verify that reviewed implementation. Also reconcile “every roster instrument exits clean” with rollback triggered only by class-(a) failures: a retained class-(b) crash otherwise has no prescribed disposition. Exit status alone is insufficient: [closure returns zero even when its reported band fails](/Users/abhi_macbook/Projects/apps/autoanim/tools/compare/d4_glb_closure.py:140).

   Freeze explicit close-out outcomes and rollback scope, including restoring the rig delivery and compatible roster. Reverting the default alone leaves MHR files in the delivery directory, which the new mixed-schema refusal should reject.

2. **The roster contract conflates observed behaviour, final disposition and schema policy.** [Line 25](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:25) classifies the existing instruments before modification; [line 44](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:44) calls silhouette class (a) *after* modification. Its existing exporter selects MPFB meshes, so those are different observations. Meanwhile, class (b)’s capture-side scope overlaps class (d)’s unsupported-delivery claim, without precedence.

   Freeze separate fields for **observed class**, **final action/replacement**, and **post-migration class**, with evidence and exact commands. An unsupported delivered-capability GREEN takes precedence over relabelling as capture-side work. Measure unexpected outcomes instead of forcing the predictions.

   Also qualify [must-fail (i)](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:34): an explicitly **MHR-scoped invocation** must reject a rig track before interpreting payload fields. A schema-aware, dual-mode silhouette tool cannot both support rig measurements and universally refuse rig inputs. Apply the refusal test to every final class-(a) entry, including replacements.

Everything else below is **debt or a claim limitation, not an additional dispatch blocker**.

- **N5.1 exclusion is correct for this delivery.** The preview obtains its track from [MAMMA parameters](/Users/abhi_macbook/Projects/apps/autoanim/scripts/build_mamma_gnm_character_preview.py:79); the video acting build [projects SOMA into its own track](/Users/abhi_macbook/Projects/apps/autoanim/scripts/build_video_acting_shot.py:153). The shared `BodyTrack` machinery still [accepts only the supported rig skeletons](/Users/abhi_macbook/Projects/apps/autoanim/src/autoanim_gnm/body.py:398). Record the result as:

  > MHR is the default body for `build_commercial_multiview_comparison.py` and this commercial-multiview delivery’s migrated instruments. D4i does not integrate MHR into N5.1, unified character composition, GNM binding, or the other acting builds.

  “The body model is integrated” without that scope is unsupported.

- **The exactness arm is right.** [Line 20](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:20) tests the correct proposition: changing selection must preserve what explicit MHR already produces. Exact B1 reproduction separately checks the instrument migration. Preserve the full file manifest, reference hashes, subject identities, draws and numeric populations; equality of rounded summaries is weaker than equality of the scored rows. The intentionally corrected run report is outside the per-subject byte oracle.

  There is concrete cache-provenance debt: [silhouette accepts its mesh cache by modification time](/Users/abhi_macbook/Projects/apps/autoanim/tools/compare/silhouette.py:219). An old cache can reproduce old numbers. Bind mesh exports to GLB hashes and exporter settings, or demonstrate a fresh export. Archive and hash the D7c baseline **before Phase 1**, including its potentially destructive legacy commands.

- **Removing class (d) is right.** A GREEN about a head solve absent from the delivered body must not survive as delivery acceptance with a banner. Historical rig diagnostics may remain separately named. The predictions for `head_gate` and `bootstrap_margin` remain predictions until measured; a crash belongs in observed class (c).

- **Must-fails i–iv target the right silent hazards, but do not establish instrument sensitivity.** The shared root key really is hazardous: MHR writes [the captured Z-up root joint](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:410). Schema refusal, mixed-directory rejection and false-head-report rejection are appropriate. The mean-body mutation is a **negative control**: the result should remain unchanged when that unused field changes.

  A constant instrument can pass that negative control, schema rejection and named-population reporting. Per-conjunct fuzz must mutate underlying evidence, not merely supplied PASS flags. Positive controls that alter scored joints or mesh motion should change the applicable reading. These are instrument debt unless assigned explicit acceptance consequences.

- **Naming and counting a population does not enforce its completeness.** The deciding [verdict line 46](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:46) requires reporting populations, not matching a frozen expected set. B3 currently [takes the shorter frame sequence and removes nonfinite distances](/Users/abhi_macbook/Projects/apps/autoanim/tools/compare/d4_b3_placement.py:83). That can honestly report a smaller population. Record expected versus observed subjects, frames, mappings and exclusions; add missing-subject, truncated-frame and substituted-reference mutations if you want them banded. Do not later reject under an unstated completeness band.

- **B2 is rostered but its success is not explicitly conjoined.** It can [report FAIL and return zero](/Users/abhi_macbook/Projects/apps/autoanim/tools/compare/d4_b2_same_denominator.py:138). Likewise, B3/B4 are reported measurements, not newly licensed quality gates. Distinguish execution/schema failure from a scientific result outside an existing band; otherwise “any class-(a) instrument fails” invites post-hoc thresholds.

- **The spine attribution overclaims.** [Card line 5](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-flip-card-2026-09-25.md:5) calls `1.102` a “measured consequence” of missing `Spine1`. The excursion is measured; that cause is not isolated. Say “observed excursion; missing spine information/convention mismatch is a hypothesis.” Preserve D4d’s existing accuracy and generalisation limitations.

- **The interpreter move is in scope.** A default path needs a durable runtime; the current [default points into `/tmp`](/Users/abhi_macbook/Projects/apps/autoanim/scripts/build_commercial_multiview_comparison.py:361). The bootstrap should record the Python/platform and resolved dependencies alongside the pinned momentum version. It must still satisfy the exact-byte oracle; dependency migration does not license relaxed equality.