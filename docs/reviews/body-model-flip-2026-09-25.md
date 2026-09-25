# D4i: the body model becomes the default, and the instrument roster follows it. The measurement (2026-09-25)

**VERDICT: FAIL, on the oracle's JSON leg, read literally. Records only: the default is NOT flipped on main.**

The gate (`tools/compare/d4i_flip_gate.py`) prints one line:

    VERDICT: FAIL (failed: oracle, coordinator_migration_recorded)

**Why it fails.** The card requires the flipped default (no `--body` flag) to rebuild D4d's gated `--body mhr` delivery
BYTE-IDENTICAL, every delivered file per subject, the track JSON included. It does not, and it cannot from any checkout
but D4d's own:
* the frozen fitter, `tools/fitter/mhr_delivery.py`, writes `body_model.assets = str(ROOT / ".cache/mhr/assets")`, where
  ROOT is the fitter's own checkout;
* so D4d's delivery carries `.../worktrees/ladder-D4d/.cache/mhr/assets` and this branch's build carries
  `.../worktrees/ladder-D4i/.cache/mhr/assets`;
* the two `subject-XX.body-track.json` files therefore differ in exactly that one leaf, and in nothing else.

The coordinator's ruling: the leg is read literally, no normalisation is admitted into this card's verdict, and a failed
conjunct is never "an exception" (CLAUDE.md, D4). So D4i is FAIL, and every other figure is reported as measured.

**What the record must say about that failure, all five points:**
1. **The other 10 per-subject files are byte-identical:** both GLBs, both track npz, both `markers.npz`, both calibration
   start and passes sidecars. So are the camera rig and the three converter inputs.
2. **The leaf was predicted before the build.** `docs/reviews/body-model-flip-records/stage3-predictions.json` was
   committed at `af69599`; the default build ran at stage 3 (`59a4e4c`). The prediction held exactly: one leaf per file,
   the worktree segment `ladder-D4d` to `ladder-D4i`.
3. **D4d's tripwire named the same leaf.** It is D4d's one track-JSON normalisation
   (`docs/reviews/body-model-twopass-2026-09-25.md` §7).
4. **It is the coordinator's pre-registration error, the lane's sixth.** The card listed a file carrying a
   checkout-absolute path among its byte-identical files.
5. **The same leaf would have tripped the close-out rollback.** An in-place rebuild on main writes
   `/Users/abhi_macbook/Projects/apps/AutoAnim/.cache/mhr/assets`. The card's frozen close-out ("byte-identical to the
   branch build") would then have fired the FULL ROLLBACK after the merge.

**The counterfactual, REPORTED and never admitted into the verdict** (`fuzz.json`): put D4d's own track-JSON bytes in
place of the default's, and the oracle conjunct holds. The FAIL is those two files and nothing else.

**The second failed conjunct is timing, not measurement.** The card conjoins "the coordinator's post_merge.sh /
ladder.py changes on the branch". `post_merge.sh` is installed (`1242ed0`) and its final roster run is recorded.
`ladder.py` is the coordinator's, edited after this stage by the coordinator's own order. Rerun the gate then; it cannot
move the oracle.

**D4i-b** re-registers the flip with that leaf named: it must equal the building checkout's own
`<root>/.cache/mhr/assets`. It reuses this step's tooling, and B1's arm rename to `delivered_MHR` is D4i-b's.

| stage | commit | what |
|---|---|---|
| 0 | `2146a22` | The D7c silhouette baseline (`artifacts/compare/i6`) archived, read-only, and hashed BEFORE Phase 1 |
| 1 | `5403001` | Phase 1: the OBSERVED class of every roster instrument on D4d's delivery, from a shadow root. The 4530 shared live reports are unchanged |
| 2 | `af69599` | The code changes and their tests; `stage3-predictions.json` |
| 3 | `59a4e4c` | The oracle build and its readings |
| 4 | `188ac7c` | Hygiene |
| 5 | `5464f29` | Must-fails i-v and the population mutations; the final actions; the proposed roster |
| coordinator | `1242ed0` | `post_merge.sh` installed from the proposal; the final roster run recorded |
| 6 | `ea11026` | Post-migration classes; the gate and its fuzz |
| 7 | this commit | Tests, this review, the extractor stub |

The records are in `docs/reviews/body-model-flip-records/`. The logs are in `artifacts/compare/d4i-flip/logs/`.
`src/` and `tools/fitter/mhr_delivery.py` are byte-identical to the step's base `acceb55`.

---

## 1. The pre-registration

The card is the "D4i the body model becomes the default" row of `docs/LADDER_EXECUTION_PLAN.md` §2 (lines 161–224).
`docs/reviews/body-model-flip-card-2026-09-25.md` is byte-identical to those lines (checked with `cmp`). It is
reproduced verbatim here:


| **D4i the body model becomes the default, and the instrument roster follows it** | **Why, and what changed in the plan.** D4 closed on 2026-09-25: `--body mhr` carries D4c's landmark start plus D4d's second pass, and the default is still `rig`. D4's review §6 and the resume brief listed the flip's work as the compositor (`unified_gltf`, the N5.1 assembly), schema-aware checks, the instruments and an end-to-end rebuild. A read-only map of the code (2026-09-25) retires the first item. Nothing in `src/autoanim_gnm` reads this delivery: `unified_gltf`, `body_compositor` and `body_binding` are fed by `build_mamma_gnm_character_preview.py`, `build_audio_acting_shot.py` and `build_video_acting_shot.py` with in-memory `BodyTrack`s, and they need the MPFB mesh and GNM binding that MHR lacks. Binding GNM onto MHR's head is its own step. So D4i is **a default flip plus an instrument-roster migration**, and the first step since D3 whose DEFAULT output bytes change on purpose.

**The change:**
- `scripts/build_commercial_multiview_comparison.py` defaults to `--body mhr`.
- In MHR mode, `run-report.json` marks `head_orientation`, `toe_triangulation`, `spine_triangulation`, `pelvis_frame` and `contact_frames` as NOT CONSUMED by the delivered body (today it reports them as solved in both modes). Its `limitations` state what the MHR delivery does not carry: the multi-frame head solve, toe input, `Spine1`, foot contacts and the ground projection. The real-take spine at 1.102 against its 1.1 limit is an OBSERVED excursion; missing spine information or a convention mismatch is a hypothesis, not a measured cause. It also states that the track's `rest_positions_z_up_m` is the MEAN body, never a rest; that note goes in the run report, not `write_track`, which D4c's and D4d's tests pin.
- The build REFUSES to write into a directory that holds the other schema's files; today a stale `subject-XX.mapping.npz` survives a rebuild.
- The momentum interpreter moves off `/tmp`, which has been wiped once. `--mhr-python` defaults to a gitignored `.venv-mhr/` created by a new `scripts/bootstrap_mhr.sh` (pymomentum-cpu 0.1.114.post0), and a missing interpreter fails with that recipe. The bootstrap records the Python version, the platform and the resolved dependencies beside the pinned momentum version, and the exact-byte oracle must still hold on the new interpreter.
- `scripts/verify_commercial_multiview_artifact.py` gains an MHR branch: schema `2.0-mhr`, 127 joints, pose-value shape, finite joints, and landmark placement through `landmark_to_joint`.
- `tools/compare/silhouette.py` selects the Blender exporter by schema, and its MHR runs use their own `--work`. The D7c rig mesh under `artifacts/compare/i6`, the comparator every B1 since D4 has used, moves to a named read-only baseline and is never overwritten.
- `tests/test_body_model.py` is re-pinned to `mhr`, and new tests cover the schema checks.

Byte-identical across the step: `src/` (including `body.py` and `body_export.py`, which the user's uncommitted tests exercise) and `tools/fitter/mhr_delivery.py`.

**Ownership, and ORDER:** `post_merge.sh` and `ladder.py` are the coordinator's. The agent delivers a ROSTER JSON. The coordinator edits both ON THE CANDIDATE BRANCH, before the single merge review. `post_merge.sh` gains an explicit delivery-dir argument, so the final roster runs against the branch build and never writes into the shipped delivery. The final roster run is recorded on the branch. The merge review therefore sees the whole migration, and the close-out only verifies it. **The recorded scope claim (Astra):** "MHR is the default body for `build_commercial_multiview_comparison.py` and this commercial-multiview delivery's migrated instruments. D4i does not integrate MHR into N5.1, unified character composition, GNM binding, or the other acting builds." "The body model is integrated" is never claimed beyond that.

**Frozen historical producers** (`d7*`, `d8*`, `d9*`, `d9b*`, `precard/d7c_take_build.py`) call `build.main()` without `--body`, so they now get MHR. They are listed, and any rerun passes `--body rig`.

**Rig rungs on the ladder** that read `scoreboard-commercial-multiview-soma77.json` and `facing-location.json` are frozen at D7c's close-out and DATED as such on the pages. B4 (`d4_b4_mamma_arm`) gets its own extractor.

Card reviewed by Astra GPT6 once, at medium effort (`docs/reviews/body-model-flip-astra-review-2026-09-25.md`): two blockers (the migration after the verdict; the roster's conflated fields), both adopted, and the debt adopted into this text. Window 0. | 1, 7 | **THE oracle:** the flipped default (no `--body` flag) rebuilds D4d's gated `--body mhr` delivery BYTE-IDENTICAL, every delivered file per subject: the GLB, the track npz and json, `markers.npz`, and the calibration start and passes sidecars. **B1 reproduced EXACTLY:** on those identical bytes, the schema-aware `silhouette.py` and `d4_silhouette_paired.py` with identical draws reproduce D4d's B1 to the digit (not "within CI": same bytes, same seeds, same numbers), and likewise the closure and B5. **Hygiene:**
- `--body rig --body-run artifacts/compare/d1-fix/body-run-regenerated` rebuilds the D7c delivery 8/8 byte-identical;
- `src/` and `mhr_delivery.py` are byte-identical;
- the D7c silhouette baseline's sha256 is unchanged after a FULL roster run.

**Before Phase 1:** the D7c silhouette baseline (`artifacts/compare/i6`) is archived and its sha256 recorded, because the legacy commands can overwrite it. The mesh cache is bound to the GLB hashes and exporter settings (today `silhouette.py` accepts its cache by modification time), or every export is fresh and said so.

**Phase 1, the roster by a FROZEN rule, measured.** Each instrument gets THREE separate fields, each with its evidence (log path) and exact command:
1. **OBSERVED class, pre-migration:** the instrument as it stands today, run on D4d's gated MHR delivery.
   - (a) it runs and scores the delivered body;
   - (b) it runs but scores a rig arm or a capture-side quantity;
   - (c) it crashes or refuses;
   - (d) it runs GREEN on a delivered capability the MHR delivery does not carry.
   **(d) takes precedence over (b):** an unsupported delivered-capability GREEN is never relabelled as capture-side work.
2. **FINAL ACTION:** keep, relabel, replace (naming the replacement) or remove. (d) is always removed from the MHR roster. Historical rig diagnostics may survive only as separately named rig-arm entries run against an explicit `--body rig` build at `artifacts/commercial-multiview-soma77-rig`.
3. **POST-MIGRATION class:** the final entry measured again, after the code changes, on the default build.

The explorer's code readings are the PREDICTIONS. Unexpected outcomes are measured, never forced.

**Exit status is not a verdict.** `d4_glb_closure.py` and `d4_b2_same_denominator.py` return 0 while reporting a failed band. Every roster entry records its OWN verdict field, and the gate reads that field, never the exit code. **Execution/schema failure is separated from a scientific result:** B3 and B4 remain REPORTED measurements, and no new quality threshold is licensed by this step.

**THE population, frozen:** 2 subjects × 150 frames × the declared landmark-to-joint mapping (17), and for B1 150 frames × 4 cameras × 2 performers. Every class-(a) entry records EXPECTED against OBSERVED subjects, frames, mappings and exclusions, and a mismatch FAILS. B3 today takes the shorter frame sequence and drops nonfinite distances, so missing-subject, truncated-frame and substituted-reference mutations must each turn it.

**Must-fails, each demonstrated:**
- (i) every final class-(a) entry, replacements included, invoked in MHR SCOPE, rejects a rig-schema track by `schema_version` before interpreting any payload field. `root_translation_m` shares its key across schemas with a different frame. A dual-mode tool (silhouette) rejects when called in MHR scope, and accepts rig only in rig scope;
- (ii) the build refuses a directory holding the other schema's files, and the verifier fails a mixed directory, so a stale `mapping.npz` must FAIL;
- (iii) a NEGATIVE CONTROL: mutating the MHR track's `rest_positions_z_up_m` (the mean body) leaves every class-(a) reading unchanged;
- (iv) a run report that claims a solved head on an MHR build fails the verifier;
- (v) POSITIVE CONTROLS: perturbing the scored joints (B3) and the mesh motion (B1) changes the applicable reading, so a constant instrument cannot pass.

**Prediction, fixed before numbers:**
- the oracle is byte-identical, and B1 reproduces exactly;
- observed: `head_gate` and `bootstrap_margin` (d), or (c) if they crash; `captured_limb_stability`, `retarget_cost`, `oracle_2d` and `fit_smplx_pose` (b); `d3_skeleton_gate`, `facing_location`, `mamma_scoreboard`, `delivered_vs_capture`, `delivered_foot_is_fiction` and the current `silhouette` (c);
- final: silhouette (schema-aware), B2, B3, B4, closure and B5 in (a).

**ONE verdict:** PASS iff the oracle (the full per-subject file manifest, reference hashes, subject identities, draws and the SCORED ROWS equal, not rounded summaries; the intentionally corrected run report is outside it) ∧ B1 reproduced exactly ∧ B2 PASS ∧ closure PASS ∧ hygiene ∧ must-fails i–v ∧ every final class-(a) entry matches the frozen population and reports its own verdict ∧ the roster JSON complete with all three fields for every instrument ∧ the coordinator's `post_merge.sh`/`ladder.py` changes on the branch, with the final roster run recorded. The gate is `tools/compare/d4i_flip_gate.py`. Its per-conjunct fuzz mutates the UNDERLYING EVIDENCE (the files, arrays and verdict fields), never supplied PASS flags, and keeps CRASH as its own class.

**Merge rule, fixed before numbers:** PASS → merge (one merge review sees the whole migration). **The close-out, with frozen outcomes:** the new `post_merge.sh` rebuilds the MHR default in place, archiving the rig delivery under `delivered-before-D4i-<date>`, and reruns the final roster. It PASSES iff the rebuild is byte-identical to the branch build ∧ no roster entry crashes ∧ every entry's own verdict field equals the merge-review record. Anything else triggers a FULL ROLLBACK, all of it:
- `git revert` of the flip, roster and `ladder.py` commits;
- the archived rig delivery restored into `artifacts/commercial-multiview-soma77`, so no MHR file remains to trip the mixed-directory refusal;
- the previous `post_merge.sh` and roster restored;
- the pages republished;
- the step recorded as FAIL.

Non-PASS → records only on main. |

## 2. The clause table (`docs/reviews/body-model-flip-records/gate.json`)

| clause | predicted | measured | verdict |
|---|---|---|---|
| **THE oracle: per-subject files BYTE-IDENTICAL** | the card: all 12. Stage 3's pre-registration (`af69599`): 10, with the two track JSONs differing in `body_model.assets` only | **10/12.** `subject-00/01.body-track.json` differ in `/body_model/assets` only (`ladder-D4d` → `ladder-D4i`) | **FAIL**, read literally (the card's prediction FAILED; stage 3's held) |
| oracle: the manifest | equal | equal (outside `work/`) | holds |
| oracle: the reference hashes | equal | camera rig and the three converter inputs byte-identical | holds |
| oracle: the subject identities | equal | the silhouette's identity block is equal; B4's subject map is our 0 → body_id-01, our 1 → body_id-00 | holds |
| oracle: the draws | equal | B1 seed 20260921, block 15, 2000 resamples; the arm sha256s are equal | holds |
| oracle: the SCORED ROWS | equal | the B1 report is equal to every value except its arm paths; the silhouette report is equal in every section (the two new fields `scope` and `mesh_cache` excluded) | holds |
| oracle: closure, B2, B5 reproduced | exactly | the closure and B2 equal D4d's records; B5 equals its Phase-1 reading on D4d's delivery (D4d never ran B5) | holds |
| **B1 reproduced exactly** | to the digit | every value equal; lower CI of delivered minus D7c **+0.1295 / +0.0836**; on a FRESH bound export whose mesh sha256 `2b7ba78d…` equals D4d's | **HOLDS** |
| B2 PASS | PASS | B2's own `verdict` PASS, both subjects (against this step's rig arm) | HOLDS |
| closure PASS | PASS | `all_within_band`, worst 2.293e-06 m against 1e-4 | HOLDS |
| hygiene | the rig arm 8/8; `src/` and the fitter byte-identical; i6 unchanged after a FULL roster run | 8/8 against the shipped D7c delivery; `src/` and the fitter byte-identical to `acceb55`; i6 7/7 after the coordinator's full roster run | HOLDS |
| must-fails i–v and the population mutations | each turns | 70 cases, every mutation turned its entry (or left the negative control unchanged); the gate re-derives 62 of them from their entry reports and reads 11 from the record | HOLDS (one prediction's CLASS failed, §4) |
| every final class-(a) entry: frozen population and its own verdict | yes | 8/8 entries: population matched (0 mismatches), own verdict field reported, scope checked, and bound BY CONTENT to the default build. The coordinator's recorded run agrees | HOLDS |
| the roster JSON complete, three fields | yes | 19/19 instruments; every (d) removed | HOLDS |
| the coordinator's `post_merge.sh` / `ladder.py` changes on the branch, the final roster run recorded | yes | `post_merge.sh` yes (`1242ed0`); the run recorded (every MHR entry PASS, no crash, i6 7/7); **`ladder.py` not yet edited** (the coordinator's, after this stage) | **NOT MET at gate time** |
| observed classes (Phase 1) | head_gate and bootstrap_margin (d) or (c); four (b); six (c) | every one as predicted (§3); the verifier, not predicted, (c) | held |
| final class (a) | silhouette (schema-aware), B2, B3, B4, closure, B5 | all (a), plus B1 and the verifier | held |
| **overall** | PASS | **FAIL (oracle; coordinator_migration_recorded pending)** | **FAIL** |

**The fuzz** (`tools/compare/d4i_flip_gate_fuzz.py`, `fuzz.json`) mutates the underlying evidence on copies: delivered
bytes, the instruments' own reports, the entry reports, the roster and the tree.
- **39 of 39** targeted mutations TURN their leg.
- **3 CRASH cases** fail closed and are counted as their own class.
- **4 negative controls** leave every leg unmoved: every delivered file's mtime moved, the review page edited, the run
  log appended to, and B3's REPORTED figures edited in the ENTRY report.
- **It found one hole, fixed before this reading.** The population leg bound each entry to the default build by PATH, so
  a copy with only its mtimes moved read as "another delivery". It now binds by the per-subject file hashes.
- **The stage-5 mutations found another hole.** The silhouette's mesh sidecar bound the GLBs and the exporter but not the
  mesh file itself; it now carries the mesh's own sha256.

## 3. The roster (`docs/reviews/body-model-flip-records/roster.json`)

The three fields are kept separate, each with its evidence and its exact command in the JSON:
- **OBSERVED:** Phase 1 on D4d's delivery, the instruments as they stood;
- **FINAL ACTION;**
- **POST-MIGRATION:** the coordinator's recorded final roster run, `post_merge.sh roster artifacts/compare/d4i-flip/default
  artifacts/compare/soma77-rig-d4i artifacts/compare/d4i-flip/final-roster`.

(d) takes precedence over (b). Exit status is never a verdict; each entry's own field is read.

| instrument | OBSERVED (Phase 1) | FINAL ACTION | POST-MIGRATION (the coordinator's run) |
|---|---|---|---|
| `silhouette` | (c) none written (crash) | keep [MHR] | (a) PASS |
| `b1_paired` | (a) no verdict word | replace → d4i_mhr_roster.py b1 [MHR] | (a) PASS |
| `b2` | (a) verdict = PASS (both subjects PASS) | replace → d4i_mhr_roster.py b2 [MHR] | (a) PASS |
| `b3` | (a) no verdict field (REPORTED) | replace → d4i_mhr_roster.py b3 [MHR] | (a) PASS |
| `b4` | (a) no verdict field (REPORTED, never selected) | replace → d4i_mhr_roster.py b4 [MHR] | (a) PASS |
| `closure` | (a) all_within_band = true, worst 2.293e-06 m | replace → d4i_mhr_roster.py closure [MHR] | (a) PASS |
| `b5` | (a) verdict_bytes = PASS (both subjects PASS) | replace → d4i_mhr_roster.py b5 [MHR] | (a) PASS |
| `verifier` | (c) status = fail | replace → d4i_mhr_roster.py verifier (verify_commercial_multiview_artifact.py --scope mhr, its new MHR branch) | (a) PASS |
| `captured_limb_stability` | (b) verdict = REPORTED (limb-stability.json) | relabel → CAPTURE-SIDE [CAPTURE] | (b) REPORTED |
| `oracle_2d` | (b) no verdict field (report-only) | relabel → CAPTURE-SIDE [CAPTURE] | (b) REPORTED (report written) |
| `retarget_cost` | (b) no verdict field (report-only) | relabel → RIG-ARM: rig_arm_retarget_cost [RIG-ARM] | (b) REPORTED (report written) |
| `fit_smplx_pose` | (b) no verdict field (report-only) | remove | (not in the roster (removed)) |
| `delivered_vs_capture` | (c) none written (crash before any report) | replace → d4i_mhr_roster.py b3 | (not in the roster (removed)) |
| `mamma_scoreboard` | (c) none written (crash) | replace → d4i_mhr_roster.py b4 | (not in the roster (removed)) |
| `d3_skeleton_gate` | (c) none written (crash) | replace → d4i_mhr_roster.py closure (its delivered-file closure); the rest is rig-only and removed | (not in the roster (removed)) |
| `facing_location` | (c) none written (crash) | remove | (not in the roster (removed)) |
| `delivered_foot` | (c) none written (crash) | remove | (not in the roster (removed)) |
| `head_gate` | (d) per-arm verdict column: candidate_multiview_fit PASS on both subjects  | remove | (not in the roster (removed)) |
| `bootstrap_margin` | (d) P(pass) per subject and block: candidate 0.91-0.92 / 1.00 | remove | (not in the roster (removed)) |

**How the roster's own field is read:**
- The MHR entries run through `tools/compare/d4i_mhr_roster.py`, in this order: refusal by `schema_version` before any
  payload is read; CRASH kept apart; the frozen population, expected against observed; then the instrument's own verdict
  field.
- **B3 and B4 remain REPORTED.** Their entry verdict is execution and population only; no quality threshold was written.

**The coordinator's rulings (2026-09-25):**
- **The wrappers are ACCEPTED as the final class-(a) entries.** Must-fail (i) is defined on the roster's final entries.
- **Debt:** B2, B3, B4, B5 and the closure, invoked DIRECTLY outside the roster, crash on a missing key instead of
  refusing a rig track.

**Frozen historical producers.** `d7*`, `d8*`, `d9*`, `d9b*` and `precard/d7c_take_build.py` call `build.main()` without
`--body`. Had the flip merged they would get MHR, and any rerun passes `--body rig`. Since D4i does not merge, this
belongs to D4i-b.

## 4. Every FAILED prediction, attributed

1. **"The oracle is byte-identical" (the card).** FAILED on `subject-00.body-track.json` and `subject-01.body-track.json`,
   in the leaf `/body_model/assets`. Attribution: the frozen fitter embeds its own checkout's absolute path. This is the
   coordinator's pre-registration error, the lane's sixth. It was predicted before the build, D4d's tripwire had named the
   same leaf, and the same leaf would have tripped the close-out rollback. It is not a fitter-output difference: the GLBs,
   the npz and every figure are identical.
2. **Must-fail population, "a substituted reference turns B4 to FAIL" (stage 5).** Observed **CRASH**, not FAIL.
   - The mutation substitutes subject 01's captured array for subject 00's, so both performers' pelvis tracks are
     identical.
   - B4's own subject map (`tools/head/subject_map.py`, which needs a 5x margin) refuses: "subject correspondence is
     ambiguous (margin 1.00x) -- refusing to guess".
   - The mutation turns the entry (it is not PASS); the predicted CLASS was wrong.
   - The 1 mm nudge on a single value, which leaves the subject map resolvable, reads FAIL as predicted.

Every other prediction held:
- every observed class;
- every final class;
- B1 exact;
- B2, closure and B5 reproduced;
- rig-arm hygiene 8/8;
- stage 3's one-leaf prediction;
- 69 of 70 must-fail classes.

## 5. What the MHR delivery does not carry

The run report records all of this in MHR mode (`not_consumed_by_delivered_body`, `limitations`), and the verifier fails
a report that claims otherwise (must-fail iv):
- **the multi-frame head solve, the toe input, SOMA-77 `Spine1`, the foot contacts and the ground projection.** The
  capture still solves the head, the toes, the spine, the pelvis frame and the contacts, and the report keeps them,
  marked NOT CONSUMED by the delivered body.
- **articulated fingers or toes;**
- **a rest of this performer.** `rest_positions_z_up_m` is MHR's MEAN body, and changing it moves no roster reading
  (must-fail iii).
- **the MPFB mapping (`mapping.npz`), the GNM binding, the N5.1 assembly.** Nothing in `src/autoanim_gnm` reads this
  delivery; binding GNM onto MHR's head is its own step.
- **evidence of accuracy for its identity.** The fitted spine-length channel reads 1.102 on performer 0, past the soft
  limit 1.1. That is an OBSERVED excursion; missing spine information or a convention mismatch is a hypothesis, not a
  measured cause. D4d's claim limits carry over unchanged.

## 6. Astra's scope claim, verbatim

> MHR is the default body for `build_commercial_multiview_comparison.py` and this commercial-multiview delivery’s migrated instruments. D4i does not integrate MHR into N5.1, unified character composition, GNM binding, or the other acting builds.

D4i FAILED, so not even this claim is made: on main the default stays `rig`. It is the claim D4i-b would carry.

## 7. What is open

- **D4i-b: the flip, re-registered.** The oracle names `body_model.assets` as the building checkout's own
  `<root>/.cache/mhr/assets`, and the close-out rule is written to match. It reuses this step's tooling:
  - `scripts/body_delivery_schema.py`, the verifier's MHR branch and the schema-aware `silhouette.py`;
  - `d4i_mhr_roster.py`, `d4i_mustfail.py`, `d4i_flip_gate.py` (and its fuzz) and `d4i_live_reports.py`;
  - the bootstrap and `post_merge.sh`.

  The B1 arm rename to `delivered_MHR` is D4i-b's, and this stub must change with it.
- **Fitter debt:** an absolute checkout path in a delivered file. A relative or hashed asset reference belongs in a later
  fitter change, with its own registration.
- **Instrument debt (the coordinator's ruling):** B2, B3, B4, B5 and the closure, invoked directly, crash on a missing key
  instead of refusing a rig track by `schema_version`.
- **`ladder.py`** is the coordinator's, after this stage. The rig rungs that read
  `scoreboard-commercial-multiview-soma77.json` and `facing-location.json` stay frozen at D7c's close-out and are dated
  as such. B4 now has its own extractor (`tools/compare/extractors/d4i_flip.py`, `x_body_model_flip_b4`).
- **Neither B1 nor the silhouette stores per-cell rows.** "Scored rows equal" rests on byte-identical inputs (the mesh
  npz, the mask cache, the same code) plus every derived statistic equal at full precision.
- **The silhouette's resolution check** (`HALF_RES`) still reads the D7c rig's half-resolution report under `i6`, even
  in MHR scope. It is a rasteriser check across two different bodies (debt; the value reproduced D4d's exactly).
- **`captured_limb_stability --landmarks-from`** takes the landmarks from the delivery under test, but its observations,
  rig and association still come from the shipped build. They are schema-free and byte-identical here; it is still a
  path it does not take from its argument.
- **`fit_smplx_pose` was removed**, because it has no `--out` and writes the live `smplx-pose-fit.json`. It can return as
  capture-side once it gains an `--out`.
- **The main checkout needs `.venv-mhr`** (`scripts/bootstrap_mhr.sh`, pinned by `scripts/mhr-requirements.lock`; the
  record is in `mhr-bootstrap-record.json`). The new interpreter reproduced every delivered byte except the path leaf.
- **The detections were reused.** Every build here reused the shipped delivery's cached detections (copied in with
  `cp -Rp`); the SOMA-77 detector was not re-run.
- **Nothing is merged or pushed.** Per the card's merge rule, non-PASS means records only on main.

## 8. Reproduce

```
cp -Rp artifacts/compare/i6 artifacts/compare/d4i-flip/i6-baseline-archive                   # stage 0
zsh docs/reviews/body-model-flip-records/phase1_run.sh all                                    # stage 1 (the shadow root: its header)
zsh scripts/bootstrap_mhr.sh                                                                   # .venv-mhr
cp -Rp artifacts/commercial-multiview-soma77/work artifacts/compare/d4i-flip/default/
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos \
    --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml --detector soma77 --output artifacts/compare/d4i-flip/default
cp -Rp artifacts/commercial-multiview-soma77/work artifacts/compare/soma77-rig-d4i/
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py ... --body rig --body-run artifacts/compare/d1-fix/body-run-regenerated \
    --output artifacts/compare/soma77-rig-d4i
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_mustfail.py --out docs/reviews/body-model-flip-records/mustfail.json
zsh tools/compare/post_merge.sh roster artifacts/compare/d4i-flip/default artifacts/compare/soma77-rig-d4i artifacts/compare/d4i-flip/final-roster
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_flip_gate.py --out artifacts/compare/d4i-flip/gate.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_flip_gate_fuzz.py --out artifacts/compare/d4i-flip/fuzz.json
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_d4i_flip.py tests/test_d4i_flip_gate.py tests/test_body_model.py
```

The full suite after stage 7 (`artifacts/compare/d4i-flip/logs/21-full-suite-stage7.log`): **4 failed, 1301 passed, 43 skipped**. The four are D4d's pre-existing failures (the `test_body_compositor` unified preview, the `test_body_export` GLB hash-bound test, and two in `test_phase4_app`). The D4i tests pass: 27 in `tests/test_d4i_flip.py`, 8 in `tests/test_d4i_flip_gate.py`, and `tests/test_body_model.py` with its two build-default re-pins.
