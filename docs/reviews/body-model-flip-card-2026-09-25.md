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
