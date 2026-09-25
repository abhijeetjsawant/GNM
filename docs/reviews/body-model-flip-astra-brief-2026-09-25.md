# Review brief for Astra GPT6: the D4i card, the default flip to the MHR body (2026-09-25). ONE ROUND.

You are the reviewer of record for the AutoAnim body-capture lane. The rule is one card review and one merge review per
step. Findings about an instrument that do not reach a card-banded verdict count as instrument debt. Answer adversarially and
cite the deciding line. List ONLY what would block dispatch, then list the rest as debt.

## State
D4 closed 2026-09-25 through D4d (docs/reviews/body-model-twopass-2026-09-25.md; your merge ruling is
docs/reviews/body-model-twopass-astra-merge-review-2026-09-25.md). `--body mhr` carries the combined fitter; the default is
`rig`. A read-only code map (below, from an explorer agent: code readings, not measurements) found that nothing in src/
reads this delivery, and that the N5.1 assembly is fed by other builds. So the card scopes the compositor out and makes the
step a default flip plus an instrument-roster migration.

Read:
- scripts/build_commercial_multiview_comparison.py (`--body` at about :345-354; the MHR branch :446-486; the rig branch
  :487-516; the run report);
- tools/compare/post_merge.sh;
- tools/compare/silhouette.py (`--work` default and the MPFB mesh filter);
- scripts/verify_commercial_multiview_artifact.py;
- tools/fitter/mhr_delivery.py `write_track`;
- src/autoanim_gnm/body.py (`skeleton_for_joint_names`, `BodyTrack.from_dict`);
- tools/compare/ladder.py's reads of scoreboard-commercial-multiview-soma77.json and facing-location.json.

## The explorer's findings (summarised)
- Only `body.py` is a choke point for the rig-schema instruments.
- `unified_gltf`, `body_compositor` and `body_binding` are not on this delivery's path.
- Instruments reading only landmark arrays (`captured_limb_stability`, `retarget_cost`, `oracle_2d`, `fit_smplx_pose`) run on
  MHR, but `retarget_cost` re-runs the RIG converter.
- `head_gate` and `bootstrap_margin` would run GREEN on a head the MHR delivery does not carry.
- `d3_skeleton_gate`, `facing_location`, `mamma_scoreboard`, `delivered_vs_capture` and `delivered_foot_is_fiction` crash on
  the MHR schema.
- `silhouette.py` keeps only meshes named MPFB and would overwrite the D7c baseline under artifacts/compare/i6.
- The same key `root_translation_m` has a different frame in each schema.
- `rest_positions_z_up_m` in the MHR track is the mean body.
- The run report claims head, toe and spine solves in both modes.
- Frozen historical producers call build.main() without --body.

## The card, verbatim (docs/LADDER_EXECUTION_PLAN.md §2, the row after D4d; also docs/reviews/body-model-flip-card-2026-09-25.md)
| **D4i the body model becomes the default, and the instrument roster follows it** | **Why, and what changed in the plan.** D4 closed on 2026-09-25: `--body mhr` carries D4c's landmark start plus D4d's second pass, and the default is still `rig`. D4's review §6 and the resume brief listed the flip's work as the compositor (`unified_gltf`, the N5.1 assembly), schema-aware checks, the instruments and an end-to-end rebuild. A read-only map of the code (2026-09-25) retires the first item. Nothing in `src/autoanim_gnm` reads this delivery: `unified_gltf`, `body_compositor` and `body_binding` are fed by `build_mamma_gnm_character_preview.py`, `build_audio_acting_shot.py` and `build_video_acting_shot.py` with in-memory `BodyTrack`s, and they need the MPFB mesh and GNM binding that MHR lacks. Binding GNM onto MHR's head is its own step. So D4i is **a default flip plus an instrument-roster migration**, and the first step since D3 whose DEFAULT output bytes change on purpose.

**The change:**
- `scripts/build_commercial_multiview_comparison.py` defaults to `--body mhr`.
- In MHR mode, `run-report.json` marks `head_orientation`, `toe_triangulation`, `spine_triangulation`, `pelvis_frame` and `contact_frames` as NOT CONSUMED by the delivered body (today it reports them as solved in both modes). Its `limitations` state what the MHR delivery does not carry: the multi-frame head solve, toe input, `Spine1`, foot contacts and the ground projection. The real-take spine at 1.102 against its 1.1 limit is the measured consequence of having no `Spine1`. It also states that the track's `rest_positions_z_up_m` is the MEAN body, never a rest; that note goes in the run report, not `write_track`, which D4c's and D4d's tests pin.
- The build REFUSES to write into a directory that holds the other schema's files; today a stale `subject-XX.mapping.npz` survives a rebuild.
- The momentum interpreter moves off `/tmp`, which has been wiped once. `--mhr-python` defaults to a gitignored `.venv-mhr/` created by a new `scripts/bootstrap_mhr.sh` (pymomentum-cpu 0.1.114.post0), and a missing interpreter fails with that recipe.
- `scripts/verify_commercial_multiview_artifact.py` gains an MHR branch: schema `2.0-mhr`, 127 joints, pose-value shape, finite joints, and landmark placement through `landmark_to_joint`.
- `tools/compare/silhouette.py` selects the Blender exporter by schema, and its MHR runs use their own `--work`. The D7c rig mesh under `artifacts/compare/i6`, the comparator every B1 since D4 has used, moves to a named read-only baseline and is never overwritten.
- `tests/test_body_model.py` is re-pinned to `mhr`, and new tests cover the schema checks.

Byte-identical across the step: `src/` (including `body.py` and `body_export.py`, which the user's uncommitted tests exercise) and `tools/fitter/mhr_delivery.py`.

**Ownership:** `post_merge.sh` and `ladder.py` are the coordinator's. The agent delivers a ROSTER JSON (instrument, class, reason, exact command), and the coordinator edits both at the merge.

**Frozen historical producers** (`d7*`, `d8*`, `d9*`, `d9b*`, `precard/d7c_take_build.py`) call `build.main()` without `--body`, so they now get MHR. They are listed, and any rerun passes `--body rig`.

**Rig rungs on the ladder** that read `scoreboard-commercial-multiview-soma77.json` and `facing-location.json` are frozen at D7c's close-out and DATED as such on the pages. B4 (`d4_b4_mamma_arm`) gets its own extractor.

Window 0. | 1, 7 | **THE oracle:** the flipped default (no `--body` flag) rebuilds D4d's gated `--body mhr` delivery BYTE-IDENTICAL, every delivered file per subject: the GLB, the track npz and json, `markers.npz`, and the calibration start and passes sidecars. **B1 reproduced EXACTLY:** on those identical bytes, the schema-aware `silhouette.py` and `d4_silhouette_paired.py` with identical draws reproduce D4d's B1 to the digit (not "within CI": same bytes, same seeds, same numbers), and likewise the closure and B5. **Hygiene:**
- `--body rig --body-run artifacts/compare/d1-fix/body-run-regenerated` rebuilds the D7c delivery 8/8 byte-identical;
- `src/` and `mhr_delivery.py` are byte-identical;
- the D7c silhouette baseline's sha256 is unchanged after a FULL roster run.

**Phase 1, the roster by a FROZEN rule, measured before the flip:** every instrument `post_merge.sh` runs today, plus `d4_b3_placement`, `d4_b4_mamma_arm`, `d4_glb_closure`, `d4_b5_delivered_bytes` and `d4_b2_same_denominator`, is RUN on D4d's existing MHR delivery and classified by observed behaviour:
- **(a)** it runs and scores the delivered body: it stays on the roster, and must name and count its population;
- **(b)** it runs but scores a rig arm or a capture-side quantity: it stays, RELABELLED as such;
- **(c)** it crashes or refuses: it is replaced by its named D4 analogue or scoped out with a reason;
- **(d)** it runs GREEN on something the delivery does not carry: it comes OFF the MHR roster, never kept with a banner.

The explorer's code readings are the predictions, and this phase measures them. Rig-arm instruments that stay meaningful run against an explicit `--body rig` build kept at `artifacts/commercial-multiview-soma77-rig`.

**Must-fails, each demonstrated:**
- (i) a rig-schema track handed to any class-(a) instrument is REFUSED by `schema_version` before any field is read. `root_translation_m` shares its key across schemas with a different frame (the rig's Y-up root against MHR's Z-up root joint), so a by-key read would be a correct measurement carrying a claim it does not support;
- (ii) the build refuses a directory holding the other schema's files, and the verifier fails a mixed directory, so a stale `mapping.npz` must FAIL;
- (iii) no class-(a) instrument reads `rest_positions_z_up_m` as a rest; a test proves it by mutation;
- (iv) a run report that claims a solved head on an MHR build fails the verifier.

**Prediction, fixed before numbers:**
- the oracle is byte-identical, and B1 reproduces exactly;
- `head_gate` and `bootstrap_margin` fall in (d);
- `captured_limb_stability`, `retarget_cost`, `oracle_2d` and `fit_smplx_pose` fall in (b);
- `d3_skeleton_gate`, `facing_location`, `mamma_scoreboard`, `delivered_vs_capture` and `delivered_foot_is_fiction` fall in (c), replaced by B3, B4, closure and B5 or scoped out;
- `silhouette` falls in (a) once schema-aware.

**ONE verdict:** PASS iff the oracle ∧ B1 reproduced ∧ hygiene ∧ must-fails i–iv ∧ every class-(a) instrument reports its named population on the MHR delivery ∧ the roster JSON is complete (every current instrument classified). The gate is `tools/compare/d4i_flip_gate.py`, with a per-conjunct input-mutation fuzz and CRASH kept as its own class.

**Merge rule, fixed before numbers:** PASS → merge. The coordinator then edits `post_merge.sh` to the roster and rewires `ladder.py`, and the close-out runs the new `post_merge.sh`: the MHR default rebuilds in place (the rig delivery archived) and every roster instrument exits clean. If the close-out's rebuild is not byte-identical to the branch build, or any class-(a) instrument fails, the flip commit is REVERTED (the default returns to `rig`) and the step is recorded as FAIL. Non-PASS → records only on main. |

## Questions
1. Dispatchable? If not, list ONLY what blocks, in order.
2. Is scoping N5.1 (unified_gltf, the compositor, the binding) out of the flip correct, given what reads the delivery?
   What must the record say so that nothing claims "the body model is integrated" beyond this delivery?
3. Is the oracle (default rebuild = D4d's gated --body mhr delivery, byte-identical) plus B1 reproduced exactly the right
   exactness arm for a flip that should change nothing MHR produces?
4. Phase 1's four-class rule: complete, and frozen well enough that the roster is measured rather than chosen? Is "class (d)
   comes OFF the roster, never a banner" right for head_gate and bootstrap_margin?
5. The ownership split (the agent writes a roster JSON; the coordinator edits post_merge.sh and ladder.py) and the
   close-out revert rule: sound, or does the revert rule create a verdict that depends on work done after the gate?
6. Are must-fails i–iv the right silent-hazard checks? Anything a constant can pass, or any population or provenance
   hole of the D4 to D4d kind?
7. Moving the momentum interpreter to a gitignored .venv-mhr/ with a bootstrap script: in scope for a default flip, or
   scope creep?
