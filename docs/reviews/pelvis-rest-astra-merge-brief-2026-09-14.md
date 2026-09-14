# Merge review for Astra GPT6 — D7c, the pelvis on the rig's own rest — 2026-09-14

You reviewed this card in seven rounds (`docs/reviews/pelvis-rest-astra-review-2026-09-14.md`). The agent has finished on
branch `ladder/D7c` (worktree `/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`, commits d146aa3 …
9dda9ac) and reports MERGE on all ten conjuncts. Review the MERGE before it lands: read the agent's review
`docs/reviews/pelvis-rest-2026-09-14.md` in that worktree (§3–3C the two stops and the reread, §4A the src change and the
tripwire, §5A the delivery and the bands, §5 the clause table — note §5 still carries the first-stop table and §5A the
delivery figures), the gate `artifacts/compare/d7c-pelvis-rest/gate.json`, `selector.json`, `selector-calibrated.json`,
`selector-calibrated-amended.json`, `selector-reread-sigma0.335546875.json`, the logs under
`artifacts/compare/d7c-pelvis-rest/logs/` (01–22), and the src diff below. Read-only; adversarial; cite the line that
decides; verify every number you rely on against the artifacts.

## The src change (the whole diff of `src/autoanim_gnm/commercial_multiview.py`, 7e35dd0..HEAD)

```diff
diff --git a/src/autoanim_gnm/commercial_multiview.py b/src/autoanim_gnm/commercial_multiview.py
index 901cc6b..a0795a9 100644
--- a/src/autoanim_gnm/commercial_multiview.py
+++ b/src/autoanim_gnm/commercial_multiview.py
@@ -2172,12 +2172,44 @@ SOMA77_REST_PELVIS_TEMPLATE_M = (
     (-0.0960559993982315, -0.055917683988809586, 0.017343545332551003),     # root -> RightLeg
 )
 
+# The rig-rest modes. Both read the CALLER'S OWN `rest` and no `SOMA77_REST_*` constant; the
+# lever guard below runs under these two and never under A/B/C.
+RIG_REST_PELVIS_MODES = ("D_rig_rest_hipline", "E_rig_rest_kabsch")
+
 # Which construction ships. SELECTED ON SYNTHETIC TRUTH ONLY, on the NOISY arm -- the clean
 # arm cannot select, because every rigid candidate is exact on it by construction. The
 # MAMMA arm reported beside it and selected nothing.
-# PROVENANCE: `tools/compare/d7_pelvis_synthetic.py` ->
-# `artifacts/compare/d7-pelvis-frame/synthetic.json`.
-PELVIS_FRAME_SOURCE = "C_kabsch_pelvis"
+#
+# D7 selected `C_kabsch_pelvis` under SOMA-77's template geometry, where the spine lever is
+# 39 mm and the hips dominate the fit. D7c re-ran that selection under the RIG's geometry,
+# where the same un-centred Kabsch gives `Spine` a 197 mm lever, and it reverses: the mode
+# that ships is `E_rig_rest_kabsch`, the same construction on the rig's own rest offsets
+# about the observed hip midpoint. On the D3 gate's six exact-skeleton bodies the shipped
+# C fit reads 6.865 deg of pitch about the hip line on EVERY frame of EVERY seed -- a
+# constant of SOMA-77's convention, not noise -- and both rig modes recover the truth
+# exactly (1.5-2.4e-7 m of unnormalised residual). The `root` landmark (SOMA's `Hips`, a
+# pelvis-interior joint 74-77 mm above the hip midpoint on this take) is no longer read.
+#
+# PROVENANCE, and the constant's history IS its provenance:
+#   `tools/compare/d7c_pelvis_synthetic.py` (the selector) ->
+#   `artifacts/compare/d7c-pelvis-rest/selector.json`
+#       S at the card's own fixture: STOPPED on the frozen-pitch-follower clause, 5 of 6
+#       bodies below the 2x separation the card requires. Recorded, never moved.
+#   `artifacts/compare/d7c-pelvis-rest/selector-calibrated.json`
+#       the fixture calibrated to the take's own guard-kept pelvis-lever spread (8.7636 mm,
+#       measured at S's own processing stage): UNREACHABLE under that amendment's frozen
+#       monotonicity precondition, on one 0.0135 mm decrease. Recorded, never moved.
+#   `artifacts/compare/d7c-pelvis-rest/selector-calibrated-amended.json`
+#       the SAME frozen evaluations under Astra GPT6 round 7's amended admissibility rule:
+#       REACHED at sigma = 0.335546875. An observed tolerance match, not monotonicity and
+#       not uniqueness. The amendment is POST HOC and is recorded as such.
+#   `artifacts/compare/d7c-pelvis-rest/selector-reread-sigma0.335546875.json`
+#       all of S reread there: (a) `E_rig_rest_kabsch` better than (b) in all six cells,
+#       better than C-on-SOMA in all six, the frozen-pitch follower 2.56-3.20x on every
+#       body, G2's guard 7.27 deg against 76.67 unguarded. PROCEED.
+#   (a) beat (b) in all six cells at sigma 1.00, 0.50, 0.35, 0.25 AND 0.3355 -- the shipping
+#   selection is stable across every fixture tested and is not a product of the calibration.
+PELVIS_FRAME_SOURCE = "E_rig_rest_kabsch"
 
 # Temporal window on the pelvis FRAME, smoothed as a rotation in the tangent space about
 # the take's mean -- the same treatment `_thorax_frames` applies, and for the same reason:
@@ -2196,10 +2228,59 @@ PELVIS_SMOOTHING_FRAMES = 0
 PELVIS_MINIMUM_RESOLVED_FRACTION = 0.5
 
 
+def _pelvis_lever_guard(
+    spine: np.ndarray,
+    hip_mid: np.ndarray,
+    ceiling_fraction: float = SEGMENT_LENGTH_CEILING_FRACTION,
+) -> tuple[np.ndarray, dict[str, Any]]:
+    """Discard the Spine1 sample on frames whose pelvis LEVER is off the subject's median.
+
+    D7c, and it is a NEW MECHANISM, not D8b/D8c's demote. D8b demotes a landmark's RAYS and
+    lets the sequence solve keep them; this discards a solved WORLD POINT and hands the gap
+    to the spine's existing `np.interp` path a dozen lines below. The shared ceiling
+    validates nothing about it, so it is scored on synthetic truth WITH its gaps
+    (`tools/compare/d7c_pelvis_synthetic.py`, trials G1 and G2).
+
+    THE DENOMINATOR IS FROZEN. The median is taken over the finite frames of the array as it
+    ARRIVES, before any frame is discarded, so it cannot move with the mask it produces --
+    D8b's moving-denominator defect (CLAUDE.md) cannot enter. `np.median` over an array
+    carrying NaN returns NaN, which would make the ceiling infinite and the mask empty -- a
+    SILENT PASS -- so the finite mask is explicit and is not an accident of dtype.
+
+    `ceiling_fraction` is `SEGMENT_LENGTH_CEILING_FRACTION`, D8b's 0.15, NOT re-selected
+    here. A length rule is blind to direction by its own docstring: a same-length rotation
+    of the lever passes it untouched.
+
+    Returns the spine array with the rejected frames set to NaN (a COPY; the caller's array
+    is never written) and the report the diagnostics publish.
+    """
+
+    spine = np.asarray(spine, dtype=np.float64)
+    hip_mid = np.asarray(hip_mid, dtype=np.float64)
+    finite = np.isfinite(spine).all(axis=1)
+    lever = np.full(len(spine), np.nan)
+    lever[finite] = np.linalg.norm(spine[finite] - hip_mid[finite], axis=1)
+    off = np.zeros(len(spine), dtype=bool)
+    median = float(np.median(lever[finite])) if finite.any() else float("nan")
+    if finite.any() and median > 0.0:
+        off[finite] = np.abs(lever[finite] / median - 1.0) > ceiling_fraction
+    masked = spine.copy()
+    masked[off] = np.nan
+    return masked, {
+        "applied": True,
+        "ceiling_fraction": ceiling_fraction,
+        "pre_guard_median_mm": round(1e3 * median, 4) if finite.any() else None,
+        "demoted_frames": [int(index) for index in np.flatnonzero(off)],
+        "demoted_count": int(off.sum()),
+        "finite_frames_before_the_guard": int(finite.sum()),
+    }
+
+
 def _pelvis_world_frames(
     points_rig_y_up_m: np.ndarray,
     spine1_rig_y_up_m: np.ndarray,
     *,
+    rest: dict[str, np.ndarray] | None = None,
     mode: str | None = None,
     smoothing_frames: int | None = None,
 ) -> tuple[np.ndarray | None, dict[str, Any]]:
@@ -2229,6 +2310,26 @@ def _pelvis_world_frames(
     frames = len(points)
     if spine.shape != (frames, 3):
         raise CommercialMultiviewError("Spine positions must be [frame, 3]")
+    # D7c. The lever guard runs under the RIG-REST modes ONLY. A/B/C read the spine array
+    # untouched, which is exactly what the refactor tripwire asserts: with the mode held at
+    # `C_kabsch_pelvis` this function reproduces D9b bit for bit. Production uses precisely
+    # the guarding the selector evaluated -- both rig modes, never the legacy three.
+    lever_guard: dict[str, Any] = {
+        "applied": False,
+        "reason": ("the pelvis lever guard runs under the rig-rest modes only; A, B and C "
+                   "read the spine array untouched"),
+        "demoted_frames": [],
+        "demoted_count": 0,
+    }
+    if mode in RIG_REST_PELVIS_MODES:
+        if rest is None:
+            raise CommercialMultiviewError(
+                f"{mode} reads the caller's own rest skeleton and none was supplied")
+        spine, lever_guard = _pelvis_lever_guard(
+            spine,
+            0.5 * (points[:, JOINT_INDEX["left_hip"]]
+                   + points[:, JOINT_INDEX["right_hip"]]),
+        )
     valid = np.isfinite(spine).all(axis=1)
     resolved = float(valid.mean())
     if resolved < PELVIS_MINIMUM_RESOLVED_FRACTION or int(valid.sum()) < 2:
@@ -2241,6 +2342,7 @@ def _pelvis_world_frames(
             ),
             "resolved_fraction": resolved,
             "mode": mode,
+            "lever_guard": lever_guard,
         }
     filled = spine.copy()
     axis = np.arange(frames, dtype=np.float64)
@@ -2254,7 +2356,65 @@ def _pelvis_world_frames(
     hip_across = left_hip - right_hip
 
     quaternions = np.zeros((frames, 4), dtype=np.float64)
-    if mode == "C_kabsch_pelvis":
+    if mode in RIG_REST_PELVIS_MODES:
+        # D7c. NO CONSTANT ENTERS. Every vector below is read from the caller's own `rest`
+        # -- the per-performer skeleton D3 stamps on the track -- exactly as
+        # `_leg_root_offset` reads it, and the observed set is taken about the captured HIP
+        # MIDPOINT rather than about the `root` landmark, whose place relative to the hip
+        # joints is SOMA-77's convention and not this rig's.
+        #
+        # On an exact rig the two triangles are CONGRUENT by construction: forward
+        # kinematics puts the leg roots at `root + rest[Hips] + R . rest[L|R]` and `Spine`
+        # at `root + rest[Hips] + R . rest[Spine]`, so subtracting the leg-root midpoint
+        # from both leaves `R` acting on exactly these three vectors. That congruence is
+        # why the oracle clause is an EXACTNESS clause and not a fit quality.
+        mid = 0.5 * (
+            np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
+            + np.asarray(rest["RightUpperLeg"], dtype=np.float64)
+        )
+        if mode == "D_rig_rest_hipline":
+            # The observed hip line is the EXACT primary axis and Spine1 sets only the pitch
+            # about it. `_frame_alignment` normalises both source axes, so the rest supplies
+            # two DIRECTIONS and its lengths do not weight the fit at all.
+            source_primary = (
+                np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
+                - np.asarray(rest["RightUpperLeg"], dtype=np.float64)
+            )
+            source_secondary = np.asarray(rest["Spine"], dtype=np.float64) - mid
+            for frame in range(frames):
+                # `_frame` orthogonalises its SECONDARY in place and `np.asarray` does not
+                # copy a float64 array, so a persistent source vector handed to it every
+                # frame would be mutated on the first one. Hand it a fresh copy.
+                quaternions[frame] = _frame_alignment(
+                    source_primary,
+                    np.array(source_secondary),
+                    hip_across[frame],
+                    filled[frame] - hip_mid[frame],
+                )
+        else:
+            # `E_rig_rest_kabsch`, the mode `PELVIS_FRAME_SOURCE` selects: the C branch's
+            # un-centred rotation-only SVD with the rig's own rest triangle in place of
+            # SOMA-77's and the hip midpoint in place of the `root` landmark. Here the rest
+            # LENGTHS do weight the fit, and that is the whole difference from D: the 197 mm
+            # `Spine` lever pulls against a wrongly-rotated hip line instead of following it.
+            template = np.stack(
+                (
+                    np.asarray(rest["Spine"], dtype=np.float64) - mid,
+                    np.asarray(rest["LeftUpperLeg"], dtype=np.float64) - mid,
+                    np.asarray(rest["RightUpperLeg"], dtype=np.float64) - mid,
+                )
+            )
+            for frame in range(frames):
+                observed = np.stack(
+                    (filled[frame] - hip_mid[frame],
+                     left_hip[frame] - hip_mid[frame],
+                     right_hip[frame] - hip_mid[frame])
+                )
+                u, _, vt = np.linalg.svd(template.T @ observed)
+                sign = float(np.sign(np.linalg.det(vt.T @ u.T)))
+                rotation = vt.T @ np.diag((1.0, 1.0, sign)) @ u.T
+                quaternions[frame] = Rotation.from_matrix(rotation).as_quat()
+    elif mode == "C_kabsch_pelvis":
         template = np.asarray(SOMA77_REST_PELVIS_TEMPLATE_M, dtype=np.float64)
         for frame in range(frames):
             observed = np.stack(
@@ -2302,6 +2462,7 @@ def _pelvis_world_frames(
         "smoothing_frames": int(smoothing_frames),
         "resolved_fraction": resolved,
         "interpolated_frames": int(frames - int(valid.sum())),
+        "lever_guard": lever_guard,
     }
 
 
@@ -2827,7 +2988,10 @@ def positions_to_body_track(
             raise CommercialMultiviewError("Spine positions must be [frame, 3]")
         spine = spine[..., (0, 2, 1)].copy()
         spine[..., 2] *= -1.0
-        pelvis_world, pelvis_report = _pelvis_world_frames(points, spine)
+        # D7c: the pelvis's rest frame comes from the CALLER'S OWN `rest`, exactly as
+        # `_leg_root_offset` and `_joint_origin` already read it, so a per-performer sized
+        # skeleton is honoured and no rest geometry is written down in this module.
+        pelvis_world, pelvis_report = _pelvis_world_frames(points, spine, rest=rest)
     if pelvis_report_out is not None:
         pelvis_report_out.clear()
         pelvis_report_out.update(pelvis_report)
```

## What the agent reports
- Hygiene 8/8. Tripwire (mode C held through the refactored signature) 8/8 byte-identical to D9b, 274 s; the same six-body C
  execution reads 6.8650–6.8651° vs exact truth (the must-fail). O1: 0.0001° / 0.0001 mm / 0.0002 mm / torso 0.00 / residual
  1.5–1.8e-7 m. O2: legs 0.054–0.077 mm, contacts identical, hoist ≤ 0.032 mm. O3 arms 1.32–2.72 → 0.07–0.60 (report).
- S at σ 1.0: STOP (follower 1.45–2.74×, 5/6 under 2×) — recorded, immutable. Calibration under its first wording: UNREACHABLE
  (the 0.0135 mm dip) — recorded, immutable. Under your round-7 wording: REACHED at σ 0.335546875 exactly. Reread at that σ:
  (a) `E_rig_rest_kabsch` wins 6/6 cells, beats C-on-SOMA, the follower 2.56–3.20× on every body; G1 array claim holds; G2
  7.27° vs 76.67°, both metrics every body. PROCEED; (a) ships; bit-parity 12/12 between the src branch and the frozen
  instrument-side estimator on S's own draws.
- Containment: a test deletes the four `SOMA77_REST_*` constants; both rig modes bit-identical on the resolved and the
  missing-data path, mode C raises.
- Delivery: both landmark arrays byte-identical (same denominator PASS); run-report records the mode and the 29-frame mask.
  P1 PASS on the take (track authenticated against the GLB's `body_track_sha256`) and on 6/6 oracle bodies; P2 4.5e-7 /
  2.9e-7 m vs 1e-5; P3 51 intervals reported. Control 1 (restore pre-projection foot locals) is REFUSED by
  `validate_body_track` (left foot contact moved 0.0088 m) so it cannot reach a delivered file; its detection is exercised
  offline (P1 fails on local::LeftFoot / RightFoot); control 2 (clear the mask) fails P1 on `foot_contacts`. An instrument
  defect found and fixed: the watcher snapshotted AFTER a control's mutation; the delivery's P1 was re-read against the old
  snapshots and still passes.
- B1: 8/8 PASS on `ci95[1] >= 0`; torso+legs ROSE with the CI clear of zero on three cells (perf 0 whole +0.0072
  [0.0049, 0.0116], bent +0.0079 [0.0056, 0.0133]; perf 1 whole +0.0030 [0.0007, 0.0075]) — improvement not predicted,
  reported as unexplained, no credit taken. MAMMA mesh oracle 0.0.
- The (a) restatement from the GUARDED delivery: pelvis pitch −8.78 / −9.22°, root 12.40 / 13.06 mm hoist-subtracted
  (−12.21 / −12.81 fore-aft), Spine 30.5 / 30.1, Neck 12.4 / 9.7, leg-root midpoint 0.0002 mm, hip residual full p95
  12.1 / 14.9 mm (unguarded (a) 11.9 / 31.9; D9b 6.4 / 12.0) — a REPORT, no band; hoist p95 12.54 → 13.12 / 8.72 → 8.09;
  contacts (38,51) → (36,36) / (11,18) → (5,18); pelvis step p95 14.1 / 13.3°; frames over 800°/s 0 / 1.
- B5: head gate PASS; the delivered `Head` WORLD rotation identical between builds (105.0914° / 83.5315°). B6: LINEAR
  samplers, norms 1 ± 4e-8, zero negative adjacent dots, track→GLB closure 0.0005 mm; Root / eyes / finger proximals
  bit-identical. **The mesh-deformation reading the card put in B6 was NOT made** — "handed to D6".
- Tests: `tests/test_pelvis_rest.py` new; four pins in `tests/test_pelvis_frame.py` fail (one moved-by-design: the
  SOMASKEL77-posed round trip reads 7.568°; three signature pins calling `_pelvis_world_frames` without `rest`) and are
  re-pinned in the new file, the old file untouched by the agent (the coordinator re-pins in place at the merge, as at D9b).
  The full suite is running now; the agent ran four files (4 failed, 47 passed).
- FAILED predictions, attributed: the σ-1.0 follower clause (fixture noise, matched inflation 1.43×); the calibration's
  monotonicity precondition (keep-mask moving with σ); "zero transverse hip residual" does not carry under (a); B1's torso
  rise. Open: the three B1 rises; the hip residual as a permanent report; one frame over 800°/s; both amendments POST HOC;
  the convention to lane H; G1 5/6 and 4/6 at the sensitivity σ; the constants' move not made; the mesh reading not made.

## Questions
1. MERGE, or not? If not, ONLY what blocks, in order.
2. The src diff: does `E_rig_rest_kabsch` implement exactly the estimator S selected (the un-centred SVD about the hip
   midpoint on {rest[Spine]−mid, rest[L]−mid, rest[R]−mid}), is the guard scoped to the two rig modes only, does the
   missing-data path stay whole-subject (no per-frame definition switch), and is anything in the diff a hidden constant?
3. B6's mesh-deformation reading was a card commitment (a report) and was not made. Is a missing report clause a merge
   blocker, or a recorded omission handed to D6?
4. The three B1 rises with the CI clear of zero on a change the card said could not be seen by the masks: does the
   unexplained direction change the merge, and what one measurement would you require before the close-out?
5. Anything in the agent's review that overclaims, and any number you could not reproduce from the artifacts.
