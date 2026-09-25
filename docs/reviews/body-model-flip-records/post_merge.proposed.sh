#!/bin/zsh
# D4i PROPOSED post_merge.sh (the agent's proposal; the coordinator owns tools/compare/post_merge.sh and edits it).
#
# Two modes. Both take their delivery explicitly: nothing here writes into the shipped delivery except `closeout`,
# which moves it aside first. Every entry's OWN verdict field is read from its report (roster-verdicts.json); the exit
# status is logged beside it and is never read as a verdict. A missing report is CRASH, a refusal is REFUSED, and
# both are kept apart from a FAIL.
#
#   post_merge.proposed.sh roster   <delivery-dir> <rig-build-dir> <out-dir>
#       the final roster against an explicit MHR delivery (the branch run: artifacts/compare/d4i-flip/default,
#       rig build artifacts/compare/soma77-rig-d4i)
#   post_merge.proposed.sh closeout <ID> <branch-delivery-dir> <branch-rig-build-dir>
#       the close-out: MOVE the shipped delivery aside (the mixed-directory refusal forbids rebuilding MHR over rig),
#       copy its work/ (the cached detections) into a fresh shipped dir, rebuild the default IN PLACE, build the rig
#       arm at artifacts/commercial-multiview-soma77-rig, byte-check both against the branch, then run `roster`.
#
# SECTIONS of the roster (roster.json, final actions):
#   MHR       the delivered body, class (a): silhouette, b1, b2, b3, b4, closure, b5, verifier -- each through
#             tools/compare/d4i_mhr_roster.py (MHR scope, refusal before any payload, the frozen population, the
#             instrument's own verdict field). The step verdict reads these.
#   CAPTURE   relabelled (b), capture-side: captured_limb_stability, oracle_2d. They score the triangulated landmarks
#             or the camera rig, which are byte-identical across the two schemas (B2 1a/1b); never a delivered-body
#             claim. Their own verdict field is recorded (REPORTED / none).
#   RIG-ARM   relabelled (b), a rig diagnostic run against the explicit --body rig build: rig_arm_retarget_cost.
#   removed   head_gate, bootstrap_margin (d); delivered_vs_capture, mamma_scoreboard, d3_skeleton_gate,
#             facing_location, delivered_foot (c, rig-only; replacements named in roster.json); fit_smplx_pose
#             (capture-side, but it writes the live artifacts/compare/smplx-pose-fit.json with no --out).
set -u
cd "$(dirname "$0")/../../.."          # the repository root (this file lives in docs/reviews/body-model-flip-records/)
export PYTHONPATH=$PWD/src; PY=.venv/bin/python
SHIPPED=artifacts/commercial-multiview-soma77
SHIPPED_RIG=artifacts/commercial-multiview-soma77-rig
VIDEOS=.cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos
CALIB=.cache/mamma/configs/examples/calib/iphones_outdoors.yaml
BODY_RUN=artifacts/compare/d1-fix/body-run-regenerated

roster() {
  local DELIVERY=$1 RIGBUILD=$2 OUT=$3
  mkdir -p $OUT
  local SWORK=$OUT/silhouette-work        # a FRESH work dir per run: the mesh is exported from THIS delivery's GLBs
  run() { local name=$1; shift; echo "--- $name: $*"
    { echo "# command: $*"; echo "# started: $(date '+%F %T')"; } > $OUT/$name.log
    "$@" >> $OUT/$name.log 2>&1; local rc=$?; echo "# exit: $rc (logged, never a verdict)" >> $OUT/$name.log; }
  R=tools/compare/d4i_mhr_roster.py
  # MHR (the silhouette first: b1 and b5 consume its bound export)
  run silhouette $PY $R silhouette --delivery $DELIVERY --work $SWORK --out $OUT/silhouette.json
  run b1         $PY $R b1 --delivery $DELIVERY --work $SWORK --candidate-name D4d_fitted_MHR_lod2 \
                     --extra-arm D4c_fitted_MHR_lod2=artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz \
                     --extra-pair D4d_fitted_MHR_lod2,D4c_fitted_MHR_lod2 --out $OUT/b1.json
  run b2         $PY $R b2 --delivery $DELIVERY --rig-build $RIGBUILD --out $OUT/b2.json
  run b3         $PY $R b3 --delivery $DELIVERY --out $OUT/b3.json
  run b4         $PY $R b4 --delivery $DELIVERY --out $OUT/b4.json
  run closure    $PY $R closure --delivery $DELIVERY --out $OUT/closure.json
  run b5         $PY $R b5 --delivery $DELIVERY --mesh $SWORK/delivered-mesh.npz --out $OUT/b5.json
  run verifier   $PY $R verifier --delivery $DELIVERY --out $OUT/verifier.json
  # CAPTURE (relabelled): the capture side of the same delivery, reported
  run captured_limb_stability $PY tools/compare/captured_limb_stability.py --landmarks-from $DELIVERY \
                     --skip-reproduction --out $OUT/limb-stability.json
  run oracle_2d  $PY tools/compare/oracle_2d.py --out $OUT/oracle-2d.json
  # RIG-ARM (relabelled): against the explicit --body rig build; --tracks/--out are relative to the repository root
  run rig_arm_retarget_cost python3 tools/swap-harness/retarget_cost.py --tracks $RIGBUILD \
                     --out $OUT/rig-arm-retarget-cost.json
  # The verdict table: each entry's OWN field, from its own report.
  $PY - $OUT <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def read(name):
    p = out / name
    return json.loads(p.read_text()) if p.is_file() else None
rows = {}
for e in ("silhouette", "b1", "b2", "b3", "b4", "closure", "b5", "verifier"):
    r = read(f"{e}.json")
    rows[e] = {"section": "MHR", "field": "verdict", "value": r["verdict"] if r else "CRASH (no entry report)",
               "instrument_verdict_field": (r or {}).get("instrument_verdict_field"),
               "population_mismatches": (r or {}).get("population", {}).get("mismatches")}
r = read("limb-stability.json")
rows["captured_limb_stability"] = {"section": "CAPTURE", "field": "verdict",
                                   "value": r.get("verdict") if r else "CRASH (no report)"}
r = read("oracle-2d.json")
rows["oracle_2d"] = {"section": "CAPTURE", "field": None, "value": "REPORTED (report written)" if r else "CRASH (no report)"}
r = read("rig-arm-retarget-cost.json")
rows["rig_arm_retarget_cost"] = {"section": "RIG-ARM", "field": None,
                                 "value": "REPORTED (report written)" if r else "CRASH (no report)"}
mhr = [v["value"] for v in rows.values() if v["section"] == "MHR"]
summary = {"entries": rows, "mhr_all_pass": all(v == "PASS" for v in mhr),
           "no_entry_crashed": not any(str(v["value"]).startswith("CRASH") for v in rows.values())}
(out / "roster-verdicts.json").write_text(json.dumps(summary, indent=1))
for k, v in rows.items():
    print(f"{v['section']:8s} {k:26s} {v['value']}")
print("MHR ROSTER:", "every entry PASS" if summary["mhr_all_pass"] else "NOT every entry PASS",
      "| no entry crashed" if summary["no_entry_crashed"] else "| AN ENTRY CRASHED")
PY
  echo "--- D7c silhouette baseline (i6) against its archived hashes:"
  # absolute: artifacts/ is a symlink in a worktree, so a relative ../../../docs would resolve into the main checkout
  local SHAFILE=$PWD/docs/reviews/body-model-flip-records/i6-baseline.sha256
  (cd artifacts/compare/i6 && shasum -a 256 -c $SHAFILE)
}

closeout() {
  local ID=$1 BRANCH=$2 BRANCH_RIG=$3 DATE=$(date +%F)
  local ARCH=artifacts/compare/delivered-before-$ID-$DATE LOGS=artifacts/compare/post-merge-$ID
  mkdir -p $LOGS
  echo "== 1. move the shipped delivery aside -> $ARCH (a copy would leave rig files for the mixed-directory refusal)"
  if [ ! -d $ARCH ]; then mv $SHIPPED $ARCH && mkdir -p $SHIPPED && cp -Rp $ARCH/work $SHIPPED/; fi
  echo "== 2. rebuild the default (MHR) IN PLACE from the cached detections"
  # the momentum interpreter is gitignored: a fresh checkout creates it (pinned; its record lands in .venv-mhr/)
  [ -x .venv-mhr/bin/python ] || zsh scripts/bootstrap_mhr.sh > $LOGS/bootstrap-mhr.log 2>&1
  $PY scripts/build_commercial_multiview_comparison.py --videos $VIDEOS --calibration-yaml $CALIB --detector soma77 \
      --output $SHIPPED > $LOGS/rebuild.log 2>&1; echo "    exit $?"
  echo "== 3. the rig arm at $SHIPPED_RIG (fresh dir, the same cached detections)"
  mkdir -p $SHIPPED_RIG && cp -Rp $SHIPPED/work $SHIPPED_RIG/ 2>/dev/null
  $PY scripts/build_commercial_multiview_comparison.py --videos $VIDEOS --calibration-yaml $CALIB --detector soma77 \
      --body rig --body-run $BODY_RUN --output $SHIPPED_RIG > $LOGS/rebuild-rig-arm.log 2>&1; echo "    exit $?"
  echo "== 4. byte identity against the branch builds"
  for f in subject-00.glb subject-01.glb subject-00.body-track.npz subject-01.body-track.npz \
           subject-00.body-track.json subject-01.body-track.json subject-00.markers.npz subject-01.markers.npz \
           subject-00.calibration-start.json subject-01.calibration-start.json \
           subject-00.calibration-passes.json subject-01.calibration-passes.json; do
    a=$(shasum -a 256 $SHIPPED/$f | cut -c1-16); b=$(shasum -a 256 $BRANCH/$f | cut -c1-16)
    [ "$a" = "$b" ] && echo "$f: identical" || echo "$f: DIFFERENT $a $b"
  done
  # KNOWN, pre-registered (stage3-predictions.json): the fitter writes body_model.assets = its checkout's absolute
  # path, so the two track JSONs built on main differ from the branch's in that one leaf. Shown, never normalised:
  $PY - $SHIPPED $BRANCH <<'PY'
import json, sys
def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, f"{pre}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, o
for s in (0, 1):
    a, b = (dict(leaves(json.load(open(f"{d}/subject-{s:02d}.body-track.json")))) for d in sys.argv[1:3])
    print(f"subject-{s:02d}.body-track.json differing leaves:", sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))
PY
  for f in subject-00.glb subject-01.glb subject-00.body-track.npz subject-01.body-track.npz \
           subject-00.body-track.json subject-01.body-track.json subject-00.mapping.npz subject-01.mapping.npz; do
    a=$(shasum -a 256 $SHIPPED_RIG/$f | cut -c1-16); b=$(shasum -a 256 $BRANCH_RIG/$f | cut -c1-16)
    [ "$a" = "$b" ] && echo "rig arm $f: identical" || echo "rig arm $f: DIFFERENT $a $b"
  done
  echo "== 5. the roster (full logs under $LOGS/roster)"
  roster $SHIPPED $SHIPPED_RIG $LOGS/roster
}

case ${1:-} in
  roster) shift; roster "$@" ;;
  closeout) shift; closeout "$@" ;;
  *) echo "usage: $0 roster <delivery-dir> <rig-build-dir> <out-dir> | closeout <ID> <branch-delivery> <branch-rig-build>"; exit 2 ;;
esac
