#!/bin/zsh
# D4i Phase 1: the OBSERVED class of every roster instrument, as it stands at commit 2146a22 (tools
# unchanged since fc1a052), run on D4d's gated `--body mhr` delivery.
#
# THE MECHANISM (why a shadow root). Most legacy instruments hardcode `ROOT/artifacts/commercial-multiview-soma77`
# (ROOT = parents of their own file) and write fixed report paths the ladder reads. `artifacts/` is shared with the
# main checkout. So they run from a SHADOW ROOT:
#   SHADOW = artifacts/compare/d4i-flip/phase1/shadow
#   * the repository tree at HEAD, extracted with `git archive HEAD | tar -x` (a COPY: `Path(__file__).resolve()` follows
#     symlinks, so a symlinked tools/ would resolve ROOT back to the worktree);
#   * `.cache` and `.venv` symlinked to the main checkout's (read-only use);
#   * `artifacts/` built from APFS CLONES (`cp -Rc`) of every shared top-level entry and every `artifacts/compare/*`
#     entry except `compare/d4i-flip` (clones are independent files, so no write can reach the shared tree), with
#     the five parked bear-experiment dirs (sv4d bullettime videodepth prompthmr mocapanything) symlinked;
#   * `artifacts/commercial-multiview-soma77` REPLACED by a clone of `artifacts/compare/d4d-twopass/delivery` (D4d's
#     gated MHR delivery), and the shipped rig delivery cloned to `artifacts/compare/delivered-before-D4i-observed`
#     (the ARCH that post_merge.sh's delivered_vs_capture line compares against).
# The post_merge.sh step-4 lines below are VERBATIM (ID=D4i-observed), run with cwd = SHADOW. The D4 instruments and the
# verifier are added with their D4d arguments. Every line of every instrument is logged; the exit status is recorded
# and is NOT a verdict.
#
# Usage: phase1_run.sh NAME...   (names below; `all` runs every entry)
set -u
SHADOW=/Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/compare/d4i-flip/phase1/shadow
LOGS=/Users/abhi_macbook/Projects/apps/AutoAnim/artifacts/compare/d4i-flip/logs/phase1
mkdir -p $LOGS
cd $SHADOW
export PYTHONPATH=$PWD/src; PY=.venv/bin/python; ID=D4i-observed
ARCH=artifacts/compare/delivered-before-$ID
mkdir -p artifacts/compare/post-merge-$ID
OUT=artifacts/compare/post-merge-$ID
run() { name=$1; shift; echo "--- $name: $*"; { echo "# cwd: $PWD"; echo "# command: $*"; echo "# started: $(date '+%F %T')"; } > $LOGS/$name.log
  "$@" >> $LOGS/$name.log 2>&1; rc=$?; echo "# exit: $rc  finished: $(date '+%F %T')" >> $LOGS/$name.log
  echo "    exit $rc ; $(grep -c -E 'FAIL' $LOGS/$name.log) FAIL lines, $(grep -c -E 'PASS' $LOGS/$name.log) PASS lines"; }
entry() {
  case $1 in
    delivered_vs_capture) run delivered_vs_capture $PY tools/compare/delivered_vs_capture.py --delivery before=$ARCH --delivery after=artifacts/commercial-multiview-soma77 --out artifacts/compare/post-merge-$ID/delivered-vs-capture.json ;;
    captured_limb_stability) run captured_limb_stability $PY tools/compare/captured_limb_stability.py --out artifacts/compare/post-merge-$ID/limb-stability.json --skip-reproduction ;;
    mamma_scoreboard) run mamma_scoreboard $PY tools/compare/mamma_scoreboard.py ;;
    retarget_cost) run retarget_cost python3 tools/swap-harness/retarget_cost.py ;;
    d3_skeleton_gate) run d3_skeleton_gate $PY tools/compare/d3_skeleton_gate.py ;;
    facing_location) run facing_location $PY tools/compare/facing_location.py ;;
    silhouette) run silhouette $PY tools/compare/silhouette.py ;;
    delivered_foot) run delivered_foot $PY tools/feet/delivered_foot_is_fiction.py ;;
    head_gate) run head_gate $PY tools/head/head_gate.py ;;
    bootstrap_margin) run bootstrap_margin $PY tools/head/bootstrap_margin.py ;;
    oracle_2d) run oracle_2d $PY tools/compare/oracle_2d.py ;;
    fit_smplx_pose) run fit_smplx_pose $PY tools/compare/fit_smplx_pose.py ;;
    # --- the D4 instruments, with D4d's own arguments, pointed at the (substituted) delivery
    b2) run b2 $PY tools/compare/d4_b2_same_denominator.py --delivery artifacts/commercial-multiview-soma77 --rig-build artifacts/compare/d4d-twopass/hygiene --out $OUT/b2-same-denominator.json ;;
    b3) run b3 $PY tools/compare/d4_b3_placement.py --delivery artifacts/commercial-multiview-soma77 --out $OUT/b3-placement.json ;;
    b4) run b4 $PY tools/compare/d4_b4_mamma_arm.py --delivery artifacts/commercial-multiview-soma77 --out $OUT/b4-mamma-arm.json ;;
    closure) run closure $PY tools/compare/d4_glb_closure.py --pair subject_00=artifacts/commercial-multiview-soma77/subject-00.glb,artifacts/commercial-multiview-soma77/subject-00.body-track.npz --pair subject_01=artifacts/commercial-multiview-soma77/subject-01.glb,artifacts/commercial-multiview-soma77/subject-01.body-track.npz --out $OUT/delivery-closure.json ;;
    b5) run b5 $PY tools/compare/d4_b5_delivered_bytes.py --delivery artifacts/commercial-multiview-soma77 --reference artifacts/compare/d4-body/mhr-reference-lod2.npz --mesh artifacts/compare/d4d-twopass/work-delivery/delivered-mesh.npz --out $OUT/b5-delivered-bytes.json ;;
    b1_paired) run b1_paired $PY tools/compare/d4_silhouette_paired.py --out $OUT/b1-paired.json --arm baseline_D7c_rig=artifacts/compare/i6/delivered-mesh.npz --arm D4c_fitted_MHR_lod2=artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz --arm D4d_fitted_MHR_lod2=artifacts/compare/d4d-twopass/work-delivery/delivered-mesh.npz --pair D4d_fitted_MHR_lod2,baseline_D7c_rig --pair D4d_fitted_MHR_lod2,D4c_fitted_MHR_lod2 ;;
    verifier) run verifier $PY scripts/verify_commercial_multiview_artifact.py artifacts/commercial-multiview-soma77 ;;
    *) echo "unknown entry $1"; return 2 ;;
  esac
}
if [ "$1" = all ]; then set -- delivered_vs_capture captured_limb_stability mamma_scoreboard retarget_cost d3_skeleton_gate facing_location silhouette delivered_foot head_gate bootstrap_margin oracle_2d fit_smplx_pose b2 b3 b4 closure b5 b1_paired verifier; fi
for name in "$@"; do entry $name; done
