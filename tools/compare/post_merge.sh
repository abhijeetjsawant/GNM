#!/bin/zsh
# The post-merge pass for a delivery step on the body-capture ladder (CLAUDE.md, body lane):
#   1. keep the previous build under artifacts/compare/delivered-before-<ID>-<date>
#   2. rebuild the delivery IN PLACE from the cached detections (nothing re-detects)
#   3. byte-check the eight delivered files against the gated branch build
#   4. rerun every instrument that reads the delivery, logging EVERY line of each into
#      artifacts/compare/post-merge-<ID>/<instrument>.log (2026-09-06: the D3 gate's
#      exact-skeleton oracle failed on the legs for two close-outs behind a `tail -4`)
# Usage: tools/compare/post_merge.sh <ID> <branch-delivery-dir> [extra instrument commands...]
set -u
ID=$1; BRANCH_DELIVERY=$2; shift 2
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/src; PY=.venv/bin/python; DATE=$(date +%F)
ARCH=artifacts/compare/delivered-before-$ID-$DATE; LOGS=artifacts/compare/post-merge-$ID; mkdir -p $LOGS
echo "== 1. keep the previous build -> $ARCH"
[ -d $ARCH ] || cp -R artifacts/commercial-multiview-soma77 $ARCH
echo "== 2. rebuild the delivery IN PLACE from the cached detections"
$PY scripts/build_commercial_multiview_comparison.py --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml --detector soma77 --output artifacts/commercial-multiview-soma77 > $LOGS/rebuild.log 2>&1; tail -3 $LOGS/rebuild.log
echo "== 3. byte-identity against $BRANCH_DELIVERY"
for f in subject-00.glb subject-01.glb subject-00.body-track.npz subject-01.body-track.npz subject-00.body-track.json subject-01.body-track.json subject-00.mapping.npz subject-01.mapping.npz; do
  a=$(shasum -a 256 artifacts/commercial-multiview-soma77/$f | cut -c1-16); b=$(shasum -a 256 $BRANCH_DELIVERY/$f | cut -c1-16)
  [ "$a" = "$b" ] && echo "$f: identical" || echo "$f: DIFFERENT $a $b"
done
echo "== 4. instruments that read the delivery (full logs under $LOGS)"
run() { name=$1; shift; echo "--- $name: $*"; "$@" > $LOGS/$name.log 2>&1; echo "    exit $? ; $(grep -c -E 'FAIL' $LOGS/$name.log) FAIL lines, $(grep -c -E 'PASS' $LOGS/$name.log) PASS lines"; grep -E '^(FAIL|PASS)|verdict' $LOGS/$name.log | head -30; }
run delivered_vs_capture $PY tools/compare/delivered_vs_capture.py --delivery before=$ARCH --delivery after=artifacts/commercial-multiview-soma77 --out artifacts/compare/post-merge-$ID/delivered-vs-capture.json
run captured_limb_stability $PY tools/compare/captured_limb_stability.py --out artifacts/compare/post-merge-$ID/limb-stability.json --skip-reproduction
run mamma_scoreboard $PY tools/compare/mamma_scoreboard.py
run retarget_cost python3 tools/swap-harness/retarget_cost.py
run d3_skeleton_gate $PY tools/compare/d3_skeleton_gate.py
run facing_location $PY tools/compare/facing_location.py
run silhouette $PY tools/compare/silhouette.py
run delivered_foot $PY tools/feet/delivered_foot_is_fiction.py
run head_gate $PY tools/head/head_gate.py
run bootstrap_margin $PY tools/head/bootstrap_margin.py
run oracle_2d $PY tools/compare/oracle_2d.py
run fit_smplx_pose $PY tools/compare/fit_smplx_pose.py
for extra in "$@"; do run extra-$(echo $extra | tr ' /' '__' | cut -c1-40) sh -c "$extra"; done
echo "== done"
