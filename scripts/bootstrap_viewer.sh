#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=${AUTOANIM_CACHE_DIR:-"$PROJECT_DIR/.cache/autoanim_gnm"}
THREE_VERSION="0.183.2"
THREE_ARCHIVE="three-$THREE_VERSION.tgz"
THREE_URL="https://registry.npmjs.org/three/-/$THREE_ARCHIVE"
THREE_SHA256="435d785005cec60a2e9dcca23b69c6a4c048278098f2669d3d879881dff9ce4d"
TARGET_DIR=${AUTOANIM_VIEWER_VENDOR_DIR:-"$CACHE_DIR/viewer/three-$THREE_VERSION"}

command -v curl >/dev/null 2>&1 || { echo "curl is required on PATH." >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "tar is required on PATH." >&2; exit 1; }
command -v shasum >/dev/null 2>&1 || { echo "shasum is required on PATH." >&2; exit 1; }

bundle_is_valid() {
  [[ -f "$TARGET_DIR/three.core.js" ]] &&
    [[ -f "$TARGET_DIR/three.module.js" ]] &&
    [[ -f "$TARGET_DIR/addons/controls/OrbitControls.js" ]] &&
    [[ -f "$TARGET_DIR/addons/loaders/GLTFLoader.js" ]] &&
    [[ -f "$TARGET_DIR/addons/utils/BufferGeometryUtils.js" ]] &&
    [[ -f "$TARGET_DIR/addons/utils/SkeletonUtils.js" ]] &&
    [[ -f "$TARGET_DIR/LICENSE" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/three.core.js" | awk '{print $1}') == "6a7fc83437818534d5e30ce8c8e0ce76230ca2245446ea24744bb3d88c436583" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/three.module.js" | awk '{print $1}') == "e8ac51bc2f6b7eb17bc88c6540eb0a1fee872f848949373de39d55f34b5c5a8f" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/addons/controls/OrbitControls.js" | awk '{print $1}') == "09673b997864b8091943d2673637c0f31f7cf67daeddd0902fce9bf098a8d093" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/addons/loaders/GLTFLoader.js" | awk '{print $1}') == "e4e692923224a10bbc00be04365d6356b5e8c48b2bbd1f22a7fe929591646fb8" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/addons/utils/BufferGeometryUtils.js" | awk '{print $1}') == "fda7e946b8e0b5ab39b779206589e7a1079a22eb24efb89d7223e03fdfb1f751" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/addons/utils/SkeletonUtils.js" | awk '{print $1}') == "0761b1e003917b215d25dd81439d854b36034b5a57ce84c2cfe3b02428d7b253" ]] &&
    [[ $(shasum -a 256 "$TARGET_DIR/LICENSE" | awk '{print $1}') == "8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc" ]]
}

if bundle_is_valid; then
  echo "Three.js $THREE_VERSION viewer bundle already verified at $TARGET_DIR"
  exit 0
fi

mkdir -p "$CACHE_DIR" "$TARGET_DIR/addons/controls" "$TARGET_DIR/addons/loaders" "$TARGET_DIR/addons/utils"
if [[ ! -f "$CACHE_DIR/$THREE_ARCHIVE" ]] ||
  [[ $(shasum -a 256 "$CACHE_DIR/$THREE_ARCHIVE" | awk '{print $1}') != "$THREE_SHA256" ]]; then
  curl -L --fail --silent --show-error "$THREE_URL" -o "$CACHE_DIR/$THREE_ARCHIVE.partial"
  echo "$THREE_SHA256  $CACHE_DIR/$THREE_ARCHIVE.partial" | shasum -a 256 -c -
  mv "$CACHE_DIR/$THREE_ARCHIVE.partial" "$CACHE_DIR/$THREE_ARCHIVE"
fi

STAGE_DIR=$(mktemp -d "$CACHE_DIR/viewer-stage.XXXXXX")
trap 'rm -rf "$STAGE_DIR"' EXIT
tar -xzf "$CACHE_DIR/$THREE_ARCHIVE" -C "$STAGE_DIR"

cp "$STAGE_DIR/package/build/three.core.js" "$TARGET_DIR/three.core.js"
cp "$STAGE_DIR/package/build/three.module.js" "$TARGET_DIR/three.module.js"
cp "$STAGE_DIR/package/examples/jsm/controls/OrbitControls.js" "$TARGET_DIR/addons/controls/OrbitControls.js"
cp "$STAGE_DIR/package/examples/jsm/loaders/GLTFLoader.js" "$TARGET_DIR/addons/loaders/GLTFLoader.js"
cp "$STAGE_DIR/package/examples/jsm/utils/BufferGeometryUtils.js" "$TARGET_DIR/addons/utils/BufferGeometryUtils.js"
cp "$STAGE_DIR/package/examples/jsm/utils/SkeletonUtils.js" "$TARGET_DIR/addons/utils/SkeletonUtils.js"
cp "$STAGE_DIR/package/LICENSE" "$TARGET_DIR/LICENSE"

if ! bundle_is_valid; then
  echo "Extracted Three.js bundle failed module checksum verification." >&2
  exit 1
fi

echo "Three.js $THREE_VERSION viewer bundle ready at $TARGET_DIR"
