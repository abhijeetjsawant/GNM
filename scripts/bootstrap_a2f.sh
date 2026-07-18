#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=${AUTOANIM_CACHE_DIR:-"$PROJECT_DIR/.cache/autoanim_gnm"}
ASSET_DIR=${AUTOANIM_A2F_ASSET_DIR:-"$CACHE_DIR/a2f-claire"}
MODEL_REPO="https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire/resolve/main"
RUNNER_DIR="$PROJECT_DIR/native/a2f-runner"

if [[ $(uname -s) != "Darwin" || $(uname -m) != "arm64" ]]; then
  echo "The local MLX Audio2Face backend requires Apple Silicon macOS." >&2
  exit 1
fi
command -v swift >/dev/null 2>&1 || { echo "Swift 6+ is required." >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
command -v shasum >/dev/null 2>&1 || { echo "shasum is required." >&2; exit 1; }

mkdir -p "$ASSET_DIR"
download_checked() {
  local name=$1 sha=$2 target="$ASSET_DIR/$1"
  if [[ ! -f "$target" ]]; then
    curl -L --fail --silent --show-error "$MODEL_REPO/$name?download=true" -o "$target.partial"
    mv "$target.partial" "$target"
  fi
  echo "$sha  $target" | shasum -a 256 -c -
}

download_checked model_data.npz 4c2205365533790add8219170b9505960e31d2fca82708f2e3db31c5ccf092a4
download_checked bs_skin.npz cc94a937af9438f007b811b3f686015d7e96fc0ce964974905e130724ff1c4db
download_checked bs_skin_config.json 8580023e66336320e3802efe5c9be8a72fd5aa6162561d27225a056565345e4c
download_checked bs_tongue.npz 82032cb3a4f08d1f6e543ce80e8d9587aca8394b8cc009384ef21db19a90450e
download_checked bs_tongue_config.json 9ef4d5ee664cb0975624de11b689b374c69d74b99126297ac4bdd58b9470a750
download_checked README.md 17385721f7940e3e4d4d2e266a54ace3aff5dadea8518334135d05a716cb3518
cp "$PROJECT_DIR/assets/notices/NVIDIA_Audio2Face_Claire_NOTICE.txt" "$ASSET_DIR/NVIDIA_MODEL_NOTICE.txt"

swift build --package-path "$RUNNER_DIR" -c release --product a2f-runner
METAL_SCRIPT="$RUNNER_DIR/.build/checkouts/speech-swift/scripts/build_mlx_metallib.sh"
if ! xcrun --find metal >/dev/null 2>&1; then
  echo "Xcode's optional Metal toolchain is missing." >&2
  echo "Install it with: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi
env BUILD_DIR="$RUNNER_DIR/.build" "$METAL_SCRIPT" release

echo "Audio2Face runner and checked Claire retarget assets are ready."
echo "The MLX Claire weights download automatically on the first learned run."
