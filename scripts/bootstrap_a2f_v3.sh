#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=${AUTOANIM_CACHE_DIR:-"$PROJECT_DIR/.cache/autoanim_gnm"}
PROFILE_DIR=${AUTOANIM_A2F_V3_PROFILE:-"$CACHE_DIR/a2f-v3-claire-profile"}
MODEL_REVISION="b74132732fd9a9d29b237bec193ded64c9745e91"
MODEL_REPO="https://huggingface.co/nvidia/Audio2Face-3D-v3.0/resolve/$MODEL_REVISION"

command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
command -v shasum >/dev/null 2>&1 || { echo "shasum is required." >&2; exit 1; }

mkdir -p "$PROFILE_DIR"
download_checked() {
  local name=$1 sha=$2 target="$PROFILE_DIR/$1"
  if [[ ! -f "$target" ]]; then
    curl -L --fail --silent --show-error "$MODEL_REPO/$name?download=true" -o "$target.partial"
    mv "$target.partial" "$target"
  fi
  echo "$sha  $target" | shasum -a 256 -c -
}

download_checked network.onnx db47c2701ca849de443c9e9f25657210f829a74fc458ee6fed603a8a501253a8
download_checked network_info.json 5524cdbe96a6bc89c78f06f32ae959e2302c50c663f407cb2b392c0ecac5975d
download_checked model_data_Claire.npz 4f05331263fa609321335e55c20922f4d6709d33160d368c3b537f019429ea4f
download_checked model_config_Claire.json 0819530451ad28ef42c1a478398850dc91e32475a49f9899ed37216309107fb4
download_checked bs_skin_Claire.npz bcb1fde2c7384fe9ec3cf9932b0fdeeda01fe4a1e42bba3817bba14e7f1716d3
download_checked bs_skin_config_Claire.json e2b508c5d17f1fb01c3a5b0292072d09e66e8c55bc23fcbe0c9aee8f8eae1713
download_checked bs_tongue_Claire.npz 812f10c34edb6ab6f36aedfe1d59a79d8190a5a8ee0a6071382f6bae9e3413b6
download_checked bs_tongue_config_Claire.json ace4b0b6b9be280f96a66568bd13ac4ea1fddf9c690464ab450fe339d9752e98
download_checked README.md cb824ad69b31f99e91f7f9b93ff0c19b5f00f296fdf0834ed359bb6191925905
cp "$PROJECT_DIR/assets/notices/NVIDIA_Audio2Face_v3_NOTICE.txt" "$PROFILE_DIR/NVIDIA_MODEL_NOTICE.txt"

echo "Pinned Audio2Face v3 Claire model and interpretation profile are ready."
echo "Profile: $PROFILE_DIR"
echo "This enables AutoAnim's local ONNX candidate, not NVIDIA SDK parity or production approval."
