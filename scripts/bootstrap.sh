#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR=${AUTOANIM_CACHE_DIR:-"$PROJECT_DIR/.cache/autoanim_gnm"}
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_SHA="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python 3.12 is required (set PYTHON_BIN)." >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required on PATH." >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe is required on PATH." >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required on PATH." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required on PATH." >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "tar is required on PATH." >&2; exit 1; }
command -v shasum >/dev/null 2>&1 || { echo "shasum is required on PATH." >&2; exit 1; }

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --no-deps -e "$PROJECT_DIR/gnm/shape"
"$VENV_DIR/bin/python" -m pip install -e "$PROJECT_DIR[dev]"

mkdir -p "$CACHE_DIR"
curl -L --fail --silent --show-error "$MODEL_URL" -o "$CACHE_DIR/face_landmarker.task.partial"
echo "$MODEL_SHA  $CACHE_DIR/face_landmarker.task.partial" | shasum -a 256 -c -
mv "$CACHE_DIR/face_landmarker.task.partial" "$CACHE_DIR/face_landmarker.task"

SYSTEM_NAME=$(uname -s)
case "$SYSTEM_NAME" in
  Darwin)
    RHUBARB_ARCHIVE="Rhubarb-Lip-Sync-1.14.0-macOS.zip"
    RHUBARB_URL="https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/$RHUBARB_ARCHIVE"
    RHUBARB_SHA="f991deacac6c973a14a4431a16a58b842f436531e120cfaea142c87c0d3ab4c5"
    ;;
  Linux)
    RHUBARB_ARCHIVE="Rhubarb-Lip-Sync-1.14.0-Linux.zip"
    RHUBARB_URL="https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/$RHUBARB_ARCHIVE"
    RHUBARB_SHA="a9a9074862cff47b2d59b8bf399a678a3b0b74f9452ad6ad94cb292913dd8667"
    ;;
  *) echo "Install Rhubarb 1.14 manually and set RHUBARB_BIN." >&2; exit 0 ;;
esac

mkdir -p "$CACHE_DIR/rhubarb"
curl -L --fail --silent --show-error "$RHUBARB_URL" -o "$CACHE_DIR/$RHUBARB_ARCHIVE.partial"
if [[ -n "$RHUBARB_SHA" ]]; then
  echo "$RHUBARB_SHA  $CACHE_DIR/$RHUBARB_ARCHIVE.partial" | shasum -a 256 -c -
fi
mv "$CACHE_DIR/$RHUBARB_ARCHIVE.partial" "$CACHE_DIR/$RHUBARB_ARCHIVE"
RHUBARB_STAGE=$(mktemp -d "$CACHE_DIR/rhubarb-stage.XXXXXX")
trap 'rm -rf "$RHUBARB_STAGE"' EXIT
unzip -q "$CACHE_DIR/$RHUBARB_ARCHIVE" -d "$RHUBARB_STAGE"
RHUBARB_SOURCE=$(find "$RHUBARB_STAGE" -type f -name rhubarb -print -quit)
if [[ -z "$RHUBARB_SOURCE" ]]; then
  echo "The Rhubarb archive does not contain an executable." >&2
  exit 1
fi
RHUBARB_SOURCE_DIR=$(dirname "$RHUBARB_SOURCE")
# Rhubarb resolves PocketSphinx data relative to the executable, so retain the
# entire release bundle (especially res/sphinx), not just the binary.
cp -R "$RHUBARB_SOURCE_DIR"/. "$CACHE_DIR/rhubarb/"
chmod +x "$CACHE_DIR/rhubarb/rhubarb"
if [[ "$SYSTEM_NAME" == "Darwin" ]]; then
  command -v codesign >/dev/null 2>&1 || {
    echo "codesign is required to qualify the cached Rhubarb executable on macOS." >&2
    exit 1
  }
  # The upstream 1.14 macOS archive is unsigned. Verify the published archive
  # hash first, then apply a local ad-hoc signature so AMFI does not stall or
  # SIGKILL this cached third-party executable when Developer Mode is disabled.
  codesign --force --sign - --timestamp=none "$CACHE_DIR/rhubarb/rhubarb"
  codesign --verify --strict "$CACHE_DIR/rhubarb/rhubarb"
fi
if [[ ! -f "$CACHE_DIR/rhubarb/res/sphinx/cmudict-en-us.dict" ]]; then
  echo "The Rhubarb resource bundle is incomplete." >&2
  exit 1
fi

AUTOANIM_CACHE_DIR="$CACHE_DIR" "$PROJECT_DIR/scripts/bootstrap_viewer.sh"

echo "Ready. Activate: source $VENV_DIR/bin/activate"
echo "Rhubarb: export RHUBARB_BIN=$CACHE_DIR/rhubarb/rhubarb"
