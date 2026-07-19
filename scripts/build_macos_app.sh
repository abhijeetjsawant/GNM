#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE_DIR="$PROJECT_DIR/native/autoanim-macos"
OUTPUT_DIR=${AUTOANIM_MACOS_OUTPUT_DIR:-"$PACKAGE_DIR/dist"}
APP_PATH="$OUTPUT_DIR/AutoAnim.app"
INFO_TEMPLATE="$PACKAGE_DIR/Resources/Info.plist"
HELPER_SOURCE="$PACKAGE_DIR/Support/source_runtime_service.py"

if [[ ! -x "$PROJECT_DIR/.venv/bin/autoanim-gnm" ]] || [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "The source checkout's .venv is required; run scripts/bootstrap.sh first." >&2
  exit 1
fi
if [[ ! -f "$INFO_TEMPLATE" ]] || [[ ! -f "$HELPER_SOURCE" ]]; then
  echo "The native app bundle templates are incomplete." >&2
  exit 1
fi

swift build --package-path "$PACKAGE_DIR" -c release --product AutoAnimMac
BIN_DIR=$(swift build --package-path "$PACKAGE_DIR" -c release --show-bin-path)
BINARY="$BIN_DIR/AutoAnimMac"
if [[ ! -x "$BINARY" ]]; then
  echo "Release executable was not produced at $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
STAGE_DIR=$(mktemp -d "$OUTPUT_DIR/.autoanim-app-stage.XXXXXX")
trap 'rm -rf "$STAGE_DIR"' EXIT
STAGE_APP="$STAGE_DIR/AutoAnim.app"
mkdir -p "$STAGE_APP/Contents/MacOS" "$STAGE_APP/Contents/Resources"
cp "$BINARY" "$STAGE_APP/Contents/MacOS/AutoAnimMac"
cp "$INFO_TEMPLATE" "$STAGE_APP/Contents/Info.plist"
cp "$HELPER_SOURCE" "$STAGE_APP/Contents/Resources/source_runtime_service.py"

SOURCE_REVISION=$(git -C "$PROJECT_DIR" rev-parse HEAD)
plutil -replace AutoAnimSourceRoot -string "$PROJECT_DIR" "$STAGE_APP/Contents/Info.plist"
plutil -replace AutoAnimSourceRevision -string "$SOURCE_REVISION" "$STAGE_APP/Contents/Info.plist"
plutil -lint "$STAGE_APP/Contents/Info.plist"

# Keep local source builds ad-hoc by default. Distribution/release automation
# must select a trusted identity explicitly; it must never depend on whichever
# personal certificate happens to be installed on the build host.
SIGN_IDENTITY=${AUTOANIM_CODE_SIGN_IDENTITY:-}
if [[ -n "$SIGN_IDENTITY" ]]; then
  codesign --force --sign "$SIGN_IDENTITY" --options runtime --timestamp=none "$STAGE_APP"
  SIGNING_DESCRIPTION="Apple Development identity $SIGN_IDENTITY (hardened runtime)"
else
  codesign --force --sign - --timestamp=none "$STAGE_APP"
  SIGNING_DESCRIPTION="ad-hoc identity (non-hardened development fallback)"
fi
codesign --verify --deep --strict --verbose=2 "$STAGE_APP"

if [[ -e "$APP_PATH" ]]; then
  rm -rf "$APP_PATH"
fi
mv "$STAGE_APP" "$APP_PATH"

echo "$APP_PATH"
echo "Signed with $SIGNING_DESCRIPTION."
echo "Development-only: this app requires $PROJECT_DIR/.venv and checkout assets."
