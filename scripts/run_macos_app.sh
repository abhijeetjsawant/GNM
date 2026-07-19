#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE_DIR="$PROJECT_DIR/native/autoanim-macos"
HELPER="$PACKAGE_DIR/Support/source_runtime_service.py"

swift build --package-path "$PACKAGE_DIR" -c release --product AutoAnimMac
BIN_DIR=$(swift build --package-path "$PACKAGE_DIR" -c release --show-bin-path)
EXECUTABLE="$BIN_DIR/AutoAnimMac"

if [[ ! -x "$EXECUTABLE" ]] || [[ ! -f "$HELPER" ]]; then
  echo "The native development executable or authenticated helper is missing." >&2
  exit 1
fi

echo "Launching the source-runtime-dependent native AutoAnim UI."
exec env \
  AUTOANIM_SOURCE_ROOT="$PROJECT_DIR" \
  AUTOANIM_NATIVE_HELPER="$HELPER" \
  "$EXECUTABLE"
