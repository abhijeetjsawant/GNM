#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE_DIR="$PROJECT_DIR/native/autoanim-macos"
APP_PATH=${AUTOANIM_MACOS_APP_PATH:-"$PACKAGE_DIR/dist/AutoAnim.app"}
INFO="$APP_PATH/Contents/Info.plist"
HELPER="$APP_PATH/Contents/Resources/source_runtime_service.py"

if [[ ! -x "$APP_PATH/Contents/MacOS/AutoAnimMac" ]] || [[ ! -f "$INFO" ]] || [[ ! -f "$HELPER" ]]; then
  echo "AutoAnim.app is incomplete; run scripts/build_macos_app.sh first." >&2
  exit 1
fi

plutil -lint "$INFO"
DEPENDENT=$(plutil -extract AutoAnimSourceRuntimeDependent raw "$INFO")
SOURCE_ROOT=$(plutil -extract AutoAnimSourceRoot raw "$INFO")
if [[ "$DEPENDENT" != "true" ]] || [[ "$SOURCE_ROOT" != "$PROJECT_DIR" ]]; then
  echo "The development bundle does not truthfully identify its source runtime." >&2
  exit 1
fi
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

SMOKE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/autoanim-macos-smoke.XXXXXX")
SERVICE_PID=""
GUI_PID=""
GUI_CHILD_PID=""
cleanup() {
  if [[ -n "$GUI_PID" ]]; then kill -TERM "$GUI_PID" 2>/dev/null || true; fi
  if [[ -n "$GUI_CHILD_PID" ]]; then kill -KILL "$GUI_CHILD_PID" 2>/dev/null || true; fi
  if [[ -n "$SERVICE_PID" ]]; then kill -TERM "$SERVICE_PID" 2>/dev/null || true; fi
  rm -rf "$SMOKE_DIR"
}
trap cleanup EXIT

helper_is_live() {
  local state command
  state=$(ps -p "$GUI_CHILD_PID" -o state= 2>/dev/null | tr -d '[:space:]' || true)
  command=$(ps -p "$GUI_CHILD_PID" -o command= 2>/dev/null || true)
  [[ -n "$state" ]] && [[ "$state" != Z* ]] && [[ "$command" == *source_runtime_service.py* ]]
}

TOKEN=$(openssl rand -hex 32)
"$PROJECT_DIR/.venv/bin/python" "$HELPER" \
  --source-root "$PROJECT_DIR" \
  --artifacts "$PROJECT_DIR/artifacts/jobs" \
  --model-path "$PROJECT_DIR/.cache/autoanim_gnm/face_landmarker.task" \
  --rhubarb-bin "$PROJECT_DIR/.cache/autoanim_gnm/rhubarb/rhubarb" \
  --a2f-runner "$PROJECT_DIR/native/a2f-runner/.build/arm64-apple-macosx/release/a2f-runner" \
  --a2f-assets "$PROJECT_DIR/.cache/autoanim_gnm/a2f-claire" \
  --viewer-vendor "$PROJECT_DIR/.cache/autoanim_gnm/viewer/three-0.183.2" \
  --native-parent-pid "$$" \
  --session-token "$TOKEN" >"$SMOKE_DIR/service.stdout" 2>"$SMOKE_DIR/service.stderr" &
SERVICE_PID=$!

for _ in $(seq 1 300); do
  if [[ -s "$SMOKE_DIR/service.stdout" ]]; then break; fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    cat "$SMOKE_DIR/service.stderr" >&2
    exit 1
  fi
  sleep 0.1
done
READY_LINE=$(sed -n '1p' "$SMOKE_DIR/service.stdout")
BASE_URL=$(printf '%s' "$READY_LINE" | sed -E 's/.*"url":"([^"]+)".*/\1/')
if [[ ! "$BASE_URL" =~ ^http://127\.0\.0\.1:[0-9]+/$ ]]; then
  cat "$SMOKE_DIR/service.stdout" >&2
  cat "$SMOKE_DIR/service.stderr" >&2
  echo "Authenticated source runtime did not publish a valid loopback URL." >&2
  exit 1
fi

UNAUTHORIZED_STATUS=$(curl -sS -o "$SMOKE_DIR/unauthorized.json" -w '%{http_code}' "${BASE_URL}api/health")
if [[ "$UNAUTHORIZED_STATUS" != "401" ]]; then
  echo "Unauthenticated health request returned $UNAUTHORIZED_STATUS, expected 401." >&2
  exit 1
fi
AUTHORIZED_STATUS=$(curl -sS -o "$SMOKE_DIR/health.json" -w '%{http_code}' \
  -H "X-AutoAnim-Token: $TOKEN" "${BASE_URL}api/health")
if [[ "$AUTHORIZED_STATUS" != "200" ]]; then
  cat "$SMOKE_DIR/health.json" >&2
  echo "Authenticated health request returned $AUTHORIZED_STATUS." >&2
  exit 1
fi

kill -TERM "$SERVICE_PID"
set +e
wait "$SERVICE_PID"
SERVICE_STATUS=$?
set -e
if [[ "$SERVICE_STATUS" != "0" ]] && [[ "$SERVICE_STATUS" != "130" ]] && [[ "$SERVICE_STATUS" != "143" ]]; then
  echo "Source runtime exited with unexpected status $SERVICE_STATUS during shutdown." >&2
  exit 1
fi
SERVICE_PID=""

if [[ "${AUTOANIM_GUI_SMOKE:-0}" == "1" ]]; then
  AUTOANIM_SOURCE_ROOT="$PROJECT_DIR" "$APP_PATH/Contents/MacOS/AutoAnimMac" \
    >"$SMOKE_DIR/app.stdout" 2>"$SMOKE_DIR/app.stderr" &
  GUI_PID=$!
  for _ in $(seq 1 450); do
    if ! kill -0 "$GUI_PID" 2>/dev/null; then
      cat "$SMOKE_DIR/app.stderr" >&2
      echo "Native app exited during GUI smoke." >&2
      exit 1
    fi
    GUI_CHILD_PID=$(
      pgrep -P "$GUI_PID" -f 'source_runtime_service.py' | sed -n '1p' || true
    )
    if [[ -n "$GUI_CHILD_PID" ]]; then break; fi
    sleep 0.1
  done
  if [[ -z "$GUI_CHILD_PID" ]]; then
    cat "$SMOKE_DIR/app.stderr" >&2
    echo "Native app did not supervise its authenticated source runtime within 45 seconds." >&2
    exit 1
  fi
  kill -TERM "$GUI_PID"
  wait "$GUI_PID" || true
  GUI_PID=""
  for _ in $(seq 1 30); do
    if ! helper_is_live; then break; fi
    sleep 0.1
  done
  if helper_is_live; then
    ps -p "$GUI_CHILD_PID" -o pid,ppid,state,command >&2 || true
    echo "Authenticated source runtime outlived the native app." >&2
    exit 1
  fi
  GUI_CHILD_PID=""
fi

echo "macOS source-runtime smoke passed: signed bundle, token rejection, authenticated health, clean shutdown."
