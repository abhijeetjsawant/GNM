#!/bin/zsh
# D4i: create the momentum interpreter the MHR body fit runs on (`build_commercial_multiview_comparison.py`'s default
# `--body mhr` shells out to it, one process per performer; .venv has no pymomentum).
#
#   scripts/bootstrap_mhr.sh            # -> .venv-mhr/ (gitignored) beside the checkout
#   MHR_VENV=/path scripts/bootstrap_mhr.sh
#
# It installs EXACTLY scripts/mhr-requirements.lock (pymomentum-cpu 0.1.114.post0 and its resolved set, frozen from
# the /tmp/momenv every D4-D4d delivery ran on), with --no-deps, on CPython 3.12.13 via uv. It then records the
# Python version, the platform, the resolved set and the lock's sha256 in .venv-mhr/bootstrap-record.json, and
# imports pymomentum to prove the interpreter works. The exact-byte oracle (D4i) is what proves it reproduces the
# deliveries; this script does not claim that.
set -eu
cd "$(dirname "$0")/.."
VENV=${MHR_VENV:-$PWD/.venv-mhr}
LOCK=$PWD/scripts/mhr-requirements.lock
PYVER=3.12.13
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)"; exit 1; }
uv venv --python $PYVER "$VENV"
uv pip install --python "$VENV/bin/python" --no-deps -r "$LOCK"
"$VENV/bin/python" - "$VENV" "$LOCK" <<'PY'
import hashlib, json, platform, subprocess, sys
from pathlib import Path
venv, lock = Path(sys.argv[1]), Path(sys.argv[2])
import pymomentum.geometry  # noqa: F401  -- the interpreter must actually carry momentum
freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], check=True, capture_output=True,
                        text=True).stdout.split()
pins = [l.strip() for l in lock.read_text().splitlines() if l.strip() and not l.startswith("#")]
record = {"python": sys.version, "implementation": platform.python_implementation(),
          "platform": platform.platform(), "machine": platform.machine(),
          "momentum": next(p for p in freeze if p.startswith("pymomentum-cpu==")),
          "resolved": freeze, "lock": str(lock), "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
          "resolved_equals_lock": sorted(freeze) == sorted(pins)}
(venv / "bootstrap-record.json").write_text(json.dumps(record, indent=1))
print(json.dumps(record, indent=1))
if not record["resolved_equals_lock"]:
    raise SystemExit("the resolved set differs from the lock")
PY
echo "momentum interpreter: $VENV/bin/python"
