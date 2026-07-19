"""Authenticated development launcher for the source-dependent macOS shell.

This file deliberately lives with the native development target rather than in
the production Python API.  It binds the existing FastAPI application to an
ephemeral IPv4 loopback socket and requires the per-launch native session token
on every request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import select
import socket
import sys
import threading


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoanim-source-runtime-service")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--session-token", required=True)
    parser.add_argument("--native-parent-pid", type=int, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rhubarb-bin", type=Path, required=True)
    parser.add_argument("--a2f-runner", type=Path, required=True)
    parser.add_argument("--a2f-assets", type=Path, required=True)
    parser.add_argument("--viewer-vendor", type=Path, required=True)
    return parser


def arm_parent_exit_guard(expected_parent_pid: int) -> None:
    """Exit even if the Swift host dies while Python is still importing."""

    if expected_parent_pid <= 1 or os.getppid() != expected_parent_pid:
        raise RuntimeError("The authenticated native parent is no longer running")

    def watch() -> None:
        try:
            queue = select.kqueue()
            event = select.kevent(
                expected_parent_pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
                fflags=select.KQ_NOTE_EXIT,
            )
            queue.control([event], 1, None)
        finally:
            os._exit(0)

    threading.Thread(target=watch, name="native-parent-guard", daemon=True).start()


async def serve(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve(strict=True)
    if not (source_root / ".venv/bin/autoanim-gnm").is_file():
        raise RuntimeError("The source checkout does not contain .venv/bin/autoanim-gnm")
    if len(args.session_token) < 32:
        raise RuntimeError("The native session token is too short")

    # The source runtime is installed editable in this checkout's virtualenv.
    # Keep the working directory stable for the pipeline's retained relative
    # resource references until the packaged resource-locator phase replaces it.
    os.chdir(source_root)
    from autoanim_gnm.api import create_app
    import uvicorn

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    application = create_app(
        args.artifacts,
        model_path=args.model_path,
        rhubarb_bin=args.rhubarb_bin,
        a2f_runner=args.a2f_runner,
        a2f_asset_dir=args.a2f_assets,
        a2f_offline=True,
        viewer_vendor_root=args.viewer_vendor,
        session_token=args.session_token,
    )
    configuration = uvicorn.Config(
        application,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(configuration)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started and not task.done():
        await asyncio.sleep(0.01)
    if task.done():
        return int(await task or 1)
    print(
        json.dumps(
            {
                "event": "ready",
                "url": f"http://127.0.0.1:{port}/",
                "source_runtime_dependent": True,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    await task
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        arm_parent_exit_guard(args.native_parent_pid)
        return asyncio.run(serve(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"source runtime failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
