"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import create_app
from .errors import AutoAnimError
from .service import ApplicationService, default_model_path
from .viewer import default_viewer_vendor_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoanim-gnm")
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--rhubarb-bin", type=Path)
    parser.add_argument("--a2f-runner", type=Path)
    parser.add_argument("--a2f-assets", type=Path)
    parser.add_argument("--a2f-offline", action="store_true")
    parser.add_argument("--viewer-vendor", type=Path, default=default_viewer_vendor_root())
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("health")
    health.add_argument("--json", action="store_true")
    health.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    audio = subparsers.add_parser("audio")
    audio.add_argument("input", type=Path)
    audio.add_argument("--out", type=Path, required=True)
    audio.add_argument("--fps", type=int, default=30)
    audio.add_argument("--emotion", default="auto")
    audio.add_argument("--emotion-strength", type=float, default=0.65)
    audio.add_argument("--backend", choices=("auto", "learned", "fallback"), default="auto")
    audio.add_argument("--dialog")
    image = subparsers.add_parser("image")
    image.add_argument("input", type=Path)
    image.add_argument("--out", type=Path, required=True)
    image.add_argument("--modes", type=int, choices=(10, 20), default=20)
    image.add_argument("--allow-low-confidence", action="store_true")
    multiview = subparsers.add_parser("multiview")
    multiview.add_argument("inputs", type=Path, nargs="+")
    multiview.add_argument("--out", type=Path, required=True)
    multiview.add_argument("--roles", help="Comma-separated roles in input order")
    multiview.add_argument("--texture-size", type=int, choices=(128, 256, 512, 1024), default=256)
    multiview.add_argument("--focal-scale", type=float, default=1.25)
    multiview.add_argument(
        "--calibration",
        type=Path,
        help="Versioned OpenCV camera-bundle JSON with fit/held-out view assignments",
    )
    multiview.add_argument(
        "--mirror-fill",
        action="store_true",
        help="Disabled for GNM: anatomical UV tiles are not horizontal mirror pairs",
    )
    video = subparsers.add_parser("video")
    video.add_argument("input", type=Path)
    video.add_argument("--out", type=Path, required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = getattr(args, "out", None) or getattr(args, "artifacts", Path("artifacts/jobs"))
    service = ApplicationService(
        artifact_root,
        model_path=args.model_path,
        rhubarb_bin=args.rhubarb_bin,
        a2f_runner=args.a2f_runner,
        a2f_asset_dir=args.a2f_assets,
        a2f_offline=args.a2f_offline,
        viewer_vendor_root=args.viewer_vendor,
    )
    try:
        if args.command == "health":
            result = service.health()
        elif args.command == "audio":
            result = service.audio(
                args.input,
                fps=args.fps,
                emotion=args.emotion,
                emotion_strength=args.emotion_strength,
                backend=args.backend,
                dialog=args.dialog,
            )
        elif args.command == "image":
            result = service.image(
                args.input,
                modes=args.modes,
                allow_low_confidence=args.allow_low_confidence,
            )
        elif args.command == "multiview":
            roles = (
                tuple(value.strip() for value in args.roles.split(",") if value.strip())
                if args.roles
                else None
            )
            result = service.multiview(
                args.inputs,
                roles=roles,
                texture_size=args.texture_size,
                focal_scale=args.focal_scale,
                mirror_fill=args.mirror_fill,
                camera_bundle_path=args.calibration,
                input_names=tuple(path.name for path in args.inputs),
            )
        elif args.command == "video":
            result = service.video(args.input)
        else:
            import uvicorn

            uvicorn.run(
                create_app(
                    args.artifacts,
                    model_path=args.model_path,
                    rhubarb_bin=args.rhubarb_bin,
                    a2f_runner=args.a2f_runner,
                    a2f_asset_dir=args.a2f_assets,
                    a2f_offline=args.a2f_offline,
                    viewer_vendor_root=args.viewer_vendor,
                ),
                host=args.host,
                port=args.port,
                workers=1,
            )
            return 0
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except AutoAnimError as exc:
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 3 if exc.code == "DEPENDENCY_MISSING" else (1 if exc.code == "INTERNAL_ERROR" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
