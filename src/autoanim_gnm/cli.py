"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import create_app
from .artifacts import sha256
from .errors import AutoAnimError
from .materials import MaterialValidationError, validate_material_package
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
    parser.add_argument(
        "--characters",
        type=Path,
        help="Character library root (defaults beside the selected jobs root)",
    )
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
    audio.add_argument("--character", help="Saved character ID to apply")
    audio.add_argument("--character-revision", help="Exact saved character revision ID")
    audio.add_argument(
        "--usage-scope",
        choices=("personal", "production", "commercial", "research"),
        default="production",
    )
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
    video.add_argument("--character", help="Saved target character ID")
    video.add_argument("--character-revision", help="Exact saved character revision ID")
    video.add_argument(
        "--usage-scope",
        choices=("personal", "production", "commercial", "research"),
        default="production",
    )
    character = subparsers.add_parser("character")
    character.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    character_actions = character.add_subparsers(dest="character_command", required=True)
    character_actions.add_parser("list")
    character_show = character_actions.add_parser("show")
    character_show.add_argument("character_id")
    character_promote = character_actions.add_parser("promote")
    character_promote.add_argument("job_id")
    character_promote.add_argument("--name", required=True)
    character_promote.add_argument("--consent", action="store_true", required=True)
    character_promote.add_argument("--consent-subject", required=True)
    character_promote.add_argument("--consent-attester", required=True)
    character_promote.add_argument(
        "--consent-scope",
        choices=("personal", "production", "commercial", "research"),
        required=True,
    )
    character_promote.add_argument("--consent-evidence-ref", required=True)
    character_promote.add_argument("--consent-evidence", type=Path, required=True)
    character_promote.add_argument("--consent-expires-at")
    character_promote.add_argument("--consent-note")
    character_revoke = character_actions.add_parser("revoke")
    character_revoke.add_argument("character_id")
    character_revoke.add_argument("--reason", required=True)
    character_revoke.add_argument("--revoked-by", required=True)
    direct = subparsers.add_parser("direct")
    direct.add_argument("source_job_id")
    direct.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    direct.add_argument("--provider", choices=("codex", "claude"), default="codex")
    direct.add_argument("--instructions", default="")
    direct.add_argument("--instructions-file", type=Path)
    direct.add_argument("--transcript", default="")
    direct.add_argument("--transcript-file", type=Path)
    direct.add_argument("--character")
    direct.add_argument("--character-revision")
    direct.add_argument(
        "--usage-scope",
        choices=("personal", "production", "commercial", "research"),
        default="production",
    )
    direct.add_argument("--provider-model")
    direct.add_argument("--timeout", type=int, default=180)
    direct.add_argument("--max-budget-usd", type=float)
    job = subparsers.add_parser("job")
    job.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    job_actions = job.add_subparsers(dest="job_command", required=True)
    job_seal = job_actions.add_parser("seal-legacy")
    job_seal.add_argument("job_id")
    job_seal.add_argument("--attested-by", required=True)
    job_seal.add_argument("--reason", required=True)
    material = subparsers.add_parser("material")
    material.add_argument("package_root", type=Path)
    material.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON with package_id, inventory, capture, provenance, rights, and claims",
    )
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
        character_root=args.characters,
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
                character_id=args.character,
                character_revision_id=args.character_revision,
                usage_scope=args.usage_scope,
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
            result = service.video(
                args.input,
                character_id=args.character,
                character_revision_id=args.character_revision,
                usage_scope=args.usage_scope,
            )
        elif args.command == "character":
            if args.character_command == "list":
                result = {"characters": service.characters.list()}
            elif args.character_command == "show":
                try:
                    result = service.characters.read(args.character_id)
                except FileNotFoundError as exc:
                    raise AutoAnimError(
                        "CHARACTER_NOT_FOUND", "Character was not found"
                    ) from exc
            elif args.character_command == "promote":
                result = service.promote_character(
                    args.job_id,
                    name=args.name,
                    consent_attested=args.consent,
                    consent_subject=args.consent_subject,
                    consent_attester=args.consent_attester,
                    consent_scope=args.consent_scope,
                    consent_evidence_ref=args.consent_evidence_ref,
                    consent_evidence_sha256=sha256(args.consent_evidence),
                    consent_expires_at=args.consent_expires_at,
                    consent_note=args.consent_note,
                )
            else:
                result = service.characters.revoke(
                    args.character_id,
                    reason=args.reason,
                    revoked_by=args.revoked_by,
                )
        elif args.command == "direct":
            instructions = (
                args.instructions_file.read_text(encoding="utf-8")
                if args.instructions_file is not None
                else args.instructions
            )
            transcript = (
                args.transcript_file.read_text(encoding="utf-8")
                if args.transcript_file is not None
                else args.transcript
            )
            result = service.direct(
                args.source_job_id,
                provider=args.provider,
                instructions=instructions,
                transcript=transcript,
                character_id=args.character,
                character_revision_id=args.character_revision,
                usage_scope=args.usage_scope,
                model=args.provider_model,
                timeout_seconds=args.timeout,
                max_budget_usd=args.max_budget_usd,
            )
        elif args.command == "job":
            result = service.store.seal_legacy(
                args.job_id,
                attested_by=args.attested_by,
                reason=args.reason,
            )
        elif args.command == "material":
            try:
                specification = json.loads(args.spec.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AutoAnimError(
                    "INPUT_INVALID", "Material specification is not readable JSON"
                ) from exc
            if not isinstance(specification, dict) or set(specification) != {
                "package_id",
                "inventory",
                "capture",
                "provenance",
                "rights",
                "claims",
            }:
                raise AutoAnimError(
                    "INPUT_INVALID", "Material specification fields are missing or unknown"
                )
            result = validate_material_package(
                args.package_root,
                package_id=specification["package_id"],
                inventory=specification["inventory"],
                capture=specification["capture"],
                provenance=specification["provenance"],
                rights=specification["rights"],
                claims=specification["claims"],
            )
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
                    character_root=args.characters,
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
    except MaterialValidationError as exc:
        print(
            json.dumps(
                {"code": exc.code, "message": str(exc), "field": exc.field},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
