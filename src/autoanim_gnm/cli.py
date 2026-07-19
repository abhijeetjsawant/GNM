"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import create_app
from .artifacts import sha256
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .lipsync_qualification import (
    LipsyncQualificationError,
    evaluate_controls_qualification,
    load_identity_artifact,
)
from .materials import MaterialValidationError, validate_material_package
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json
from .service import ApplicationService, default_model_path
from .viewer import default_viewer_vendor_root


MATERIAL_SPEC_FIELDS = frozenset(
    {"package_id", "inventory", "capture", "provenance", "rights", "claims"}
)


def _read_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoAnimError("INPUT_INVALID", f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise AutoAnimError("INPUT_INVALID", f"{label} must be a JSON object")
    return value


def _required_file_sha256(path: Path, *, label: str) -> str:
    try:
        return sha256(path)
    except OSError as exc:
        raise AutoAnimError(
            "INPUT_INVALID", f"{label} file is missing or unreadable"
        ) from exc


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
    qualify = subparsers.add_parser("qualify-lipsync")
    qualify.add_argument("profile", type=Path)
    qualify.add_argument("--controls", type=Path, required=True)
    qualify.add_argument("--source-audio", type=Path, required=True)
    qualify.add_argument("--character-manifest", type=Path, required=True)
    qualify.add_argument("--identity-artifact", type=Path, required=True)
    qualify.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="ARTIFACT_ID=PATH",
        help="Repeat for every exact annotation/prototype provenance artifact",
    )
    qualify.add_argument("--out", type=Path, required=True)
    qualify.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
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
    character_material_template = character_actions.add_parser("material-template")
    character_material_template.add_argument("character_id")
    character_material_template.add_argument("--character-revision", required=True)
    character_material_template.add_argument("--package-root", type=Path, required=True)
    character_material_template.add_argument("--spec", type=Path, required=True)
    character_material_template.add_argument("--attester", required=True)
    character_material_template.add_argument("--evidence-ref", required=True)
    character_material_template.add_argument("--evidence", type=Path, required=True)
    character_material_template.add_argument("--package-subject", required=True)
    character_material_template.add_argument(
        "--same-subject-attested", action="store_true", required=True
    )
    character_material_template.add_argument(
        "--authored-for-attested", action="store_true", required=True
    )
    character_material_template.add_argument(
        "--usage-scope",
        choices=("personal", "production", "commercial", "research"),
        default="production",
    )
    character_material_template.add_argument(
        "--displacement-midpoint", type=float, required=True
    )
    character_material_template.add_argument(
        "--displacement-scale-m", type=float, required=True
    )
    character_material_template.add_argument("--out", type=Path, required=True)
    character_material_import = character_actions.add_parser("import-material")
    character_material_import.add_argument("character_id")
    character_material_import.add_argument("--character-revision", required=True)
    character_material_import.add_argument("--package-root", type=Path, required=True)
    character_material_import.add_argument("--spec", type=Path, required=True)
    character_material_import.add_argument("--attachment", type=Path, required=True)
    character_material_import.add_argument(
        "--usage-scope",
        choices=("personal", "production", "commercial", "research"),
        default="production",
    )
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
    # Character subcommands can have both the job-store --artifacts directory
    # and a file-valued --out (for example material-template). Prefer the
    # explicit artifact store so an attachment output path is never created as
    # a directory by ApplicationService initialization.
    artifact_root = (
        getattr(args, "artifacts", None)
        or getattr(args, "out", None)
        or Path("artifacts/jobs")
    )
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
        elif args.command == "qualify-lipsync":
            evidence: dict[str, Path] = {}
            for value in args.evidence:
                artifact_id, separator, supplied_path = value.partition("=")
                if (
                    not separator
                    or not artifact_id
                    or not supplied_path
                    or artifact_id in evidence
                ):
                    raise LipsyncQualificationError(
                        "INVALID_PROVENANCE",
                        "Every --evidence must be one unique ARTIFACT_ID=PATH",
                        field="evidence",
                    )
                evidence[artifact_id] = Path(supplied_path)
            identity = load_identity_artifact(args.identity_artifact)
            adapter = GNMAdapter()
            decoder = ExpressionDecoder(
                Path(__file__).resolve().parents[2]
                / "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
            )
            report = evaluate_controls_qualification(
                args.profile,
                controls_path=args.controls,
                source_audio_path=args.source_audio,
                character_manifest_path=args.character_manifest,
                identity_artifact_path=args.identity_artifact,
                provenance_artifacts=evidence,
                rig=ControlRig(adapter, decoder, identity=identity),
            )
            result = service.store.signer.sign(report.as_dict())
            write_json(args.out, result)
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
                    consent_evidence_sha256=_required_file_sha256(
                        args.consent_evidence, label="Consent evidence"
                    ),
                    consent_expires_at=args.consent_expires_at,
                    consent_note=args.consent_note,
                )
            elif args.character_command == "material-template":
                specification = _read_json_object(
                    args.spec, label="Material specification"
                )
                prepared = service.prepare_character_material_attachment(
                    args.character_id,
                    args.package_root,
                    specification=specification,
                    base_revision_id=args.character_revision,
                    usage_scope=args.usage_scope,
                    attester=args.attester,
                    evidence_ref=args.evidence_ref,
                    evidence_sha256=_required_file_sha256(
                        args.evidence, label="Material binding evidence"
                    ),
                    package_subject=args.package_subject,
                    same_subject_attested=args.same_subject_attested,
                    authored_for_attested=args.authored_for_attested,
                    displacement_midpoint=args.displacement_midpoint,
                    displacement_scale_m=args.displacement_scale_m,
                )
                write_json(args.out, prepared["attachment"])
                result = {
                    "status": "validated",
                    "attachment": str(args.out),
                    "attachment_sha256": sha256(args.out),
                    "attachment_payload_sha256": prepared["attachment"][
                        "attachment_payload_sha256"
                    ],
                    "material_manifest_payload_sha256": prepared[
                        "material_manifest"
                    ]["manifest_payload_sha256"],
                }
            elif args.character_command == "import-material":
                specification = _read_json_object(
                    args.spec, label="Material specification"
                )
                attachment = _read_json_object(
                    args.attachment, label="Material attachment"
                )
                result = service.import_character_material(
                    args.character_id,
                    args.package_root,
                    specification=specification,
                    attachment=attachment,
                    base_revision_id=args.character_revision,
                    usage_scope=args.usage_scope,
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
            specification = _read_json_object(
                args.spec, label="Material specification"
            )
            if set(specification) != set(MATERIAL_SPEC_FIELDS):
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
    except LipsyncQualificationError as exc:
        print(
            json.dumps(
                {"code": exc.code, "message": str(exc), "field": exc.field},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
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
