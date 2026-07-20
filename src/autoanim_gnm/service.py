"""Shared application service used identically by CLI and HTTP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping
import wave
import zipfile

import cv2
import mediapipe
import numpy as np
import scipy

from . import __version__
from .a2f import resolve_a2f_runner
from .a2f_v3_local import default_local_v3_profile_directory
from .a2f_v3_profile import load_official_v3_claire_profile
from .acting import ActingDirector, TICKS_PER_SECOND
from .animation import calibrate_lip_contact
from .artifacts import JobStore, sha256
from .capture_session import load_verified_video_capture_session
from .characters import CharacterRevision, CharacterStore
from .audio import resolve_rhubarb
from .audio_pipeline import _resolve_a2f_assets, run_audio_pipeline
from .body import (
    ATTACHMENT_SCHEMA_VERSION,
    BODY_TRACK_LIMITATIONS,
    BODY_TRACK_SCHEMA_VERSION,
    CANONICAL_HUMANOID,
    MAX_DURATION_TICKS,
    SKELETON_SCHEMA_VERSION,
    attachment_contract,
    compile_body_track,
)
from .production_readiness import evaluate_production_readiness
from .phone_events import (
    PHONE_EVENT_SCHEMA_VERSION,
    PHONE_TIMING_REPORT_SCHEMA_VERSION,
    evaluate_bilabial_timing,
    load_textgrid_phone_events,
)
from .phone_articulation import (
    PHONE_ARTICULATION_REPORT_SCHEMA_VERSION,
    PHONE_ARTICULATION_VERIFIER_ALGORITHM,
    articulation_evidence_bindings,
    diagnostic_articulation_calibration,
    evaluate_phone_articulation,
    measure_articulation_geometry_from_controls,
    summarize_phone_articulation,
)
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .image import validate_model
from .image_pipeline import run_image_pipeline
from .multiview_pipeline import run_multiview_pipeline
from .oral_validation import validate_controls_npz
from .video_pipeline import run_video_pipeline
from .video_capture import load_capture_npz, load_verified_capture_jsonl
from .video_evidence import load_verified_performance_evidence
from .video_observation import (
    load_pixel_observations,
    load_verified_observation_v3_summary,
)
from .viewer import default_viewer_vendor_root, viewer_vendor_health
from .serialization import write_json, write_npz
from .sequence_provider import local_a2f_v3_worker_preflight
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_AUDIO_CONTROL_NPZ_MEMBERS = frozenset(
    {
        "expression.npy",
        "rotations.npy",
        "translation.npy",
        "timestamps.npy",
        "fps.npy",
        "viseme_weights.npy",
        "speech_activity.npy",
        "energy.npy",
        "pitch_semitones.npy",
        "accent.npy",
        "phrase_id.npy",
        "emotion_intensity.npy",
        "mouth_speed_limited.npy",
        "lip_contact_confidence.npy",
        "lip_contact_target_gap.npy",
        "contact_correction_applied.npy",
        "lip_contact_attained.npy",
        "contact_continuity_restored.npy",
        "contact_corrected.npy",
        "lip_order_repaired.npy",
        "mouth_aperture_edit_eligible.npy",
        "mouth_aperture_edit_protected_contact.npy",
        "mouth_aperture_edit_applied.npy",
        "mouth_aperture_edit_target_attained.npy",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _preflight_audio_controls_npz(path: Path, *, maximum_frames: int) -> None:
    """Reject archive amplification before NumPy materializes control arrays."""

    if maximum_frames < 2:
        raise ValueError("Audio control frame bound is invalid")
    maximum_uncompressed_bytes = maximum_frames * 2048 + 2 * 1024 * 1024
    if path.stat().st_size <= 0 or path.stat().st_size > maximum_uncompressed_bytes:
        raise ValueError("Audio controls archive exceeds its compressed byte bound")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(set(names))
                or set(names) != _AUDIO_CONTROL_NPZ_MEMBERS
                or any(member.flag_bits & 0x1 for member in members)
                or sum(member.file_size for member in members)
                > maximum_uncompressed_bytes
            ):
                raise ValueError(
                    "Audio controls archive members or expanded bytes exceed the schema"
                )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("Audio controls are not a safe numeric NPZ") from exc


def default_model_path() -> Path:
    configured = os.environ.get("AUTOANIM_FACE_LANDMARKER")
    return Path(configured) if configured else PROJECT_ROOT / ".cache/autoanim_gnm/face_landmarker.task"


def runtime_versions() -> dict[str, str]:
    versions = {
        "autoanim": __version__,
        "gnm": "3.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "mediapipe": mediapipe.__version__,
    }
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        versions["onnxruntime"] = "missing"
    try:
        versions["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        versions["git_dirty"] = str(
            bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        ).lower()
    except Exception:
        versions["git"] = "unknown"
        versions["git_dirty"] = "unknown"
    try:
        versions["ffmpeg"] = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except Exception:
        versions["ffmpeg"] = "missing"
    return versions


class ApplicationService:
    def __init__(
        self,
        artifact_root: str | Path,
        *,
        model_path: str | Path | None = None,
        rhubarb_bin: str | Path | None = None,
        a2f_runner: str | Path | None = None,
        a2f_asset_dir: str | Path | None = None,
        a2f_offline: bool = False,
        viewer_vendor_root: str | Path | None = None,
        character_root: str | Path | None = None,
    ):
        self.store = JobStore(artifact_root)
        self.characters = CharacterStore(
            (
                Path(character_root)
                if character_root is not None
                else self.store.root.parent / "characters"
            ),
            self.store,
        )
        self.model_path = Path(model_path) if model_path is not None else default_model_path()
        self.rhubarb_bin = Path(rhubarb_bin) if rhubarb_bin is not None else None
        self.a2f_runner = Path(a2f_runner) if a2f_runner is not None else None
        self.a2f_asset_dir = Path(a2f_asset_dir) if a2f_asset_dir is not None else None
        self.a2f_offline = a2f_offline
        self.viewer_vendor_root = (
            Path(viewer_vendor_root)
            if viewer_vendor_root is not None
            else default_viewer_vendor_root()
        )
        self._readiness_lock = threading.Lock()

    def health(self) -> dict:
        checks: dict[str, dict[str, str | bool]] = {}
        try:
            model = GNMAdapter()
            checks["gnm"] = {"ready": True, "detail": model.model.version.value}
        except Exception as exc:
            checks["gnm"] = {"ready": False, "detail": str(exc)}
        for tool in ("ffmpeg", "ffprobe"):
            path = shutil.which(tool)
            checks[tool] = {"ready": bool(path), "detail": path or "not on PATH"}
        try:
            path = validate_model(self.model_path)
            checks["mediapipe_model"] = {"ready": True, "detail": str(path)}
        except AutoAnimError as exc:
            checks["mediapipe_model"] = {"ready": False, "detail": exc.message}
        try:
            path = resolve_rhubarb(self.rhubarb_bin)
            checks["rhubarb"] = {"ready": True, "detail": str(path)}
        except AutoAnimError as exc:
            checks["rhubarb"] = {"ready": False, "detail": exc.message}
        try:
            path = resolve_a2f_runner(self.a2f_runner)
            metallib = path.parent / "mlx.metallib"
            checks["a2f_runner"] = {
                "ready": metallib.is_file(),
                "detail": str(path) if metallib.is_file() else f"MLX Metal library missing beside {path}",
            }
        except Exception as exc:
            checks["a2f_runner"] = {"ready": False, "detail": str(exc)}
        try:
            path = _resolve_a2f_assets(self.a2f_asset_dir)
            checks["a2f_assets"] = {"ready": True, "detail": str(path)}
            provenance = (path / "README.md", path / "NVIDIA_MODEL_NOTICE.txt")
            missing_provenance = [item.name for item in provenance if not item.is_file()]
            checks["a2f_provenance"] = {
                "ready": not missing_provenance,
                "detail": (
                    str(path)
                    if not missing_provenance
                    else f"missing {', '.join(missing_provenance)} under {path}"
                ),
            }
        except Exception as exc:
            checks["a2f_assets"] = {"ready": False, "detail": str(exc)}
            checks["a2f_provenance"] = {"ready": False, "detail": str(exc)}
        v3_preflight = local_a2f_v3_worker_preflight()
        checks["a2f_v3_worker"] = {
            "ready": v3_preflight.can_execute_locally,
            "detail": v3_preflight.blocker,
        }
        try:
            import onnxruntime

            providers = tuple(onnxruntime.get_available_providers())
            profile_path = default_local_v3_profile_directory()
            profile = load_official_v3_claire_profile(
                profile_path, verify_network=True
            )
            network_path = profile.root / "network.onnx"
            ready = "CPUExecutionProvider" in providers and network_path.is_file()
            checks["a2f_v3_local"] = {
                "ready": ready,
                "detail": (
                    f"{network_path} · onnxruntime {onnxruntime.__version__} · "
                    f"providers {', '.join(providers)}"
                    if ready
                    else "Pinned network or CPUExecutionProvider is unavailable"
                ),
            }
        except Exception as exc:
            checks["a2f_v3_local"] = {"ready": False, "detail": str(exc)}
        checks["viewer_bundle"] = viewer_vendor_health(self.viewer_vendor_root)
        required = (
            "gnm", "ffmpeg", "ffprobe", "mediapipe_model", "rhubarb",
            "a2f_runner", "a2f_assets", "a2f_provenance", "a2f_v3_local",
            "viewer_bundle",
        )
        return {
            "status": "ready" if all(bool(checks[name]["ready"]) for name in required) else "degraded",
            "checks": checks,
            "versions": runtime_versions(),
        }

    def audio(
        self,
        input_path: str | Path,
        *,
        fps: int = 30,
        emotion: str = "auto",
        dialog: str | None = None,
        backend: str = "auto",
        emotion_strength: float = 0.65,
        mouth_aperture_gain: float = 1.0,
        mouth_aperture_author: str | None = None,
        mouth_aperture_reason: str | None = None,
        input_name: str | None = None,
        character_id: str | None = None,
        character_revision_id: str | None = None,
        usage_scope: str = "production",
        a2f_v3_request_path: str | Path | None = None,
        a2f_v3_response_path: str | Path | None = None,
        a2f_v3_model_path: str | Path | None = None,
        a2f_v3_runtime_path: str | Path | None = None,
        a2f_v3_identity_path: str | Path | None = None,
        a2f_v3_schema_path: str | Path | None = None,
        a2f_v3_profile_dir: str | Path | None = None,
        a2f_v3_local_seed: int = 0,
        phone_annotation_path: str | Path | None = None,
        phone_annotation_name: str | None = None,
        phone_annotations_independently_reviewed: bool = False,
        phone_annotation_reviewer: str | None = None,
    ) -> dict:
        character = self._resolve_character(
            character_id, character_revision_id, usage_scope=usage_scope
        )
        annotation_snapshot: Path | None = None
        if phone_annotation_path is not None:
            annotation_source = Path(phone_annotation_path)
            if not annotation_source.is_file():
                raise AutoAnimError(
                    "INPUT_INVALID", "Phone TextGrid annotation is not a file"
                )
            with tempfile.NamedTemporaryFile(
                "wb", suffix=".TextGrid", delete=False
            ) as snapshot_handle:
                annotation_snapshot = Path(snapshot_handle.name)
            try:
                shutil.copy2(annotation_source, annotation_snapshot)
            except Exception:
                annotation_snapshot.unlink(missing_ok=True)
                raise
        configuration = {
            "fps": fps,
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "mouth_aperture_gain": mouth_aperture_gain,
            "mouth_aperture_author": mouth_aperture_author,
            "mouth_aperture_reason": mouth_aperture_reason,
            "dialog": dialog,
            "backend": backend,
            "character_id": character.character_id if character is not None else None,
            "character_revision_id": character.revision_id if character is not None else None,
            "usage_scope": usage_scope,
            "a2f_v3_import": backend == "a2f-v3",
            "a2f_v3_local": backend == "a2f-v3-local",
            "a2f_v3_local_seed": a2f_v3_local_seed,
            "phone_evidence": (
                {
                    "present": True,
                    "source_name": Path(
                        phone_annotation_name or str(phone_annotation_path)
                    ).name,
                    "source_sha256": sha256(annotation_snapshot),
                    "independently_reviewed": phone_annotations_independently_reviewed,
                    "reviewer": phone_annotation_reviewer,
                    "motion_authority": False,
                }
                if annotation_snapshot is not None
                else {
                    "present": False,
                    "independently_reviewed": False,
                    "reviewer": None,
                    "motion_authority": False,
                }
            ),
        }
        try:
            job_id, job_dir, retained, manifest = self.store.start(
                "audio_animation",
                input_path,
                configuration,
                original_name=input_name,
                attachments=(
                    {"phone_annotations": annotation_snapshot}
                    if annotation_snapshot is not None
                    else None
                ),
            )
        finally:
            if annotation_snapshot is not None:
                annotation_snapshot.unlink(missing_ok=True)
        retained_phone_annotation: Path | None = None
        if phone_annotation_path is not None:
            attachment = next(
                (
                    item
                    for item in manifest.get("attachments", [])
                    if item.get("logical_name") == "phone_annotations"
                ),
                None,
            )
            if not isinstance(attachment, dict) or not isinstance(
                attachment.get("retained_name"), str
            ):
                error = AutoAnimError(
                    "INTERNAL_ERROR", "Phone annotation attachment was not retained"
                )
                self.store.fail(
                    manifest, job_dir, error.as_dict(), runtime_versions()
                )
                raise error
            retained_phone_annotation = job_dir / attachment["retained_name"]
            if attachment.get("sha256") != configuration["phone_evidence"].get(
                "source_sha256"
            ):
                error = AutoAnimError(
                    "INPUT_CHANGED",
                    "Phone TextGrid changed while its immutable job attachment was created",
                )
                self.store.fail(manifest, job_dir, error.as_dict(), runtime_versions())
                raise error
        versions = runtime_versions()
        try:
            result = run_audio_pipeline(
                retained,
                job_dir,
                fps=fps,
                emotion=emotion,
                dialog=dialog,
                rhubarb_bin=self.rhubarb_bin,
                backend=backend,
                emotion_strength=emotion_strength,
                mouth_aperture_gain=mouth_aperture_gain,
                mouth_aperture_author=mouth_aperture_author,
                mouth_aperture_reason=mouth_aperture_reason,
                a2f_runner=self.a2f_runner,
                a2f_asset_dir=self.a2f_asset_dir,
                a2f_offline=self.a2f_offline,
                identity=character.identity if character is not None else None,
                texture_path=(
                    character.texture_path
                    if character is not None and not character.runtime_material_paths
                    else None
                ),
                runtime_material_paths=(
                    character.runtime_material_paths if character is not None else None
                ),
                texture_triangle_uvs=(
                    character.triangle_uvs if character is not None else None
                ),
                character_ref=self._character_ref(character),
                a2f_v3_request_path=a2f_v3_request_path,
                a2f_v3_response_path=a2f_v3_response_path,
                a2f_v3_model_path=a2f_v3_model_path,
                a2f_v3_runtime_path=a2f_v3_runtime_path,
                a2f_v3_identity_path=a2f_v3_identity_path,
                a2f_v3_schema_path=a2f_v3_schema_path,
                a2f_v3_profile_dir=a2f_v3_profile_dir,
                a2f_v3_local_seed=a2f_v3_local_seed,
                phone_annotation_path=retained_phone_annotation,
                phone_annotations_independently_reviewed=(
                    phone_annotations_independently_reviewed
                ),
                phone_annotation_reviewer=phone_annotation_reviewer,
            )
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc

    def image(
        self,
        input_path: str | Path,
        *,
        modes: int = 20,
        allow_low_confidence: bool = False,
        input_name: str | None = None,
    ) -> dict:
        configuration = {"modes": modes, "allow_low_confidence": allow_low_confidence}
        job_id, job_dir, retained, manifest = self.store.start(
            "image_fit", input_path, configuration, original_name=input_name
        )
        versions = runtime_versions()
        try:
            result = run_image_pipeline(
                retained,
                job_dir,
                model_path=self.model_path,
                modes=modes,
                allow_low_confidence=allow_low_confidence,
            )
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc

    def video(
        self,
        input_path: str | Path,
        *,
        input_name: str | None = None,
        character_id: str | None = None,
        character_revision_id: str | None = None,
        usage_scope: str = "production",
        audio_visual_repair: bool = False,
        mouth_aperture_gain: float = 1.0,
        mouth_aperture_author: str | None = None,
        mouth_aperture_reason: str | None = None,
    ) -> dict:
        character = self._resolve_character(
            character_id, character_revision_id, usage_scope=usage_scope
        )
        configuration = {
            "backend": "mediapipe",
            "retargeter": (
                "geometry_calibrated_dense_contact_aperture_v3"
                if self.a2f_asset_dir is not None
                else "semantic_prototype_contact_aperture_v3_fallback"
            ),
            "neutral_baseline_seconds": 0.2,
            "profile": "offline",
            "character_id": character.character_id if character is not None else None,
            "character_revision_id": character.revision_id if character is not None else None,
            "usage_scope": usage_scope,
            "audio_visual_repair": audio_visual_repair,
            "mouth_aperture_gain": mouth_aperture_gain,
            "mouth_aperture_author": mouth_aperture_author,
            "mouth_aperture_reason": mouth_aperture_reason,
        }
        _, job_dir, retained, manifest = self.store.start(
            "video_performance", input_path, configuration, original_name=input_name
        )
        versions = runtime_versions()
        try:
            result = run_video_pipeline(
                retained,
                job_dir,
                model_path=self.model_path,
                a2f_asset_dir=self.a2f_asset_dir,
                identity=character.identity if character is not None else None,
                texture_path=(
                    character.texture_path
                    if character is not None and not character.runtime_material_paths
                    else None
                ),
                runtime_material_paths=(
                    character.runtime_material_paths if character is not None else None
                ),
                texture_triangle_uvs=(
                    character.triangle_uvs if character is not None else None
                ),
                character_ref=self._character_ref(character),
                require_audio_visual_repair=audio_visual_repair,
                rhubarb_bin=self.rhubarb_bin,
                a2f_runner=self.a2f_runner,
                a2f_offline=self.a2f_offline,
                mouth_aperture_gain=mouth_aperture_gain,
                mouth_aperture_author=mouth_aperture_author,
                mouth_aperture_reason=mouth_aperture_reason,
            )
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc

    def production_readiness(
        self,
        performance_job_id: str,
        *,
        direction_job_id: str | None = None,
        require_acting: bool = False,
        require_body: bool = False,
        require_pbr: bool = True,
    ) -> dict[str, Any]:
        """Consolidate release evidence without mutating or approving a job."""

        if not self._readiness_lock.acquire(blocking=False):
            raise AutoAnimError(
                "BUSY",
                "Production evidence verification is already running",
                retryable=True,
            )
        try:
            return self._production_readiness_unlocked(
                performance_job_id,
                direction_job_id=direction_job_id,
                require_acting=require_acting,
                require_body=require_body,
                require_pbr=require_pbr,
            )
        finally:
            self._readiness_lock.release()

    def _production_readiness_unlocked(
        self,
        performance_job_id: str,
        *,
        direction_job_id: str | None = None,
        require_acting: bool = False,
        require_body: bool = False,
        require_pbr: bool = True,
    ) -> dict[str, Any]:
        """Perform one single-flight verification with bounded GNM mesh batches."""

        try:
            performance = self.store.read(performance_job_id)
        except FileNotFoundError as exc:
            raise AutoAnimError("JOB_NOT_FOUND", "Performance job was not found") from exc

        performance_manifest_verified = self.store.signer.verify(performance)
        source_input_verified = False
        input_ledger = performance.get("input")
        retained_inputs = list(self.store.job_dir(performance_job_id).glob("input.*"))
        if isinstance(input_ledger, dict) and len(retained_inputs) == 1:
            retained_input = retained_inputs[0]
            source_input_verified = bool(
                retained_input.is_file()
                and retained_input.stat().st_size == input_ledger.get("bytes")
                and sha256(retained_input) == input_ledger.get("sha256")
            )
        delivery_artifact_verified = False
        artifact_ledger = performance.get("artifacts")
        glb_ledger = (
            artifact_ledger.get("glb") if isinstance(artifact_ledger, dict) else None
        )
        if isinstance(glb_ledger, dict) and isinstance(glb_ledger.get("name"), str):
            try:
                self.store.artifact(performance_job_id, glb_ledger["name"])
                delivery_artifact_verified = True
            except FileNotFoundError:
                pass
        performance_evidence_artifact_verified = False
        evidence_ledger = (
            artifact_ledger.get("performance_evidence")
            if isinstance(artifact_ledger, dict)
            else None
        )
        if (
            performance_manifest_verified
            and isinstance(evidence_ledger, dict)
            and isinstance(evidence_ledger.get("name"), str)
        ):
            try:
                evidence_path = self.store.artifact(
                    performance_job_id, evidence_ledger["name"]
                )
                input_sha256 = (
                    input_ledger.get("sha256") if isinstance(input_ledger, dict) else None
                )
                capture = performance.get("capture")
                capture_frames = (
                    capture.get("frames") if isinstance(capture, dict) else None
                )
                if isinstance(input_sha256, str) and isinstance(capture_frames, int):
                    load_verified_performance_evidence(
                        evidence_path,
                        expected_source_sha256=input_sha256,
                        expected_frame_count=capture_frames,
                    )
                    performance_evidence_artifact_verified = (
                        performance.get("kind") != "video_performance"
                    )
            except (FileNotFoundError, OSError, ValueError):
                pass

        observation_v3_artifacts_verified = False
        capture_session_artifact_verified = False
        capture_session_production_claims_verified = False
        if (
            performance.get("kind") == "video_performance"
            and performance_manifest_verified
        ):
            try:
                self.store.require_sealed(performance_job_id)
                video_paths: dict[str, Path] = {}
                for logical_name in (
                    "capture",
                    "capture_jsonl",
                    "performance_evidence",
                    "pixel_observations",
                    "observation_v3",
                    "capture_session",
                ):
                    entry = _mapping(_mapping(artifact_ledger).get(logical_name))
                    name = entry.get("name")
                    if not isinstance(name, str):
                        raise FileNotFoundError(logical_name)
                    artifact_path = self.store.artifact(
                        performance_job_id, name
                    )
                    if (
                        artifact_path.stat().st_size != entry.get("bytes")
                        or sha256(artifact_path) != entry.get("sha256")
                    ):
                        raise FileNotFoundError(logical_name)
                    video_paths[logical_name] = artifact_path
                capture_track = load_capture_npz(video_paths["capture"])
                load_verified_capture_jsonl(
                    video_paths["capture_jsonl"], capture_track
                )
                pixel_observations = load_pixel_observations(
                    video_paths["pixel_observations"]
                )
                input_sha256 = (
                    input_ledger.get("sha256")
                    if isinstance(input_ledger, dict)
                    else None
                )
                input_bytes = (
                    input_ledger.get("bytes")
                    if isinstance(input_ledger, dict)
                    else None
                )
                capture_summary = _mapping(performance.get("capture"))
                if (
                    capture_track.provenance.source_sha256 != input_sha256
                    or capture_track.provenance.source_bytes != input_bytes
                    or capture_track.frame_count != capture_summary.get("frames")
                ):
                    raise ValueError("Capture source binding does not match the sealed job")
                pixel_observations.validate_capture(capture_track)
                load_verified_performance_evidence(
                    video_paths["performance_evidence"],
                    expected_source_sha256=(
                        capture_track.provenance.source_sha256
                    ),
                    expected_frame_count=capture_track.frame_count,
                    expected_capture=capture_track,
                )
                performance_evidence_artifact_verified = source_input_verified
                load_verified_observation_v3_summary(
                    video_paths["observation_v3"],
                    pixel_observations_path=video_paths["pixel_observations"],
                    capture_artifact_path=video_paths["capture"],
                    expected_capture=capture_track,
                    expected_observations=pixel_observations,
                )
                observation_v3_artifacts_verified = source_input_verified
                capture_session = load_verified_video_capture_session(
                    video_paths["capture_session"],
                    expected_capture=capture_track,
                    expected_observations=pixel_observations,
                    artifact_paths={
                        name: video_paths[name]
                        for name in (
                            "capture",
                            "capture_jsonl",
                            "performance_evidence",
                            "pixel_observations",
                            "observation_v3",
                        )
                    },
                )
                capture_session_artifact_verified = source_input_verified
                session_claims = _mapping(capture_session.get("claims"))
                session_assessments = _mapping(
                    capture_session.get("assessments")
                )
                capture_session_production_claims_verified = bool(
                    source_input_verified
                    and session_claims.get("production_validated")
                    and session_claims.get("identity_continuity_verified")
                    and session_claims.get("neutrality_independently_confirmed")
                    and _mapping(capture_session.get("subject_binding")).get(
                        "state"
                    )
                    == "bound"
                    and _mapping(session_assessments.get("neutrality")).get(
                        "state"
                    )
                    == "confirmed_neutral"
                    and _mapping(
                        session_assessments.get("identity_continuity")
                    ).get("state")
                    == "verified_consistent"
                )
            except (AutoAnimError, FileNotFoundError, OSError, ValueError):
                pass

        phone_evidence_artifacts_verified = False
        phone_evidence_artifact_failure_reason: str | None = None
        if (
            performance.get("kind") == "audio_animation"
            and not performance_manifest_verified
        ):
            phone_evidence_artifact_failure_reason = "legacy_unsealed_phone_evidence"
        if (
            performance.get("kind") == "audio_animation"
            and performance_manifest_verified
        ):
            phone_evidence_artifact_failure_reason = (
                "phone_evidence_missing_tampered_or_unreconstructable"
            )
            try:
                self.store.require_sealed(performance_job_id)
                phone_paths: dict[str, Path] = {}
                for logical_name in (
                    "phone_annotations",
                    "phone_events",
                    "phone_timing_report",
                    "phone_articulation_report",
                    "normalized_audio",
                    "controls",
                ):
                    entry = _mapping(_mapping(artifact_ledger).get(logical_name))
                    name = entry.get("name")
                    if not isinstance(name, str):
                        raise FileNotFoundError(logical_name)
                    phone_paths[logical_name] = self.store.artifact(
                        performance_job_id, name
                    )
                event_document = json.loads(
                    phone_paths["phone_events"].read_text(encoding="utf-8")
                )
                timing_document = json.loads(
                    phone_paths["phone_timing_report"].read_text(encoding="utf-8")
                )
                articulation_document = json.loads(
                    phone_paths["phone_articulation_report"].read_text(
                        encoding="utf-8"
                    )
                )
                event_bindings = _mapping(
                    _mapping(event_document).get("bindings")
                )
                timing_bindings = _mapping(
                    _mapping(timing_document).get("annotation_bindings")
                )
                articulation_bindings = _mapping(
                    _mapping(articulation_document).get("annotation_bindings")
                )
                articulation_evidence = _mapping(
                    _mapping(articulation_document).get("evidence_bindings")
                )
                verifier_bundle = hashlib.sha256()
                for verifier_source_name in (
                    "gnm_adapter.py",
                    "oral_validation.py",
                    "phone_articulation.py",
                    "phone_events.py",
                ):
                    verifier_bundle.update(verifier_source_name.encode("utf-8"))
                    verifier_bundle.update(
                        bytes.fromhex(
                            sha256(
                                PROJECT_ROOT
                                / "src/autoanim_gnm"
                                / verifier_source_name
                            )
                        )
                    )
                current_verifier = {
                    "verifier_algorithm": PHONE_ARTICULATION_VERIFIER_ALGORITHM,
                    "verifier_source_sha256": sha256(
                        PROJECT_ROOT / "src/autoanim_gnm/phone_articulation.py"
                    ),
                    "verifier_bundle_sha256": verifier_bundle.hexdigest(),
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    "gnm_head_asset_sha256": sha256(
                        PROJECT_ROOT / "gnm/shape/data/versions/v3_0/gnm_head.npz"
                    ),
                    "landmark_regressor_sha256": sha256(
                        PROJECT_ROOT / "gnm/shape/data/landmarks/head_sparse_68.txt"
                    ),
                    "expression_decoder_sha256": sha256(
                        PROJECT_ROOT
                        / "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
                    ),
                }
                if any(
                    articulation_evidence.get(name) != value
                    for name, value in current_verifier.items()
                ):
                    phone_evidence_artifact_failure_reason = (
                        "historical_phone_articulation_verifier_unavailable"
                    )
                    raise ValueError(
                        "Phone articulation verifier or numerical runtime differs"
                    )
                phone_analysis = _mapping(
                    _mapping(performance.get("analysis")).get("phone_evidence")
                )
                phone_review = _mapping(
                    _mapping(event_document).get("review")
                )
                phone_configuration = _mapping(
                    _mapping(performance.get("configuration")).get(
                        "phone_evidence"
                    )
                )
                expected_audio_sha256 = (
                    input_ledger.get("sha256")
                    if isinstance(input_ledger, dict)
                    else None
                )
                textgrid_sha256 = sha256(phone_paths["phone_annotations"])
                attachment = next(
                    (
                        item
                        for item in performance.get("attachments", [])
                        if isinstance(item, dict)
                        and item.get("logical_name") == "phone_annotations"
                    ),
                    None,
                )
                attachment_name = (
                    attachment.get("retained_name")
                    if isinstance(attachment, dict)
                    else None
                )
                attachment_path = (
                    self.store.job_dir(performance_job_id) / attachment_name
                    if isinstance(attachment_name, str)
                    else None
                )
                attachment_verified = bool(
                    isinstance(attachment, dict)
                    and attachment_path is not None
                    and Path(attachment_name).name == attachment_name
                    and attachment_path.is_file()
                    and attachment_path.stat().st_size == attachment.get("bytes")
                    and sha256(attachment_path) == attachment.get("sha256")
                    and attachment.get("sha256") == textgrid_sha256
                )
                with wave.open(str(phone_paths["normalized_audio"]), "rb") as audio:
                    if (
                        audio.getnchannels() != 1
                        or audio.getframerate() != 16_000
                        or audio.getsampwidth() != 2
                    ):
                        raise ValueError("Normalized phone-evidence clock is invalid")
                    normalized_duration = audio.getnframes() / audio.getframerate()
                _preflight_audio_controls_npz(
                    phone_paths["controls"],
                    maximum_frames=int(np.ceil(normalized_duration * 60.0)) + 1,
                )
                annotations = load_textgrid_phone_events(
                    phone_paths["phone_annotations"],
                    audio_path=retained_inputs[0],
                    duration_seconds=normalized_duration,
                    independently_reviewed=(
                        phone_configuration.get("independently_reviewed") is True
                    ),
                    reviewer=phone_configuration.get("reviewer"),
                )
                expected_event_document = annotations.as_dict()

                identity = np.zeros(253, dtype=np.float32)
                resolved_character_manifest_sha256: str | None = None
                character_ref = _mapping(
                    _mapping(performance.get("model")).get("character")
                )
                character_id = character_ref.get("character_id")
                character_revision_id = character_ref.get("revision_id")
                if isinstance(character_id, str) and isinstance(
                    character_revision_id, str
                ):
                    usage_scope = _mapping(performance.get("configuration")).get(
                        "usage_scope", "production"
                    )
                    resolved_character = self.characters.resolve(
                        character_id,
                        character_revision_id,
                        usage_scope=(
                            usage_scope
                            if isinstance(usage_scope, str)
                            else "production"
                        ),
                    )
                    identity = resolved_character.identity
                    resolved_character_manifest_sha256 = (
                        resolved_character.manifest_sha256
                    )
                with np.load(phone_paths["controls"], allow_pickle=False) as controls:
                    expression = np.asarray(controls["expression"], dtype=np.float32)
                    timestamps = np.asarray(controls["timestamps"], dtype=np.float64)
                adapter = GNMAdapter()
                rig = ControlRig(
                    adapter,
                    ExpressionDecoder(
                        PROJECT_ROOT
                        / "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
                    ),
                    identity=np.asarray(identity, dtype=np.float32),
                )
                landmarks = np.stack(
                    [rig.compact_landmarks(frame) for frame in expression]
                )
                interocular = float(
                    np.linalg.norm(landmarks[0, 36] - landmarks[0, 45])
                )
                lip_gap_interocular = np.mean(
                    np.stack(
                        [
                            np.linalg.norm(
                                landmarks[:, upper] - landmarks[:, lower], axis=1
                            )
                            for upper, lower in ((61, 67), (62, 66), (63, 65))
                        ],
                        axis=1,
                    ),
                    axis=1,
                ) / max(interocular, 1e-8)
                contact_calibration = calibrate_lip_contact(rig)
                expected_timing_document = evaluate_bilabial_timing(
                    annotations,
                    timestamps_seconds=timestamps,
                    lip_gap_interocular=lip_gap_interocular,
                    contact_threshold_interocular=min(
                        contact_calibration.neutral_gap_interocular,
                        max(
                            0.006,
                            contact_calibration.seal_gap_interocular + 0.003,
                        ),
                    ),
                )
                articulation_geometry = measure_articulation_geometry_from_controls(
                    expression,
                    identity,
                    landmarks,
                    adapter=adapter,
                )
                oral_geometry = validate_controls_npz(
                    phone_paths["controls"],
                    adapter=adapter,
                    identity=identity,
                )
                articulation_calibration = diagnostic_articulation_calibration(
                    bilabial_gap_interocular=min(
                        contact_calibration.neutral_gap_interocular,
                        max(
                            0.006,
                            contact_calibration.seal_gap_interocular + 0.003,
                        ),
                    ),
                    neutral_landmarks=rig.compact_landmarks(rig.viseme("X")),
                    rounded_landmarks=rig.compact_landmarks(rig.viseme("F")),
                )
                expected_articulation_document = evaluate_phone_articulation(
                    annotations,
                    timestamps_seconds=timestamps,
                    lip_gap_interocular=oral_geometry.lip_gap_interocular,
                    labiodental_gap_interocular=(
                        articulation_geometry.labiodental_gap_interocular
                    ),
                    tongue_upper_teeth_gap_interocular=(
                        oral_geometry.tongue_upper_teeth_gap_interocular
                    ),
                    mouth_width_interocular=(
                        articulation_geometry.mouth_width_interocular
                    ),
                    calibration=articulation_calibration,
                    evidence_bindings=articulation_evidence_bindings(
                        controls_path=phone_paths["controls"],
                        identity=identity,
                        gnm_asset_path=(
                            PROJECT_ROOT
                            / "gnm/shape/data/versions/v3_0/gnm_head.npz"
                        ),
                        landmark_regressor_path=(
                            PROJECT_ROOT
                            / "gnm/shape/data/landmarks/head_sparse_68.txt"
                        ),
                        expression_decoder_path=(
                            PROJECT_ROOT
                            / "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
                        ),
                        character_revision_manifest_sha256=(
                            resolved_character_manifest_sha256
                        ),
                    ),
                )
                phone_evidence_artifacts_verified = bool(
                    attachment_verified
                    and _mapping(event_document).get("schema_version")
                    == PHONE_EVENT_SCHEMA_VERSION
                    and event_document == expected_event_document
                    and _mapping(timing_document).get("schema_version")
                    == PHONE_TIMING_REPORT_SCHEMA_VERSION
                    and timing_document == expected_timing_document
                    and _mapping(articulation_document).get("schema_version")
                    == PHONE_ARTICULATION_REPORT_SCHEMA_VERSION
                    and articulation_document == expected_articulation_document
                    and isinstance(expected_audio_sha256, str)
                    and event_bindings.get("audio_sha256")
                    == expected_audio_sha256
                    and timing_bindings.get("audio_sha256")
                    == expected_audio_sha256
                    and articulation_bindings.get("audio_sha256")
                    == expected_audio_sha256
                    and event_bindings.get("textgrid_sha256")
                    == textgrid_sha256
                    and timing_bindings.get("textgrid_sha256")
                    == textgrid_sha256
                    and articulation_bindings.get("textgrid_sha256")
                    == textgrid_sha256
                    and phone_configuration.get("source_sha256")
                    == textgrid_sha256
                    and phone_configuration.get("motion_authority") is False
                    and timing_document == performance.get("phone_timing")
                    and summarize_phone_articulation(articulation_document)
                    == performance.get("phone_articulation")
                    and phone_analysis.get("present") is True
                    and phone_analysis.get("motion_authored_by_annotations")
                    is False
                    and phone_analysis.get("event_count")
                    == _mapping(event_document).get("event_count")
                    and phone_analysis.get("production_review_complete")
                    == phone_review.get("production_review_complete")
                    and phone_analysis.get("independently_reviewed")
                    == phone_review.get("independently_reviewed")
                )
                if phone_evidence_artifacts_verified:
                    phone_evidence_artifact_failure_reason = None
            except (
                AutoAnimError,
                FileNotFoundError,
                KeyError,
                IndexError,
                OSError,
                TypeError,
                AttributeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
                wave.Error,
            ):
                pass

        audio_visual_repair_artifacts_verified = False
        audio_visual_repair_qualification_verified = False
        repair = _mapping(
            _mapping(performance.get("retargeting")).get(
                "audio_visual_repair"
            )
        )
        if (
            performance.get("kind") == "video_performance"
            and repair.get("status") not in (None, "disabled")
        ):
            required_repair_artifacts = (
                "audio_visual_source",
                "audio_visual_repair",
                "audio_visual_repair_arrays",
                "audio_visual_source_controls",
                "audio_visual_source_arkit_controls",
                "audio_visual_source_normalized_audio",
                "audio_visual_source_raw",
                "audio_visual_source_retarget_calibration",
                "audio_visual_source_rhubarb",
                "audio_visual_source_cues",
                "audio_visual_source_timeline",
                "audio_video_timing",
                "audio_visual_timing_consumption",
                "performance_revision_chain",
            )
            try:
                for logical_name in required_repair_artifacts:
                    entry = _mapping(_mapping(artifact_ledger).get(logical_name))
                    name = entry.get("name")
                    if not isinstance(name, str):
                        raise FileNotFoundError(logical_name)
                    self.store.artifact(performance_job_id, name)
                audio_visual_repair_artifacts_verified = True
            except FileNotFoundError:
                pass
            qualification_hash = _mapping(repair.get("claims")).get(
                "qualificationProfileSha256"
            )
            qualification_entry = _mapping(
                _mapping(artifact_ledger).get("audio_visual_repair_qualification")
            )
            qualification_name = qualification_entry.get("name")
            if (
                isinstance(qualification_hash, str)
                and len(qualification_hash) == 64
                and isinstance(qualification_name, str)
                and qualification_entry.get("sha256") == qualification_hash
            ):
                try:
                    self.store.artifact(performance_job_id, qualification_name)
                    audio_visual_repair_qualification_verified = True
                except FileNotFoundError:
                    pass

        direction = None
        direction_manifest_verified = False
        if direction_job_id is not None and direction_job_id.strip():
            try:
                direction = self.store.read(direction_job_id.strip())
                direction_manifest_verified = self.store.signer.verify(direction)
            except FileNotFoundError as exc:
                raise AutoAnimError("JOB_NOT_FOUND", "Acting-direction job was not found") from exc

        character_revision: dict[str, Any] | None = None
        character_resolution_error: str | None = None
        model = performance.get("model")
        character_ref = (
            model.get("character") if isinstance(model, dict) else None
        )
        if isinstance(character_ref, dict):
            character_id = character_ref.get("character_id")
            revision_id = character_ref.get("revision_id")
            configuration = performance.get("configuration")
            usage_scope = (
                configuration.get("usage_scope", "production")
                if isinstance(configuration, dict)
                else "production"
            )
            if isinstance(character_id, str) and isinstance(revision_id, str):
                try:
                    resolved = self.characters.resolve(
                        character_id,
                        revision_id,
                        usage_scope=(
                            usage_scope if isinstance(usage_scope, str) else "production"
                        ),
                    )
                    character_revision = dict(resolved.manifest)
                    character_revision["_manifest_sha256"] = resolved.manifest_sha256
                    character_revision["_runtime_material_sha256s"] = dict(
                        resolved.runtime_material_sha256s
                    )
                except (AutoAnimError, FileNotFoundError) as exc:
                    character_resolution_error = (
                        exc.code if isinstance(exc, AutoAnimError) else "CHARACTER_NOT_FOUND"
                    )

        return evaluate_production_readiness(
            performance,
            performance_manifest_verified=performance_manifest_verified,
            source_input_verified=source_input_verified,
            delivery_artifact_verified=delivery_artifact_verified,
            performance_evidence_artifact_verified=(
                performance_evidence_artifact_verified
            ),
            observation_v3_artifacts_verified=(
                observation_v3_artifacts_verified
            ),
            capture_session_artifact_verified=(
                capture_session_artifact_verified
            ),
            capture_session_production_claims_verified=(
                capture_session_production_claims_verified
            ),
            phone_evidence_artifacts_verified=(
                phone_evidence_artifacts_verified
            ),
            phone_evidence_artifact_failure_reason=(
                phone_evidence_artifact_failure_reason
            ),
            audio_visual_repair_artifacts_verified=(
                audio_visual_repair_artifacts_verified
            ),
            audio_visual_repair_qualification_verified=(
                audio_visual_repair_qualification_verified
            ),
            character_revision=character_revision,
            character_resolution_error=character_resolution_error,
            direction=direction,
            direction_manifest_verified=direction_manifest_verified,
            require_acting=require_acting,
            require_body=require_body,
            require_pbr=require_pbr,
        )

    def promote_character(
        self,
        job_id: str,
        *,
        name: str,
        consent_attested: bool,
        consent_subject: str,
        consent_attester: str,
        consent_scope: str,
        consent_evidence_ref: str,
        consent_evidence_sha256: str,
        consent_expires_at: str | None = None,
        consent_note: str | None = None,
    ) -> dict:
        return self.characters.promote(
            job_id,
            name=name,
            consent_attested=consent_attested,
            consent_subject=consent_subject,
            consent_attester=consent_attester,
            consent_scope=consent_scope,
            consent_evidence_ref=consent_evidence_ref,
            consent_evidence_sha256=consent_evidence_sha256,
            consent_expires_at=consent_expires_at,
            consent_note=consent_note,
        )

    def import_character_material(
        self,
        character_id: str,
        package_root: str | Path,
        *,
        specification: Mapping[str, Any],
        attachment: Mapping[str, Any],
        base_revision_id: str,
        usage_scope: str = "production",
    ) -> dict:
        return self.characters.attach_material(
            character_id,
            package_root,
            specification=specification,
            attachment=attachment,
            base_revision_id=base_revision_id,
            usage_scope=usage_scope,
        )

    def prepare_character_material_attachment(
        self,
        character_id: str,
        package_root: str | Path,
        *,
        specification: Mapping[str, Any],
        base_revision_id: str,
        usage_scope: str,
        attester: str,
        evidence_ref: str,
        evidence_sha256: str,
        package_subject: str,
        same_subject_attested: bool,
        authored_for_attested: bool,
        displacement_midpoint: float,
        displacement_scale_m: float,
    ) -> dict:
        return self.characters.prepare_material_attachment(
            character_id,
            package_root,
            specification=specification,
            base_revision_id=base_revision_id,
            usage_scope=usage_scope,
            attester=attester,
            evidence_ref=evidence_ref,
            evidence_sha256=evidence_sha256,
            package_subject=package_subject,
            same_subject_attested=same_subject_attested,
            authored_for_attested=authored_for_attested,
            displacement_midpoint=displacement_midpoint,
            displacement_scale_m=displacement_scale_m,
        )

    def direct(
        self,
        source_job_id: str,
        *,
        provider: str,
        instructions: str,
        transcript: str = "",
        character_id: str | None = None,
        character_revision_id: str | None = None,
        usage_scope: str = "production",
        model: str | None = None,
        timeout_seconds: int = 180,
        provider_executable: str | Path | None = None,
        max_budget_usd: float | None = None,
    ) -> dict:
        try:
            source = self.store.require_sealed(source_job_id)
        except FileNotFoundError as exc:
            raise AutoAnimError("JOB_NOT_FOUND", "Source performance job was not found") from exc
        if source.get("status") != "succeeded" or source.get("kind") not in {
            "audio_animation",
            "video_performance",
        }:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Acting direction requires a successful audio-animation or video-performance job",
            )
        character = self._resolve_character(
            character_id, character_revision_id, usage_scope=usage_scope
        )
        duration = (
            source.get("audio", {}).get("duration_s")
            if source.get("kind") == "audio_animation"
            else source.get("capture", {}).get("duration_s")
        )
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise AutoAnimError("INTERNAL_ERROR", "Source job has no valid performance duration")
        duration_ticks = max(1, int(round(float(duration) * TICKS_PER_SECOND)))
        if duration_ticks > MAX_DURATION_TICKS:
            raise AutoAnimError(
                "LIMIT_EXCEEDED",
                "Acting/body preview is limited to 30 minutes per take",
                {
                    "duration_ticks": duration_ticks,
                    "maximum_duration_ticks": MAX_DURATION_TICKS,
                },
            )
        performance_context = self._acting_context(source_job_id, source)
        context_document = {
            "schema_version": "autoanim.direction-input/1.0",
            "source_job_id": source_job_id,
            "source_job_kind": source["kind"],
            "source_input_sha256": source.get("input", {}).get("sha256"),
            "duration_seconds": duration,
            "performance": performance_context,
            "character": self._character_ref(character),
        }
        with tempfile.TemporaryDirectory(prefix="autoanim-direction-input-") as temporary:
            context_path = write_json(Path(temporary) / "direction-context.json", context_document)
            _, job_dir, _, manifest = self.store.start(
                "acting_direction",
                context_path,
                {
                    "source_job_id": source_job_id,
                    "provider": provider,
                    "model": model,
                    "timeout_seconds": timeout_seconds,
                    "character_id": character.character_id if character is not None else None,
                    "character_revision_id": character.revision_id if character is not None else None,
                    "usage_scope": usage_scope,
                },
                original_name="direction-context.json",
            )
        versions = runtime_versions()
        try:
            directed = ActingDirector(
                provider,  # type: ignore[arg-type]
                executable=provider_executable,
                timeout_seconds=timeout_seconds,
                model=model,
                max_budget_usd=max_budget_usd,
            ).direct(
                job_dir,
                duration_seconds=float(duration),
                transcript=transcript,
                instructions=instructions,
                performance_context=performance_context,
                character_ref=self._character_ref(character),
            )
            body_track = compile_body_track(
                directed.plan,
                duration_ticks=duration_ticks,
            )
            body_arrays_path = write_npz(
                job_dir / "body-track.npz",
                ticks=body_track.ticks,
                root_translation_m=body_track.root_translation_m,
                local_rotations_xyzw=body_track.local_rotations_xyzw,
                foot_contacts=body_track.foot_contacts,
                gaze_direction_body=body_track.gaze_direction_body,
                gaze_strength=body_track.gaze_strength,
                gnm_eye_rotations_xyzw=body_track.gnm_eye_rotations_xyzw,
            )
            body_manifest = {
                "schema_version": BODY_TRACK_SCHEMA_VERSION,
                "skeleton_schema_version": SKELETON_SCHEMA_VERSION,
                "attachment_schema_version": ATTACHMENT_SCHEMA_VERSION,
                "approval_status": "unapproved_preview",
                "timebase": {
                    "ticks_per_second": body_track.ticks_per_second,
                    "duration_ticks": body_track.duration_ticks,
                    "sample_rate_hz": body_track.sample_rate_hz,
                },
                "joint_names": list(body_track.joint_names),
                "source_plan_sha256": body_track.source_plan_sha256,
                "limitations": list(BODY_TRACK_LIMITATIONS),
                "arrays": {
                    "artifact": "body-track.npz",
                    "sha256": sha256(body_arrays_path),
                    "bytes": body_arrays_path.stat().st_size,
                    "names": {
                        "ticks": list(body_track.ticks.shape),
                        "root_translation_m": list(body_track.root_translation_m.shape),
                        "local_rotations_xyzw": list(
                            body_track.local_rotations_xyzw.shape
                        ),
                        "foot_contacts": list(body_track.foot_contacts.shape),
                        "gaze_direction_body": list(
                            body_track.gaze_direction_body.shape
                        ),
                        "gaze_strength": list(body_track.gaze_strength.shape),
                        "gnm_eye_rotations_xyzw": list(
                            body_track.gnm_eye_rotations_xyzw.shape
                        ),
                    },
                },
            }
            write_json(job_dir / "body-track.json", body_manifest)
            write_json(job_dir / "humanoid-skeleton.json", CANONICAL_HUMANOID.as_dict())
            write_json(job_dir / "gnm-body-attachment.json", attachment_contract())
            source_repair = _mapping(
                _mapping(source.get("retargeting")).get("audio_visual_repair")
            )
            source_repair_claims = _mapping(source_repair.get("claims"))
            source_repair_changes_motion = bool(
                source_repair_claims.get("changesFinalGNMMotion", False)
            )
            result = {
                "kind": "acting_direction",
                "status": "succeeded",
                "source": {
                    "job_id": source_job_id,
                    "kind": source["kind"],
                    "motion_evidence": performance_context["motion_evidence"],
                    "audio_is_animation_source": bool(
                        source["kind"] == "audio_animation"
                        or source_repair_changes_motion
                    ),
                    "video_visual_tracking_is_animation_source": source["kind"] == "video_performance",
                    "video_audio_repair_is_animation_source": bool(
                        source["kind"] == "video_performance"
                        and source_repair_changes_motion
                    ),
                    "video_motion_authority": (
                        "mixed_visual_primary_audio_repair"
                        if source["kind"] == "video_performance"
                        and source_repair_changes_motion
                        else "visual_only"
                        if source["kind"] == "video_performance"
                        else "audio"
                    ),
                },
                "model": {
                    "provider": provider,
                    "requested_model": model,
                    "character": self._character_ref(character),
                },
                "direction": {
                    "summary": directed.plan["summary"],
                    "beat_count": len(directed.plan["beats"]),
                    "ticks_per_second": directed.envelope["ticks_per_second"],
                    "duration_ticks": directed.envelope["duration_ticks"],
                    "tools_allowed": False,
                    "lipsync_override_allowed": False,
                    "body_preview_compiled": True,
                    "body_preview_approval_status": "unapproved_preview",
                    "body_track_sample_rate_hz": body_track.sample_rate_hz,
                    "production_validated": False,
                },
                "metrics": {
                    "provider_duration_seconds": directed.envelope["duration_seconds"],
                    "performance_window_count": len(performance_context["windows"]),
                    "body_track_sample_count": int(body_track.ticks.size),
                    "body_foot_contact_fraction": float(
                        np.mean(body_track.foot_contacts)
                    ),
                },
                "artifacts": {
                    **directed.artifacts,
                    "body_track": "body-track.npz",
                    "body_track_manifest": "body-track.json",
                    "humanoid_skeleton": "humanoid-skeleton.json",
                    "gnm_body_attachment": "gnm-body-attachment.json",
                },
                "warnings": [
                    "LLM_DIRECTION_IS_A_PROPOSAL: the body track is an unapproved deterministic preview; approve or edit beats and recompile before publish.",
                    "The LLM authors declarative acting intent only; it does not generate or override lipsync controls.",
                    "BODY_TRACK_IS_FOUNDATIONAL: deterministic upper-body/gaze intent is compiled, but no body mesh, mocap reconstruction, locomotion, or production validation is included.",
                ],
            }
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc

    def _acting_context(self, job_id: str, result: dict) -> dict:
        artifacts = result.get("artifacts", {})
        controls = artifacts.get("controls")
        if not isinstance(controls, dict) or not isinstance(controls.get("name"), str):
            raise AutoAnimError("INTERNAL_ERROR", "Source job has no allowlisted control artifact")
        repair_context: dict[str, Any] | None = None
        try:
            controls_path = self.store.artifact(job_id, controls["name"])
            with np.load(controls_path, allow_pickle=False) as values:
                if result["kind"] == "audio_animation":
                    timestamps = np.asarray(values["timestamps"], dtype=np.float64)
                    expression = np.asarray(values["expression"], dtype=np.float64)
                    rotations = np.asarray(values["rotations"], dtype=np.float64)
                    series = {
                        "speech": np.asarray(values["speech_activity"], dtype=np.float64),
                        "energy": np.asarray(values["energy"], dtype=np.float64),
                        "accent": np.asarray(values["accent"], dtype=np.float64),
                        "pitch_semitones": np.asarray(values["pitch_semitones"], dtype=np.float64),
                        "emotion_intensity": np.asarray(values["emotion_intensity"], dtype=np.float64),
                    }
                    evidence = "audio_inference_and_prosody"
                else:
                    timestamps = np.asarray(values["timestamps_seconds"], dtype=np.float64)
                    expression = np.asarray(values["expression"], dtype=np.float64)
                    rotations = np.asarray(values["rotations"], dtype=np.float64)
                    series = {
                        "tracking_quality": np.asarray(values["effective_quality"], dtype=np.float64),
                        "source_lip_gap": np.asarray(values["source_lip_gap_interocular"], dtype=np.float64),
                        "source_lip_contact": np.asarray(values["source_lip_contact_confidence"], dtype=np.float64),
                    }
                    repair = _mapping(
                        _mapping(result.get("retargeting")).get(
                            "audio_visual_repair"
                        )
                    )
                    repair_changes_motion = bool(
                        _mapping(repair.get("claims")).get(
                            "changesFinalGNMMotion", False
                        )
                    )
                    if repair_changes_motion:
                        repair_artifact = artifacts.get(
                            "audio_visual_repair_arrays"
                        )
                        if not isinstance(repair_artifact, dict) or not isinstance(
                            repair_artifact.get("name"), str
                        ):
                            raise AutoAnimError(
                                "INTERNAL_ERROR",
                                "Mixed video performance has no allowlisted audio-repair evidence",
                            )
                        repair_path = self.store.artifact(
                            job_id, repair_artifact["name"]
                        )
                        with np.load(repair_path, allow_pickle=False) as repair_values:
                            repair_pts = np.asarray(
                                repair_values["source_pts"], dtype=np.int64
                            )
                            control_pts = np.asarray(
                                values["source_pts"], dtype=np.int64
                            )
                            audio_lower_weight = np.asarray(
                                repair_values["lower_face_audio_weight"],
                                dtype=np.float64,
                            )
                            audio_tongue_weight = np.asarray(
                                repair_values["tongue_audio_weight"],
                                dtype=np.float64,
                            )
                            contact_disagreement = np.asarray(
                                repair_values["audio_visual_contact_conflict"],
                                dtype=np.float64,
                            )
                        if (
                            not np.array_equal(repair_pts, control_pts)
                            or audio_lower_weight.shape != timestamps.shape
                            or audio_tongue_weight.shape != timestamps.shape
                            or contact_disagreement.shape != timestamps.shape
                        ):
                            raise AutoAnimError(
                                "INTERNAL_ERROR",
                                "Audio-repair acting evidence does not bind to final video source PTS",
                            )
                        series.update(
                            {
                                "audio_lower_face_weight": audio_lower_weight,
                                "audio_tongue_weight": audio_tongue_weight,
                                "audio_visual_contact_disagreement": contact_disagreement,
                            }
                        )
                        evidence = (
                            "visual_video_tracking_primary_exact_pts; "
                            "learned_audio_lower_face_repair_and_tongue"
                        )
                        repair_context = {
                            "schema_version": repair.get("schemaVersion"),
                            "policy": repair.get("policy"),
                            "status": repair.get("status"),
                            "bindings": repair.get("bindings"),
                            "source_authority": repair.get("sourceAuthority"),
                            "production_validated": _mapping(
                                repair.get("claims")
                            ).get("productionValidated", False),
                            "final_revision_chain_sha256": _mapping(
                                artifacts.get("performance_revision_chain")
                            ).get("sha256"),
                            "authored_mouth_aperture_revision": _mapping(
                                _mapping(result.get("retargeting")).get(
                                    "mouth_aperture_artist_edit"
                                )
                            ),
                        }
                    else:
                        evidence = (
                            "visual_video_tracking_exact_pts; audio_not_used_for_motion"
                        )
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            raise AutoAnimError("INTERNAL_ERROR", "Source control artifact is unreadable") from exc
        if (
            timestamps.ndim != 1
            or not len(timestamps)
            or expression.shape[0] != len(timestamps)
            or rotations.shape[0] != len(timestamps)
            or not np.isfinite(timestamps).all()
            or not np.isfinite(expression).all()
            or not np.isfinite(rotations).all()
            or any(array.shape != timestamps.shape or not np.isfinite(array).all() for array in series.values())
        ):
            raise AutoAnimError("INTERNAL_ERROR", "Source control arrays are inconsistent")
        expression_speed = np.zeros(len(timestamps), dtype=np.float64)
        head_speed = np.zeros(len(timestamps), dtype=np.float64)
        if len(timestamps) > 1:
            delta = np.maximum(np.diff(timestamps), 1e-6)
            expression_speed[1:] = np.linalg.norm(np.diff(expression, axis=0), axis=1) / delta
            head_speed[1:] = np.rad2deg(
                np.linalg.norm(np.diff(rotations[:, 1], axis=0), axis=1) / delta
            )
        series["expression_speed"] = expression_speed
        series["head_speed_degrees_per_second"] = head_speed
        series["head_pose_degrees"] = np.rad2deg(np.linalg.norm(rotations[:, 1], axis=1))
        series["gaze_degrees"] = np.rad2deg(np.max(np.linalg.norm(rotations[:, 2:4], axis=2), axis=1))
        window_seconds = 0.5
        indices = np.floor((timestamps - timestamps[0]) / window_seconds).astype(np.int64)
        windows: list[dict] = []
        for index in range(int(indices.max(initial=0)) + 1):
            selected = indices == index
            if not np.any(selected):
                continue
            window: dict[str, float | int] = {
                "start_tick": int(round(float(timestamps[selected][0]) * 48_000)),
                "end_tick": int(round(float(timestamps[selected][-1]) * 48_000)),
                "samples": int(np.count_nonzero(selected)),
            }
            for name, array in series.items():
                window[f"{name}_mean"] = round(float(np.mean(array[selected])), 6)
                window[f"{name}_peak"] = round(float(np.max(array[selected])), 6)
            windows.append(window)
        return {
            "motion_evidence": evidence,
            "timebase": {"ticks_per_second": 48_000, "window_seconds": window_seconds},
            "windows": windows,
            "source_analysis": {
                "emotion": result.get("analysis", {}).get("emotion"),
                "emotion_validated": result.get("analysis", {}).get("emotion_validated"),
                "neutral_baseline_method": result.get("retargeting", {}).get("neutral_baseline_method"),
                "audio_visual_repair": repair_context,
                "warnings": result.get("warnings", [])[:12],
            },
        }

    def _resolve_character(
        self,
        character_id: str | None,
        revision_id: str | None,
        *,
        usage_scope: str = "production",
    ) -> CharacterRevision | None:
        if character_id is None or not character_id.strip():
            if revision_id is not None and revision_id.strip():
                raise AutoAnimError(
                    "INPUT_INVALID", "A character revision cannot be selected without a character"
                )
            return None
        try:
            return self.characters.resolve(
                character_id.strip(), revision_id or None, usage_scope=usage_scope
            )
        except FileNotFoundError as exc:
            raise AutoAnimError("CHARACTER_NOT_FOUND", "Character or revision was not found") from exc

    @staticmethod
    def _character_ref(character: CharacterRevision | None) -> dict[str, Any] | None:
        if character is None:
            return None
        consent = character.manifest.get("consent", {})
        appearance = character.manifest.get("appearance", {})
        appearance_summary = (
            {
                key: appearance.get(key)
                for key in (
                    "material_package_id",
                    "capture_class",
                    "resolution_label",
                    "material_rights_expires_at",
                    "pore_claim_gate_passed",
                    "relightable_claim_gate_passed",
                    "production_validated",
                )
                if key in appearance
            }
            if isinstance(appearance, dict)
            else {}
        )
        return {
            "character_id": character.character_id,
            "revision_id": character.revision_id,
            "name": character.name,
            "revision_manifest_sha256": character.manifest_sha256,
            "identity_sha256": character.identity_sha256,
            "base_color_sha256": character.texture_sha256,
            "texture_uvs_sha256": character.texture_uvs_sha256,
            "texture_uvs_array_sha256": character.texture_uvs_array_sha256,
            "material_descriptor_sha256": character.material_manifest_sha256,
            "material_map_sha256s": dict(character.material_sha256s),
            "runtime_material_sha256s": dict(character.runtime_material_sha256s),
            # Job manifests need reproducibility hashes and coarse claims, not
            # the full UV attestation/evidence envelope from the character.
            "appearance": appearance_summary,
            "consent_scope": consent.get("scope") if isinstance(consent, dict) else None,
            "consent_evidence_sha256": (
                consent.get("evidence_sha256") if isinstance(consent, dict) else None
            ),
            "consent_expires_at": (
                consent.get("expires_at") if isinstance(consent, dict) else None
            ),
        }

    def multiview(
        self,
        input_paths: list[str | Path] | tuple[str | Path, ...],
        *,
        roles: list[str] | tuple[str, ...] | None = None,
        texture_size: int = 256,
        focal_scale: float = 1.25,
        mirror_fill: bool = False,
        input_names: list[str] | tuple[str, ...] | None = None,
        camera_bundle_path: str | Path | None = None,
    ) -> dict:
        bundle_source = Path(camera_bundle_path) if camera_bundle_path is not None else None
        if bundle_source is not None and not bundle_source.is_file():
            raise AutoAnimError("INPUT_INVALID", "Camera calibration sidecar is not a file")
        if bundle_source is not None and bundle_source.stat().st_size > 1_000_000:
            raise AutoAnimError(
                "LIMIT_EXCEEDED", "Camera calibration sidecar exceeds 1 MB"
            )
        configuration = {
            "roles": list(roles or ()),
            "texture_size": texture_size,
            "focal_scale": focal_scale,
            "mirror_fill": mirror_fill,
        }
        _, job_dir, retained, manifest = self.store.start_many(
            "multiview_reconstruction",
            input_paths,
            configuration,
            original_names=input_names,
            attachments=(
                {"camera_calibration": bundle_source} if bundle_source is not None else None
            ),
        )
        retained_bundle = None
        if bundle_source is not None:
            attachment = next(
                value
                for value in manifest.get("attachments", ())
                if value["logical_name"] == "camera_calibration"
            )
            retained_bundle = job_dir / attachment["retained_name"]
            configuration["calibration_sha256"] = attachment["sha256"]
        versions = runtime_versions()
        try:
            result = run_multiview_pipeline(
                retained,
                job_dir,
                model_path=self.model_path,
                roles=roles,
                texture_size=texture_size,
                focal_scale=focal_scale,
                mirror_fill=mirror_fill,
                camera_bundle_path=retained_bundle,
                input_names=input_names,
            )
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc
