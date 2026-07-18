"""Shared application service used identically by CLI and HTTP."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import cv2
import mediapipe
import numpy as np

from . import __version__
from .a2f import resolve_a2f_runner
from .artifacts import JobStore
from .audio import resolve_rhubarb
from .audio_pipeline import _resolve_a2f_assets, run_audio_pipeline
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .image import validate_model
from .image_pipeline import run_image_pipeline
from .multiview_pipeline import run_multiview_pipeline
from .video_pipeline import run_video_pipeline
from .viewer import default_viewer_vendor_root, viewer_vendor_health


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    configured = os.environ.get("AUTOANIM_FACE_LANDMARKER")
    return Path(configured) if configured else PROJECT_ROOT / ".cache/autoanim_gnm/face_landmarker.task"


def runtime_versions() -> dict[str, str]:
    versions = {
        "autoanim": __version__,
        "gnm": "3.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "mediapipe": mediapipe.__version__,
    }
    try:
        versions["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        versions["git"] = "unknown"
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
    ):
        self.store = JobStore(artifact_root)
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
        checks["viewer_bundle"] = viewer_vendor_health(self.viewer_vendor_root)
        required = (
            "gnm", "ffmpeg", "ffprobe", "mediapipe_model", "rhubarb",
            "a2f_runner", "a2f_assets", "a2f_provenance", "viewer_bundle",
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
        input_name: str | None = None,
    ) -> dict:
        configuration = {
            "fps": fps,
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "dialog": dialog,
            "backend": backend,
        }
        job_id, job_dir, retained, manifest = self.store.start(
            "audio_animation", input_path, configuration, original_name=input_name
        )
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
                a2f_runner=self.a2f_runner,
                a2f_asset_dir=self.a2f_asset_dir,
                a2f_offline=self.a2f_offline,
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
    ) -> dict:
        configuration = {
            "backend": "mediapipe",
            "retargeter": (
                "geometry_calibrated_dense_contact_v2"
                if self.a2f_asset_dir is not None
                else "semantic_prototype_contact_v2_fallback"
            ),
            "neutral_baseline_seconds": 0.2,
            "profile": "offline",
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
            )
            return self.store.finish(manifest, job_dir, result, versions)
        except AutoAnimError as exc:
            self.store.fail(manifest, job_dir, exc.as_dict(), versions)
            raise
        except Exception as exc:
            error = AutoAnimError("INTERNAL_ERROR", str(exc))
            self.store.fail(manifest, job_dir, error.as_dict(), versions)
            raise error from exc

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
