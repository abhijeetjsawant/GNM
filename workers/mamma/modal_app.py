"""Pinned Modal execution surface for the MAMMA multiview body pipeline.

The gated checkpoints and SMPL-X files are uploaded from the user's local
MAMMA checkout. Credentials never enter Modal. The worker runs the official
five-stage pipeline at a pinned source revision and downloads only the run
artifacts and logs.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # Local tests import constants without Modal.
    modal = None


APP_NAME = "autoanim-mamma"
MAMMA_COMMIT = "588492f18876e2ed6888b2d26929047cb6b575e7"
GPU_FALLBACK = ["L40S", "A100-40GB"]
DATA_VOLUME_NAME = "autoanim-mamma-data-v1"
RUNS_VOLUME_NAME = "autoanim-mamma-runs-v1"
MAMMA_ROOT = Path("/opt/mamma")
DATA_ROOT = Path("/data")
RUNS_ROOT = Path("/runs")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CAMERAS = ("A001", "B001", "C001", "D001")
# Official run_ma_masks uses basename(--ma_cap_dir) as this intermediate
# dataset directory unless a separate --dataset_name is supplied.
MASK_DATASET_LEVEL = "ma_cap"
# Five seconds at the example cameras' 30 Hz rate.  Keep the public worker
# bounded: longer takes are split into review windows instead of risking an
# unbounded full-sequence Modal job.
MAX_REVIEW_WINDOW_FRAMES = 150


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_mamma_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".cache" / "mamma"


def _required_local_assets(root: Path) -> dict[str, Path]:
    videos = root / "data" / "mamma_example" / "pushing_and_lifting_from_ground" / "videos"
    assets = {
        "weights/ma_2d/mamma_mask_full_cvpr.ckpt": root / "data" / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt",
        "weights/sam2/sam2.1_hiera_large.pt": root / "data" / "weights" / "sam2" / "sam2.1_hiera_large.pt",
        "weights/yolo/yolo12x.pt": root / "data" / "weights" / "yolo" / "yolo12x.pt",
        "body_models/downsampled_verts/verts_512.pkl": root / "data" / "body_models" / "downsampled_verts" / "verts_512.pkl",
        "body_models/downsampled_verts/smplx_512_body_parts.npz": root / "data" / "body_models" / "downsampled_verts" / "smplx_512_body_parts.npz",
        "body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz": root / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz",
        "body_models/smplx_locked_head/smplx/SMPLX_MALE.npz": root / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_MALE.npz",
        "body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz": root / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_FEMALE.npz",
    }
    for camera in CAMERAS:
        assets[f"inputs/videos/{camera}.mp4"] = (videos / f"{camera}.mp4").resolve()
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MAMMA assets:\n" + "\n".join(missing))
    return assets


def _convert_downloaded_body_tracks(root: Path, *, sample_rate_hz: int = 30) -> list[str]:
    """Convert every downloaded official MAMMA subject into AutoAnim-55 JSON."""
    try:
        from autoanim_gnm.mamma_motion import load_mamma_body_track
    except ModuleNotFoundError as error:
        # The Modal CLI's own Python intentionally has few project deps. Use
        # the checked-out application's venv for the local-only conversion.
        project_python = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
        converter = Path(__file__).with_name("convert_tracks.py")
        # Compare launcher paths, not resolved interpreter binaries: uv venvs
        # commonly symlink both launchers to the same base CPython executable.
        if not project_python.is_file() or Path(sys.executable).absolute() == project_python.absolute():
            raise error
        completed = subprocess.run(
            [
                str(project_python),
                str(converter),
                str(root),
                "--sample-rate-hz",
                str(sample_rate_hz),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "MAMMA conversion via the project environment failed:\n"
                + completed.stderr[-4000:]
            )
        return list(json.loads(completed.stdout))

    converted: list[str] = []
    for parameter_path in sorted(root.rglob("smplx_params_body_id-*.npz")):
        if not re.fullmatch(r"smplx_params_body_id-\d{2}\.npz", parameter_path.name):
            continue
        track = load_mamma_body_track(parameter_path, sample_rate_hz=sample_rate_hz)
        target = parameter_path.with_suffix(".autoanim-body-track.json")
        target.write_bytes(track.canonical_json_bytes() + b"\n")
        converted.append(target.relative_to(root).as_posix())
    if not converted:
        raise FileNotFoundError("Downloaded MAMMA run contains no SMPL-X parameter files")
    return converted


def _run(command: list[str], cwd: Path, log_root: Path, name: str) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": str(MAMMA_ROOT),
        },
    )
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    result = {
        "name": name,
        "returncode": completed.returncode,
        "runtime_seconds": time.monotonic() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


if modal is not None:
    worker_root = Path(__file__).resolve().parent
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(
            "build-essential",
            "cmake",
            "ffmpeg",
            "git",
            "libegl1",
            "libgl1",
            "libglib2.0-0",
            "libgomp1",
            "ninja-build",
        )
        .uv_pip_install(
            "torch==2.5.1+cu124",
            "torchvision==0.20.1+cu124",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install_from_requirements(str(worker_root / "requirements-modal.txt"))
        # Image builds do not have an attached GPU.  Tell PyTorch's extension
        # builder which architectures our runtime fallback can actually use.
        .env(
            {
                "CC": "/usr/bin/gcc",
                "CXX": "/usr/bin/g++",
                "TORCH_CUDA_ARCH_LIST": "8.0;8.9",
            }
        )
        .run_commands(
            "git clone https://github.com/cuevhv/mamma.git /opt/mamma",
            f"git -C /opt/mamma checkout --detach {MAMMA_COMMIT}",
            "python -m pip install git+https://github.com/facebookresearch/sam2.git",
            "python -m pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git",
            # Detectron2 pins a slightly older iopath, while SAM2 requires this
            # compatible newer release at runtime.
            "python -m pip install iopath==0.1.10",
            "python -m pip install --no-build-isolation git+https://github.com/cuevhv/pytorch_sdf.git@torch2.5-cu124",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "MPLCONFIGDIR": "/tmp/matplotlib",
                "PYOPENGL_PLATFORM": "egl",
                "PYTHONPATH": "/opt/mamma",
            }
        )
    )
    app = modal.App(APP_NAME)
    data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

    @app.function(image=image, gpu=GPU_FALLBACK, timeout=7200, cpu=8, memory=65536)
    def runtime_probe() -> dict[str, Any]:
        import torch

        revision = subprocess.run(
            ["git", "-C", str(MAMMA_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "schema_version": "autoanim.mamma-runtime/1.0",
            "mamma_commit": revision,
            # torch.__version__ is a TorchVersion (a str subclass).  Coerce it
            # to a builtin str so the lightweight local Modal client does not
            # need PyTorch merely to unpickle this diagnostic response.
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_pinned": revision == MAMMA_COMMIT,
        }

    @app.function(
        image=image,
        gpu=GPU_FALLBACK,
        timeout=7200,
        cpu=8,
        memory=65536,
        volumes={str(DATA_ROOT): data_volume, str(RUNS_ROOT): runs_volume},
    )
    def run_pipeline(run_id: str, start_frame: int = 60, end_frame: int = 90) -> dict[str, Any]:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Unsafe MAMMA run_id")
        if (
            start_frame < 0
            or end_frame <= start_frame
            or end_frame - start_frame > MAX_REVIEW_WINDOW_FRAMES
        ):
            raise ValueError("Invalid MAMMA frame range")
        data_volume.reload()
        runs_volume.reload()
        run_root = RUNS_ROOT / run_id
        if run_root.exists():
            raise FileExistsError(f"MAMMA run already exists: {run_id}")
        logs = run_root / "logs"
        outputs = run_root / "output"
        videos = DATA_ROOT / "inputs" / "videos"
        seq = "pushing_and_lifting_from_ground"
        cameras = list(CAMERAS)
        calibration = MAMMA_ROOT / "configs" / "examples" / "calib" / "iphones_outdoors.yaml"
        steps: list[dict[str, Any]] = []

        steps.append(_run([
            "python", "run_ma_cap.py", "--videos_dir", str(videos),
            "--calibration", str(calibration), "--seq_name", seq,
            "--cam_names", *cameras, "--out", str(outputs / "ma_cap"),
            "--start", str(start_frame), "--end", str(end_frame), "-v",
        ], MAMMA_ROOT / "capture", logs, "ma_cap"))

        steps.append(_run([
            "python", "run_ma_masks.py", "--ma_cap_dir", str(outputs / "ma_cap"),
            "--seq_name", seq, "--out", str(outputs / "ma_masks"),
            "--sam_version", "sam2", "--sam_checkpoint",
            str(DATA_ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt"),
            "--yolo-checkpoint", str(DATA_ROOT / "weights" / "yolo" / "yolo12x.pt"),
            "--cfg", "configs/sam2.yaml", "--expected_subjects", "2",
            "--cam_names", *cameras, "--skip_collage", "--skip_masked_outputs",
        ], MAMMA_ROOT / "segmentation", logs, "ma_masks"))

        steps.append(_run([
            "python", "run_ma_2d.py", "--img_folder", str(outputs / "ma_cap"),
            "--config_path", "configs/train/models_2d/config_mammanet_mask_512.yaml",
            "--weights", str(DATA_ROOT / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt"),
            "--out_folder", str(outputs / "ma_2d"), "--seq_name", seq,
            # run_ma_masks defaults its dataset level to basename(ma_cap_dir),
            # yielding ma_masks/ma_cap/<sequence>/<camera>/masks.  Point the
            # downstream official landmark stage at that dataset level.
            "--mask_path", str(outputs / "ma_masks" / MASK_DATASET_LEVEL), "--cam_names", *cameras,
            "--downsampled-verts", str(DATA_ROOT / "body_models" / "downsampled_verts" / "verts_512.pkl"),
            "--no-save_cam_output",
        ], MAMMA_ROOT / "landmarks", logs, "ma_2d"))

        steps.append(_run([
            "python", "run_ma_3d.py", "--seq_name", seq,
            "--ma_2d_dir", str(outputs / "ma_2d"), "--ma_cap_dir", str(outputs / "ma_cap"),
            "--config_file", "config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml",
            "--cam_names", *cameras, "--out_path", str(outputs / "ma_3d"),
            "--smplx-models", str(DATA_ROOT / "body_models" / "smplx_locked_head"),
            "--downsampled-verts", str(DATA_ROOT / "body_models" / "downsampled_verts" / "verts_512.pkl"),
            "--skip_scene_videos",
        ], MAMMA_ROOT / "optimization", logs, "ma_3d"))

        steps.append(_run([
            "python", "run_ma_vis.py", "--seq-name", seq,
            "--ma-cap-dir", str(outputs / "ma_cap"), "--ma-2d-dir", str(outputs / "ma_2d"),
            "--ma-3d-dir", str(outputs / "ma_3d"), "--out-path", str(outputs / "ma_vis"),
            "--cam-names-overlay", "A001", "B001", "--overlay-resolution", "720",
            "--overlay-max-frames", str(end_frame - start_frame), "--overlay-num-workers", "1",
            "--no-rerun-images", "-v",
        ], MAMMA_ROOT / "visualization", logs, "ma_vis"))

        manifest_files = []
        for path in sorted(run_root.rglob("*")):
            if path.is_file():
                manifest_files.append({
                    "path": path.relative_to(run_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        report = {
            "schema_version": "autoanim.mamma-run/1.0",
            "run_id": run_id,
            "mamma_commit": MAMMA_COMMIT,
            "frame_range": [start_frame, end_frame],
            "cameras": cameras,
            "steps": steps,
            "files": manifest_files,
            "completed": True,
        }
        (run_root / "run-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        runs_volume.commit()
        return report

    @app.function(
        image=image,
        gpu=GPU_FALLBACK,
        timeout=7200,
        cpu=8,
        memory=65536,
        volumes={str(DATA_ROOT): data_volume, str(RUNS_ROOT): runs_volume},
    )
    def run_custom_pipeline(
        input_id: str,
        run_id: str,
        camera_names: list[str],
        start_frame: int = 0,
        end_frame: int = 120,
        cam_init: str | None = None,
    ) -> dict[str, Any]:
        """Run a bounded, explicitly uncalibrated/custom multiview trial.

        ``cam_init`` names the camera MAMMA's mask stage uses to establish subject
        identity (its ``--cam_init``); identity then propagates to the other views.
        Experimental footage where a per-camera person detector fails on some views
        (a back view of a costume) needs this. None keeps MAMMA's default.

        Custom inputs live under ``/data/custom/<input_id>`` and never replace
        the pinned four-camera example fixture.  The caller must provide a
        calibration YAML and one same-named MP4 per camera.  This surface is
        intended for experimental footage whose calibration provenance is
        recorded alongside the videos; it does not make an approximate camera
        solution production-grade.
        """
        if not RUN_ID_RE.fullmatch(input_id) or not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Unsafe MAMMA input_id or run_id")
        if (
            start_frame < 0
            or end_frame <= start_frame
            or end_frame - start_frame > MAX_REVIEW_WINDOW_FRAMES
        ):
            raise ValueError("Invalid MAMMA frame range")
        if not 2 <= len(camera_names) <= 4:
            raise ValueError("Custom MAMMA trials require two to four cameras")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError("Custom MAMMA camera names must be unique")
        if any(not RUN_ID_RE.fullmatch(name) for name in camera_names):
            raise ValueError("Unsafe custom MAMMA camera name")

        data_volume.reload()
        runs_volume.reload()
        input_root = DATA_ROOT / "custom" / input_id
        videos = input_root / "videos"
        calibration = input_root / "calibration.yaml"
        expected = [calibration, *(videos / f"{name}.mp4" for name in camera_names)]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing custom MAMMA inputs:\n" + "\n".join(missing))

        run_root = RUNS_ROOT / run_id
        if run_root.exists():
            raise FileExistsError(f"MAMMA run already exists: {run_id}")
        logs = run_root / "logs"
        outputs = run_root / "output"
        seq = input_id
        cameras = list(camera_names)
        steps: list[dict[str, Any]] = []

        steps.append(_run([
            "python", "run_ma_cap.py", "--videos_dir", str(videos),
            "--calibration", str(calibration), "--seq_name", seq,
            "--cam_names", *cameras, "--out", str(outputs / "ma_cap"),
            "--start", str(start_frame), "--end", str(end_frame), "-v",
        ], MAMMA_ROOT / "capture", logs, "ma_cap"))

        steps.append(_run([
            "python", "run_ma_masks.py", "--ma_cap_dir", str(outputs / "ma_cap"),
            "--seq_name", seq, "--out", str(outputs / "ma_masks"),
            "--sam_version", "sam2", "--sam_checkpoint",
            str(DATA_ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt"),
            "--yolo-checkpoint", str(DATA_ROOT / "weights" / "yolo" / "yolo12x.pt"),
            "--cfg", "configs/sam2.yaml", "--expected_subjects", "1",
            "--cam_names", *cameras, "--skip_collage", "--skip_masked_outputs",
            *(["--cam_init", cam_init] if cam_init else []),
        ], MAMMA_ROOT / "segmentation", logs, "ma_masks"))

        steps.append(_run([
            "python", "run_ma_2d.py", "--img_folder", str(outputs / "ma_cap"),
            "--config_path", "configs/train/models_2d/config_mammanet_mask_512.yaml",
            "--weights", str(DATA_ROOT / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt"),
            "--out_folder", str(outputs / "ma_2d"), "--seq_name", seq,
            "--mask_path", str(outputs / "ma_masks" / MASK_DATASET_LEVEL),
            "--cam_names", *cameras,
            "--downsampled-verts", str(DATA_ROOT / "body_models" / "downsampled_verts" / "verts_512.pkl"),
            "--no-save_cam_output",
        ], MAMMA_ROOT / "landmarks", logs, "ma_2d"))

        steps.append(_run([
            "python", "run_ma_3d.py", "--seq_name", seq,
            "--ma_2d_dir", str(outputs / "ma_2d"), "--ma_cap_dir", str(outputs / "ma_cap"),
            "--config_file", "config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml",
            "--cam_names", *cameras, "--out_path", str(outputs / "ma_3d"),
            "--smplx-models", str(DATA_ROOT / "body_models" / "smplx_locked_head"),
            "--downsampled-verts", str(DATA_ROOT / "body_models" / "downsampled_verts" / "verts_512.pkl"),
            "--skip_scene_videos",
        ], MAMMA_ROOT / "optimization", logs, "ma_3d"))

        steps.append(_run([
            "python", "run_ma_vis.py", "--seq-name", seq,
            "--ma-cap-dir", str(outputs / "ma_cap"), "--ma-2d-dir", str(outputs / "ma_2d"),
            "--ma-3d-dir", str(outputs / "ma_3d"), "--out-path", str(outputs / "ma_vis"),
            "--cam-names-overlay", *cameras[:2], "--overlay-resolution", "720",
            "--overlay-max-frames", str(end_frame - start_frame), "--overlay-num-workers", "1",
            "--no-rerun-images", "-v",
        ], MAMMA_ROOT / "visualization", logs, "ma_vis"))

        manifest_files = []
        for path in sorted(run_root.rglob("*")):
            if path.is_file():
                manifest_files.append({
                    "path": path.relative_to(run_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        report = {
            "schema_version": "autoanim.mamma-custom-run/1.0",
            "run_id": run_id,
            "input_id": input_id,
            "mamma_commit": MAMMA_COMMIT,
            "frame_range": [start_frame, end_frame],
            "cameras": cameras,
            "calibration_quality": "caller-supplied-experimental",
            "cam_init": cam_init,
            "steps": steps,
            "files": manifest_files,
            "completed": True,
        }
        (run_root / "run-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        runs_volume.commit()
        return report

    @app.local_entrypoint(name="probe")
    def probe_entrypoint() -> None:
        print(json.dumps(runtime_probe.remote(), indent=2, sort_keys=True))

    @app.local_entrypoint(name="upload-assets")
    def upload_assets_entrypoint() -> None:
        local_root = _local_mamma_root()
        assets = _required_local_assets(local_root)
        with data_volume.batch_upload(force=True) as batch:
            for relative, source in assets.items():
                batch.put_file(source, f"/{relative}")
        print(json.dumps({
            "uploaded": len(assets),
            "bytes": sum(path.stat().st_size for path in assets.values()),
            "files": sorted(assets),
        }, indent=2, sort_keys=True))

    @app.local_entrypoint(name="upload-custom")
    def upload_custom_entrypoint(input_id: str, input_directory: str) -> None:
        if not RUN_ID_RE.fullmatch(input_id):
            raise ValueError("Unsafe MAMMA input_id")
        source_root = Path(input_directory).resolve(strict=True)
        calibration = source_root / "calibration.yaml"
        videos = sorted((source_root / "videos").glob("*.mp4"))
        if not calibration.is_file() or len(videos) < 2 or len(videos) > 4:
            raise FileNotFoundError(
                "Custom input requires calibration.yaml and two to four videos/*.mp4 files"
            )
        files = [calibration, *videos]
        with data_volume.batch_upload(force=True) as batch:
            for source in files:
                relative = source.relative_to(source_root).as_posix()
                batch.put_file(source, f"/custom/{input_id}/{relative}")
        print(json.dumps({
            "input_id": input_id,
            "uploaded": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "files": [path.relative_to(source_root).as_posix() for path in files],
        }, indent=2, sort_keys=True))

    @app.local_entrypoint(name="run")
    def run_entrypoint(run_id: str, output_directory: str, start_frame: int = 60, end_frame: int = 90) -> None:
        report = run_pipeline.remote(run_id, start_frame, end_frame)
        destination_root = Path(output_directory).resolve() / run_id
        if destination_root.exists():
            raise FileExistsError(f"Local response already exists: {destination_root}")
        destination_root.mkdir(parents=True)
        remote_root = f"/{run_id}"
        downloaded = []
        for entry in runs_volume.iterdir(remote_root, recursive=True):
            if entry.type.name != "FILE":
                continue
            relative = Path(entry.path).relative_to(remote_root.lstrip("/"))
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                for block in runs_volume.read_file(entry.path):
                    target.write(block)
            downloaded.append(relative.as_posix())
        converted = _convert_downloaded_body_tracks(destination_root)
        print(json.dumps({
            "report": report,
            "local_response": str(destination_root),
            "downloaded": downloaded,
            "autoanim_body_tracks": converted,
        }, indent=2, sort_keys=True))

    @app.local_entrypoint(name="run-custom")
    def run_custom_entrypoint(
        input_id: str,
        run_id: str,
        output_directory: str,
        camera_names: str,
        start_frame: int = 0,
        end_frame: int = 120,
        cam_init: str = "",
    ) -> None:
        cameras = [name.strip() for name in camera_names.split(",") if name.strip()]
        if cam_init and cam_init not in cameras:
            raise ValueError("cam_init must be one of the camera names")
        report = run_custom_pipeline.remote(
            input_id, run_id, cameras, start_frame, end_frame, cam_init or None
        )
        destination_root = Path(output_directory).resolve() / run_id
        if destination_root.exists():
            raise FileExistsError(f"Local response already exists: {destination_root}")
        destination_root.mkdir(parents=True)
        remote_root = f"/{run_id}"
        downloaded = []
        for entry in runs_volume.iterdir(remote_root, recursive=True):
            if entry.type.name != "FILE":
                continue
            relative = Path(entry.path).relative_to(remote_root.lstrip("/"))
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                for block in runs_volume.read_file(entry.path):
                    target.write(block)
            downloaded.append(relative.as_posix())
        converted = _convert_downloaded_body_tracks(destination_root)
        print(json.dumps({
            "report": report,
            "local_response": str(destination_root),
            "downloaded": downloaded,
            "autoanim_body_tracks": converted,
        }, indent=2, sort_keys=True))


__all__ = [
    "APP_NAME",
    "CAMERAS",
    "DATA_VOLUME_NAME",
    "GPU_FALLBACK",
    "MAMMA_COMMIT",
    "RUNS_VOLUME_NAME",
]
