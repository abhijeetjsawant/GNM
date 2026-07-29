"""Run GEM-X's ONNX denoiser and SOMA decoder on CPU.

The pinned GEM-X macOS demo currently selects Core ML for dynamic ONNX graphs
that fail at first inference, then unconditionally moves denoiser outputs and
the lightweight EnDecoder to CUDA.  This provider-internal worker keeps the
official data preparation, model configuration, checkpoint loader, decoder,
and camera math, while selecting ONNX Runtime's CPU provider and keeping every
PyTorch tensor on CPU.

The resulting ``hpe_results.pt`` is trusted only inside the quarantined
provider environment.  AutoAnim must export it through the reviewed JSON/NPZ
response boundary before importing it into the application process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import onnxruntime as ort
import torch

from scripts.demo import demo_soma_onnx as demo


class CpuOnnxRunner:
    """Small runner matching GEM-X's runner protocol without Core ML fallback."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"GEM-X denoiser does not exist: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path.resolve()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = tuple(value.name for value in self.session.get_inputs())
        self.output_names = tuple(value.name for value in self.session.get_outputs())

    def __call__(self, **values: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            value = values[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            feed[name] = np.ascontiguousarray(value)
        outputs = self.session.run(list(self.output_names), feed)
        return {
            name: torch.from_numpy(value)
            for name, value in zip(self.output_names, outputs, strict=True)
        }


def _configuration(video_path: Path, output_root: Path, checkpoint: Path) -> object:
    arguments = SimpleNamespace(
        video=str(video_path),
        output_root=str(output_root),
        static_cam=True,
        verbose=False,
        ckpt=str(checkpoint),
        exp="gem_soma_regression",
        force_pytorch=False,
        retarget=False,
        no_imgfeat=True,
        ddim=False,
    )
    return demo._build_cfg(arguments)


@torch.inference_mode()
def run_cpu_motion(
    *,
    video_path: Path,
    output_root: Path,
    denoiser_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    for label, path in (
        ("Input video", video_path),
        ("GEM-X checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    demo._ensure_pipeline_deps()
    configuration = _configuration(video_path, output_root, checkpoint_path)
    required_preprocess = (
        Path(configuration.paths.bbx),
        Path(configuration.paths.vitpose),
    )
    for path in required_preprocess:
        if not path.is_file():
            raise FileNotFoundError(f"Required GEM-X preprocessing is missing: {path}")

    data = demo.load_data_dict(configuration, no_imgfeat=True)
    runner = CpuOnnxRunner(denoiser_path)
    batch = {
        "obs": data["kp2d"].unsqueeze(0),
        "bbx_xys": data["bbx_xys"].unsqueeze(0),
        "K_fullimg": data["K_fullimg"].unsqueeze(0),
        "f_imgseq": data["f_imgseq"].unsqueeze(0),
        "f_cam_angvel": data["cam_angvel"].unsqueeze(0),
    }
    outputs = runner(**batch)
    pred_x = outputs["pred_x"].float()
    pred_cam = outputs["pred_cam"].float()

    # The ONNX graph already contains all learned denoiser weights.  GEM-X
    # excludes EnDecoder from its checkpoint and constructs it entirely from
    # the pinned config/statistics module.  Instantiating the top-level GEM
    # object is both unnecessary and broken on macOS because its constructor
    # hardcodes a CUDA SOMA layer.
    endecoder = hydra.utils.instantiate(configuration.endecoder).eval().cpu()
    if endecoder.obs_indices_dict is None:
        endecoder.build_obs_indices_dict()
    decoded = endecoder.decode(pred_x)
    if "scale_params" in decoded:
        scale_parameters = decoded["scale_params"].clone()
        scale_parameters[..., 0] = scale_parameters[..., 0].clamp(0.7, 1.0)
        decoded["scale_params"] = scale_parameters

    translation_in_camera = demo.compute_transl_full_cam(
        pred_cam,
        data["bbx_xys"].unsqueeze(0),
        data["K_fullimg"].unsqueeze(0),
    )[0]
    body_in_camera = {
        "body_pose": decoded["body_pose"][0],
        "global_orient": decoded["global_orient"][0],
        "transl": translation_in_camera,
    }
    for name in ("identity_coeffs", "scale_params"):
        if name in decoded:
            body_in_camera[name] = decoded[name][0]

    if "global_orient_gv" in decoded and "local_transl_vel" in decoded:
        from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2

        solved_global = get_body_params_w_Rt_v2(
            global_orient_gv=decoded["global_orient_gv"],
            local_transl_vel=decoded["local_transl_vel"],
            global_orient_c=decoded["global_orient"],
            cam_angvel=data["cam_angvel"].unsqueeze(0),
        )
        body_global = {
            "body_pose": decoded["body_pose"][0],
            "global_orient": solved_global["global_orient"][0],
            "transl": solved_global["transl"][0],
        }
        for name in ("identity_coeffs", "scale_params"):
            if name in decoded:
                body_global[name] = decoded[name][0]
    else:
        raise RuntimeError(
            "GEM-X decode omitted global_orient_gv/local_transl_vel; refusing "
            "to relabel camera-space motion as global motion"
        )

    prediction = {
        "body_params_incam": demo.detach_to_cpu(body_in_camera),
        "body_params_global": demo.detach_to_cpu(body_global),
        "K_fullimg": data["K_fullimg"].cpu(),
    }
    output_path = Path(configuration.paths.hpe_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    torch.save(prediction, temporary_path)
    temporary_path.replace(output_path)

    return {
        "execution_provider": "CPUExecutionProvider",
        "runtime_class": "apple_silicon_preview",
        "camera_model": "static_camera_assumed",
        "frame_count": int(data["length"]),
        "denoiser_inputs": list(runner.input_names),
        "denoiser_outputs": list(runner.output_names),
        "pred_x_shape": list(pred_x.shape),
        "pred_cam_shape": list(pred_cam.shape),
        "body_pose_shape": list(body_global["body_pose"].shape),
        "global_orient_shape": list(body_global["global_orient"].shape),
        "translation_shape": list(body_global["transl"].shape),
        "output_path": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--denoiser", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_cpu_motion(
        video_path=arguments.video.resolve(),
        output_root=arguments.output_root.resolve(),
        denoiser_path=arguments.denoiser.resolve(),
        checkpoint_path=arguments.checkpoint.resolve(),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
