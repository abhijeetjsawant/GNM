"""Strict opt-in ctypes binding for the AutoAnim Rust P0 physics core.

This module does not participate in the audio or video pipelines.  Callers
must explicitly construct :class:`PhysicsSimulator`; missing libraries and
invalid native calls raise typed exceptions and never select a Python fallback.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Literal

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_ENV = "AUTOANIM_PHYSICS_LIBRARY"
_KERNELS = {"auto": 0, "scalar": 1, "stable": 2, "neon": 3}


class PhysicsError(RuntimeError):
    """Base error for the native physics boundary."""


class PhysicsLibraryError(PhysicsError):
    """The requested release library is absent or ABI-incompatible."""


class PhysicsInputError(PhysicsError, ValueError):
    """A caller-owned array does not satisfy the native ABI contract."""


class _AaPhysicsConfig(ctypes.Structure):
    _fields_ = (
        ("frames_per_second", ctypes.c_float),
        ("substeps", ctypes.c_uint32),
        ("iterations", ctypes.c_uint32),
        ("velocity_retention", ctypes.c_float),
        ("stretch_compliance", ctypes.c_float),
        ("tether_compliance", ctypes.c_float),
        ("max_displacement_m", ctypes.c_float),
        ("jacobi_relaxation", ctypes.c_float),
    )


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    """C-ABI-compatible solver configuration.

    The constructor uses the release library's native defaults when ``config``
    is omitted.  These field defaults mirror P0 for convenient explicit edits.
    """

    frames_per_second: float = 60.0
    substeps: int = 2
    iterations: int = 5
    velocity_retention: float = 0.88
    stretch_compliance: float = 2.0e-7
    tether_compliance: float = 8.0e-6
    max_displacement_m: float = 0.00075
    jacobi_relaxation: float = 0.8

    @classmethod
    def _from_native(cls, value: _AaPhysicsConfig) -> PhysicsConfig:
        return cls(
            frames_per_second=float(value.frames_per_second),
            substeps=int(value.substeps),
            iterations=int(value.iterations),
            velocity_retention=float(value.velocity_retention),
            stretch_compliance=float(value.stretch_compliance),
            tether_compliance=float(value.tether_compliance),
            max_displacement_m=float(value.max_displacement_m),
            jacobi_relaxation=float(value.jacobi_relaxation),
        )

    def _as_native(self) -> _AaPhysicsConfig:
        try:
            return _AaPhysicsConfig(
                self.frames_per_second,
                self.substeps,
                self.iterations,
                self.velocity_retention,
                self.stretch_compliance,
                self.tether_compliance,
                self.max_displacement_m,
                self.jacobi_relaxation,
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise PhysicsInputError("PhysicsConfig contains an ABI-incompatible value") from exc


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libautoanim_physics_capi.dylib"
    if sys.platform == "win32":
        return "autoanim_physics_capi.dll"
    return "libautoanim_physics_capi.so"


def find_local_release_library() -> Path | None:
    """Return the repository-local release library, without loading it."""
    filename = _library_filename()
    release = _PROJECT_ROOT / "native" / "autoanim-physics" / "target" / "release"
    for candidate in (release / filename, release / "deps" / filename):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_library_path(library_path: str | Path | None) -> Path:
    configured: str | Path | None = library_path
    source = "supplied"
    if configured is None:
        configured = os.environ.get(_LIBRARY_ENV)
        source = _LIBRARY_ENV
    if configured is not None:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise PhysicsLibraryError(f"{source} physics release library is absent: {path}")
        return path.resolve()
    local = find_local_release_library()
    if local is None:
        raise PhysicsLibraryError(
            "AutoAnim physics release library is absent; build or explicitly supply the "
            f"{_library_filename()} path"
        )
    return local


class _Bindings:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.library = ctypes.CDLL(str(path))
            self.library.aa_physics_default_config.argtypes = ()
            self.library.aa_physics_default_config.restype = _AaPhysicsConfig
            self.library.aa_physics_last_error_message.argtypes = ()
            self.library.aa_physics_last_error_message.restype = ctypes.c_char_p
            self.library.aa_physics_topology_create.argtypes = (
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
            )
            self.library.aa_physics_topology_create.restype = ctypes.c_void_p
            self.library.aa_physics_topology_destroy.argtypes = (ctypes.c_void_p,)
            self.library.aa_physics_topology_destroy.restype = None
            self.library.aa_physics_simulator_create.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                _AaPhysicsConfig,
                ctypes.c_size_t,
                ctypes.c_uint32,
            )
            self.library.aa_physics_simulator_create.restype = ctypes.c_void_p
            self.library.aa_physics_simulator_destroy.argtypes = (ctypes.c_void_p,)
            self.library.aa_physics_simulator_destroy.restype = None
            self.library.aa_physics_simulate_chunk.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            )
            self.library.aa_physics_simulate_chunk.restype = ctypes.c_int32
            self.library.aa_physics_report_json.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            )
            self.library.aa_physics_report_json.restype = ctypes.c_size_t
        except (AttributeError, OSError) as exc:
            raise PhysicsLibraryError(
                f"Physics release library could not satisfy the P0 ABI: {path}: {exc}"
            ) from exc

    def last_error(self) -> str:
        raw = self.library.aa_physics_last_error_message()
        if raw is None:
            return "native physics call failed without an error message"
        return raw.decode("utf-8", errors="replace")


def _require_array(
    value: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PhysicsInputError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise PhysicsInputError(
            f"{name} must have dtype {dtype.name}, received {value.dtype}"
        )
    if value.ndim != ndim:
        raise PhysicsInputError(f"{name} must have {ndim} dimensions, received {value.ndim}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise PhysicsInputError(f"{name} must be C-contiguous and naturally aligned")
    return value


def _float_pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class PhysicsSimulator:
    """Own one native topology and stateful P0 target-relative simulator."""

    def __init__(
        self,
        triangles: np.ndarray,
        motion_weights: np.ndarray,
        *,
        library_path: str | Path | None = None,
        config: PhysicsConfig | None = None,
        threads: int = 4,
        kernel: Literal["auto", "scalar", "stable", "neon"] = "auto",
    ) -> None:
        triangles = _require_array(
            triangles, name="triangles", dtype=np.dtype(np.uint32), ndim=2
        )
        weights = _require_array(
            motion_weights,
            name="motion_weights",
            dtype=np.dtype(np.float32),
            ndim=1,
        )
        if triangles.shape[1:] != (3,) or triangles.shape[0] == 0:
            raise PhysicsInputError("triangles must have nonempty shape [triangle, 3]")
        if weights.size == 0:
            raise PhysicsInputError("motion_weights must contain at least one vertex")
        if not np.isfinite(weights).all() or np.any((weights < 0.0) | (weights > 1.0)):
            raise PhysicsInputError("motion_weights must be finite and in 0..=1")
        if (
            not isinstance(threads, int)
            or isinstance(threads, bool)
            or not 1 <= threads <= 256
        ):
            raise PhysicsInputError("threads must be an integer in 1..=256")
        if kernel not in _KERNELS:
            raise PhysicsInputError(
                f"kernel must be one of {tuple(_KERNELS)}, received {kernel!r}"
            )
        if config is not None and not isinstance(config, PhysicsConfig):
            raise PhysicsInputError("config must be a PhysicsConfig instance or None")

        self._lock = threading.RLock()
        self._bindings = _Bindings(_resolve_library_path(library_path))
        self._topology: int | None = None
        self._simulator: int | None = None
        self._vertex_count = int(weights.size)
        self._triangle_count = int(triangles.shape[0])
        native_config = (
            self._bindings.library.aa_physics_default_config()
            if config is None
            else config._as_native()
        )
        self.config = PhysicsConfig._from_native(native_config)

        topology = self._bindings.library.aa_physics_topology_create(
            self._vertex_count,
            triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._triangle_count,
        )
        if not topology:
            raise PhysicsInputError(self._bindings.last_error())
        self._topology = topology
        try:
            simulator = self._bindings.library.aa_physics_simulator_create(
                topology,
                _float_pointer(weights),
                weights.size,
                native_config,
                threads,
                _KERNELS[kernel],
            )
            if not simulator:
                raise PhysicsInputError(self._bindings.last_error())
            self._simulator = simulator
        except BaseException:
            self._bindings.library.aa_physics_topology_destroy(topology)
            self._topology = None
            raise

    @property
    def library_path(self) -> Path:
        return self._bindings.path

    @property
    def closed(self) -> bool:
        return self._simulator is None

    def _require_open(self) -> int:
        if self._simulator is None:
            raise PhysicsError("PhysicsSimulator is closed")
        return self._simulator

    def simulate(
        self,
        targets: np.ndarray,
        accelerations: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate one ``[V,3]`` target or a contiguous ``[F,V,3]`` chunk."""
        if not isinstance(targets, np.ndarray):
            raise PhysicsInputError("targets must be a NumPy array")
        target_ndim = targets.ndim
        if target_ndim not in (2, 3):
            raise PhysicsInputError("targets must have shape [V,3] or [F,V,3]")
        targets = _require_array(
            targets, name="targets", dtype=np.dtype(np.float32), ndim=target_ndim
        )
        if target_ndim == 2:
            expected = (self._vertex_count, 3)
            frame_count = 1
        else:
            expected = (targets.shape[0], self._vertex_count, 3)
            frame_count = int(targets.shape[0])
        if targets.shape != expected or frame_count == 0:
            raise PhysicsInputError(f"targets must have nonempty shape {expected}")
        if not np.isfinite(targets).all():
            raise PhysicsInputError("targets must contain only finite values")

        acceleration_pointer: ctypes.POINTER(ctypes.c_float) | None = None
        acceleration_count = 0
        if accelerations is not None:
            acceleration_ndim = (
                accelerations.ndim if isinstance(accelerations, np.ndarray) else 0
            )
            if acceleration_ndim not in (1, 2):
                raise PhysicsInputError("accelerations must have shape [3] or [F,3]")
            accelerations = _require_array(
                accelerations,
                name="accelerations",
                dtype=np.dtype(np.float32),
                ndim=acceleration_ndim,
            )
            expected_acceleration = (3,) if target_ndim == 2 else (frame_count, 3)
            if accelerations.shape != expected_acceleration:
                raise PhysicsInputError(
                    f"accelerations must have shape {expected_acceleration}"
                )
            if not np.isfinite(accelerations).all():
                raise PhysicsInputError("accelerations must contain only finite values")
            acceleration_pointer = _float_pointer(accelerations)
            acceleration_count = int(accelerations.size)

        output = np.empty_like(targets)
        with self._lock:
            simulator = self._require_open()
            status = self._bindings.library.aa_physics_simulate_chunk(
                simulator,
                _float_pointer(targets),
                targets.size,
                acceleration_pointer,
                acceleration_count,
                _float_pointer(output),
                output.size,
            )
            if status != 0:
                raise PhysicsError(self._bindings.last_error())
        return output

    def report(self) -> dict[str, Any]:
        """Return the native audit report plus the ABI-owned triangle count."""
        with self._lock:
            simulator = self._require_open()
            required = self._bindings.library.aa_physics_report_json(simulator, None, 0)
            if required == 0:
                raise PhysicsError(self._bindings.last_error())
            buffer = ctypes.create_string_buffer(required)
            written = self._bindings.library.aa_physics_report_json(
                simulator, buffer, required
            )
            if written != required:
                raise PhysicsError(self._bindings.last_error())
            try:
                report = json.loads(buffer.value.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PhysicsLibraryError("Native physics report is not valid JSON") from exc
        if not isinstance(report, dict):
            raise PhysicsLibraryError("Native physics report must be a JSON object")
        report["triangle_count"] = self._triangle_count
        return report

    def close(self) -> None:
        """Destroy native state in dependency order; safe to call repeatedly."""
        with self._lock:
            if self._simulator is not None:
                self._bindings.library.aa_physics_simulator_destroy(self._simulator)
                self._simulator = None
            if self._topology is not None:
                self._bindings.library.aa_physics_topology_destroy(self._topology)
                self._topology = None

    def __enter__(self) -> PhysicsSimulator:
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can tear down ctypes before finalizers run.
            pass
