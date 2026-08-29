"""Strict import of MAMMA multi-view SMPL-X motion into AutoAnim-55."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import zipfile

import numpy as np

from .acting import TICKS_PER_SECOND
from .body import MAX_SAMPLE_RATE_HZ, BodyTrack
from .speech_motion import (
    MAX_NATIVE_FRAMES,
    SMPLX55_GNM_OWNED,
    SMPLX55_NAMES,
    SpeechMotionValidationError,
    axis_angle_to_quaternion,
    retarget_smplx55_samples_to_autoanim55,
)


MAMMA_PROVIDER_ID = "mamma_multiview_smplx"
MAMMA_COMMIT = "588492f18876e2ed6888b2d26929047cb6b575e7"
MAMMA_PARAMETER_KEYS = frozenset(
    {
        "smplx_pose",
        "smplx_betas",
        "smplx_translation",
        "triangulated_3d_pts",
        "smplx_contact",
        "smplx_floor_contact",
        "v_template_pred",
    }
)
MAX_MAMMA_PARAMETER_BYTES = 512 * 1024 * 1024
MAX_MAMMA_EXPANDED_BYTES = 1024 * 1024 * 1024


class MammaMotionValidationError(ValueError):
    """A MAMMA parameter artifact violated the pinned import contract."""


def rebase_mamma_body_track(track: BodyTrack) -> BodyTrack:
    """Move a capture-world MAMMA take into character-local space.

    MAMMA translations are expressed in its calibrated capture world.  A
    reusable character should instead begin at its own origin while retaining
    every world-space delta within the take.  The new provenance digest makes
    that coordinate change observable rather than silently overwriting the raw
    import identity.
    """

    origin = np.asarray(track.root_translation_m[0], dtype=np.float32)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise MammaMotionValidationError("MAMMA root origin is invalid")
    translations = np.asarray(track.root_translation_m, dtype=np.float32) - origin
    digest = sha256()
    digest.update(b"autoanim.mamma-root-rebase/1.0\0")
    digest.update(track.source_plan_sha256.encode("ascii"))
    digest.update(origin.astype("<f4", copy=False).tobytes())
    return replace(
        track,
        root_translation_m=translations.astype(np.float32),
        source_plan_sha256=digest.hexdigest(),
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_npz_envelope(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MammaMotionValidationError(f"MAMMA parameter file is missing: {path}")
    if path.stat().st_size > MAX_MAMMA_PARAMETER_BYTES:
        raise MammaMotionValidationError("MAMMA parameter file exceeds the size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected = {f"{key}.npy" for key in MAMMA_PARAMETER_KEYS}
            if len(names) != len(set(names)) or set(names) != expected:
                raise MammaMotionValidationError(
                    "MAMMA parameter archive members differ from the pinned schema"
                )
            if any(
                member.flag_bits & 0x1
                or member.compress_type != zipfile.ZIP_STORED
                or member.file_size < 0
                for member in members
            ):
                raise MammaMotionValidationError(
                    "MAMMA parameter archive must be unencrypted and uncompressed"
                )
            if sum(member.file_size for member in members) > MAX_MAMMA_EXPANDED_BYTES:
                raise MammaMotionValidationError(
                    "MAMMA parameter archive exceeds the expanded-size limit"
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise MammaMotionValidationError("MAMMA parameter archive is invalid") from exc


def load_mamma_body_track(
    parameter_path: str | Path,
    *,
    sample_rate_hz: int = 30,
    source_up_axis: str = "z",
) -> BodyTrack:
    """Load one official ``smplx_params_body_id-XX.npz`` as AutoAnim-55.

    MAMMA owns body and hand motion. Its jaw and eye pose slots are reset before
    retargeting because Google GNM remains the sole owner of facial expression,
    oral motion, and eye rotations in the assembled character.
    """

    path = Path(parameter_path).expanduser().resolve()
    _validate_npz_envelope(path)
    if (
        type(sample_rate_hz) is not int
        or sample_rate_hz <= 0
        or sample_rate_hz > MAX_SAMPLE_RATE_HZ
        or TICKS_PER_SECOND % sample_rate_hz
    ):
        raise MammaMotionValidationError(
            "MAMMA sample_rate_hz must divide 48 kHz and be at most 120"
        )
    if source_up_axis not in {"y", "z"}:
        raise MammaMotionValidationError("MAMMA source_up_axis must be 'y' or 'z'")
    try:
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != MAMMA_PARAMETER_KEYS:
                raise MammaMotionValidationError(
                    "MAMMA parameter arrays differ from the pinned schema"
                )
            pose = np.asarray(values["smplx_pose"])
            translation = np.asarray(values["smplx_translation"])
    except (OSError, ValueError, KeyError) as exc:
        raise MammaMotionValidationError(
            "MAMMA numeric pose or translation arrays are invalid"
        ) from exc
    if pose.ndim == 2 and pose.shape[1] == 165:
        pose = pose.reshape(pose.shape[0], 55, 3)
    frames = pose.shape[0] if pose.ndim == 3 else 0
    if (
        frames < 2
        or frames > MAX_NATIVE_FRAMES
        or pose.shape != (frames, 55, 3)
        or translation.shape != (frames, 3)
        or pose.dtype.kind not in "fc"
        or translation.dtype.kind not in "fc"
        or not np.isfinite(pose).all()
        or not np.isfinite(translation).all()
    ):
        raise MammaMotionValidationError(
            "MAMMA pose must be [frame,165] and translation [frame,3] finite floats"
        )
    pose = np.array(pose, dtype=np.float32, copy=True)
    for name in SMPLX55_GNM_OWNED:
        pose[:, SMPLX55_NAMES.index(name), :] = 0.0
    try:
        rotations = axis_angle_to_quaternion(pose)
    except SpeechMotionValidationError as exc:
        raise MammaMotionValidationError(str(exc)) from exc
    rest_world = np.zeros((55, 4), dtype=np.float32)
    rest_world[:, 3] = 1.0
    tick_step = TICKS_PER_SECOND // sample_rate_hz
    ticks = np.arange(frames, dtype=np.int64) * tick_step
    # MAMMA's shipped multiview visualization and example calibration are
    # Z-up. Rotate that capture world -90 degrees around X so Z-up becomes
    # AutoAnim's +Y-up; local anatomical SMPL-X rotations retain their joint
    # semantics. Custom Y-up captures can opt out explicitly.
    source_to_canonical = (
        np.asarray((-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32)
        if source_up_axis == "z"
        else np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    )
    return retarget_smplx55_samples_to_autoanim55(
        root_translation_m=translation.astype(np.float32, copy=False),
        local_rotations_xyzw=rotations,
        rest_world_rotations_xyzw=rest_world,
        ticks=ticks,
        duration_ticks=int(ticks[-1]),
        sample_rate_hz=sample_rate_hz,
        source_sha256=_sha256(path),
        source_to_canonical_rotation_xyzw=source_to_canonical,
        source_orientation_mode="world_pre_rotate" if source_up_axis == "z" else "conjugate",
    )


__all__ = [
    "MAMMA_COMMIT",
    "MAMMA_PARAMETER_KEYS",
    "MAMMA_PROVIDER_ID",
    "MammaMotionValidationError",
    "load_mamma_body_track",
    "rebase_mamma_body_track",
]
