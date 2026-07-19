"""Pinned NVIDIA Audio2Face-v3 Claire import profile.

The official v3 network emits Claire geometry.  A separately provisioned
NVIDIA worker is responsible for inference and for running NVIDIA's skin and
tongue blendshape solvers.  This module verifies the exact public model assets
needed to interpret those post-solver controls and then validates a generic
``SequenceProviderTrack`` against that profile.

It intentionally performs no network access, no inference, and no production
promotion.  The model/SDK revisions are integrity anchors, not a substitute
for worker authentication or independent perceptual qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256
from .sequence_provider import (
    A2F_V3_IDENTITY,
    A2F_V3_IDENTITY_INDEX,
    A2F_V3_MODEL_REVISION,
    A2F_V3_NETWORK_VERSION,
    A2F_V3_PUBLIC_MODEL_VERSION,
    A2F_V3_SDK_REVISION,
    OfficialV3TrackValidation,
    SequenceProviderError,
    SequenceProviderTrack,
    validate_official_v3_sequence_track,
)


OFFICIAL_V3_TONGUE_CONTROL_NAMES = (
    "tongueTipUp",
    "tongueTipDown",
    "tongueTipLeft",
    "tongueTipRight",
    "tongueRollUp",
    "tongueRollDown",
    "tongueRollLeft",
    "tongueRollRight",
    "tongueUp",
    "tongueDown",
    "tongueLeft",
    "tongueRight",
    "tongueIn",
    "tongueStretch",
    "tongueWide",
    "tongueNarrow",
)

# Hashes are from the immutable public v3 revision above.  Pinning the small
# interpretation bundle avoids accepting a different rig/config merely because
# it uses the same filenames.  ``network.onnx`` is checked by the existing
# request model binding and its official hash is exposed for deployment tools.
OFFICIAL_V3_ASSET_SHA256 = {
    "network.onnx": "db47c2701ca849de443c9e9f25657210f829a74fc458ee6fed603a8a501253a8",
    "network_info.json": "5524cdbe96a6bc89c78f06f32ae959e2302c50c663f407cb2b392c0ecac5975d",
    "model_data_Claire.npz": "4f05331263fa609321335e55c20922f4d6709d33160d368c3b537f019429ea4f",
    "model_config_Claire.json": "0819530451ad28ef42c1a478398850dc91e32475a49f9899ed37216309107fb4",
    "bs_skin_Claire.npz": "bcb1fde2c7384fe9ec3cf9932b0fdeeda01fe4a1e42bba3817bba14e7f1716d3",
    "bs_skin_config_Claire.json": "e2b508c5d17f1fb01c3a5b0292072d09e66e8c55bc23fcbe0c9aee8f8eae1713",
    "bs_tongue_Claire.npz": "812f10c34edb6ab6f36aedfe1d59a79d8190a5a8ee0a6071382f6bae9e3413b6",
    "bs_tongue_config_Claire.json": "ace4b0b6b9be280f96a66568bd13ac4ea1fddf9c690464ab450fe339d9752e98",
}

_INTERPRETATION_FILES = tuple(
    name for name in OFFICIAL_V3_ASSET_SHA256 if name != "network.onnx"
)


@dataclass(frozen=True, slots=True)
class OfficialV3ClaireProfile:
    root: Path
    public_model_version: str
    network_version: str
    identity: str
    identity_index: int
    model_revision: str
    required_sdk_revision: str
    skin_pose_names: tuple[str, ...]
    tongue_pose_names: tuple[str, ...]
    skin_minimums: tuple[float, ...]
    skin_maximums: tuple[float, ...]
    tongue_minimums: tuple[float, ...]
    tongue_maximums: tuple[float, ...]
    interpretation_asset_sha256: dict[str, str]
    network_sha256: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Artifact manifests expose the immutable file hashes, not a host's
        # absolute cache path.
        value.pop("root", None)
        return value

    def validate_track(self, track: SequenceProviderTrack) -> OfficialV3TrackValidation:
        expected_identity = OFFICIAL_V3_ASSET_SHA256["model_data_Claire.npz"]
        if track.bindings.model_sha256 != self.network_sha256:
            raise SequenceProviderError(
                "OFFICIAL_ASSET_HASH_MISMATCH",
                "Worker request is not bound to the pinned v3 network.onnx",
                field="bindings.model_sha256",
            )
        if track.bindings.identity_sha256 != expected_identity:
            raise SequenceProviderError(
                "OFFICIAL_ASSET_HASH_MISMATCH",
                "Worker request is not bound to the pinned Claire v3 identity data",
                field="bindings.identity_sha256",
            )
        return validate_official_v3_sequence_track(
            track,
            skin_pose_names=self.skin_pose_names,
            skin_minimums=self.skin_minimums,
            skin_maximums=self.skin_maximums,
            tongue_pose_names=self.tongue_pose_names,
            tongue_minimums=self.tongue_minimums,
            tongue_maximums=self.tongue_maximums,
            public_model_version=self.public_model_version,
            network_version=self.network_version,
            identity=self.identity,
            identity_index=self.identity_index,
            model_revision=self.model_revision,
            sdk_revision=self.required_sdk_revision,
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} is not readable JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} must be a JSON object"
        )
    return value


def _pose_names(path: Path, *, expected_count: int, label: str) -> tuple[str, ...]:
    try:
        with np.load(path, allow_pickle=False) as values:
            raw = np.asarray(values["poseNames"])
    except (OSError, ValueError, KeyError) as exc:
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} pose names are unreadable"
        ) from exc
    if raw.ndim != 1:
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} poseNames must be one vector"
        )
    names = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw.tolist()
    )
    if (
        len(names) != expected_count + 1
        or names[0] != "neutral"
        or len(set(names)) != len(names)
        or any(not name for name in names)
    ):
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID",
            f"{label} must contain neutral plus {expected_count} unique controls",
        )
    return names[1:]


def _solver_ranges(
    path: Path, *, expected_count: int, label: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    document = _read_json(path, label)
    try:
        parameters = document["blendshape_params"]
        multipliers = np.asarray(parameters["bsWeightMultipliers"], dtype=np.float64)
        offsets = np.asarray(parameters["bsWeightOffsets"], dtype=np.float64)
        active = np.asarray(parameters["bsSolveActivePoses"], dtype=np.float64)
        count = int(parameters["numPoses"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} solver ranges are invalid"
        ) from exc
    if (
        count != expected_count
        or multipliers.shape != (expected_count,)
        or offsets.shape != (expected_count,)
        or active.shape != (expected_count,)
        or not np.isfinite(multipliers).all()
        or not np.isfinite(offsets).all()
        or np.any((active != 0.0) & (active != 1.0))
    ):
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", f"{label} solver range widths differ"
        )
    transformed = multipliers * active + offsets
    minimums = np.minimum(offsets, transformed)
    maximums = np.maximum(offsets, transformed)
    return (
        tuple(float(value) for value in minimums),
        tuple(float(value) for value in maximums),
    )


def load_official_v3_claire_profile(
    directory: str | Path,
    *,
    verify_network: bool = False,
) -> OfficialV3ClaireProfile:
    """Load and hash-check the pinned official Claire interpretation bundle.

    The 725 MB ONNX file is optional at the orchestration/retarget host because
    inference runs on an external NVIDIA worker.  Set ``verify_network`` on the
    worker or deployment audit host to require and hash it as well.
    """

    root = Path(directory).expanduser().resolve()
    required = _INTERPRETATION_FILES + (("network.onnx",) if verify_network else ())
    actual_hashes: dict[str, str] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise SequenceProviderError(
                "OFFICIAL_ASSET_MISSING", f"Pinned v3 asset is missing: {name}"
            )
        actual = sha256(path)
        expected = OFFICIAL_V3_ASSET_SHA256[name]
        if actual != expected:
            raise SequenceProviderError(
                "OFFICIAL_ASSET_HASH_MISMATCH",
                f"Pinned v3 asset hash differs: {name}",
                field=name,
            )
        actual_hashes[name] = actual

    network = _read_json(root / "network_info.json", "network_info.json")
    try:
        identifier = network["id"]
        parameters = network["params"]
        audio = network["audio_params"]
        identities = tuple(str(value) for value in parameters["identities"])
    except (KeyError, TypeError) as exc:
        raise SequenceProviderError(
            "OFFICIAL_ASSET_INVALID", "network_info.json lacks the v3 profile"
        ) from exc
    expected_network = (
        isinstance(identifier, dict)
        and identifier.get("type") == "diffusion"
        and identifier.get("actor") == "multi"
        and identifier.get("output") == "geometry"
        and str(identifier.get("version")) == A2F_V3_NETWORK_VERSION
        and identities[A2F_V3_IDENTITY_INDEX : A2F_V3_IDENTITY_INDEX + 1]
        == (A2F_V3_IDENTITY,)
        and parameters.get("skin_size") == 72_006
        and parameters.get("tongue_size") == 16_806
        and parameters.get("jaw_size") == 15
        and parameters.get("eyes_size") == 4
        and parameters.get("num_frames_left_truncate") == 15
        and parameters.get("num_frames_right_truncate") == 15
        and parameters.get("num_frames_center") == 30
        and isinstance(audio, dict)
        and audio.get("buffer_len") == 16_000
        and audio.get("padding_left") == 16_000
        and audio.get("padding_right") == 16_000
        and audio.get("samplerate") == 16_000
    )
    if not expected_network:
        raise SequenceProviderError(
            "OFFICIAL_PROFILE_MISMATCH",
            "network_info.json differs from the pinned Claire v3 geometry profile",
        )

    skin_names = _pose_names(
        root / "bs_skin_Claire.npz", expected_count=52, label="Claire skin"
    )
    tongue_names = _pose_names(
        root / "bs_tongue_Claire.npz", expected_count=16, label="Claire tongue"
    )
    if tongue_names != OFFICIAL_V3_TONGUE_CONTROL_NAMES:
        raise SequenceProviderError(
            "OFFICIAL_CONTROL_SCHEMA_MISMATCH",
            "Claire tongue control ordering differs from the pinned worker ABI",
        )
    tongue_minimums, tongue_maximums = _solver_ranges(
        root / "bs_tongue_config_Claire.json",
        expected_count=16,
        label="Claire tongue",
    )
    skin_minimums, skin_maximums = _solver_ranges(
        root / "bs_skin_config_Claire.json",
        expected_count=52,
        label="Claire skin",
    )
    return OfficialV3ClaireProfile(
        root=root,
        public_model_version=A2F_V3_PUBLIC_MODEL_VERSION,
        network_version=A2F_V3_NETWORK_VERSION,
        identity=A2F_V3_IDENTITY,
        identity_index=A2F_V3_IDENTITY_INDEX,
        model_revision=A2F_V3_MODEL_REVISION,
        required_sdk_revision=A2F_V3_SDK_REVISION,
        skin_pose_names=skin_names,
        tongue_pose_names=tongue_names,
        skin_minimums=skin_minimums,
        skin_maximums=skin_maximums,
        tongue_minimums=tongue_minimums,
        tongue_maximums=tongue_maximums,
        interpretation_asset_sha256=actual_hashes,
        network_sha256=OFFICIAL_V3_ASSET_SHA256["network.onnx"],
    )


__all__ = [
    "OFFICIAL_V3_ASSET_SHA256",
    "OFFICIAL_V3_TONGUE_CONTROL_NAMES",
    "OfficialV3ClaireProfile",
    "load_official_v3_claire_profile",
]
