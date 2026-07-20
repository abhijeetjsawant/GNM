from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest

from autoanim_gnm.review_bundle import (
    BRIDGE_MESSAGE_TYPES,
    CLOSEUP_REGIONS,
    LAYER_ORDER,
    MAX_DOCUMENT_BYTES,
    MAX_FRAMES,
    ReviewBundleError,
    SCHEMA_VERSION,
    build_review_bundle,
    load_review_bundle,
    review_bundle_payload_sha256,
)
from autoanim_gnm.serialization import write_npz
from autoanim_gnm.video_capture import (
    LANDMARK_COUNT,
    CaptureProvenance,
    CaptureTrack,
    write_capture_npz,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline_array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _capture(path: Path, *, source_sha256: str, source_bytes: int) -> Path:
    pts = np.asarray((100, 102, 105), dtype=np.int64)
    time_base = Fraction(2, 60)
    seconds = np.asarray(
        [float(Fraction(int(value - pts[0])) * time_base) for value in pts],
        dtype=np.float64,
    )
    milliseconds = np.asarray(
        [int(round(value * 1_000)) for value in seconds], dtype=np.int64
    )
    count = len(pts)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], count, axis=0)
    track = CaptureTrack(
        source_pts=pts,
        timestamps_seconds=seconds,
        mediapipe_timestamps_ms=milliseconds,
        detected=np.ones(count, dtype=np.bool_),
        landmarks_xyz=np.zeros((count, LANDMARK_COUNT, 3), dtype=np.float32),
        landmark_visibility=np.ones((count, LANDMARK_COUNT), dtype=np.float32),
        landmark_presence=np.ones((count, LANDMARK_COUNT), dtype=np.float32),
        blendshape_names=("_neutral",),
        blendshape_scores=np.zeros((count, 1), dtype=np.float32),
        facial_transforms=transforms,
        face_confidence=np.ones(count, dtype=np.float32),
        tracking_quality=np.ones(count, dtype=np.float32),
        width=640,
        height=480,
        provenance=CaptureProvenance(
            source_name="performance.mov",
            source_sha256=source_sha256,
            source_bytes=source_bytes,
            model_name="face-landmarker.task",
            model_sha256=_digest("model"),
            mediapipe_version="0.10",
            ffprobe_version="test",
            ffmpeg_version="test",
            codec="h264",
            time_base_numerator=2,
            time_base_denominator=60,
            source_start_pts=int(pts[0]),
            display_rotation_degrees=0,
            ffprobe_command=("ffprobe", "performance.mov"),
            ffmpeg_command=("ffmpeg", "performance.mov"),
        ),
    )
    return write_capture_npz(path, track)


def _performance(path: Path) -> Path:
    count = 3
    zeros = np.zeros(count, dtype=np.float32)
    return write_npz(
        path,
        schema_version=np.asarray("autoanim.gnm-performance.v3"),
        identity=np.linspace(-0.25, 0.25, 253, dtype=np.float32),
        expression=np.zeros((count, 383), dtype=np.float32),
        rotations=np.zeros((count, 4, 3), dtype=np.float32),
        translation=np.zeros((count, 3), dtype=np.float32),
        timestamps_seconds=np.asarray((0.0, 2 / 30, 5 / 30), dtype=np.float64),
        source_pts=np.asarray((100, 102, 105), dtype=np.int64),
        detected=np.ones(count, dtype=np.bool_),
        effective_quality=np.ones(count, dtype=np.float32),
        source_lip_geometry_valid=np.ones(count, dtype=np.bool_),
        source_lip_gap_interocular=zeros,
        source_lip_contact_confidence=zeros,
        lip_contact_target_gap_interocular=zeros,
        contact_correction_applied=np.zeros(count, dtype=np.bool_),
        lip_contact_attained=np.ones(count, dtype=np.bool_),
        lip_aperture_target_gap_interocular=zeros,
        lip_aperture_correction_applied=np.zeros(count, dtype=np.bool_),
        lip_aperture_target_attained=np.ones(count, dtype=np.bool_),
        provenance_json=np.asarray(json.dumps({"fixture": "review-bundle"})),
    )


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    source_bytes = b"retained-source"
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    files: dict[str, Path] = {}
    files["capture"] = _capture(
        tmp_path / "capture.npz",
        source_sha256=source_sha256,
        source_bytes=len(source_bytes),
    )
    files["controls"] = _performance(tmp_path / "performance.npz")
    payloads = {
        "viewer_media": ("source-proxy.mp4", b"proxy"),
        "audio_visual_repair_arrays": ("audio-visual-repair.npz", b"repair arrays"),
        "acting_track": ("acting-track.bin", b"acting"),
        "mouth_aperture_edit": ("mouth-aperture-edit.json", b"authored"),
        "mouth_aperture_edit_arrays": ("mouth-aperture-edit.npz", b"authored arrays"),
        "physics_track": ("physics-track.bin", b"physics"),
        "glb": ("performance.glb", b"glTF"),
        "material_base_color": ("base-color.png", b"base color"),
    }
    for logical_name, (name, data) in payloads.items():
        path = tmp_path / name
        path.write_bytes(data)
        files[logical_name] = path
    identity = np.linspace(-0.25, 0.25, 253, dtype=np.float32)
    final_expression_sha256 = _pipeline_array_digest(
        np.zeros((3, 383), dtype=np.float32)
    )
    visual_expression_sha256 = _digest("visual-expression-before-repair")
    repair_report = {
        "bindings": {
            "identitySha256": _pipeline_array_digest(identity),
            "inputExpressionSha256": visual_expression_sha256,
            "outputExpressionSha256": final_expression_sha256,
        },
        "caveats": [],
        "claims": {
            "changesFinalGNMMotion": True,
            "productionValidated": False,
        },
        "clockJoin": {},
        "config": {},
        "locks": {},
        "metrics": {},
        "outputRole": "intermediate_pre_artist_mouth_aperture_revision",
        "policy": "video_authoritative_conservative_audio_repair_v1",
        "schemaVersion": "autoanim.audio-visual-repair.v1",
        "sourceAuthority": {},
        "status": "repaired",
    }
    repair_path = tmp_path / "audio-visual-repair.json"
    repair_path.write_text(json.dumps(repair_report), encoding="utf-8")
    files["audio_visual_repair"] = repair_path
    revision_chain = {
        "chainConsistent": True,
        "finalPerformanceExpressionSha256": final_expression_sha256,
        "immutableSourceMediaSha256": source_sha256,
        "productionValidated": False,
        "revisions": [
            {
                "inputAuthority": "immutable_video_snapshot",
                "name": "visual_video_retarget",
                "outputExpressionSha256": visual_expression_sha256,
            },
            {
                "applied": True,
                "inputExpressionSha256": visual_expression_sha256,
                "name": "learned_audio_visual_repair",
                "outputExpressionSha256": final_expression_sha256,
                "reportSha256": _file_digest(repair_path),
            },
            {
                "applied": False,
                "compositeInputSha256": _digest("composite-input"),
                "compositeOutputSha256": _digest("composite-output"),
                "inputExpressionSha256": final_expression_sha256,
                "name": "authored_mouth_aperture",
                "outputExpressionSha256": final_expression_sha256,
                "reportSha256": None,
            },
        ],
        "schemaVersion": "autoanim.performance-revision-chain.v1",
        "sourcePtsSha256": _pipeline_array_digest(
            np.asarray((100, 102, 105), dtype=np.int64)
        ),
        "status": "candidate_unqualified",
    }
    chain_path = tmp_path / "performance-revision-chain.json"
    chain_path.write_text(json.dumps(revision_chain), encoding="utf-8")
    files["performance_revision_chain"] = chain_path
    artifacts = {
        logical_name: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_digest(path),
            "media_type": (
                "application/json"
                if path.suffix == ".json"
                else "model/gltf-binary"
                if path.suffix == ".glb"
                else "application/octet-stream"
            ),
        }
        for logical_name, path in files.items()
    }
    manifest = {
        "schema_version": "1.0",
        "job_id": "01reviewbundlefixture00000000",
        "kind": "video_performance",
        "status": "succeeded",
        "created_at": "2026-07-20T10:00:00Z",
        "updated_at": "2026-07-20T10:01:00Z",
        "input": {
            "name": "performance.mov",
            "sha256": source_sha256,
            "bytes": len(source_bytes),
            "media_type": "video/quicktime",
        },
        "capture": {"frames": 3},
        "model": {
            "gnm_version": "3.0",
            "character": {
                "runtime_material_sha256s": {
                    "base_color": artifacts["material_base_color"]["sha256"],
                    "normal": _digest("normal-reference"),
                }
            }
        },
        "artifacts": artifacts,
        "warnings": [],
        "integrity": {
            "schema": "autoanim.hmac-sha256.v1",
            "key_id": "0123456789abcdef",
            "signature": _digest("manifest-signature"),
        },
    }
    return manifest, files


@pytest.fixture
def built_bundle(tmp_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    manifest, paths = _fixture(tmp_path)
    return build_review_bundle(manifest, artifact_paths=paths), manifest, paths


def _rehash(document: dict) -> dict:
    document["bundle_sha256"] = review_bundle_payload_sha256(document)
    return document


def _rehash_clock(document: dict) -> dict:
    clock = document["clock"]
    payload = deepcopy(clock)
    payload.pop("clock_sha256", None)
    clock["clock_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return _rehash(document)


def test_builder_is_deterministic_and_reconstructs_native_review_contract(
    tmp_path: Path,
) -> None:
    manifest, paths = _fixture(tmp_path)

    first = build_review_bundle(manifest, artifact_paths=paths)
    second = build_review_bundle(deepcopy(manifest), artifact_paths=paths)

    assert first == second
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["clock"]["time_base"] == [1, 30]
    assert first["clock"]["source_pts"] == [100, 102, 105]
    assert first["clock"]["first_display_time_exact_rational"] == [0, 1]
    assert first["clock"]["source_start_time_exact_rational"] == [10, 3]
    assert first["clock"]["duration_exact_rational"] == [1, 6]
    assert [value["layer_id"] for value in first["layers"]] == list(LAYER_ORDER)
    assert all(value["availability"] == "available" for value in first["layers"])
    assert all(value["production_motion_authority"] == "none" for value in first["layers"])
    assert all(value["production_validated"] is False for value in first["layers"])
    assert [value["region_id"] for value in first["closeups"]] == list(CLOSEUP_REGIONS)
    assert all(value["renderable"] is False for value in first["closeups"])
    assert len(first["revision_graph"]["nodes"]) == 7
    assert first["revision_graph"]["ab_pairs"] == []
    assert first["revision_graph"]["ab_scope"] == (
        "cross_bundle_same_comparison_key_only"
    )
    assert first["revision_graph"]["renderable_revisions"] == [
        {
            "revision_id": first["layers"][-1]["revision_id"],
            "artifact_logical_name": "glb",
            "render_role": "final_textured_animation",
            "production_validated": False,
            "approval_status": "unapproved",
        }
    ]
    comparison = first["comparison_key"]
    assert comparison["input_sha256"] == first["source_manifest"]["input"]["sha256"]
    assert comparison["clock_sha256"] == first["clock"]["clock_sha256"]
    assert comparison["viewer_media_sha256"] == next(
        item["sha256"]
        for item in first["artifacts"]
        if item["logical_name"] == "viewer_media"
    )
    assert comparison["gnm_version"] == "3.0"
    assert len(comparison["controls_identity_sha256"]) == 64
    assert first["bridge"]["allowed_message_types"] == list(BRIDGE_MESSAGE_TYPES)
    assert "correction" not in first["bridge"]["allowed_message_types"]
    assert first["correction_eligibility"]["candidate_request_eligible"] is False
    assert first["correction_eligibility"]["writer_implemented"] is False
    assert first["correction_eligibility"]["production_revision_eligible"] is False
    assert first["claims"]["artifact_ledger_bytes_verified"] is True
    assert first["claims"]["exact_rational_pts_clock_verified"] is True
    assert first["claims"]["manifest_signature_verified"] is False
    assert first["claims"]["production_validated"] is False
    assert first["claims"]["publishable"] is False
    assert load_review_bundle(first) == first


def test_artifacts_materials_and_motion_consumption_are_explicit(
    built_bundle: tuple[dict, dict, dict[str, Path]],
) -> None:
    bundle, _, _ = built_bundle
    artifacts = bundle["artifacts"]
    assert [value["logical_name"] for value in artifacts] == sorted(
        value["logical_name"] for value in artifacts
    )
    assert all(value["bytes_verified"] is True for value in artifacts)
    layers = {value["layer_id"]: value for value in bundle["layers"]}
    assert layers["source"]["motion_authority"] == "reference_only"
    assert layers["visual_base"]["motion_authority"] == "candidate_visual_retarget"
    assert layers["audio_repair"]["motion_authority"] == (
        "candidate_lower_face_and_tongue_repair"
    )
    assert layers["audio_repair"]["consumption"]["consumed_by_final_reported"] is True
    assert layers["audio_repair"]["consumption"]["consumption_independently_verified"] is False
    materials = {value["channel"]: value for value in bundle["material_channels"]}
    assert materials["base_color"]["status"] == "sealed_artifact"
    assert materials["base_color"]["isolatable"] is True
    assert materials["normal"]["status"] == "hash_reference_only"
    assert materials["normal"]["isolatable"] is False
    assert materials["displacement"]["status"] == "unavailable"
    assert all(value["measured"] is False for value in materials.values())
    assert all(value["production_validated"] is False for value in materials.values())


def test_builder_requires_exact_sealed_artifact_set_and_bytes(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    missing = dict(paths)
    missing.pop("glb")
    with pytest.raises(ReviewBundleError) as missing_error:
        build_review_bundle(manifest, artifact_paths=missing)
    assert missing_error.value.code == "ARTIFACT_SET_MISMATCH"

    extra = dict(paths)
    extra["unexpected"] = paths["glb"]
    with pytest.raises(ReviewBundleError) as extra_error:
        build_review_bundle(manifest, artifact_paths=extra)
    assert extra_error.value.code == "ARTIFACT_SET_MISMATCH"

    paths["glb"].write_bytes(b"tampered glb")
    with pytest.raises(ReviewBundleError) as tampered:
        build_review_bundle(manifest, artifact_paths=paths)
    assert tampered.value.code == "ARTIFACT_INTEGRITY"


def _reseal_artifact(manifest: dict, paths: dict[str, Path], logical_name: str) -> None:
    path = paths[logical_name]
    manifest["artifacts"][logical_name].update(
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_digest(path),
        }
    )


def test_controls_identity_reader_is_strict_bounded_and_clock_bound(
    tmp_path: Path,
) -> None:
    manifest, paths = _fixture(tmp_path)
    controls_path = paths["controls"]
    with np.load(controls_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["source_pts"] = np.asarray((100, 103, 105), dtype=np.int64)
    write_npz(controls_path, **arrays)
    _reseal_artifact(manifest, paths, "controls")
    with pytest.raises(ReviewBundleError) as clock:
        build_review_bundle(manifest, artifact_paths=paths)
    assert clock.value.code == "CONTROLS_CLOCK_MISMATCH"

    manifest, paths = _fixture(tmp_path / "nonfinite")
    controls_path = paths["controls"]
    with np.load(controls_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["identity"][0] = np.nan
    write_npz(controls_path, **arrays)
    _reseal_artifact(manifest, paths, "controls")
    with pytest.raises(ReviewBundleError) as nonfinite:
        build_review_bundle(manifest, artifact_paths=paths)
    assert nonfinite.value.code == "CONTROLS_NONFINITE"

    manifest, paths = _fixture(tmp_path / "duplicate")
    controls_path = paths["controls"]
    with zipfile.ZipFile(controls_path) as archive:
        duplicate_payload = archive.read("identity.npy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(controls_path, "a") as archive:
            archive.writestr("identity.npy", duplicate_payload)
    _reseal_artifact(manifest, paths, "controls")
    with pytest.raises(ReviewBundleError) as duplicate:
        build_review_bundle(manifest, artifact_paths=paths)
    assert duplicate.value.code == "CONTROLS_ARCHIVE_INVALID"


def test_timing_and_audio_source_evidence_do_not_claim_repair_motion(
    tmp_path: Path,
) -> None:
    manifest, paths = _fixture(tmp_path)
    for logical_name in ("audio_visual_repair", "audio_visual_repair_arrays"):
        paths.pop(logical_name)
        manifest["artifacts"].pop(logical_name)
    for logical_name, filename in (
        ("audio_video_timing", "audio-video-timing.json"),
        ("audio_visual_source", "audio-visual-source.json"),
        ("audio_visual_source_controls", "audio-source-controls.npz"),
    ):
        path = tmp_path / filename
        path.write_bytes(logical_name.encode("ascii"))
        paths[logical_name] = path
        manifest["artifacts"][logical_name] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_digest(path),
            "media_type": "application/octet-stream",
        }

    bundle = build_review_bundle(manifest, artifact_paths=paths)
    layers = {layer["layer_id"]: layer for layer in bundle["layers"]}
    assert layers["audio_repair"]["availability"] == "unavailable"
    assert layers["audio_repair"]["motion_authority"] == "none"
    assert layers["audio_repair"]["changes_motion_reported"] is False
    assert {
        "audio_video_timing",
        "audio_visual_source",
        "audio_visual_source_controls",
    }.issubset(layers["source"]["artifact_logical_names"])
    assert layers["source"]["motion_authority"] == "reference_only"


def test_audio_repair_requires_report_and_arrays_together(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    paths.pop("audio_visual_repair_arrays")
    manifest["artifacts"].pop("audio_visual_repair_arrays")

    bundle = build_review_bundle(manifest, artifact_paths=paths)
    repair = next(layer for layer in bundle["layers"] if layer["layer_id"] == "audio_repair")
    assert repair["artifact_logical_names"] == ["audio_visual_repair"]
    assert repair["availability"] == "unavailable"
    assert repair["changes_motion_reported"] is False


def test_current_v2_audio_repair_schema_reports_applied_motion(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    report_path = paths["audio_visual_repair"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "schemaVersion": "autoanim.audio-visual-repair.v2",
            "policy": "video_authoritative_conservative_audio_repair_v2",
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _reseal_artifact(manifest, paths, "audio_visual_repair")
    chain_path = paths["performance_revision_chain"]
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    chain["revisions"][1]["reportSha256"] = _file_digest(report_path)
    chain_path.write_text(json.dumps(chain), encoding="utf-8")
    _reseal_artifact(manifest, paths, "performance_revision_chain")

    bundle = build_review_bundle(manifest, artifact_paths=paths)
    repair = next(
        layer for layer in bundle["layers"] if layer["layer_id"] == "audio_repair"
    )
    assert repair["availability"] == "available"
    assert repair["changes_motion_reported"] is True
    assert repair["motion_authority"] == "candidate_lower_face_and_tongue_repair"
    assert repair["consumption"]["consumed_by_final_reported"] is True


def test_retained_noop_reports_do_not_claim_applied_motion(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    chain_path = paths["performance_revision_chain"]
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    final_expression = chain["finalPerformanceExpressionSha256"]
    chain["revisions"][0]["outputExpressionSha256"] = final_expression
    chain["revisions"][1].update(
        {
            "applied": False,
            "inputExpressionSha256": final_expression,
            "outputExpressionSha256": final_expression,
            "reportSha256": None,
        }
    )
    chain["revisions"][2].update(
        {
            "applied": False,
            "inputExpressionSha256": final_expression,
            "outputExpressionSha256": final_expression,
        }
    )
    chain_path.write_text(json.dumps(chain), encoding="utf-8")
    _reseal_artifact(manifest, paths, "performance_revision_chain")

    bundle = build_review_bundle(manifest, artifact_paths=paths)
    layers = {layer["layer_id"]: layer for layer in bundle["layers"]}
    assert layers["audio_repair"]["availability"] == "available"
    assert layers["audio_repair"]["changes_motion_reported"] is False
    assert layers["audio_repair"]["motion_authority"] == "none"
    assert layers["audio_repair"]["consumption"]["consumed_by_final_reported"] is False
    assert layers["authored_correction"]["availability"] == "available"
    assert layers["authored_correction"]["changes_motion_reported"] is False


def test_builder_rejects_unsealed_wrong_kind_and_unbound_capture(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    unsealed = deepcopy(manifest)
    unsealed.pop("integrity")
    with pytest.raises(ReviewBundleError) as seal:
        build_review_bundle(unsealed, artifact_paths=paths)
    assert seal.value.code in {"INVALID_TYPE", "MANIFEST_UNSEALED"}

    wrong_kind = deepcopy(manifest)
    wrong_kind["kind"] = "audio_animation"
    with pytest.raises(ReviewBundleError) as kind:
        build_review_bundle(wrong_kind, artifact_paths=paths)
    assert kind.value.code == "UNSUPPORTED_PERFORMANCE"

    unbound = deepcopy(manifest)
    unbound["input"]["sha256"] = _digest("different-source")
    with pytest.raises(ReviewBundleError) as clock:
        build_review_bundle(unbound, artifact_paths=paths)
    assert clock.value.code == "CLOCK_SOURCE_MISMATCH"


def test_loader_rejects_duplicate_nonfinite_unknown_and_oversized_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReviewBundleError) as duplicate:
        load_review_bundle(b'{"schema_version":"a","schema_version":"b"}')
    assert duplicate.value.code == "DUPLICATE_KEY"

    with pytest.raises(ReviewBundleError) as nonfinite:
        load_review_bundle(b'{"schema_version":NaN}')
    assert nonfinite.value.code == "NONFINITE_NUMBER"

    manifest, paths = _fixture(tmp_path)
    bundle = build_review_bundle(manifest, artifact_paths=paths)
    unknown = deepcopy(bundle)
    unknown["unknown"] = True
    unknown["bundle_sha256"] = review_bundle_payload_sha256(unknown)
    with pytest.raises(ReviewBundleError) as unknown_error:
        load_review_bundle(unknown)
    assert unknown_error.value.code == "INVALID_FIELDS"

    oversized = tmp_path / "oversized-review.json"
    oversized.write_bytes(b"{" + b" " * MAX_DOCUMENT_BYTES + b"}")
    with pytest.raises(ReviewBundleError) as size:
        load_review_bundle(oversized)
    assert size.value.code == "DOCUMENT_SIZE"


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (
            lambda value: value["claims"].__setitem__("production_validated", True),
            "UNSUPPORTED_CLAIM",
        ),
        (
            lambda value: value["layers"][2].__setitem__(
                "production_motion_authority", "approved"
            ),
            "UNSUPPORTED_CLAIM",
        ),
        (
            lambda value: value["revision_graph"]["ab_pairs"].append(
                {
                    "a_revision_id": value["revision_graph"]["nodes"][0]["revision_id"],
                    "b_revision_id": value["revision_graph"]["nodes"][1]["revision_id"],
                }
            ),
            "INVALID_REVISION_GRAPH",
        ),
        (
            lambda value: value["closeups"][0].__setitem__("renderable", True),
            "INVALID_CLOSEUP",
        ),
        (
            lambda value: value["material_channels"][0].__setitem__("measured", True),
            "UNSUPPORTED_CLAIM",
        ),
        (
            lambda value: value["correction_eligibility"].__setitem__(
                "writer_implemented", True
            ),
            "UNSUPPORTED_CLAIM",
        ),
        (
            lambda value: value["bridge"].__setitem__(
                "arbitrary_script_messages_allowed", True
            ),
            "INVALID_BRIDGE",
        ),
    ),
)
def test_loader_rejects_rehashed_authority_and_approval_escalation(
    built_bundle: tuple[dict, dict, dict[str, Path]], mutation, code: str
) -> None:
    bundle, _, _ = built_bundle
    forged = deepcopy(bundle)
    mutation(forged)
    _rehash(forged)

    with pytest.raises(ReviewBundleError) as caught:
        load_review_bundle(forged)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["clock"]["source_pts"].__setitem__(1, 100),
        lambda value: value["clock"]["source_pts"].__setitem__(1, True),
        lambda value: value["clock"].__setitem__("frame_count", 4),
        lambda value: value["clock"].__setitem__("time_base", [2, 60]),
        lambda value: value["clock"].__setitem__(
            "duration_exact_rational", [1, 5]
        ),
    ),
)
def test_loader_rejects_rehashed_nonexact_rational_pts_clock(
    built_bundle: tuple[dict, dict, dict[str, Path]], mutation
) -> None:
    bundle, _, _ = built_bundle
    forged = deepcopy(bundle)
    mutation(forged)
    _rehash_clock(forged)

    with pytest.raises(ReviewBundleError) as caught:
        load_review_bundle(forged)
    assert caught.value.code in {"INVALID_CLOCK", "INVALID_RATIONAL", "INVALID_TYPE"}


def test_loader_rejects_clock_beyond_u1_bound(
    built_bundle: tuple[dict, dict, dict[str, Path]],
) -> None:
    bundle, _, _ = built_bundle
    forged = deepcopy(bundle)
    pts = list(range(MAX_FRAMES + 1))
    forged["clock"].update(
        {
            "source_pts": pts,
            "frame_count": len(pts),
            "first_source_pts": 0,
            "last_source_pts": MAX_FRAMES,
            "first_display_time_exact_rational": [0, 1],
            "source_start_time_exact_rational": [0, 1],
            "duration_exact_rational": [60, 1],
        }
    )
    _rehash_clock(forged)

    with pytest.raises(ReviewBundleError) as caught:
        load_review_bundle(forged)
    assert caught.value.code == "CLOCK_BOUNDS"


@pytest.mark.parametrize(
    "field",
    ("input_sha256", "clock_sha256", "source_pts_sha256", "viewer_media_sha256"),
)
def test_loader_rejects_rehashed_comparison_binding_tampering(
    built_bundle: tuple[dict, dict, dict[str, Path]], field: str
) -> None:
    bundle, _, _ = built_bundle
    forged = deepcopy(bundle)
    forged["comparison_key"][field] = _digest(f"forged-{field}")
    comparison_payload = deepcopy(forged["comparison_key"])
    comparison_payload.pop("comparison_key_sha256")
    forged["comparison_key"]["comparison_key_sha256"] = hashlib.sha256(
        json.dumps(
            comparison_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _rehash(forged)

    with pytest.raises(ReviewBundleError) as caught:
        load_review_bundle(forged)
    assert caught.value.code == "COMPARISON_KEY_MISMATCH"


def test_bundle_hash_and_nested_fields_are_strict(
    built_bundle: tuple[dict, dict, dict[str, Path]],
) -> None:
    bundle, _, _ = built_bundle
    tampered = deepcopy(bundle)
    tampered["source_manifest"]["job_id"] = "different-job"
    with pytest.raises(ReviewBundleError) as digest:
        load_review_bundle(tampered)
    assert digest.value.code == "BUNDLE_HASH_MISMATCH"

    nested = deepcopy(bundle)
    nested["layers"][0]["unknown"] = True
    _rehash(nested)
    with pytest.raises(ReviewBundleError) as fields:
        load_review_bundle(nested)
    assert fields.value.code == "INVALID_FIELDS"


def test_json_file_round_trip_is_canonical(
    tmp_path: Path, built_bundle: tuple[dict, dict, dict[str, Path]]
) -> None:
    bundle, _, _ = built_bundle
    path = tmp_path / "review-bundle.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    loaded = load_review_bundle(path)

    assert loaded == bundle
    assert loaded["bundle_sha256"] == review_bundle_payload_sha256(loaded)
