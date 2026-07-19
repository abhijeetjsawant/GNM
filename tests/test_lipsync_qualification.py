from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from autoanim_gnm.artifacts import sha256
from autoanim_gnm.cli import main as cli_main
from autoanim_gnm.integrity import IntegritySigner
from autoanim_gnm.lipsync_quality import QualityThresholds
from autoanim_gnm.lipsync_qualification import (
    PROFILE_SCHEMA,
    LipsyncQualificationError,
    evaluate_controls_qualification,
    identity_array_sha256,
    landmarks_sha256,
    parse_qualification_profile,
    runtime_rig_sha256,
    seal_profile_document,
)
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.serialization import write_json, write_npz


FPS = 30
FRAME_COUNT = 110
EVENT_FRAMES = (25, 40, 55, 70)
EVENT_LABELS = ("A", "B", "F", "D")


@dataclass
class QualificationCase:
    profile: dict[str, object]
    profile_path: Path
    controls_path: Path
    audio_path: Path
    character_manifest_path: Path
    identity_artifact_path: Path
    evidence: dict[str, Path]
    expression: np.ndarray
    speech_activity: np.ndarray


def _write_controls(
    path: Path,
    expression: np.ndarray,
    speech_activity: np.ndarray,
    *,
    fps: int = FPS,
    timestamps: np.ndarray | None = None,
) -> Path:
    clock = (
        np.arange(len(expression), dtype=np.float32) / np.float32(fps)
        if timestamps is None
        else np.asarray(timestamps)
    )
    return write_npz(
        path,
        expression=np.asarray(expression, dtype=np.float32),
        timestamps=clock,
        fps=np.asarray(fps, dtype=np.int32),
        speech_activity=np.asarray(speech_activity, dtype=np.float32),
    )


def _case(tmp_path: Path, rig: ControlRig) -> QualificationCase:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"independent source audio fixture\n")
    character_manifest_path = tmp_path / "character-manifest.json"
    character_manifest_path.write_text('{"character":"fixture-v1"}\n', encoding="utf-8")
    identity_artifact_path = write_npz(tmp_path / "identity.npz", identity=rig.identity)

    annotation_evidence = tmp_path / "phones.TextGrid"
    annotation_evidence.write_text("independent manual phone tier\n", encoding="utf-8")
    prototype_source = tmp_path / "gnm-targets.npz"
    prototype_source.write_bytes(b"artist-authored GNM sparse-68 targets\n")
    prototype_approval = tmp_path / "artist-approval.pdf"
    prototype_approval.write_bytes(b"artist approval evidence\n")
    evidence = {
        "phones-textgrid": annotation_evidence,
        "gnm-target-source": prototype_source,
        "gnm-target-approval": prototype_approval,
    }

    neutral_expression = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    prototype_expressions = {
        label: np.float32(0.8) * rig.viseme(label) for label in EVENT_LABELS
    }
    prototype_landmarks = {
        label: rig.compact_landmarks(prototype_expressions[label])
        for label in EVENT_LABELS
    }
    expression = np.repeat(neutral_expression[None], FRAME_COUNT, axis=0)
    keys = [(10, neutral_expression)]
    keys.extend(
        zip(
            EVENT_FRAMES,
            (prototype_expressions[label] for label in EVENT_LABELS),
            strict=True,
        )
    )
    keys.append((85, neutral_expression))
    for (left_frame, left), (right_frame, right) in zip(keys[:-1], keys[1:], strict=True):
        for frame in range(left_frame, right_frame + 1):
            alpha = (frame - left_frame) / (right_frame - left_frame)
            expression[frame] = (1.0 - alpha) * left + alpha * right
    speech_activity = np.zeros(FRAME_COUNT, dtype=np.float32)
    speech_activity[10:85] = 1.0
    controls_path = _write_controls(
        tmp_path / "controls.npz", expression, speech_activity
    )

    thresholds = asdict(QualityThresholds())
    profile: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "binding": {
            "source_audio_sha256": sha256(audio_path),
            "character_manifest_sha256": sha256(character_manifest_path),
            "identity_artifact_sha256": sha256(identity_artifact_path),
            "identity_array_sha256": identity_array_sha256(rig.identity),
            "rig_sha256": runtime_rig_sha256(rig),
        },
        "timebase": {
            "units": "seconds",
            "fps_numerator": FPS,
            "fps_denominator": 1,
            "frame_count": FRAME_COUNT,
            "timestamp_origin_seconds": 0.0,
        },
        "profile_provenance": {
            "curator_id": "qualification-curator-01",
            "created_at": "2026-07-19T08:00:00Z",
            "protocol": "independent_lipsync_qualification_v1",
        },
        "annotation_provenance": {
            "annotator_id": "phonetics-lab-annotator-07",
            "annotator_organization": "Independent Phonetics Lab",
            "created_at": "2026-07-19T07:00:00Z",
            "method": "manual_phonetic_annotation",
            "independent_from_animation_system": True,
            "viewed_system_cues": False,
            "viewed_generated_animation": False,
            "used_system_output_as_timing_source": False,
            "evidence_artifact": {
                "artifact_id": "phones-textgrid",
                "sha256": sha256(annotation_evidence),
            },
        },
        "annotations": [
            {
                "event_id": f"event-{index + 1:03d}",
                "label": label,
                "start_seconds": (frame - 2) / FPS,
                "apex_seconds": frame / FPS,
                "release_seconds": (frame + 2) / FPS,
            }
            for index, (frame, label) in enumerate(
                zip(EVENT_FRAMES, EVENT_LABELS, strict=True)
            )
        ],
        "target_prototypes": [
            {
                "label": label,
                "landmarks": prototype_landmarks[label].astype(np.float64).tolist(),
                "landmarks_sha256": landmarks_sha256(prototype_landmarks[label]),
                "provenance": {
                    "artist_id": "facial-artist-03",
                    "artist_organization": "Independent Character Art",
                    "created_at": "2026-07-18T10:00:00Z",
                    "approved_at": "2026-07-19T06:00:00Z",
                    "authoring_tool": "GNM target review tool 1.0",
                    "coordinate_space": "gnm_head_sparse_68_3d",
                    "artist_approved": True,
                    "source_artifact": {
                        "artifact_id": "gnm-target-source",
                        "sha256": sha256(prototype_source),
                    },
                    "approval_artifact": {
                        "artifact_id": "gnm-target-approval",
                        "sha256": sha256(prototype_approval),
                    },
                },
            }
            for label in EVENT_LABELS
        ],
        "evaluator": {
            "quality_thresholds": thresholds,
            "stationary_step_interocular": 5e-4,
            "neutral_tolerance_interocular": 0.015,
            "silence_guard_frames": 2,
            "timing_search_frames": 6,
        },
    }
    profile = seal_profile_document(profile)
    profile_path = write_json(tmp_path / "qualification.json", profile)
    return QualificationCase(
        profile=profile,
        profile_path=profile_path,
        controls_path=controls_path,
        audio_path=audio_path,
        character_manifest_path=character_manifest_path,
        identity_artifact_path=identity_artifact_path,
        evidence=evidence,
        expression=expression,
        speech_activity=speech_activity,
    )


def _evaluate(case: QualificationCase, rig: ControlRig, **overrides: object):
    arguments: dict[str, object] = {
        "controls_path": case.controls_path,
        "source_audio_path": case.audio_path,
        "character_manifest_path": case.character_manifest_path,
        "identity_artifact_path": case.identity_artifact_path,
        "provenance_artifacts": case.evidence,
        "rig": rig,
    }
    arguments.update(overrides)
    return evaluate_controls_qualification(case.profile_path, **arguments)


def _reseal(profile: dict[str, object]) -> dict[str, object]:
    return seal_profile_document(deepcopy(profile))


def test_sealed_independent_profile_qualifies_existing_controls_deterministically(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)

    parsed = parse_qualification_profile(case.profile_path)
    first = _evaluate(case, rig)
    second = _evaluate(case, rig)

    assert parsed.schema_version == PROFILE_SCHEMA
    assert not parsed.target_prototypes["A"].landmarks.flags.writeable
    assert first.core_quality_gate_passed
    assert not first.production_validated
    assert first.quality.production_gate.passed
    assert first.as_dict() == second.as_dict()
    assert first.controls_sha256 == sha256(case.controls_path)
    assert first.source_audio_sha256 == sha256(case.audio_path)
    assert first.evidence_sha256s == {
        artifact_id: sha256(path) for artifact_id, path in case.evidence.items()
    }
    document = first.as_dict()
    declared = document.pop("report_sha256")
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert declared == hashlib.sha256(canonical).hexdigest()
    assert document["qualification_scope"] == (
        "independent_apex_pose_and_motion_hygiene_only"
    )
    assert document["sequence_timing_validated"] is False
    assert document["perceptual_validation_completed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("independent_from_animation_system", False),
        ("viewed_system_cues", True),
        ("viewed_generated_animation", True),
        ("used_system_output_as_timing_source", True),
        ("method", "system_generated"),
    ),
)
def test_profile_rejects_self_scored_annotation_evidence(
    tmp_path: Path,
    rig: ControlRig,
    field: str,
    value: object,
) -> None:
    case = _case(tmp_path, rig)
    profile = deepcopy(case.profile)
    profile["annotation_provenance"][field] = value  # type: ignore[index]

    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(profile))

    assert caught.value.code == "INDEPENDENCE_UNPROVEN"


def test_profile_payload_and_prototype_hashes_reject_tampering(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    payload_tamper = deepcopy(case.profile)
    payload_tamper["annotations"][0]["apex_seconds"] += 0.01  # type: ignore[index]
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(payload_tamper)
    assert caught.value.code == "PROFILE_HASH_MISMATCH"

    prototype_tamper = deepcopy(case.profile)
    prototype_tamper["target_prototypes"][0]["landmarks"][61][1] += 0.01  # type: ignore[index]
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(prototype_tamper))
    assert caught.value.code == "PROTOTYPE_HASH_MISMATCH"


def test_profile_rejects_unknown_duplicate_and_nonfinite_json(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    unknown = deepcopy(case.profile)
    unknown["unexpected"] = True
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(unknown))
    assert caught.value.code == "INVALID_PROFILE"

    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(b'{"schema_version":"a","schema_version":"b"}')
    assert caught.value.code == "DUPLICATE_KEY"

    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(b'{"schema_version":NaN}')
    assert caught.value.code == "INVALID_NUMBER"


def test_profile_requires_complete_approved_prototypes_and_in_range_events(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    missing = deepcopy(case.profile)
    missing["target_prototypes"] = missing["target_prototypes"][:-1]  # type: ignore[index]
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(missing))
    assert caught.value.code == "INCOMPLETE_PROTOTYPES"

    unapproved = deepcopy(case.profile)
    unapproved["target_prototypes"][0]["provenance"]["artist_approved"] = False  # type: ignore[index]
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(unapproved))
    assert caught.value.code == "PROTOTYPE_NOT_APPROVED"

    outside = deepcopy(case.profile)
    outside["annotations"][-1]["release_seconds"] = 999.0  # type: ignore[index]
    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(outside))
    assert caught.value.code == "INVALID_ANNOTATION"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mouth_step_max_interocular", 0.5),
        ("speech_active_stationary_fraction", 1.0),
        ("target_contrast_median", 0.0),
        ("timing_error_p95_frames", 20.0),
    ),
)
def test_profile_cannot_weaken_versioned_production_gates(
    tmp_path: Path,
    rig: ControlRig,
    field: str,
    value: float,
) -> None:
    case = _case(tmp_path, rig)
    profile = deepcopy(case.profile)
    profile["evaluator"]["quality_thresholds"][field] = value  # type: ignore[index]

    with pytest.raises(LipsyncQualificationError) as caught:
        parse_qualification_profile(_reseal(profile))

    assert caught.value.code == "INVALID_THRESHOLDS"


@pytest.mark.parametrize("binding", ("audio", "character", "identity", "rig"))
def test_evaluator_rejects_source_character_identity_and_rig_binding_mismatch(
    tmp_path: Path,
    rig: ControlRig,
    binding: str,
) -> None:
    case = _case(tmp_path, rig)
    if binding == "audio":
        substitute = tmp_path / "other.wav"
        substitute.write_bytes(b"different audio")
        kwargs = {"source_audio_path": substitute}
        profile_source: object = case.profile_path
    elif binding == "character":
        substitute = tmp_path / "other-character.json"
        substitute.write_text('{"character":"other"}\n', encoding="utf-8")
        kwargs = {"character_manifest_path": substitute}
        profile_source = case.profile_path
    elif binding == "identity":
        substitute = write_npz(
            tmp_path / "other-identity.npz",
            identity=np.ones_like(rig.identity),
        )
        kwargs = {"identity_artifact_path": substitute}
        profile_source = case.profile_path
    else:
        profile = deepcopy(case.profile)
        profile["binding"]["rig_sha256"] = "f" * 64  # type: ignore[index]
        profile_source = _reseal(profile)
        kwargs = {}

    arguments = {
        "controls_path": case.controls_path,
        "source_audio_path": case.audio_path,
        "character_manifest_path": case.character_manifest_path,
        "identity_artifact_path": case.identity_artifact_path,
        "provenance_artifacts": case.evidence,
        "rig": rig,
        **kwargs,
    }
    with pytest.raises(LipsyncQualificationError) as caught:
        evaluate_controls_qualification(profile_source, **arguments)
    assert caught.value.code == "BINDING_MISMATCH"


def test_evaluator_rejects_runtime_identity_array_mismatch(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    changed_rig = ControlRig(
        rig.adapter,
        rig.decoder,
        identity=np.full_like(rig.identity, 0.01),
    )

    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, changed_rig)

    assert caught.value.code == "BINDING_MISMATCH"
    assert caught.value.field == "identity_array"


def test_evaluator_requires_exact_provenance_artifact_set_and_hashes(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    missing = dict(case.evidence)
    missing.pop("phones-textgrid")
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, provenance_artifacts=missing)
    assert caught.value.code == "EVIDENCE_MISSING"

    substitute = tmp_path / "substitute.TextGrid"
    substitute.write_text("system-produced cue tier\n", encoding="utf-8")
    mismatch = dict(case.evidence)
    mismatch["phones-textgrid"] = substitute
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, provenance_artifacts=mismatch)
    assert caught.value.code == "EVIDENCE_MISMATCH"


def test_evaluator_rejects_missing_arrays_nonfinite_controls_and_clock_mismatch(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    missing = write_npz(
        tmp_path / "missing-controls.npz",
        expression=case.expression,
        timestamps=np.arange(FRAME_COUNT, dtype=np.float32) / FPS,
        fps=np.asarray(FPS),
    )
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, controls_path=missing)
    assert caught.value.code == "INVALID_CONTROLS"

    nonfinite_expression = case.expression.copy()
    nonfinite_expression[5, 200] = np.nan
    nonfinite = _write_controls(
        tmp_path / "nonfinite-controls.npz",
        nonfinite_expression,
        case.speech_activity,
    )
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, controls_path=nonfinite)
    assert caught.value.code == "INVALID_CONTROLS"

    wrong_clock = _write_controls(
        tmp_path / "wrong-clock.npz",
        case.expression,
        case.speech_activity,
        timestamps=np.arange(FRAME_COUNT, dtype=np.float32) / FPS + 0.01,
    )
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, controls_path=wrong_clock)
    assert caught.value.code == "TIMEBASE_MISMATCH"

    unexpected_member = tmp_path / "unexpected-member.npz"
    unexpected_member.write_bytes(case.controls_path.read_bytes())
    with zipfile.ZipFile(unexpected_member, "a") as archive:
        archive.writestr("untrusted.npy", b"not a numeric array")
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, controls_path=unexpected_member)
    assert caught.value.code == "INVALID_CONTROLS"

    wrong_fps = _write_controls(
        tmp_path / "wrong-fps.npz",
        case.expression,
        case.speech_activity,
        fps=24,
    )
    with pytest.raises(LipsyncQualificationError) as caught:
        _evaluate(case, rig, controls_path=wrong_fps)
    assert caught.value.code == "TIMEBASE_MISMATCH"


def test_bound_but_bad_controls_return_a_failed_production_gate(
    tmp_path: Path,
    rig: ControlRig,
) -> None:
    case = _case(tmp_path, rig)
    static = np.zeros_like(case.expression)
    static_path = _write_controls(
        tmp_path / "static-controls.npz", static, case.speech_activity
    )

    report = _evaluate(case, rig, controls_path=static_path)

    assert not report.production_validated
    assert not report.quality.production_gate.passed
    assert "speech_active_motion" in report.quality.production_gate.failures
    assert "target_contrast_median" in report.quality.production_gate.failures


def test_cli_qualifies_bound_existing_track_and_writes_scoped_report(
    tmp_path: Path,
    rig: ControlRig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = _case(tmp_path, rig)
    output = tmp_path / "qualification-report.json"
    arguments = [
        "qualify-lipsync",
        str(case.profile_path),
        "--controls",
        str(case.controls_path),
        "--source-audio",
        str(case.audio_path),
        "--character-manifest",
        str(case.character_manifest_path),
        "--identity-artifact",
        str(case.identity_artifact_path),
        "--out",
        str(output),
        "--artifacts",
        str(tmp_path / "jobs"),
    ]
    for artifact_id, path in case.evidence.items():
        arguments.extend(("--evidence", f"{artifact_id}={path}"))

    assert cli_main(arguments) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["core_quality_gate_passed"] is True
    assert written["production_validated"] is False
    assert written["qualification_scope"] == (
        "independent_apex_pose_and_motion_hygiene_only"
    )
    signer = IntegritySigner(tmp_path / ".autoanim-integrity" / "hmac.key")
    assert signer.verify(written)
