from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import autoanim_gnm.audio_video_timing as av_timing
from autoanim_gnm.audio_video_timing import (
    AUDIO_VIDEO_TIMING_POLICY,
    AUDIO_VIDEO_TIMING_SCHEMA_VERSION,
    build_audio_video_timing_evidence,
    load_verified_audio_video_timing_evidence,
    write_audio_video_timing_evidence,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.video_capture import probe_video
from autoanim_gnm.video_pipeline import run_video_pipeline


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
A2F_ASSETS = CACHE / "a2f-claire"
CREMA_D_ANGRY = FIXTURES / "crema-d-1001-dfa-ang.flv"
CREMA_D_ANGRY_SHA256 = "10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4"
RETAINED_CREMA_JOB = Path(
    os.environ.get(
        "AUTOANIM_RETAINED_CREMA_JOB",
        "artifacts/jobs/01kxtx72xy7z1hbmv747hgjzdc",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg_media(
    path: Path,
    *,
    audio: bool = True,
    audio_offset_seconds: Fraction = Fraction(0),
    audio_duration_seconds: Fraction = Fraction(1, 2),
    vfr: bool = True,
) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            "testsrc2=size=64x64:rate=30000/1001:duration=0.101"
            if vfr
            else "testsrc2=size=64x64:rate=25:duration=0.40"
        ),
    ]
    if audio:
        if audio_offset_seconds:
            command.extend(("-itsoffset", str(float(audio_offset_seconds))))
        command.extend(
            (
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration="
                + str(float(audio_duration_seconds)),
            )
        )
    if vfr:
        command.extend(
            (
                "-filter:v",
                "setpts='if(eq(N,0),0,if(eq(N,1),1,if(eq(N,2),3,6)))'",
            )
        )
    command.extend(("-map", "0:v:0"))
    if audio:
        command.extend(("-map", "1:a:0"))
    command.extend(("-fps_mode", "passthrough"))
    if vfr:
        command.extend(("-c:v", "libx264", "-video_track_timescale", "30000"))
    else:
        command.extend(("-c:v", "ffv1"))
    if audio:
        command.extend(("-c:a", "aac" if vfr else "pcm_s16le"))
    command.append(str(path))
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return path


def _build(source: Path, *, require_available: bool = False) -> dict:
    probe = probe_video(source)
    return build_audio_video_timing_evidence(
        source,
        expected_source_sha256=_sha256(source),
        expected_video_source_pts=probe.source_pts,
        expected_video_time_base=probe.time_base,
        require_available=require_available,
    )


def test_ffprobe_output_is_capped_before_json_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(av_timing, "MAX_FFPROBE_OUTPUT_BYTES", 128)
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 4096)",
    )
    with pytest.raises(av_timing._TimingBlocked) as exc_info:
        av_timing._probe_source(command)
    assert exc_info.value.code == "LIMIT_EXCEEDED"
    assert exc_info.value.status == "blocked_probe_output_limit"


@pytest.mark.parametrize("lane", ("probe", "decode"))
def test_tool_failure_diagnostics_redact_absolute_source_paths(
    tmp_path: Path,
    lane: str,
) -> None:
    source = tmp_path / "private retained source.mov"
    source.write_bytes(b"private")
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stderr.write(sys.argv[1]); raise SystemExit(2)",
        str(source.resolve()),
    )
    with pytest.raises(av_timing._TimingBlocked) as exc_info:
        if lane == "probe":
            av_timing._probe_source(command)
        else:
            av_timing._hash_pcm(command)
    assert str(source.resolve()) not in exc_info.value.message
    assert "<REDACTED_PATH>" in exc_info.value.message


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_tools_read_one_immutable_snapshot_when_original_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ffmpeg_media(tmp_path / "mutable-source.mkv", vfr=False)
    expected_sha256 = _sha256(source)
    probe = probe_video(source)
    original_probe = av_timing._probe_source

    def mutate_original_after_snapshot(command: tuple[str, ...]) -> dict:
        source.write_bytes(b"changed after immutable snapshot")
        return original_probe(command)

    monkeypatch.setattr(av_timing, "_probe_source", mutate_original_after_snapshot)
    report = build_audio_video_timing_evidence(
        source,
        expected_source_sha256=expected_sha256,
        expected_video_source_pts=probe.source_pts,
        expected_video_time_base=probe.time_base,
    )
    assert report["status"] == "available_observation"
    assert report["source"]["sha256"] == expected_sha256


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_fractional_vfr_display_intervals_join_exact_native_pcm_samples(
    tmp_path: Path,
) -> None:
    source = _ffmpeg_media(tmp_path / "vfr.mp4")
    probe = probe_video(source)
    assert probe.time_base == Fraction(1, 30_000)
    assert probe.source_pts.tolist() == [0, 1001, 3003, 6006]

    report = _build(source)
    assert report["schemaVersion"] == AUDIO_VIDEO_TIMING_SCHEMA_VERSION
    assert report["policy"] == AUDIO_VIDEO_TIMING_POLICY
    assert report["status"] == "available_observation"
    assert report["consumedByRetargeting"] is False
    assert report["claims"]["speechContentInferred"] is False
    serialized = json.dumps(report)
    assert str(source.resolve()) not in serialized
    assert serialized.count("<RETAINED_SOURCE>") == 2
    join = report["audioVideoJoin"]
    assert join["video"]["timeBase"] == [1, 30_000]
    assert join["audio"]["sampleRate"] == 48_000
    assert join["audio"]["monoPcm"]["sampleCount"] == 24_576
    assert join["audio"]["monoPcm"]["channels"] == 1
    assert [
        frame["coveringAudioSampleSpan"] for frame in join["video"]["frames"]
    ] == [[0, 1602], [1601, 3204], [4804, 6407], [9609, 12813]]
    assert [
        frame["displayDurationSource"] for frame in join["video"]["frames"]
    ] == ["ffprobe_frame_duration_ticks"] * 4
    assert join["sync"]["audioMinusVideoStartOffsetExactRational"] == [0, 1]
    assert join["sync"]["audioMinusVideoDurationDriftExactRational"] == [919, 3750]
    assert report["fusionGate"]["status"] == "blocked"
    assert "nonzero_av_duration_drift" in report["fusionGate"]["reasons"]


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_no_audio_is_typed_unavailable_and_only_strict_repair_fails(
    tmp_path: Path,
) -> None:
    source = _ffmpeg_media(tmp_path / "silent-video.mkv", audio=False, vfr=False)
    report = _build(source)
    assert report["status"] == "unavailable_no_audio"
    assert report["audioVideoJoin"] is None
    assert report["failure"]["code"] == "AUDIO_STREAM_UNAVAILABLE"
    assert report["fusionGate"] == {
        "status": "blocked",
        "reasons": ["AUDIO_STREAM_UNAVAILABLE"],
        "exactClockJoinAvailable": False,
        "audioSemanticEvidenceAvailable": False,
    }
    assert report["claims"]["changesFinalGNMMotion"] is False

    with pytest.raises(AutoAnimError, match="no decodable primary audio") as exc_info:
        _build(source, require_available=True)
    assert exc_info.value.code == "AUDIO_STREAM_UNAVAILABLE"


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_exact_offset_and_duration_drift_are_reported_not_applied(
    tmp_path: Path,
) -> None:
    source = _ffmpeg_media(
        tmp_path / "offset.mkv",
        audio_offset_seconds=Fraction(1, 8),
        audio_duration_seconds=Fraction(1, 5),
        vfr=False,
    )
    report = _build(source)
    sync = report["audioVideoJoin"]["sync"]
    assert sync["audioMinusVideoStartOffsetExactRational"] == [1, 8]
    assert sync["audioMinusVideoDurationDriftExactRational"] == [-1, 5]
    assert sync["audioMinusVideoEndOffsetExactRational"] == [-3, 40]
    assert sync["completeAudioCoverageForAllVideoFrames"] is False
    assert report["fusionGate"]["reasons"] == [
        "audio_content_not_classified",
        "nonzero_av_start_offset",
        "nonzero_av_duration_drift",
        "incomplete_video_audio_coverage",
    ]
    assert report["claims"]["lipSyncCorrected"] is False


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_nonmonotonic_audio_pts_fail_closed_for_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ffmpeg_media(tmp_path / "nonmonotonic.mkv", vfr=False)
    original_probe = av_timing._probe_source

    def tampered_probe(command: tuple[str, ...]) -> dict:
        payload = original_probe(command)
        audio_index = next(
            stream["index"]
            for stream in payload["streams"]
            if stream["codec_type"] == "audio"
        )
        frames = [
            frame for frame in payload["frames"] if frame["stream_index"] == audio_index
        ]
        frames[1]["best_effort_timestamp"] = frames[0]["best_effort_timestamp"]
        return payload

    monkeypatch.setattr(av_timing, "_probe_source", tampered_probe)
    report = _build(source)
    assert report["status"] == "blocked_nonmonotonic_audio_timing"
    assert report["audioVideoJoin"] is None
    assert report["fusionGate"]["exactClockJoinAvailable"] is False


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_verified_join_rejects_sample_span_and_pcm_hash_tampering(
    tmp_path: Path,
) -> None:
    source = _ffmpeg_media(tmp_path / "source.mkv", vfr=False)
    probe = probe_video(source)
    path = write_audio_video_timing_evidence(
        tmp_path / "timing.json",
        source,
        expected_source_sha256=_sha256(source),
        expected_video_source_pts=probe.source_pts,
        expected_video_time_base=probe.time_base,
    )
    original = json.loads(path.read_text(encoding="utf-8"))
    pcm_sha256 = original["audioVideoJoin"]["audio"]["monoPcm"]["sha256"]
    load_verified_audio_video_timing_evidence(
        path,
        expected_source_sha256=_sha256(source),
        expected_video_source_pts=probe.source_pts,
        expected_pcm_sha256=pcm_sha256,
    )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["audioVideoJoin"]["video"]["frames"][1][
        "coveringAudioSampleSpan"
    ][0] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sample timing"):
        load_verified_audio_video_timing_evidence(
            path,
            expected_source_sha256=_sha256(source),
            expected_video_source_pts=probe.source_pts,
            expected_pcm_sha256=pcm_sha256,
        )

    original["audioVideoJoin"]["audio"]["monoPcm"]["sha256"] = "0" * 64
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="PCM binding"):
        load_verified_audio_video_timing_evidence(
            path,
            expected_source_sha256=_sha256(source),
            expected_video_source_pts=probe.source_pts,
            expected_pcm_sha256=pcm_sha256,
        )

    original["audioVideoJoin"]["audio"]["monoPcm"]["sha256"] = pcm_sha256
    original["audioVideoJoin"]["audio"]["decodedFrames"][1]["sourcePTS"] += 1
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="audio frame 1"):
        load_verified_audio_video_timing_evidence(
            path,
            expected_source_sha256=_sha256(source),
            expected_video_source_pts=probe.source_pts,
            expected_pcm_sha256=pcm_sha256,
        )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is unavailable",
)
def test_verified_unavailable_join_rejects_status_failure_code_mismatch(
    tmp_path: Path,
) -> None:
    source = _ffmpeg_media(tmp_path / "silent-video.mkv", audio=False, vfr=False)
    probe = probe_video(source)
    path = write_audio_video_timing_evidence(
        tmp_path / "timing.json",
        source,
        expected_source_sha256=_sha256(source),
        expected_video_source_pts=probe.source_pts,
        expected_video_time_base=probe.time_base,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "unavailable_no_audio"
    payload["failure"]["code"] = "MEDIA_INVALID"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid unavailable"):
        load_verified_audio_video_timing_evidence(
            path,
            expected_source_sha256=_sha256(source),
            expected_video_source_pts=probe.source_pts,
        )


@pytest.mark.skipif(
    not CREMA_D_ANGRY.is_file()
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="retained checksum-pinned CREMA-D source is unavailable",
)
def test_real_retained_crema_audio_sample_video_pts_join() -> None:
    assert _sha256(CREMA_D_ANGRY) == CREMA_D_ANGRY_SHA256
    report = _build(CREMA_D_ANGRY)
    assert report["status"] == "available_observation"
    join = report["audioVideoJoin"]
    assert join["video"]["frameCount"] == 67
    assert join["audio"]["sampleRate"] == 44_100
    assert join["audio"]["monoPcm"]["sampleCount"] == 100_224
    assert join["audio"]["monoPcm"]["sha256"] == (
        "cad1efc3437a29cc35cd31dd97697ace663133f43afd7bb7c55c65205e58763d"
    )
    assert join["audio"]["maximumContiguousClockResidualExactRational"] == [6, 6125]
    assert join["sync"]["audioMinusVideoStartOffsetExactRational"] == [-27, 1000]
    assert join["sync"]["audioMinusVideoDurationDriftExactRational"] == [369, 9800]
    assert join["sync"]["completeAudioCoverageForAllVideoFrames"] is True
    assert report["fusionGate"]["status"] == "blocked"


@pytest.mark.skipif(
    not CREMA_D_ANGRY.is_file()
    or not MODEL.is_file()
    or not (A2F_ASSETS / "bs_skin.npz").is_file()
    or not (A2F_ASSETS / "bs_tongue.npz").is_file()
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="CREMA-D/model/Claire/FFmpeg fixtures unavailable",
)
def test_diagnostics_toggle_preserves_final_controls_and_glb_hashes(
    tmp_path: Path,
) -> None:
    without_diagnostics = tmp_path / "without"
    with_diagnostics = tmp_path / "with"
    run_video_pipeline(
        CREMA_D_ANGRY,
        without_diagnostics,
        model_path=MODEL,
        a2f_asset_dir=A2F_ASSETS,
        audio_video_timing_evidence=False,
    )
    result = run_video_pipeline(
        CREMA_D_ANGRY,
        with_diagnostics,
        model_path=MODEL,
        a2f_asset_dir=A2F_ASSETS,
        audio_video_timing_evidence=True,
    )
    for artifact in ("performance.npz", "performance.glb"):
        assert _sha256(with_diagnostics / artifact) == _sha256(
            without_diagnostics / artifact
        )
    assert not (without_diagnostics / "audio-video-timing.json").exists()
    assert (with_diagnostics / "audio-video-timing.json").is_file()
    assert result["capture"]["audio_video_timing_status"] == "available_observation"
    assert result["capture"]["audio_video_timing_consumed_by_retargeting"] is False
    assert result["artifacts"]["audio_video_timing"] == "audio-video-timing.json"
