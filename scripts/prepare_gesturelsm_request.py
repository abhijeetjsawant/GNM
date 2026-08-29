#!/usr/bin/env python3
"""Seal canonical audio, transcript and alignment into a CUDA-worker request."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import wave

from autoanim_gnm.speech_motion_provider import SpeechMotionProfile, SpeechMotionRequest
from autoanim_gnm.speech_alignment import normalize_supplied_transcript


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "workers" / "gesturelsm" / "provider-lock.json"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def prepare_request(
    output_directory: str | Path,
    *,
    request_id: str,
    audio_path: str | Path,
    transcript: str,
    alignment_path: str | Path,
    seeds: tuple[int, ...] = (7, 11, 17, 23),
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_audio = Path(audio_path).resolve(strict=True)
    source_alignment = Path(alignment_path).resolve(strict=True)
    with wave.open(str(source_audio), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getframerate() != 16_000
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("GestureLSM audio must be mono 16 kHz PCM16 WAV")
        # The durable body/face lane is sampled at 30 Hz with an inclusive
        # sample at t=0. Match the final complete 30 Hz sample exactly; the
        # audio element may hold that last pose for a sub-frame tail.
        duration_ticks = (audio.getnframes() * 30 // audio.getframerate()) * 1600
    if source_alignment.is_symlink() or not source_alignment.is_file():
        raise ValueError("A regular MFA TextGrid alignment is required")
    alignment_bytes = source_alignment.read_bytes()
    if b"TextGrid" not in alignment_bytes[:512]:
        raise ValueError("Alignment does not look like a TextGrid")
    normalized_transcript = normalize_supplied_transcript(transcript)
    audio_target = output / "normalized.wav"
    transcript_target = output / "transcript.txt"
    alignment_target = output / "alignment.TextGrid"
    shutil.copy2(source_audio, audio_target)
    transcript_target.write_text(normalized_transcript + "\n", encoding="utf-8")
    shutil.copy2(source_alignment, alignment_target)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    request = SpeechMotionRequest(
        request_id=request_id,
        duration_ticks=duration_ticks,
        audio_file_name=audio_target.name,
        audio_sha256=_sha(audio_target),
        transcript_file_name=transcript_target.name,
        transcript_sha256=_sha(transcript_target),
        alignment_file_name=alignment_target.name,
        alignment_sha256=_sha(alignment_target),
        transcript_mode="supplied",
        asr_model_revision=None,
        candidate_seeds=seeds,
        profile=SpeechMotionProfile(
            provider_git_commit_oid=lock["provider"]["git_commit_oid"],
            model_revision=lock["model"]["revision"],
            model_artifact_sha256=lock["model"]["checkpoint_sha256"],
        ),
    )
    manifest = output / "request.json"
    manifest.write_bytes(request.canonical_json_bytes())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    transcript = parser.add_mutually_exclusive_group(required=True)
    transcript.add_argument("--transcript")
    transcript.add_argument("--transcript-file", type=Path)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 17, 23))
    arguments = parser.parse_args()
    text = (
        arguments.transcript_file.read_text(encoding="utf-8")
        if arguments.transcript_file is not None
        else arguments.transcript
    )
    result = prepare_request(
        arguments.output_directory,
        request_id=arguments.request_id,
        audio_path=arguments.audio,
        transcript=text,
        alignment_path=arguments.alignment,
        seeds=tuple(arguments.seeds),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
