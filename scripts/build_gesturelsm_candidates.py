#!/usr/bin/env python3
"""Validate, retarget, project and report a set of GestureLSM responses."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.speech_motion_candidates import (
    build_speech_motion_candidates,
    candidate_report,
)
from autoanim_gnm.speech_motion_provider import (
    load_gesturelsm_response,
    load_speech_motion_request,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    request = load_speech_motion_request(arguments.request)
    motions = tuple(
        load_gesturelsm_response(path, profile=request.profile, request=request)
        for path in arguments.responses
    )
    candidates = build_speech_motion_candidates(motions)
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(candidates, start=1):
        track = candidate.projected_track
        stem = f"rank-{rank:02d}-seed-{candidate.seed}"
        write_json(output / f"{stem}.json", track.as_dict())
        write_npz(
            output / f"{stem}.npz",
            ticks=track.ticks,
            root_translation_m=track.root_translation_m,
            local_rotations_xyzw=track.local_rotations_xyzw,
            foot_contacts=track.foot_contacts,
            gaze_direction_body=track.gaze_direction_body,
            gaze_strength=track.gaze_strength,
            gnm_eye_rotations_xyzw=track.gnm_eye_rotations_xyzw,
        )
    write_json(output / "candidate-set.json", candidate_report(candidates))
    print(output / "candidate-set.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
