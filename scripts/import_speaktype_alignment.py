#!/usr/bin/env python3
"""Convert SpeakType/FluidAudio Parakeet JSON into GestureLSM supplied inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoanim_gnm.speech_alignment import (
    load_speaktype_parakeet_alignment,
    write_speech_alignment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    alignment = load_speaktype_parakeet_alignment(arguments.input)
    transcript, textgrid = write_speech_alignment(
        arguments.output_directory, alignment
    )
    print(
        json.dumps(
            {
                "engine": alignment.engine,
                "model_revision": alignment.model_revision,
                "duration_seconds": alignment.duration_seconds,
                "word_count": len(alignment.words),
                "transcript": str(transcript),
                "alignment": str(textgrid),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
