#!/usr/bin/env python3
"""Build and seal a deliberate GNM facial range-of-motion viewer job."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from autoanim_gnm.artifacts import JobStore
from autoanim_gnm.expression_showcase import build_expression_showcase
from autoanim_gnm.serialization import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/experiments/expression-showcase"),
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=None,
        help=(
            "Optional character texture. By default the viewer uses Google's "
            "anatomical GNM vertex colors so teeth, tongue, and mouth sock stay visible."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    store = JobStore(args.artifacts)
    with tempfile.TemporaryDirectory(prefix="autoanim-expression-showcase-") as temporary:
        request = Path(temporary) / "showcase-request.json"
        write_json(
            request,
            {
                "kind": "deliberate_expression_range_showcase",
                "fps": args.fps,
                "texture": str(args.texture),
            },
        )
        job_id, job_dir, _, manifest = store.start(
            "expression_showcase",
            request,
            {
                "fps": args.fps,
                "diagnostic_only": True,
                "texture_path": str(args.texture),
            },
            original_name="showcase-request.json",
        )
        build = build_expression_showcase(
            job_dir,
            texture_path=args.texture,
            fps=args.fps,
        )
        metrics = build.timeline_document["metrics"]
        result = store.finish(
            manifest,
            job_dir,
            {
                "kind": "expression_showcase",
                "metrics": metrics,
                "warnings": [
                    (
                        "RANGE_OF_MOTION_DIAGNOSTIC_ONLY: this deliberately "
                        "exaggerated animation is not inferred acting or "
                        "production-approved deformation."
                    ),
                    *build.timeline_document["limitations"],
                ],
                "viewer": {
                    "clock_artifact": "viewer_media",
                    "clock": "showcase_timeline_seconds",
                },
                "artifacts": {
                    "glb": "animation.glb",
                    "glb_mapping": "animation-glb-mapping.npz",
                    "viewer_media": "preview.mp4",
                    "preview": "preview.mp4",
                    "controls": "controls.npz",
                    "showcase_timeline": "showcase-timeline.json",
                    "oral_validation": "oral-validation.json",
                    "soft_contact_report": "soft-contact-report.json",
                },
            },
            {
                "autoanim": "0.1.0",
                "gnm": "3.0",
                "showcase_schema": build.timeline_document["schema_version"],
            },
        )
    print(result["job_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
