"""Lightweight provenance validation shared by CLI and motion pipelines."""

from __future__ import annotations

import math

from .errors import AutoAnimError


def validate_mouth_aperture_authorship(
    *,
    gain: float,
    author: str | None,
    reason: str | None,
) -> tuple[str | None, str | None]:
    """Require accountable provenance for every non-default artist edit.

    This validator intentionally depends only on the standard library so CLI
    argument errors can terminate before MediaPipe, OpenCV, or model runtimes
    are imported.
    """

    if not math.isfinite(gain) or gain < 1.0:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Mouth-aperture gain must be finite and at least 1",
        )
    normalized_author = author.strip() if author is not None else ""
    normalized_reason = reason.strip() if reason is not None else ""
    if len(normalized_author) > 160 or len(normalized_reason) > 500:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Mouth-aperture author and reason are limited to 160 and 500 characters",
        )
    if gain != 1.0 and (not normalized_author or not normalized_reason):
        raise AutoAnimError(
            "INPUT_INVALID",
            "A non-default mouth-aperture edit requires both an author and a reason",
        )
    return normalized_author or None, normalized_reason or None


__all__ = ["validate_mouth_aperture_authorship"]
