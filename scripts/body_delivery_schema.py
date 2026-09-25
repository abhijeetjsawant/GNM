"""D4i: which body schema a commercial-multiview delivery directory holds, and refusal by schema.

Two schemas can sit in `build_commercial_multiview_comparison.py`'s output directory:

  * ``rig``  -- `subject-XX.body-track.json` schema `autoanim.body-track/1.x` (AutoAnim-55 on the MPFB asset),
    with `subject-XX.mapping.npz` beside it;
  * ``mhr``  -- `subject-XX.body-track.json` schema `autoanim.body-track/2.0-mhr` (MHR through momentum), with
    `subject-XX.markers.npz`, `subject-XX.calibration-start.json`, `subject-XX.calibration-passes.json`,
    `fit-report*.json` and `converter-inputs/`.

The same key means different things in the two: `root_translation_m` is the rig's root in the rig frame, and on an
MHR track it is the captured Z-up root joint. So a consumer must decide the schema from `schema_version` BEFORE it
reads any payload field, and a directory holding files of both schemas is refused outright: a stale
`mapping.npz` surviving an MHR rebuild would let a rig instrument read a rig file beside an MHR body.

`work/` is never inspected: it holds frames and detections, which are schema-free and are deliberately carried
between builds (the cached detections).

Only `schema_version` is read from a track: from the JSON (both schemas carry it) and, for the MHR npz, the one
`schema_version` member (np.load opens the zip lazily and reads only the member asked for).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MHR_SCHEMA = "autoanim.body-track/2.0-mhr"
RIG_SCHEMA_PREFIX = "autoanim.body-track/1."
SUBJECTS = (0, 1)

# Files only one schema's build writes. Globs relative to the delivery directory.
RIG_ONLY = ("subject-*.mapping.npz",)
MHR_ONLY = ("subject-*.markers.npz", "subject-*.calibration-start.json", "subject-*.calibration-passes.json",
            "fit-report*.json", "converter-inputs")


class SchemaScopeError(ValueError):
    """A track or directory is not of the schema the caller is scoped to. Raised before any payload read."""


def track_schema(delivery: Path, subject: int) -> str:
    """`schema_version` of one subject's track, read from the JSON header field only.

    For an MHR-schema JSON the npz's own `schema_version` member must agree, so a rig npz dropped beside an MHR
    JSON (or the reverse) is caught too. Raises SchemaScopeError on a missing or unreadable header.
    """
    import json

    prefix = Path(delivery) / f"subject-{subject:02d}"
    js = prefix.with_suffix(".body-track.json")
    npz = prefix.with_suffix(".body-track.npz")
    if not js.is_file() or not npz.is_file():
        raise SchemaScopeError(f"subject {subject:02d}: track missing ({js.name} / {npz.name})")
    try:
        declared = json.loads(js.read_text(encoding="utf-8")).get("schema_version")
    except (OSError, ValueError) as error:
        raise SchemaScopeError(f"subject {subject:02d}: unreadable track JSON: {error}") from error
    if not isinstance(declared, str):
        raise SchemaScopeError(f"subject {subject:02d}: track JSON declares no schema_version")
    # allow_pickle=False: only a string member is read, and nothing else is touched.
    with np.load(npz, allow_pickle=False) as archive:
        npz_declared = str(archive["schema_version"]) if "schema_version" in archive.files else None
    if declared == MHR_SCHEMA or npz_declared is not None:
        if npz_declared != declared:
            raise SchemaScopeError(f"subject {subject:02d}: the JSON says {declared!r} and the npz says "
                                   f"{npz_declared!r}: a mixed track")
    return declared


def schema_class(declared: str) -> str:
    if declared == MHR_SCHEMA:
        return "mhr"
    if declared.startswith(RIG_SCHEMA_PREFIX):
        return "rig"
    raise SchemaScopeError(f"unknown body-track schema {declared!r}")


def other_schema_files(delivery: Path, body: str) -> list[str]:
    """Files in `delivery` (never under work/) that only the OTHER schema's build writes."""
    delivery = Path(delivery)
    patterns = RIG_ONLY if body == "mhr" else MHR_ONLY
    found: list[str] = []
    for pattern in patterns:
        found += sorted(str(p.relative_to(delivery)) for p in delivery.glob(pattern))
    return found


def directory_state(delivery: Path) -> dict:
    """What a delivery directory holds: per-subject schemas, and the files of each schema. Never raises on content."""
    delivery = Path(delivery)
    subjects: dict[str, str | None] = {}
    for subject in SUBJECTS:
        try:
            subjects[f"subject_{subject:02d}"] = schema_class(track_schema(delivery, subject))
        except SchemaScopeError as error:
            subjects[f"subject_{subject:02d}"] = None if "track missing" in str(error) else f"invalid: {error}"
    return {"subjects": subjects, "mhr_only_files": other_schema_files(delivery, "rig"),
            "rig_only_files": other_schema_files(delivery, "mhr")}


def mixed_reasons(delivery: Path) -> list[str]:
    """Every reason `delivery` is a MIXED directory (empty list = not mixed). Tracks absent is not mixed."""
    state = directory_state(delivery)
    classes = {v for v in state["subjects"].values() if v is not None}
    reasons = [f"{k}: {v}" for k, v in state["subjects"].items() if v is not None and v.startswith("invalid")]
    classes = {c for c in classes if not c.startswith("invalid")}
    if len(classes) > 1:
        reasons.append(f"the subjects carry different schemas: {state['subjects']}")
    if state["mhr_only_files"] and ("rig" in classes or state["rig_only_files"]):
        reasons.append(f"MHR-only files {state['mhr_only_files']} beside rig "
                       f"{'tracks' if 'rig' in classes else 'files ' + str(state['rig_only_files'])}")
    if state["rig_only_files"] and "mhr" in classes:
        reasons.append(f"rig-only files {state['rig_only_files']} beside MHR tracks")
    return reasons


def refuse_other_schema(delivery: Path, body: str) -> None:
    """The build's guard: refuse to write a `body` delivery into a directory holding the other schema's files."""
    delivery = Path(delivery)
    if not delivery.is_dir():
        return
    problems = [f"file only a {'rig' if body == 'mhr' else 'MHR'} build writes: {name}"
                for name in other_schema_files(delivery, body)]
    for subject in SUBJECTS:
        try:
            found = schema_class(track_schema(delivery, subject))
        except SchemaScopeError as error:
            if "track missing" in str(error):
                continue
            problems.append(str(error))
            continue
        if found != body:
            problems.append(f"subject-{subject:02d} track is {found}-schema")
    if problems:
        raise SchemaScopeError(
            f"refusing to build a --body {body} delivery into {delivery}: it holds the other schema's files "
            f"({'; '.join(problems)}). Move the old delivery aside (keeping work/ for the cached detections) "
            "and rebuild into a directory holding only work/.")


def require_scope(delivery: Path, scope: str) -> dict[str, str]:
    """An instrument's scope check: every subject's track is `scope`-schema and the directory is not mixed.

    Reads `schema_version` only. Returns {subject: declared schema}. Raises SchemaScopeError otherwise.
    """
    if scope not in ("mhr", "rig"):
        raise ValueError(f"scope must be 'mhr' or 'rig', not {scope!r}")
    declared = {}
    for subject in SUBJECTS:
        value = track_schema(delivery, subject)
        if schema_class(value) != scope:
            raise SchemaScopeError(f"called in {scope.upper()} scope on a {schema_class(value)}-schema track "
                                   f"(subject-{subject:02d}: {value!r}); refused before any payload field is read")
        declared[f"subject_{subject:02d}"] = value
    reasons = mixed_reasons(delivery)
    if reasons:
        raise SchemaScopeError(f"mixed delivery directory {delivery}: {'; '.join(reasons)}")
    return declared
