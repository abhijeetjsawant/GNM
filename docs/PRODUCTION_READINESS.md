# Unified production-readiness contract

Status: implemented fail-closed release evidence, 2026-07-20

## Purpose

AutoAnim's reconstruction, character, audio, video, oral, acting, body, and
viewer stages can all produce useful artifacts before they are production
validated. A successful job or a renderable GLB is therefore not a publish
decision. `autoanim.production-readiness/1.2` collects the evidence for one
performance take into a single machine-readable report without mutating or
approving any asset.

The HTTP interface is:

```text
GET /api/jobs/{performance_job_id}/production-readiness
    ?direction_job_id={optional_acting_job_id}
    &require_pbr=true
    &require_acting=false
    &require_body=false
```

The browser requests the default report after every Audio or Video run and
shows either a completed release-evidence state or the exact missing required
gates. Optional acting and body requirements can be made mandatory for a shot
or delivery profile. Disabling an optional requirement is recorded in the
report; it does not falsify that capability's evidence.

## Required gates

| Gate | Evidence required |
| --- | --- |
| `terminal_take` | Successful `audio_animation` or `video_performance` job. |
| `provenance_integrity` | Valid job HMAC plus retained source byte count and SHA-256. |
| `character_revision` | Exact character/revision/hash binding that still resolves under the job's intended-use scope. Revoked, expired, missing, or altered revisions fail. |
| `identity` | Independent identity/hidden-geometry validation recorded on the immutable source character revision. |
| `appearance` | When PBR is required: exact runtime base color, normal, roughness and specular maps plus pore/detail, unseen-light, and appearance production approval. |
| `oral_animation` | Every source frame and viewer reconstruction structurally validated, no reported lip-order or tongue/teeth risk, and separate phone/tongue/collision/perceptual approval. |
| `performance` | Audio: learned source, independently reviewed articulatory onset/contact/apex/release annotations, passing legacy timing and future qualified multi-articulator evidence, hash-verified artifacts, independent prototype quality, and an approved animation profile. The current A1 report has phone-span proxy diagnostics only and can never pass this release gate. Video: verified Observation-v2, regional Observation-v3 and CaptureSession artifacts, approved capture, subject calibration, and labeled-neutral calibration. Observation v3 remains diagnostic and cannot itself approve a take. |
| `delivery` | Full-track, source-clocked animated GLB whose bytes match the signed artifact ledger. |
| `acting` | When required: a sealed direction job linked to this exact performance, edited/compiled and explicitly artist-approved. An LLM proposal never passes this gate. |
| `body` | When required: attached body/head seam plus approved body motion and contacts. A canonical skeleton or preview track alone never passes. |

`publishable` is true only when every required gate passes. Proxy metrics such
as landmark NME, expression correlation, mouth smoothness, texture resolution,
or structural GLB reconstruction remain visible as evidence but never override
a missing independent approval.

CaptureSession has two deliberately separate readiness facts. Artifact
verification means its deterministic document reconstructs from the exact
sealed Capture v1, JSONL, Observation-v2 and Observation-v3 bytes. Production
claims require a bound subject plus confirmed neutrality, verified identity
continuity and explicit production approval. The current v1 emitter always
leaves those claims false/unknown, so structural integrity alone cannot make a
video take publishable.

## Integrity behavior

The report re-resolves the exact character revision on every request and
rechecks the retained source and GLB bytes. For video, it also loads Capture v1,
reconstructs Observation v3 from its bounded NPZ arrays, and reconstructs the
path-free CaptureSession against all referenced sealed artifacts. Tampering
after job completion therefore changes the relevant integrity or evidence gate
to failed even if an old UI label says otherwise. Character consent and
material-rights expiry are evaluated at request time.

For audio with a retained TextGrid, readiness also re-parses the immutable
annotation, re-evaluates complete GNM oral geometry in bounded mesh batches from
the sealed control track, and byte-compares both timing reports with their
artifact-ledger and result bindings. Exact controls, identity, GNM, 68-point
regressor, decoder, verifier-source and NumPy/SciPy bindings are included. A
runtime/verifier mismatch returns `historical_phone_articulation_verifier_unavailable`
instead of being mislabeled as artifact tampering. One process-level verifier lock prevents concurrent GETs
from multiplying reconstruction work. Unsealed legacy manifests are never
opened for A1 reconstruction. Missing older-schema articulation evidence or any
altered report fails the evidence gate; it is never inferred from a summary
boolean. Expression controls and 68-point landmarks remain full-track arrays;
their practical bound is the audio pipeline's ten-minute, 30/60 fps input cap.
Before NumPy opens a control archive, readiness checks its exact member set and
compressed/expanded byte ceiling derived from that duration cap.

Readiness evaluation is read-only. Future approval operations must create a
new immutable, signed evidence revision or review record; they must not rewrite
historical job results or flip a boolean in place.

## Current expected result

Current real Audio and Video jobs are reviewable but not publishable. They lack
an independently annotated phone/prototype qualification set, production oral
approval, subject-calibrated video truth, and a production-validated identity/
appearance revision. Acting output is deliberately an unapproved proposal and
the body track is deliberately a foundation preview. The report exposes those
facts as blockers rather than representing the working prototype as a finished
digital-human production system.

The implementation lives in `src/autoanim_gnm/production_readiness.py`; API and
browser wiring live in `src/autoanim_gnm/api.py`; adversarial coverage is in
`tests/test_production_readiness.py`.
