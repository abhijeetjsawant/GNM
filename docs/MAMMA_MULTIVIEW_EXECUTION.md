# MAMMA multiview execution and verification

## Outcome

AutoAnim now has a pinned Modal execution surface for the official five-stage
MAMMA body pipeline and a strict importer from MAMMA's per-person SMPL-X output
to AutoAnim's detailed 55-joint runtime. Google GNM remains the sole owner of
the face, jaw, eyes, and oral animation; MAMMA owns body and finger motion.

The real four-camera GPU smoke run completed successfully on 2026-08-04. It
fitted two people across frames `[60, 90)`, generated both calibrated camera
overlays, and produced two finite 30-frame SMPL-X parameter archives which were
converted to canonical AutoAnim-55 body tracks and exported as animated GLBs.
The captured output lives at
`artifacts/mamma/mamma-4cam-smoke-v3/` (344 MB, excluded from version control).

## Pinned inputs

- MAMMA Git commit: `588492f18876e2ed6888b2d26929047cb6b575e7`
- CUDA base: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Python: 3.11
- PyTorch: 2.5.1+cu124
- Torchvision: 0.20.1+cu124
- GPU order: L40S, then A100 40 GB
- Example cameras: `A001`, `B001`, `C001`, `D001`
- Smoke frame interval: `[60, 90)`

The local asset audit resolves 12 required files totaling 3,128,074,503 bytes:
the MAMMA 2D checkpoint, SAM2, YOLO, downsampled SMPL-X data, neutral/male/female
locked-head SMPL-X models, and all four example MP4 files. Credentials are not
uploaded to Modal; only the already-downloaded assets are copied into a private
Modal volume.

## Runtime path

`workers/mamma/modal_app.py` executes the official stages in order:

1. `run_ma_cap.py` extracts the exact four-camera frame interval and retains
   calibration metadata.
2. `run_ma_masks.py` assigns two people across views using YOLO plus SAM2.
3. `run_ma_2d.py` predicts dense body/hand landmarks using the released MAMMA
   checkpoint.
4. `run_ma_3d.py` performs multiview SMPL-X fitting and writes one parameter and
   one vertices/joints NPZ per person.
5. `run_ma_vis.py` produces Rerun and two-camera overlay evidence.

Every stage captures stdout/stderr, fails closed on a nonzero exit, and the run
report hashes every produced artifact. After download,
`_convert_downloaded_body_tracks` locates every
`smplx_params_body_id-XX.npz` and emits the corresponding canonical
`*.autoanim-body-track.json`.

## Retarget contract

The MAMMA pose array is the official 165-value SMPL-X order:

`global(3) + body(63) + jaw(3) + left-eye(3) + right-eye(3) + left-hand(45) + right-hand(45)`

The importer:

- requires the exact seven-array MAMMA NPZ schema;
- rejects duplicate, encrypted, compressed, oversized, nonfinite, or malformed
  archives before using their arrays;
- reshapes the pose to `[frame, 55, 3]` and converts axis-angle values to
  continuous normalized XYZW quaternions;
- zeros MAMMA jaw and eye slots before retargeting so they cannot override GNM;
- converts the shipped/example MAMMA Z-up capture world to AutoAnim +Y-up;
- applies the existing reviewed SMPL-X anatomical reflection and world-delta
  retarget to AutoAnim-55;
- preserves all 30 finger joints;
- leaves hard foot-contact flags off until a target-space contact solve proves
  stationary feet.

## Verification completed

- 165 focused face/body/provider/compositor/export tests pass.
- 102 deterministic tests reclaimed from MediaPipe-heavy files pass; only the
  six named tests that instantiate the crashing native graph were excluded.
- 259 later-stage production, oral, body, viewer, and export tests pass.
- 30 focused MAMMA/SMPL-X contract tests pass after importer integration.
- 20 Audio2Face adapter/parity tests pass, including a real LibriSpeech API/CLI
  end-to-end parity run; one optional Claire-asset test skips by design.
- The blank-image API path now returns typed `FACE_NOT_FOUND` before native graph
  construction.
- `compileall` and `git diff --check` pass for the changed execution surface.
- The MAMMA Modal probe passed on an NVIDIA A100-SXM4-40GB with CUDA 12.4,
  PyTorch `2.5.1+cu124`, and the exact pinned source revision.
- The private Modal volume upload completed: 12 audited inputs totaling
  3,128,074,503 bytes; no credentials were uploaded.
- `mamma-4cam-smoke-v3` completed every official stage with return code zero:
  capture 0.91s, masks 206.04s, 2D landmarks 182.20s, 3D fit 315.83s, and
  visualization 26.20s.
- The run produced two strict-schema `[30,165]` pose archives, finite numeric
  arrays, two canonical 55-joint/30 Hz AutoAnim tracks, two valid animated GLBs,
  and A001/B001 overlays plus the MAMMA preview.
- Visual inspection of two instants from A001 and B001 shows stable limb/root
  axes and coherent bend-to-stand motion; no detached arm, palm, or gross axis
  inversion is visible in the wide shot.
- The downloaded-track converter works both from the project environment and
  from Modal's lightweight CLI environment by delegating local conversion to
  the repository virtual environment. Focused MAMMA/body/export tests pass
  (31 tests in the final combined run).

Some groups overlap; these counts are command-level evidence, not a summed
unique test total.

## Host-runtime blockers found honestly

1. MediaPipe 0.10.35 on this headless macOS process aborts in native code. Its
   face graph creates `DrishtiMetalHelper` even when the inference delegate is
   explicitly CPU; NSGL pixel-format creation fails and the graph terminates on
   `Service is unavailable`. Python cannot catch a native abort.
2. The native Audio2Face runner aborts while MLX initializes its Metal device
   (`NSRangeException: index 0 beyond bounds for empty array`). Auto/fallback
   application paths remain usable, but tests that explicitly require the
   learned MLX backend cannot pass in this host process.
3. This smoke test is a wide, full-body shot. It demonstrates robust body
   reconstruction but cannot establish close-up individual-finger contact or
   face likeness. Those require close hand cameras / additional views, and the
   separate GNM facial performance pipeline respectively.

These are recorded as blockers, not converted into passing mocks.

## Exact continuation commands

Run these from the repository root after external-command execution is allowed:

```bash
MODAL_PROFILE=ragnarok-space modal run workers/mamma/modal_app.py::probe_entrypoint
MODAL_PROFILE=ragnarok-space modal run workers/mamma/modal_app.py::upload_assets_entrypoint
MODAL_PROFILE=ragnarok-space modal run workers/mamma/modal_app.py::run_entrypoint \
  --run-id mamma-4cam-smoke-v4 \
  --output-directory artifacts/mamma \
  --start-frame 60 \
  --end-frame 90
```

The v3 smoke result meets the stage/status, two-person, finite-array, AutoAnim
track, wide-shot overlay, no-axis-inversion, and animated-body-export gates.
It does not claim close-up palm/finger fidelity or a fully assembled GNM head;
those remain explicit next validation gates.
