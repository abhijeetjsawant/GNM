# AutoAnim

## Headless DCC
- Blender 4.2: `/Applications/Blender.app/Contents/MacOS/Blender --background --python SCRIPT -- ARGS`
- Maya 2025: `/Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy SCRIPT ARGS` — `mayaHIK` and `fbxmaya` load headless; a missing `stereoCamera` plug-in is harmless noise
- SMPL-X model lives at `.cache/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz`; `np.load` needs `allow_pickle=True` (object arrays)

## Gotchas
- Blender FBX exports carry −90° X and 100× scale on the root. Blender undoes it on import, so round-trips hide it; Maya reads the character lying along −Z. `bake_space_transform=True` bakes the mesh but not the armature — it makes things worse.
- Normalise the rows before comparing world orientation matrices. An unnormalised 100×-scaled matrix drives `acos` past its clamp and every joint reads "0.00° apart".
- To move a skinned character, offset the root joint's animation. Parenting it under a group and moving the group shifts the mesh twice and separates it from its skeleton.
- Maya's FBX plug-in ignores `file(namespace=...)`; call `cmds.namespace(set=...)` before importing.
- HumanIK headless: `hikCharacterLock(char, 1, 1)` before `hikSetCharacterInput`, and set retarget properties *after* it — `hikSetCharacterInput` resets them. A generated skeleton arrives unlocked.
- SMPL-X's `v_template` is mirror-symmetric but `J_regressor` is not; joints regressed from it are asymmetric by up to 3 cm.
- `nose` in the SOMA-77 adapter is **index 6, the `Head` skeletal joint** — a point inside
  the skull, not a surface nose. It sits *behind* the eyes. Anything inferring facing
  direction from it is wrong. SOMA-77 has no ears; `left_ear`/`right_ear` are schema-only
  and populated on zero frames.
- `artifacts/` is gitignored. Scripts must regenerate everything under it.
- momentum writes glTF keyframe times in **seconds**; Blender's importer converts them
  with the *scene* fps. A 30 fps motion imported into the default 24 fps scene runs 25%
  fast and freezes after the last converted frame. Set `scene.render.fps` before importing.
  The symptom is an overlay that drifts off the performer as the shot runs — it reads as a
  placement error and is a timebase error.

## Body capture lane
- **RESUME PROTOCOL — read `docs/LADDER_STATUS.md` first in any body-lane session.** It is
  generated from `docs/ladder-status.json` (the one source of truth for step state) and a
  SessionStart hook in `.claude/settings.json` prints it. Never hand-edit either generated
  view; move state only through `tools/compare/status.py` (`set`, `log`, `decide`, `render`).
  The two plan documents stay hand-edited: `docs/LADDER_EXECUTION_PLAN.md` (what gets built,
  in what order, gated by what) and `docs/SUBSTITUTION_LADDER.md` (what is measured and how).
  **End of every step, in this order:** `status.py set <ID> done --report <path> --note "..."`
  (every `set`/`log`/`decide` rewrites both views; `render` is only for hand-edited JSON)
  → `.venv/bin/python tools/compare/ladder.py` → republish the three
  pages to their existing URLs (ladder, board, and the progress page for the non-technical
  reader) → commit the step's files together. Worktree agents branch from the last commit,
  so uncommitted tooling is invisible to them. Pages: ladder
  <https://claude.ai/code/artifact/56361ab8-b5a0-456d-9171-4d6a09d6c132>, board
  <https://claude.ai/code/artifact/cf83ef29-a4b7-4afd-9031-0918e8eb6f35>, progress
  <https://claude.ai/code/artifact/abd3a70c-4c51-4251-8b2f-344f095998c6>.
- **Head, feet and hands: `docs/HEAD_FEET_HANDS_PLAN.md`.** All three are *input*
  problems, not solver problems — the adapter maps 17 of SOMA-77's 77 joints and drops
  every finger, toe and `HeadEnd`. Read it before touching any of those three regions.
- **Start at `docs/BODY_LANE_PLAN.md`.** It is the plan of record: what is measured,
  what has been withdrawn, what is open, and the build sequence with its bands. Read
  sections 0–2 before doing anything in this lane.
- **Measuring against MAMMA goes through the substitution ladder: `docs/SUBSTITUTION_LADDER.md`.**
  One part at a time, MAMMA's retained output supplying the rest, one instrument per rung
  with its blindness beside it. `tools/compare/ladder.py` reads only JSON reports under
  `artifacts/`, appends `docs/ladder-history.jsonl` when a figure changes, and renders
  `docs/substitution-ladder.html`, published at
  <https://claude.ai/code/artifact/56361ab8-b5a0-456d-9171-4d6a09d6c132> (republish with that
  `url`). Run it after any instrument writes a report; a number
  that lives only in a document is *instrument missing*, not a measurement.
  **Execution order and gates: `docs/LADDER_EXECUTION_PLAN.md`** — three lanes (hardware,
  instruments, delivery), one part per step, the MAMMA arm reports and never selects a
  constant. `THORAX_SMOOTHING_FRAMES` was selected on the MAMMA oracle arm and is an open
  leak (I8) until re-selected from synthetic or owned data.
- **Every part carries a picture, and the picture says which way is good.** Each rung on the
  ladder page and each part on the progress page shows a bar chart -- ours beside MAMMA's, with
  an alternative of ours and a deliberately wrong answer where they exist -- labelled in plain
  words *lower is better* or *higher is better*, with one sentence naming the reference. The
  charts are declared in `VISUALS` in `tools/compare/ladder.py` (fig keys, roles ours / mamma /
  alt / control), resolved from the reports on every run, written to `docs/ladder-figures.json`,
  and rendered by `tools/compare/visuals.py` on both pages, so the non-technical page can never
  disagree with the technical one. Bars on one chart share a unit and a reference; bars on
  different charts never do. **When an instrument adds or changes a figure, add or update its
  comparison in `VISUALS` in the same pass** -- the ladder prints `NO VISUAL` for a rung with
  figures and no chart. Roles use the validated palette in `visuals.py` (blue ours, orange MAMMA,
  aqua alternative, hatched control); do not repaint them.
- **Keep the parity board current: `docs/parity-board.html`**, published at
  <https://claude.ai/code/artifact/cf83ef29-a4b7-4afd-9031-0918e8eb6f35>. It is the
  one-page view of where each pipeline stage stands — measured and holding,
  measured with a known problem, unmeasured for want of an instrument, or not built.
  **Update it in the same pass as the plan**, whenever a measurement changes a
  stage's status, closes a question, or moves the target. Edit the file and
  republish it with the Artifact tool, passing that URL as `url` so the link stays
  stable; publishing without it creates a second board and the link people hold
  goes stale. Two rules the board itself must keep: **every stage carries what its
  instrument is blind to**, and **numbers from different references never share an
  axis** — the detector figures and the retarget figures are not comparable.
- Two standing rules, and they are not optional. **No gate a constant can pass** —
  every acceptance band ships with a demonstration that a degenerate solution fails
  it. **Same denominator** — score both arms of a comparison on the same population.
- The recurring defect here is a *correct measurement carrying a claim it does not
  support*. Before writing a conclusion, ask what the instrument is structurally
  blind to.
- `_epipolar_distance_px` returns the **symmetric** distance — the sum of two
  one-sided distances, ratio measured at 1.962. Halve it before fitting a per-view
  sigma to it.
- `gt_joints` and `gt_vertices` in MAMMA's retained output are **byte-copies of
  `pred_*`**. There is no ground truth on disk; never score against a `gt_` variable.
- **MAMMA's subject indices are not ours.** On the four-camera fixture `body_id-00` is
  our subject **1**. Both sides are two-element lists indexed 0 and 1 and nothing
  declares the order, so pairing by index silently crosses the performers — and it is
  invisible in every per-subject statistic taken *separately*, corrupting only the
  comparisons. Resolve it from 3D pelvis agreement (MAMMA's world frame **is** the
  camera-rig world frame): `tools/head/subject_map.py`.
- **Cross-view consistency is a within-frame property; rigidity is a between-frame one,
  and neither implies the other.** Apple Vision's ears are the *most* epipolar-consistent
  landmarks measured in this lane (0.29–0.76x the body control) and fit a rigid skull
  *worst* (1.3–1.6x the residual of SOMA-77's head joints). A surface landmark slides
  over the bone as the subject turns: every camera sees the same apparent point, and it
  is not a fixed place on the skeleton. This is `soma77_pose.py`'s founding argument, and
  a rigid-template fit over a take is the instrument that exposes it.
- **What fixed the head was removing freedom and repairing instruments — never adding
  capacity.** Every idea that gave the model more to work with failed (extra landmarks
  from a second detector, a support-conditioned prior, per-landmark inverse-variance
  weighting, a frame-quality flag). Every constraint drawn from anatomy paid (fit instead
  of triangulate, smooth the neck rather than the world head, anchor the skull to the
  neck), and so did every instrument repair (the crossed subject map, the gate's own
  thorax frame, the weight-selection rule). **When a fit is short, suspect the instrument
  and the free parameters before reaching for more input.**
- **A band the solver regularises is a knob setting, not evidence.** The head fit's
  temporal prior acts on exactly the quantity the gate's jitter band measures, in the
  gate's own reference frame — so that band, and the ceiling side of the spread band,
  discriminate only the *controls*. When designing a gate, ask of every band: *can the
  candidate optimise this directly?* If yes it proves nothing about the candidate, and the
  claim rests on whatever remains.
- **Whole-take medians on one 150-frame take are not robust passes.** Per-frame agreement
  here has lag-1 autocorrelation 0.99, so ordinary resampling is invalid; a moving-block
  bootstrap put a 7.54° median against an 8° band at P(fail) = 0.48. Quote a margin only
  with a block bootstrap behind it, and pair it against the oracle on identical draws so
  the fragility is attributed to the right term.
- **A tracking gate is blind to absolute orientation, and delivery is not.** A rigid fit
  determines pose only up to a constant, and a gate that mean-removes each take scores
  *tracking* — so it cannot see an 80–176° constant offset that would ship a head pointing
  sideways, smoothly, on every frame. Fix the zero from the subject's own anatomy, never
  from a reference: aligning to a research fitter's answer is a shipped constant fitted on
  a reference-derived artifact.
- **A fit criterion made of pixels is blind to what a body can do.** Minimum reprojection
  accepted a 140° single-frame head flip, because much of the flip lay along the viewing
  rays — the same blindness that makes head depth ambiguous in the first place. Add a hard
  physical reject (here 60°/frame of *neck* travel, against a human peak near 500–800°/s)
  and measure it at the joint, not in world: a head on a turning body travels in world
  with the neck perfectly still.
- **A gate needs an ORACLE arm, not just failing controls.** "No gate a constant can
  pass" has a dual: *a gate no oracle can pass is miscalibrated*. Score the reference's
  own answer through your pipeline's frames — it measures the floor your frame
  definitions impose, and it tells you whether a candidate's gap is yours or the
  instrument's. On the head it did both, and it also **condemned an earlier version of
  the gate's own reference frame**, which no control could have.
- **An oracle behaving strangely means audit the instrument, not theorise.** The head
  gate once showed the oracle degrading under a frame filter, and that got written up as
  a finding about facing direction. It was a defect in the gate's own gauge estimation —
  the mean was estimated over frames that were not scored. The oracle is the one arm
  whose true value you can predict, so **when it surprises you, the surprise is almost
  always yours.**
- **A length invariant cannot score direction.** For an axis roughly perpendicular to the
  camera's depth direction, a differential depth error between its endpoints is *first
  order* in the axis's direction and *second order* in its length — at 160 mm, a 50 mm
  differential is 17° of yaw and 8 mm of length. A segment can hold its length to 11 % and
  be useless as an orientation. The companion to "reprojection cannot score depth".
- **Wrap the pipeline to instrument it; never re-implement it.** A careful
  hand-replication of `reconstruct_multiview`'s association loop — same associator, same
  gates, written from the source — drifted 9–19 mm from the retained tracks. Wrapping the
  real function with a recording associator reproduces them at 0.0 mm.
- SAM 3D Body is CPU-only here — float64 inside its TorchScript MHR module, out of
  reach of any shim. It also hardcodes `.cuda()`; the shims live in our worker, never
  in the vendored checkout, whose licence forbids reverse engineering.
- MAMMA is a **measuring instrument, never in the shipping path**. Nothing it
  produces may enter a delivered artifact, trained weights, or a shipped calibration
  constant.

## Verification
- Each exporter writes a JSON report beside its output with input SHAs and gate results. Check the SHA chain rather than assuming a build used current inputs.
- Confirm a suspected defect with a second, independent measurement before acting. Several "defects" this session were artefacts of the metric, not the rig.
