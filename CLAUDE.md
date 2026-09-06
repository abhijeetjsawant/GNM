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
- **A lane can be self-consistent and mirrored at once.** Until 2026-09-02 the rig's `Left`
  bones sat on the mesh's anatomical right (born in `DEFAULT_MPFB_JOINT_MAP`, copied into the
  asset and `DETAILED_HUMANOID`, and independently into `CANONICAL_HEAD_AXES`), and
  `soma_motion._DELTA_MAPPING` compensated for it, so that lane shipped correct output from
  two mirrored halves. Every joint-by-name score passed. What caught it was the handedness
  triple product read two ways (from the feet and from the torso) on one rig: −1 and +1,
  which no rotation can produce. The skeleton's own published contract (`right, +Y up,
  +Z forward` ⇒ left = +X) was the shortest proof and is now a unit test.
- **A sign that measures where bones ARE cannot see what they are DOING.** The triple product
  read −1 on SOMA motion exports under the correct mapping, the swapped mapping and SOMA's
  own joints alike, because a rotation retarget puts every bone on its own side whatever
  drives it; only the distance to the joint each bone should follow discriminated (92 vs
  540 mm). The companion to "a length invariant cannot score direction".
- **Relabel, never reflect.** Fixing a left/right naming mirror by negating vertex X and
  swapping weights is a reflection: winding reverses, normals invert, signed volume changes
  sign, and no joint, forward-dot or silhouette gate can see it. Swap the names (rest X,
  joint map, asset node names) and prove it with a signed-volume test. And a curl about a
  finger's own bone axis displaces nothing: the delivered "rest curl" is inert.
- **A direction measured from the rig's own origin sees root placement; one measured from
  a landmark anchor is blind to it.** The clavicle's 36–47 mm "converter cost" was never the
  clavicle: the old anchor lived in the landmark frame, the fix aimed from the rig's shoulder
  origin, and the first rig-frame direction ever measured exposed that `Hips` sat on the
  hip-landmark midpoint with the hip joints 80 mm below. Root placement is now derived from
  the skeleton (`_leg_root_offset`); the round trip closes at ~0.5 mm. And the rig has ONE
  frame for the whole trunk, so on a bent performer that offset rides the lean; a pelvis
  frame (SOMA-77 index 0 `Hips`, `Spine1`, `Spine2` are unmapped) is its own pose step after D3
  (D3 shipped the per-performer skeleton on 2026-09-03 and left the pelvis frame to that step).
- **The round trip cannot score a temporal step.** It rebuilds its torso frame from the
  upper-arm origins that D2c's clavicle reject moves, so its second pass rejects frames its
  first pass accepted. Score temporal treatments on synthetic truth (I7's fixture with the
  detector's measured heavy-tail noise); report frames-over-ceiling, never band it, because
  a reject zeroes it by construction. A per-frame temporal rule needs a REACHABILITY test
  (accept only what the joint can reach from the last accepted frame at a physical rate),
  not a step test, which accepts the wrong plateau between two big steps; and after
  replacing a parent, RE-SOLVE the chains below it or the elbow and wrist leave their landmarks.
- **Joint instruments and the photograph instrument can disagree, and both be right.** D2
  placed the skeleton better by every joint measure (ours and MAMMA's) and the mesh bound to
  it overlapped the SAM2 masks worse in 8 of 8 cells. Three explanations were pre-registered
  and refuted in turn; what survived was the root's own share (exact by translating the same
  rendered mesh) and a larger share from the clavicle chain through skin-weight bleed at the
  shoulder cap. The silhouette scores the MESH; it is not a limb-placement gate, and a
  folded-arm control does not realise "a limb hidden inside the outline" under four cameras.
- **Blender-only instruments run under Blender:** `facing_surface_probe.py` imports `bpy`
  (`Blender --background --python tools/compare/facing_surface_probe.py -- OUT.json GLB_DIR`).
  And a waiter loop `until ! pgrep -f NAME` matches its own command line; use `[N]AME`.
- **The delivered file must be read back from its own bytes; a code-path instrument cannot see
  what the exporter wrote.** Until D3 (2026-09-03) every joint instrument scored
  `forward_kinematics_positions` on `DETAILED_HUMANOID` while `export_animated_body_glb` wrote the
  MPFB asset's own rest into the GLB: two skeletons, 81–195 mm apart, and only the silhouette
  (which renders the file) could see it. A per-performer rest now rides on the `BodyTrack`
  (`rest_translations_m`; `skeleton_for_track`, `skeleton_for_track_dict`), and the closure band
  in `tools/compare/d3_skeleton_gate.py` parses the delivered GLB and forward-kinematics it against
  the track's own rest. `skeleton_for_joint_names(track.joint_names)` is the defect pattern: it
  hands back the canonical body whatever the track carries. `tools/head/sized_skeleton.py`
  re-exports the shipped sizing, so D2's frozen gates no longer reproduce their SIZED arms.
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

- **Close-out protocol (2026-09-06):** `tools/compare/post_merge.sh <ID> <branch-delivery-dir>` archives, rebuilds in
  place, byte-checks the eight delivered files and reruns every instrument with EVERY line logged under
  `artifacts/compare/post-merge-<ID>/`. A `tail -4` hid a failing oracle for two close-outs (the D3 gate's
  exact-skeleton legs read 12.8 mm behind the verdict line from D7b to D9). Read the gate lines, not the verdict.
- **The D3 gate's exact-skeleton oracle feeds the converter the truth's SPINE and TOES** (as the delivery does);
  without them it exercises the legacy trunk-line hips frame the delivery left at D7. Its "canonical unchanged" and
  "round trip = D2c" clauses compare against references frozen at D2c/D3 that later steps moved by design: re-pin owed.
  Its residual torso (8–11 mm) is D7's SOMA-derived rest-pelvis constants tilting the pelvis ~7° about the hip line
  on an exact rig — invisible to the leg roots, visible above them ("D7c": fit the pelvis to the rig's own rest).
- **`captured_limb_stability.py` recomputes the performer's median per build** — a moving denominator; compare builds
  with a fixed reference (`--median-from`, owed). D8's 27→18 / 22→4 headline needs that recheck.
- **A step's fixture can fail a pre-registered clause for the fixture's own defects** (D8b: honest-frame noise 9× the
  take, a static collapse). Measure the discrepancy, repair the FIXTURE as a fixture parameter with src byte-identical
  across the repair, rerun the same clauses; never merge on an override and never move a band.
- **SAM 3D Body runs natively on Modal** (`workers/sam3d_body/modal_app.py`, L40S, 0.5–0.9 s/frame; CUDA has the
  float64 its MHR module needs; GPU/CPU parity 1.5 mm). The Mac CPU path is for development only. Our shipped
  detection is CPU ONNX and numpy: a bigger GPU changes detection time, not detections (resolution sweep 1280/2560/3840
  did not move the shoulder collapse).
- **Synthetic-camera rules (bear experiments, 2026-09-06):** cross-view consistency is NOT evidence when the views
  descend from one observation — the correspondence-scramble control exposes it (synthetic pairs discriminate 1.1×,
  real pairs 14–16×); score a synthetic set against a REAL held-out camera only. MAMMA's shape parameters swing
  0.03–47 on real footage with the camera set alone and are not a verdict. For a single calibrated camera the
  bottleneck is metric depth (MoGe-2 / VDA fail a floor gate by 0.5–1.5× in scale; TAPIP3D tracking is near-perfect).
  Four SEPARATE video-model generations of one prompt are not one scene; a multi-view solver "failing" on them is
  detecting that. Workers: `workers/{bullettime,sv4d,videodepth,prompthmr,mocapanything}`; page
  <https://claude.ai/code/artifact/0cc6e2fc-c544-4514-a80d-68dd99855b4e>. Parked 2026-09-06.
- **The delivered file is whole-take-coupled (D8c, 2026-09-06).** D3 sizes the per-performer rest over the whole take, so a
  change to 18 frames of landmarks moved 16 of 55 rest bones (< 1 mm) and every frame's joints with them (54 frames at 1–2 mm,
  5–41 frames from any fire). Pre-register locality at the CAPTURE stage (there it held exactly: 0 frames outside the
  Savitzky–Golay ±4 envelope); never predict a DELIVERED change to be local, and subtract each frame's hoist before reading a
  root move. The continuity horizon for a demoted slot is the smoother's window (`SMOOTHING_WINDOW_FRAMES` = 9), not the
  6-frame gap clause (that is the hold for slots the solve never saw).
- **MAMMA cannot referee a width.** Its hip line is a constant 117.6 / 114.8 mm on every frame (a rigid SMPL-X pelvis) against
  this performer's 215, so a collapsed hip line scored CLOSER to it and the repair scored worse. Its joints are conventions;
  a length comparison against them says which convention you are nearer, not which is right.
- **Sol unavailable → Cursor's Grok 4.6 in its place** (2026-09-06, D8c card and merge): `cursor-agent -p --trust --mode ask
  --model cursor-grok-4.6-medium "$(cat brief.md)" > review.md` (~5 min; without `--trust` it exits 1 asking for workspace
  trust). Same brief shape (state, code excerpts, the measurement, the card, numbered adversarial questions); verify every
  code claim against the source before adopting it. Records under `docs/reviews/*-grok-*.md`.
- **"The Solve So Far" is at the page cap** (9.30 of ~9.5 MB after v7): a v8 player needs an older tab's frames shrunk or
  dropped first. v2's JPEGs were already recompressed to q60 to pay for v7.
- **Report pages:** one version tab per update (never a stacked section), frame PLAYERS (JPEG frames in a JSON
  script + play/pause/scrub/step) not animated images — the viewer blocks `<video>` from data: and blob: URLs —
  and keep a page under ~9.5 MB (10.6 MB froze the renderer). Full-rate mp4s go to the user via SendUserFile.
  "The Solve So Far": <https://claude.ai/code/artifact/9fc29718-f55d-478a-b0e7-6f59ee770e70> (v2–v6).
- **zsh does not word-split an unquoted `$VAR`**: pass file lists as `${=VAR}` / an array, or use a glob.

## Verification
- Each exporter writes a JSON report beside its output with input SHAs and gate results. Check the SHA chain rather than assuming a build used current inputs.
- Confirm a suspected defect with a second, independent measurement before acting. Several "defects" this session were artefacts of the metric, not the rig.
- **The D3 gate's oracle score is translation-aligned (D9b, 2026-09-07).** `retarget_cost.score` subtracts the leg-root midpoint
  per frame, so the exact-skeleton oracle cannot see a root move: it read identical before and after the foot-contact projection
  on all six bodies, and it reads the CORRECT re-aim WORSE (arms 1.17 → 2.72 mm since D9b; legs 0.05–0.07 unchanged). That is the
  gauge; the band is untouched. For any arm claim on the oracle read the ABSOLUTE-frame row in `tools/compare/d9b_hoist_gate.py`
  and say which gauge a number is on. Contacts fire on every D3 body (34–67 of 150 frames hoisted): there is no "exact rig without
  ground contact", and the plant's own cost on exact truth (p95 10–14 mm on those frames) is measured and unowned.
- **After the projection, never re-run pass C or pass A's root line.** Pass C rewrites the legs, feet and toes and would overwrite
  `project_generated_foot_contacts`' `candidate_local` foot locks; the root line would wipe the hoist. The converter's quaternion
  hemisphere walk runs BEFORE the projection, so a re-solved local must be signed against the delivered local of its own frame or
  the bits of an unhoisted frame flip. The re-solve runs on every frame; 0.5 mm is a report cut, never control flow. A refactor
  that lifts converter code into a helper is proved by the tripwire: hoist forced to zero, old src vs new, 8/8 byte-identical.
- **D8c's head-gate log predates D8c's own in-place rebuild** (written 19:46, the rebuild 20:42); D9b's close-out head gate is
  line-identical to D8c's close-out log, and the drift the D9b agent measured against the earlier log is D8c's, not D9b's.
