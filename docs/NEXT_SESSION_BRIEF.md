# Resume brief — body-capture lane, written 2026-09-14 after the D9b close-out (paste this into a fresh session)

Resume the AutoAnim body-capture lane. Read, in this order: `docs/LADDER_STATUS.md` (the SessionStart hook prints it),
`docs/LADDER_EXECUTION_PLAN.md` §2 (the cards; the D7 → D9b rows are the shape of a step, D9b's row the newest) and §6,
the CLAUDE.md body-lane section (every standing rule, including the three added after D9b: the D3 gate's oracle score is
translation-aligned; never re-run pass C or pass A's root line after the projection; D8c's head-gate log predates its own
rebuild), the memory file, and `docs/reviews/hoist-reaim-2026-09-07.md` (the last step review; §8 is the clause table, §9 the
findings, §11 what is open). The same brief is committed at `docs/NEXT_SESSION_BRIEF.md`.

**STATE.** D7, D7b, D8, D9, D8b, D8c and D9b are merged, rebuilt in place, byte-checked, instrumented and pushed on
`battle0/clean-room-multiview-resolution-invariance` (last commit on the fork: see `git log -1`). D9b (2026-09-07): the
root-dependent chain (trunk frame, neck, clavicles with pass B rerun, arms, hands) is re-solved from the hoisted root after
`project_generated_foot_contacts`; every such bone now sits on its ray from the DELIVERED origin to 0.0003 mm (was 3.7–6.3 mm
median on the 67 / 22 hoisted frames, 18 mm at worst); root, contacts, hips, legs, feet and toes are byte-identical to D8c and
so is every joint on the 83 / 128 unhoisted frames; the refactor tripwire (hoist forced to zero, old src vs new) read 8/8
byte-identical. The delivered skeleton sits on the captured joints at hips 5 mm, neck 21 / 18 (the trunk-length floor), and the
arms and shoulders at their aim floors on every frame. "The Solve So Far" carries v2–v8 and is AT ITS CAP (9.27 of ~9.5 MB): a
v9 player needs an older tab shrunk or dropped first (v2 is at q60; v4/v5/v6 are at 640 px q42 already). The bear experiments
stay PARKED. Full test suite on 2026-09-07: 1272 passed; the 4 failures are all in the user's own uncommitted test files.

**REVIEWER OF RECORD: Astra GPT6**, in the seat Sol held — reviews the card before dispatch and the merge before it lands.
The brief shape is the one under `docs/reviews/hoist-reaim-grok-review-2026-09-07.md` and
`docs/reviews/hoist-reaim-grok-merge-review-2026-09-07.md`: state, code excerpts, the measurement, the card verbatim, numbered
adversarial questions; verify every code claim against the source before adopting it; record the review under
`docs/reviews/<step>-astra-review-<date>.md` with a header saying what each finding changed. If you do not know how to reach
Astra from this session, ask the user for the invocation before writing the card; fallback if Astra is unreachable: Cursor's
Grok 4.6 (`cursor-agent -p --trust --mode ask --model cursor-grok-4.6-medium "$(cat brief.md)" > review.md`, ~5 min).

**NEXT STEP: D7c — the pelvis fitted to the rig's own rest offsets instead of SOMA-derived constants.** Card first, then the
agent. The finding, from the D3 gate's exact-skeleton oracle (six synthetic bodies, exact truth): the rest pelvis D7 fits is
built from SOMA-77 proportions, not from the rig the character is built on, so on an exact rig the pelvis reads a ~7° tilt about
the hip line, the Spine origin 27 mm off, and the torso 8–11 mm — invisible to the leg roots, visible above them. D9b measured
that these torso figures are NOT hoist-contaminated (the aligned gauge read identical before and after the projection). The
card must: (1) measure first, from the oracle bodies AND the delivered files' own bytes: the pelvis tilt about the hip line
against the exact truth per seed, the Spine-origin miss, and on the take the pelvis frame's angle to the captured pelvis
landmarks and the root→hip spread on the honest mask (D8c's both-hips charge rests on that spread being +25 % loose);
(2) name the mechanism — which constants in the pelvis fit are SOMA-derived (`d7_pelvis_frame_gate.py` and the converter's
rest-pelvis construction) and what replaces them from `rest_translations_m` alone, with NO new constant; (3) bands: the oracle's
trunk and pelvis tilt return toward exact ON THE ABSOLUTE-FRAME ROW of `tools/compare/d9b_hoist_gate.py` (say which gauge every
number is on: the D3 gate's `retarget_cost.score` is leg-root-aligned and reads arms 2.72 mm since D9b by that gauge; its band
is a standing fail and is not moved); the real-take photographs (part-wise silhouette, TORSO+LEGS and ARMS, whole take and the
bent tercile, both performers, not worse, CI clear on the D7b/D8 predicate) as the band the candidate cannot optimise; the
hoist per frame before and after (report; a pelvis change moves the root through `_leg_root_offset`, so subtract each
frame's hoist before reading a root move); `delivered_vs_capture.py` D9b vs the candidate on identical draws (landmarks
byte-identical, so the same-denominator clause is an expected PASS; neck, hips, knees, ankles reported with their floors);
(4) pre-register what may move: everything below the root on every frame (a pelvis frame change is whole-take), so bit-identity
is NOT a legitimate clause here — state that up front; (5) must-fails: the D9b build itself, a pelvis frozen upright (D7's
control, which the photographs disqualified), and a fit that reproduces D7's constants; (6) afterwards re-measure the honest
root→hip spread and say whether D8c's both-hips convention still stands. Window 0. No MAMMA-referenced selection.

**THEN, IN ORDER:** D9-legs (aim thighs and shins from their own origin; the legs' ray miss on hoisted frames is 4.6–9.2 mm,
knees 1.7–2.5 mm from the pre-hoist origin, measured at D9b; exact on the oracle) → instrument debts (re-pin the D3 gate's frozen
D2c/D3 references AND replace its translation-aligned oracle gauge with the absolute-frame one; `--median-from` on
`captured_limb_stability`; recheck D8's 27 → 18 / 22 → 4 headline on a fixed denominator; a fixture that injects a two-view
DEPTH stretch; the contact model's own cost on exact truth, p95 10–14 mm on a third of the frames, needs its own step; D8c's
head-gate log predates its own rebuild) → D6 (binding, mesh-distortion instrument first) → D5 (bone lengths and the flexible
spine) → D4 (momentum on MHR).

**PROTOCOL, UNCHANGED.** One Opus agent per step in `.claude/worktrees/ladder-<ID>` on branch `ladder/<ID>`, cut from the last
commit with `artifacts`, `.cache` and `.venv` symlinked in; the brief pastes the card verbatim plus the CLAUDE.md gotchas that
bite the step; the agent commits one stage at a time (hygiene → instrument first → synthetic/oracle → the src change → delivery
and bands → gate + extractor stub + new test + review + report frames and mp4) and never runs `ladder.py`, `status.py` or
`post_merge.sh`, never publishes, pushes, or edits an existing test or plan document. Reviewer before dispatch and before the
merge. Coordinator: `status.py set <ID> in_progress` and commit BEFORE cutting the worktree; after the agent: read every gate
line (never `tail -N`), merge `--no-ff` with a message file (stash `tools/compare/ladder.py` edits; leave the user's uncommitted
tests alone), `tools/compare/post_merge.sh <ID> artifacts/compare/<step>/delivery`, wire the extractor into `RUNGS` and its
charts into `VISUALS` in `ladder.py`, run `ladder.py`, `status.py set <ID> done --report ... --note ...`, republish ladder /
board / progress to their fixed URLs (read each live page with the Artifact tool first or the publish is refused), a new
version tab with a frame player on "The Solve So Far" (shrink an older tab first), the mp4 via SendUserFile, the step's
standing rules into the CLAUDE.md body-lane section, then commit and push as separate plain commands
(`git push fork HEAD:refs/heads/battle0/clean-room-multiview-resolution-invariance`). Mind the shell cwd: a `cd` into the
worktree persists across Bash calls; use absolute paths. A full `pytest tests/` must `--ignore` the user's seven untracked
worker test files (audio/video acting shot, gem_x modal and timing, gesturelsm modal and worker, mamma modal) or it stops at
collection.

**MINE, NOT YOURS:** Lane H and the user's uncommitted tests (the modified and untracked `tests/test_*.py` in `git status`).
