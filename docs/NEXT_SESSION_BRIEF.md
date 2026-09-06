# Resume brief — body-capture lane, written 2026-09-07 at the end of the D9b close-out

**STATE.** D7, D7b, D8, D9, D8b, D8c and **D9b** are merged, rebuilt in place, byte-checked, instrumented and pushed on
`battle0/clean-room-multiview-resolution-invariance`. D9b (2026-09-07): the root-dependent chain (trunk, neck, clavicles with
pass B rerun, arms, hands) is re-solved from the hoisted root after `project_generated_foot_contacts`; every such bone sits on
its ray from the DELIVERED origin to 0.0003 mm (was 3.7–6.3 mm median on the 67 / 22 hoisted frames); root, contacts, legs,
feet byte-identical; the tripwire (hoist forced to zero, old src vs new) 8/8 byte-identical. FINDING: the D3 gate's oracle score
(`retarget_cost.score`) aligns each frame on the leg-root midpoint, so it is blind to a root move and now reads arms 2.72 mm
(was 1.17) on a correct fix — the gauge, band untouched; an absolute-frame companion row is in `d9b_hoist_gate.py`. "The Solve
So Far" carries v2–v8 (9.27 MB; v4/v5/v6 players were shrunk to 640 px q42 to pay for v8 — a v9 needs another shrink). Sol is
unavailable: Grok 4.6 via `cursor-agent -p --trust --mode ask --model cursor-grok-4.6-medium` reviews the card and the merge
(records: `docs/reviews/hoist-reaim-grok-*.md`). Reviews: `docs/reviews/hoist-reaim-2026-09-07.md` (§8 the clause table, §9 the
findings, §11 what is open).

**NEXT STEP: D7c**, the pelvis fitted to the rig's own rest offsets instead of SOMA-derived constants. Card first, measured
before it from the delivered bytes and the D3 oracle: the exact oracle reads a ~7° pelvis tilt about the hip line, Spine origin
27 mm off, torso 8–11 mm (that figure is NOT hoist-contaminated — D9b measured the aligned gauge identical pre/post projection),
arms 0.8–1.2 against the 0.5 band (now 2.72 under the aligned gauge — read the ABSOLUTE companion row for D7c's arm claim, and
say which gauge every number is on). Bands: the oracle's trunk and arms return toward exact on the absolute row; the real-take
hoist and the photographs as the bands the candidate cannot optimise; afterwards re-measure the honest root→hip spread, because
D8c's both-hips charge rests on the pelvis landmark being loose. Then D9-legs (aim thighs and shins from their own origin; the
legs' ray miss on hoisted frames is 4.6–9.2 mm, measured at D9b) → instrument debts (re-pin the D3 gate's frozen D2c/D3
references AND its translation-aligned oracle gauge; `--median-from` on `captured_limb_stability`; recheck D8's headline on a
fixed denominator; a two-view DEPTH-stretch fixture; D8c's head-gate log predates its own rebuild; the contact model's own cost
on exact truth, p95 10–14 mm on a third of the frames, needs its own step) → D6 → D5 → D4.

**Protocol, unchanged.** One Opus agent per step in `.claude/worktrees/ladder-<ID>` on branch `ladder/<ID>`, created
from the last commit with `artifacts`, `.cache`, `.venv` symlinked in; the brief pastes the card verbatim plus the
CLAUDE.md gotchas that bite the step; the agent commits one stage at a time (hygiene → instrument first → synthetic →
the src change → delivery and bands → gate + extractor stub + test + review + report frames and mp4) and never runs
`ladder.py`, `status.py`, `post_merge.sh`, never publishes, never pushes, never edits an existing test or a plan
document. Reviewer (Grok in Sol's place) before dispatch and before the merge; verify every code claim against the
source. Coordinator: `status.py set <ID> in_progress` and commit BEFORE cutting the worktree; after the agent: read the
gate lines (never `tail -N`), merge `--no-ff` with a message file (stash `tools/compare/ladder.py` edits and leave the
user's uncommitted `tests/` alone), `tools/compare/post_merge.sh <ID> artifacts/compare/<step>/delivery` (every
instrument, every line logged), wire the extractor into `RUNGS` and its charts into `VISUALS` in `ladder.py`, run
`ladder.py`, `status.py set <ID> done --report ... --note ...`, republish ladder / board / progress to their fixed URLs
(read each live page with the Artifact tool first or the publish is refused), a new version tab with a frame player on
"The Solve So Far" (JPEG frames in a JSON script; the page is at its cap — shrink an older tab first) and the mp4 via
SendUserFile, then commit and push as separate plain commands (`git push fork HEAD:refs/heads/battle0/...`; the
classifier blocks chained pushes). Commit messages end with the Claude co-author trailer and the session link.
**Mind the shell cwd:** a `cd` into the worktree persists across Bash calls in this harness; a later `git push` ran
from the worktree and failed harmlessly — use absolute paths or `cd` back.

**MINE, NOT YOURS:** Lane H and my uncommitted tests (`tests/test_*.py` modified and untracked in `git status`).
