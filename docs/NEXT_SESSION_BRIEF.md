# Resume brief — body-capture lane, written 2026-09-06 at the end of the D8c close-out

Paste this as the opening message of the next session. Same protocol as the 2026-09-06 (D8b) brief with the state
moved forward; everything it references is committed and pushed on `battle0/clean-room-multiview-resolution-invariance`
(last commit: the D8c close-out, after merge `4c1c6e2`).

---

Resume the AutoAnim body-capture lane. Read, in this order: `docs/LADDER_STATUS.md` (the SessionStart hook prints
it), `docs/LADDER_EXECUTION_PLAN.md` §2 (the cards; the D7 → D8c rows are the shape of a step, D8c's row is the
newest) and §6 (the queue after D8c), the `CLAUDE.md` body-lane section (every standing rule, including the four added
on 2026-09-06 after D8c), the memory file, and `docs/reviews/hip-line-2026-09-06.md` (the last review; §7 and §8 are
what a review must carry, §9–§11 the findings and what is open).

**State.** D7, D7b, D8, D9, D8b and D8c are merged, rebuilt in place, byte-checked, instrumented and pushed. The
delivered skeleton sits on the captured joints at the hips (5 mm), neck (21 / 18 mm, the trunk-length floor), elbows
(7 / 9 mm) and wrists (9–12 mm); the captured shoulder line and hip line no longer collapse on the falling performer
(18 → 0 and 23 → 0 frames off his own width, on fixed references). The report page "The Solve So Far" carries v2–v7 as
version tabs with frame players and is AT ITS SIZE CAP (9.30 of ~9.5 MB): a v8 player needs an older tab's frames
shrunk or dropped first. The bear side experiments are PARKED; do not reopen them unless asked. **Sol is unavailable;
use Cursor's Grok 4.6 in its place** for the card review and the merge review (`cursor-agent -p --trust --mode ask
--model cursor-grok-4.6-medium "$(cat brief.md)"`, ~5 min; the command and the brief shape are in CLAUDE.md; the two
D8c reviews under `docs/reviews/hip-line-grok-*.md` are the pattern). If Sol is back, use Sol.

**Next step: the foot-contact hoist re-aim.** Card first, then the agent. The finding, measured at D9 and confirmed at
D8c: `positions_to_body_track` ends by calling `project_generated_foot_contacts` (`src/autoanim_gnm/body_projection.py`,
called at the end of `positions_to_body_track` in `commercial_multiview.py` with `maximum_root_correction_m=0.08`,
`ground_band_m=0.08`, `velocity_threshold_m_per_s=0.30`, `minimum_contact_frames=2`), which TRANSLATES THE ROOT after
every aim pass has been taken. A translation leaves every rotation alone, so every delivered bone still points along
`target − origin_before_the_hoist` while its origin has moved: the same defect class as D7b's neck and D9's arms, one
operation later, present since D2b. Measured from the delivered files: the hoist is recovered from the four arm bones
alone as ONE rigid translation (residual 1.2e-4 mm — over-determined, eight equations in three unknowns), median ~0,
p95 12.5 / 8.8 mm (performer 0 / 1), max 19.7 / 9.7, above 0.5 mm on 67 and 19 of 150 frames (D9 review §7.1); D8c's
R4 read p95 8.75 → 8.72 mm on performer 1. The legs do NOT satisfy the same constraint (a fit including them left
5.8 / 13.2 mm) because the legs are still aimed landmark-to-landmark (D9-legs is queued after this step). The card
must: (1) measure first, from the delivered GLB's own bytes, the per-frame hoist and the per-joint miss it causes
(elbow, wrist, knee, ankle: `target − origin_after_hoist` vs the delivered direction), on both performers, whole take
and the frames above 0.5 mm; (2) choose between the two mechanisms named at D9 and pre-register the choice — apply the
foot-contact correction BEFORE the aim passes and aim from the corrected origins, or re-run the aim passes AFTER the
hoist — with the reason (the first keeps one converter pass order; the second is a re-solve after replacing a parent,
the D2c rule); (3) bands the candidate cannot optimise: the D3 gate's exact-skeleton oracle (`d3_skeleton_gate.py`;
legs 0.05–0.07 mm, arms 0.82–1.18, torso 8–11 at the D8c close-out) — on an exact rig with no ground contact the
oracle must be bit-identical, and a synthetic body with a real contact must land its legs exact; the photographs
(part-wise silhouette, ARMS and TORSO+LEGS, whole take and the bent tercile, not worse on either performer, CI clear);
`delivered_vs_capture.py` D8c vs the candidate on identical draws (elbows and wrists predicted to move by exactly the
recovered hoist on the hoisted frames and nowhere else — the smoothed landmarks are byte-identical because this is
converter-only, so the D3 same-denominator clause returns to PASS, pre-registered as an expected pass; if it reads
CHANGED the change did more than re-aim); (4) the foot contacts themselves (count, frames, ground penetration) reported
before and after, because the re-aim moves the feet and the contact detection reads the feet; (5) must-fails: the D8c
build (hoist p95 12.5 / 8.8) and a degenerate that zeroes the hoist by turning the correction off (contacts and
penetration must expose it); (6) remember the D8c lesson: the delivered file is whole-take-coupled through D3's sizing
only when LANDMARKS change — this step changes none, so bit-identity outside the hoisted frames is a legitimate clause
here (state which joints may move: everything below the root on hoisted frames; nothing on the others). No new
constant; `maximum_root_correction_m` is not re-selected. Window 0.

**Then, in order:** D7c, the pelvis fitted to the rig's own rest offsets instead of SOMA-derived constants (the exact
oracle reads a ~7° pelvis tilt about the hip line, invisible to the leg roots, Spine origin 27 mm off, neck 10.7, arms
0.8–1.2 against a 0.5 band; band: the oracle's trunk and arms return toward exact, the real-take hoist and silhouette
as the photographs; re-measure the honest root→hip spread afterwards, because D8c's both-hips charge rests on the
pelvis landmark being loose) → D9-legs (aim thighs and shins from their own origin; 3–4 mm on the take, exact on the
oracle) → instrument debts (re-pin the D3 gate's frozen D2c/D3 references; `--median-from` on
`captured_limb_stability`; recheck D8's 27 → 18 / 22 → 4 headline on a fixed denominator; a fixture that injects a
two-view DEPTH stretch, owed by D8c) → D6 (binding, mesh-distortion instrument first) → D5 (bone lengths and the
flexible spine) → D4 (momentum on MHR).

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
