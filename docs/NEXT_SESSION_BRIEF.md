# Resume brief — body-capture lane, written 2026-09-06 at the end of the D8b close-out

Paste this as the opening message of the next session. It is the same protocol as the 2026-09-04 brief with the
state moved forward; everything it references is committed on `battle0/clean-room-multiview-resolution-invariance`.

---

Resume the AutoAnim body-capture lane. Read, in this order: `docs/LADDER_STATUS.md` (the SessionStart hook prints
it), `docs/LADDER_EXECUTION_PLAN.md` §2 (the cards; D7 → D8b rows are the shape of a step), §6 (the queue after D8b),
`CLAUDE.md` body-lane section (every standing rule, including the ones added on 2026-09-06), the memory file, and
`docs/reviews/segment-length-2026-09-06.md` (the last review; §7 and §8 are what a review must carry).

**State.** D7, D7b, D8, D9 and D8b are merged, rebuilt in place, byte-checked, instrumented and pushed. The delivered
skeleton now sits on the captured joints at the hips (5 mm), neck (21 / 18 mm, the trunk-length floor), elbows
(7 / 9 mm) and wrists (9–12 mm); the captured shoulder line no longer collapses on the falling performer (18 → 0
frames off his own width). The report page "The Solve So Far" carries v2–v6 as version tabs with frame players.
The bear side experiments are PARKED (page and workers recorded in CLAUDE.md); do not reopen them unless asked.

**Next step: D8c — the hip line.** It collapses the same way the shoulder line did (23 frames off performer 1's own
width, `left_hip__right_hip` 163–245 mm against 214) and is not in D8b's segment list. Card first, then the agent.
The card must: measure the effect and classify it per camera before any band (as D8 and D8b did); state whether the
hip line is a detector collapse in agreeing views or a two-view degeneracy; ship the segment-length reject for the hip
line with BOTH hips withheld (no parent), with the root formula's dependence on the hip midpoint stated (D2b puts the
leg-root midpoint on it, so a withheld hip line moves the root: pre-register what the hoist and root do); bands the
candidate cannot optimise: photographs torso+legs on the window, raw byte-identical, MAMMA oracle report-only, the
exact-skeleton oracle unchanged; synthetic on the REPAIRED fixture v2 (D8b's, amplitude scale 0.20, moving collapse).

**Then, in order:** the foot-contact hoist re-aim (`project_generated_foot_contacts` translates the root after every
aim, p95 12.5 / 8.8 mm; correct the feet before the aims and re-aim, or re-aim after; the D3 oracle's legs 0.07 are
the band) → D7c, the pelvis fitted to the rig's own rest offsets instead of SOMA-derived constants (the exact oracle
reads a 7° pelvis tilt about the hip line, invisible to the leg roots; band: the oracle's trunk and arms return to
exact) → D9-legs (aim thighs and shins from their own origin; predicted 3–4 mm on the take, exact on the oracle) →
instrument debts (re-pin the D3 gate's frozen D2c/D3 references; add `--median-from` to `captured_limb_stability.py`
and recheck D8's 27 → 18 / 22 → 4 headline on a fixed denominator) → D6 (binding; mesh-distortion instrument first) →
D5 (bone lengths, a flexible spine: the neck's 42 mm bent-frame floor) → D4 (momentum on MHR; SAM 3D Body is on Modal
in `workers/sam3d_body`, GPU/CPU parity 1.5 mm; its four-camera fusion on the bear showed geometry fixes place and the
rig must fix shape).

**Protocol (unchanged).** Sol (the advisor) reviews the card before any src edit and reviews before any merge. One
Opus agent per step in `.claude/worktrees/ladder-<ID>` on branch `ladder/<ID>` with `artifacts`, `.cache`, `.venv`
symlinked and `PYTHONPATH=$PWD/src`; the agent builds the instrument first and reproduces the card's figures before
touching src; writes one JSON report under `artifacts/compare/<id>-<name>/`, an extractor stub with `VISUALS`, NEW
tests only, a review under `docs/reviews/`; never runs `ladder.py` / `status.py`; never writes into
`artifacts/commercial-multiview-soma77/`; rebuilds from `artifacts/commercial-multiview-soma77/work` COPIED; nothing
re-detects; every band must be one the candidate cannot optimise, with a must-fail degenerate and an oracle where one
exists; block bootstrap block 15 / 2000 draws / identical draws; MAMMA's body_id-00 is our subject 1 via
`tools/head/subject_map.py`; MAMMA reports and never selects; pre-register before numbers and record refuted
predictions; read the delivered file from its own bytes; log every instrument line. The reviewer (Fable) may rewrite
committed tests that assert a superseded contract; the agent may not. Post-merge: `tools/compare/post_merge.sh <ID>
<branch-delivery-dir>`, wire the extractor into `RUNGS`/`VISUALS`, `status.py set <ID> done`, board paragraph with
blindness stated, republish the three pages to their fixed URLs (ladder 56361ab8…, board cf83ef29…, progress
abd3a70c…), a new version tab on "The Solve So Far" (9fc29718…) with a frame player and the mp4 sent to me, commit
with the message in a file, push as a separate plain command.

**MINE, NOT YOURS.** Lane H (the rig, the marker session, performer releases; camera D001 reads one frame late on the
held-out lag test, do not correct it). My uncommitted test edits: `tests/test_body_provider.py`,
`tests/test_soma_motion.py`, `tests/test_body_export.py` still assert pre-fix contracts and fail; leave all three.
`tests/test_audio_acting_shot.py` and `tests/test_speech_motion.py` fail and are not the converter's.

**Report format.** Lead with the numbers before and after, one table per part with the reference named; every band
with PASS/FAIL; refuted predictions; what each instrument is blind to; the merge rule applied mechanically; costs
stated; what remains and who owns it.
