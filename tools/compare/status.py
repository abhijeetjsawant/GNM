#!/usr/bin/env python3
"""WHERE THE BODY LANE STANDS. One source of truth, two generated views.

    .venv/bin/python tools/compare/status.py set I1 in_progress --note "worktree ladder/I1"
    .venv/bin/python tools/compare/status.py log "retarget split written; report on disk" --step I1
    .venv/bin/python tools/compare/status.py render

`docs/ladder-status.json` is the source of truth and is edited ONLY through `set`, `log`
and `decide`. Every one of them rewrites both generated views, so the views cannot go stale;
`render` exists for the case where the JSON was edited by hand. Render is idempotent -- it
appends nothing and bumps no date -- and writes:

  docs/LADDER_STATUS.md   the resume note a fresh session reads first (a SessionStart hook
                          prints it); generated, never hand-edited
  docs/progress.html      the plain-language progress page for a non-technical reader

The two plan documents (LADDER_EXECUTION_PLAN.md, SUBSTITUTION_LADDER.md) stay hand-edited;
this file summarises them plus state, so they cannot drift apart.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "docs/ladder-status.json"
MD = ROOT / "docs/LADDER_STATUS.md"
PAGE = ROOT / "docs/progress.html"
FIGURES = ROOT / "docs/ladder-figures.json"   # written by ladder.py: the charts, resolved
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visuals import VIS_CSS, chart_svg  # noqa: E402
STATES = ("planned", "in_progress", "blocked", "done", "retired")
PLAIN_STATE = {"planned": "not started", "in_progress": "in progress", "blocked": "blocked",
               "done": "done", "retired": "retired"}
PLAIN_OWNER = {"user": "you", "fable": "Fable", "opus-agent": "an Opus agent", "sonnet": "Sonnet research"}
URLS = {
    "ladder": "https://claude.ai/code/artifact/56361ab8-b5a0-456d-9171-4d6a09d6c132",
    "board": "https://claude.ai/code/artifact/cf83ef29-a4b7-4afd-9031-0918e8eb6f35",
    "progress": "https://claude.ai/code/artifact/abd3a70c-4c51-4251-8b2f-344f095998c6",
}


def load() -> dict:
    return json.loads(STATUS.read_text())


def save(d: dict) -> None:
    """Persist the source of truth, then rewrite both generated views so they can never
    be stale relative to it. Render is idempotent, so this costs nothing."""
    d["updated"] = dt.date.today().isoformat()
    STATUS.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    write_views(d)


def find(d: dict, step_id: str) -> dict:
    for lane in d["lanes"]:
        for s in lane["steps"]:
            if s["id"] == step_id:
                return s
    raise SystemExit(f"no step {step_id!r}; known: " +
                     ", ".join(s["id"] for l in d["lanes"] for s in l["steps"]))


def cmd_set(a: argparse.Namespace) -> None:
    d = load()
    s = find(d, a.step)
    today = dt.date.today().isoformat()
    s["state"] = a.state
    if a.state == "in_progress" and not s.get("started"):
        s["started"] = today
    if a.state in ("done", "retired"):
        s["finished"] = today
        s["blocked_on"] = None          # a finished step is by definition not blocked
    if a.owner:
        s["owner"] = a.owner
    if a.report:
        s["report"] = a.report
    if a.blocked_on is not None:
        s["blocked_on"] = a.blocked_on or None
    if a.note:
        d["log"].append({"date": today, "step": a.step, "text": a.note})
    save(d)
    print(f"{a.step}: {a.state}" + (f" -- {a.note}" if a.note else ""))


def cmd_log(a: argparse.Namespace) -> None:
    d = load()
    if a.step:
        find(d, a.step)
    d["log"].append({"date": dt.date.today().isoformat(), "step": a.step, "text": a.text})
    save(d)
    print("logged")


def cmd_decide(a: argparse.Namespace) -> None:
    d = load()
    if a.remove is not None:
        d["decisions_for_user"].pop(a.remove)
    if a.add:
        d["decisions_for_user"].append(a.add)
    save(d)
    print("\n".join(f"{i}. {t}" for i, t in enumerate(d["decisions_for_user"])) or "no open decisions")


# --------------------------------------------------------------------------- render: md
def render_md(d: dict) -> str:
    steps = [s for l in d["lanes"] for s in l["steps"]]
    inflight = [s for s in steps if s["state"] == "in_progress"]
    blocked = [s for s in steps if s["state"] == "blocked"]
    done = [s for s in steps if s["state"] in ("done", "retired")]
    nxt = [s for s in steps if s["state"] == "planned" and not s.get("blocked_on")]
    recent = d["log"][-6:]
    out = [
        "# Body lane — where we are (generated, do not hand-edit)",
        "",
        f"*Rendered from `docs/ladder-status.json` on {d['updated']} by `tools/compare/status.py render`.*",
        "**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets",
        "built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).",
        "",
        "## Where we are",
        "",
        d["goal"],
        f"Done: {', '.join(s['id'] for s in done) or 'nothing yet'}. In flight: "
        f"{', '.join(s['id'] for s in inflight) or 'nothing'}. Blocked: "
        f"{', '.join(s['id'] for s in blocked) or 'nothing'}.",
        "",
        "## In flight",
        "",
    ]
    out += [f"- **{s['id']}** {s['title']} — {PLAIN_OWNER.get(s['owner'], s['owner'])}, since {s['started']}"
            for s in inflight] or ["- nothing in flight"]
    out += ["", "## Next up (unblocked, not started)", ""]
    out += [f"- **{s['id']}** {s['title']} — {PLAIN_OWNER.get(s['owner'], s['owner'])}" for s in nxt] or ["- none"]
    out += ["", "## Blocked", ""]
    out += [f"- **{s['id']}** {s['title']} — blocked on: {s['blocked_on']}" for s in blocked] or ["- none"]
    out += ["", "## Decisions waiting on the user", ""]
    out += [f"- {t}" for t in d["decisions_for_user"]] or ["- none"]
    out += ["", "## Recent log", ""]
    out += [f"- {e['date']} [{e['step'] or '—'}] {e['text']}" for e in recent]
    out += [
        "",
        "## How to resume",
        "",
        "1. Pick the step from *In flight* or *Next up*; its gate card is in `LADDER_EXECUTION_PLAN.md` §2.",
        "2. Start it: `.venv/bin/python tools/compare/status.py set <ID> in_progress --note \"...\"`.",
        "3. Instruments write a JSON report under `artifacts/` and get an extractor in `tools/compare/ladder.py`",
        "   (Fable owns that registry). Swap-harness scripts run on the *system* `python3`; everything under",
        "   `tools/compare/` on `.venv/bin/python`.",
        "4. Finish it: `status.py set <ID> done --report <path> --note \"...\"`, then `status.py render`,",
        "   then `.venv/bin/python tools/compare/ladder.py`, then republish the three pages to their URLs",
        "   (ladder, board, progress — URLs in CLAUDE.md), then commit the step's files together.",
        "5. Never select a shipped constant on a MAMMA-referenced arm. The MAMMA arm reports; it never selects.",
        "",
        f"Pages: ladder <{URLS['ladder']}> · board <{URLS['board']}>" +
        (f" · progress <{URLS['progress']}>" if URLS["progress"] else " · progress: see CLAUDE.md"),
        "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------- render: html
CSS = """
:root{--bg:#F7F6F2;--card:#FFFFFF;--rule:#DDD9D0;--ink:#1E1C18;--muted:#5E5A52;--faint:#8C8780;
 --accent:#2B5F8C;--done:#2F7A4A;--doing:#B7741C;--blocked:#B23A3A;--planned:#8C8780;--retired:#8C8780;
 --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15140F;--card:#1E1C16;--rule:#332F27;--ink:#EEEAE2;--muted:#B5AFA3;--faint:#7C766B;--accent:#7FB2DD;--done:#5FBF7E;--doing:#E0A24A;--blocked:#EE6B6B;--planned:#7C766B;--retired:#7C766B}}
:root[data-theme="dark"]{--bg:#15140F;--card:#1E1C16;--rule:#332F27;--ink:#EEEAE2;--muted:#B5AFA3;--faint:#7C766B;--accent:#7FB2DD;--done:#5FBF7E;--doing:#E0A24A;--blocked:#EE6B6B;--planned:#7C766B;--retired:#7C766B}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;font-size:17px}
.wrap{max-width:52rem;margin:0 auto;padding:2.2rem 1.2rem 4rem}
h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.1;margin:0 0 .5rem;letter-spacing:-.02em;text-wrap:balance}
h2{font-size:1.15rem;margin:2rem 0 .6rem;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:60ch;margin:0 0 1.2rem}
.stamp{font-size:.85rem;color:var(--faint);margin:0 0 1.6rem}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.7rem;margin:0 0 1.4rem}
.tile{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:.8rem 1rem}
.tile b{display:block;font-size:1.6rem;line-height:1.1;font-variant-numeric:tabular-nums}
.tile span{font-size:.85rem;color:var(--muted)}
.ask{background:var(--card);border:1px solid var(--rule);border-left:5px solid var(--accent);border-radius:8px;padding:1rem 1.2rem;margin:0 0 1.2rem}
.ask h2{margin:0 0 .5rem}.ask ol{margin:0;padding-left:1.2rem}.ask li{margin:.25rem 0}
.lane{margin:1.4rem 0}
.lane p.plain{color:var(--muted);margin:.2rem 0 .8rem;max-width:62ch}
.step{display:grid;grid-template-columns:7.2rem 1fr;gap:.2rem 1rem;align-items:start;background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:.75rem 1rem;margin:0 0 .55rem}
.chip{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.2rem .5rem;border-radius:999px;border:1.5px solid;white-space:nowrap}
.chip.done{color:var(--done);border-color:var(--done)}.chip.in_progress{color:var(--doing);border-color:var(--doing)}
.chip.blocked{color:var(--blocked);border-color:var(--blocked)}.chip.planned,.chip.retired{color:var(--planned);border-color:var(--planned)}
.step .t{font-weight:600}.step .p{color:var(--muted);font-size:.95rem;grid-column:2}
.step .m{color:var(--faint);font-size:.82rem;grid-column:2}
.log{list-style:none;padding:0;margin:0}.log li{display:grid;grid-template-columns:6.5rem 1fr;gap:1rem;padding:.45rem 0;border-top:1px solid var(--rule)}
.log time{color:var(--faint);font-variant-numeric:tabular-nums;font-size:.9rem}
footer{margin-top:2.4rem;padding-top:1rem;border-top:1px solid var(--rule);color:var(--faint);font-size:.85rem}
footer a{color:var(--accent)}
.part{margin:0 0 1.3rem}.part h3{font-size:1rem;margin:0 0 .1rem}.part h3 .n{font-size:.72rem;color:var(--faint);letter-spacing:.08em;text-transform:uppercase;margin-right:.4rem}
.part .st{font-size:.82rem;color:var(--muted);margin:0 0 .5rem}
@media (max-width:34rem){.step{grid-template-columns:1fr}.step .p,.step .m{grid-column:1}}
"""


def render_html(d: dict) -> str:
    steps = [s for l in d["lanes"] for s in l["steps"]]
    counts = {k: sum(1 for s in steps if s["state"] == k) for k in STATES}
    e = html.escape
    tiles = "".join(
        f'<div class="tile"><b>{counts[k]}</b><span>{e(PLAIN_STATE[k])}</span></div>'
        for k in ("done", "in_progress", "blocked", "planned"))
    ask = ""
    if d["decisions_for_user"]:
        ask = '<section class="ask"><h2>Waiting on you</h2><ol>' + \
              "".join(f"<li>{e(t)}</li>" for t in d["decisions_for_user"]) + "</ol></section>"
    lanes = []
    for lane in d["lanes"]:
        rows = []
        for s in lane["steps"]:
            meta = []
            if s.get("owner"):
                meta.append(PLAIN_OWNER.get(s["owner"], s["owner"]))
            if s.get("started"):
                meta.append(f"started {s['started']}")
            if s.get("finished"):
                meta.append(f"finished {s['finished']}")
            if s.get("blocked_on") and s["state"] != "done":
                meta.append(f"waits for: {s['blocked_on']}")
            rows.append(
                f'<div class="step"><span class="chip {s["state"]}">{e(PLAIN_STATE[s["state"]])}</span>'
                f'<div class="t">{e(s["title"])}</div>'
                f'<div class="p">{e(s["plain"])}</div>'
                f'<div class="m">{e(" · ".join(meta))}</div></div>')
        lanes.append(f'<section class="lane"><h2>{e(lane["title"])}</h2><p class="plain">{e(lane["plain"])}</p>'
                     + "".join(rows) + "</section>")
    board = ""
    if FIGURES.exists():
        try:
            fig = json.loads(FIGURES.read_text())
        except Exception:
            fig = None
        if fig:
            parts = []
            for r in fig["rungs"]:
                if not r.get("visuals"):
                    continue
                cards = "".join(chart_svg(c) for c in r["visuals"])
                parts.append(f'<div class="part"><h3><span class="n">part {r["n"]:02d}</span> {e(r["title"])}</h3>'
                             f'<p class="st">{e(r["status"])}</p><div class="viz-grid">{cards}</div></div>')
            if parts:
                board = ('<h2>How we compare with MAMMA, part by part</h2>'
                         '<p class="plain">One chart per comparison. Blue is ours, orange is MAMMA, the benchmark we '
                         'measure against; green is an alternative of ours; a hatched bar is a deliberately wrong '
                         'answer that must lose, there to prove the measurement can fail. Every chart says whether '
                         'lower or higher is better. Bars on one chart share a reference; bars on different charts '
                         'never do, so do not compare across charts. Nothing here is accuracy: MAMMA is an estimate '
                         'too, and only the marker session can turn agreement into accuracy. '
                         f'Figures rendered {e(fig["rendered"])}.</p>' + "".join(parts))
    log = "".join(
        f'<li><time>{e(x["date"])}</time><span>{e(x["text"])}</span></li>' for x in reversed(d["log"][-12:]))
    links = f'<a href="{URLS["ladder"]}">the measurement ladder</a> and <a href="{URLS["board"]}">the parity board</a>'
    return f'''<meta charset="utf-8">
<title>Body Capture Progress</title>
<style>{CSS}{VIS_CSS}</style>
<div class="wrap">
<h1>Body Capture Progress</h1>
<p class="lede">{e(d["goal"])}</p>
<p class="stamp">Updated {e(d["updated"])}. Three lanes: our own hardware, the measuring tools, and the changes to what we ship. Each step is either done, in progress, blocked, or not started.</p>
<div class="summary">{tiles}</div>
{ask}
{board}
{"".join(lanes)}
<h2>What changed recently</h2>
<ul class="log">{log}</ul>
<footer>For the technical detail behind every step: {links}. This page is generated from <code>docs/ladder-status.json</code> by <code>tools/compare/status.py</code>; the plans are <code>docs/LADDER_EXECUTION_PLAN.md</code> and <code>docs/SUBSTITUTION_LADDER.md</code>. MAMMA is a research-licensed reference we measure against; nothing it produces enters what we ship.</footer>
</div>
'''


def write_views(d: dict) -> None:
    MD.write_text(render_md(d))
    PAGE.write_text(render_html(d))


def cmd_render(_: argparse.Namespace) -> None:
    d = load()
    write_views(d)
    print(f"wrote {MD.relative_to(ROOT)} ({len(render_md(d).splitlines())} lines) and {PAGE.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("set", help="change a step's state")
    p.add_argument("step"); p.add_argument("state", choices=STATES)
    p.add_argument("--note"); p.add_argument("--owner"); p.add_argument("--report"); p.add_argument("--blocked-on", dest="blocked_on")
    p.set_defaults(fn=cmd_set)
    p = sub.add_parser("log", help="append a plain-language log line")
    p.add_argument("text"); p.add_argument("--step"); p.set_defaults(fn=cmd_log)
    p = sub.add_parser("decide", help="add or remove a decision waiting on the user")
    p.add_argument("--add"); p.add_argument("--remove", type=int); p.set_defaults(fn=cmd_decide)
    p = sub.add_parser("render", help="write LADDER_STATUS.md and progress.html"); p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
