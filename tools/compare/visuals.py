"""One bar chart per comparison: ours beside MAMMA's, and a plain sentence saying which way is good.

Shared by `ladder.py` (the technical page, one chart per rung) and `status.py` (the progress page
for the non-technical reader, the same charts under plain headings). Both read the SAME resolved
comparisons, so a figure can never differ between the two pages.

A comparison is a dict:
    title  -- what is being compared, in words a non-technical reader can follow
    plain  -- one sentence: what "good" looks like here and who the reference is
    better -- "lower" or "higher"
    unit   -- the unit every bar shares (bars on one chart ALWAYS share a unit and a reference)
    bars   -- [{"label": ..., "role": ours|mamma|alt|control, "value": float}]

Roles, and the one rule that keeps the chart honest: a bar is drawn only against bars of the
same reference. Roles are colour-blind-safe by construction (blue / orange / aqua are the first
three slots of a validated categorical palette); the control is a hatched outline, so identity
never rests on colour alone, and every bar carries its value and its label as text.
"""
from __future__ import annotations

import html

ROLE_WORD = {"ours": "Ours", "mamma": "MAMMA (the benchmark)", "alt": "An alternative of ours",
             "control": "A deliberately wrong answer (must lose)"}

VIS_CSS = """
.viz{padding:.8rem 1rem;border-top:1px solid var(--rule)}
.viz .vk{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);display:block;margin-bottom:.4rem}
.viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(26rem,1fr));gap:1rem}
.chart{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:.7rem .85rem .5rem}
.chart h4{font-size:.92rem;margin:0 0 .15rem;line-height:1.25;text-wrap:balance}
.chart .plain{font-size:.8rem;color:var(--muted);margin:0 0 .5rem;max-width:60ch}
.chart .dir{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:.12rem .5rem;border-radius:999px;border:1.5px solid var(--faint);color:var(--muted);margin:0 0 .5rem}
.chart svg{width:100%;height:auto;display:block;font-family:inherit}
.chart .lbl{fill:var(--ink);font-size:11.5px}
.chart .val{fill:var(--ink);font-size:11.5px;font-variant-numeric:tabular-nums;font-weight:600}
.chart .axis{stroke:var(--rule);stroke-width:1}
.chart .b-ours{fill:#2a78d6}.chart .b-mamma{fill:#eb6834}.chart .b-alt{fill:#1baf7a}
.chart .b-control{fill:url(#hatch);stroke:var(--faint);stroke-width:1}
.chart .hatch-line{stroke:var(--faint);stroke-width:1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .chart .b-ours{fill:#3987e5}:root:not([data-theme="light"]) .chart .b-mamma{fill:#d95926}:root:not([data-theme="light"]) .chart .b-alt{fill:#199e70}}
:root[data-theme="dark"] .chart .b-ours{fill:#3987e5}:root[data-theme="dark"] .chart .b-mamma{fill:#d95926}:root[data-theme="dark"] .chart .b-alt{fill:#199e70}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1rem;font-size:.76rem;color:var(--muted);margin:.4rem 0 0}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.legend i{width:12px;height:12px;border-radius:2px;display:inline-block}
.legend i.ours{background:#2a78d6}.legend i.mamma{background:#eb6834}.legend i.alt{background:#1baf7a}
.legend i.control{background:repeating-linear-gradient(135deg,var(--faint) 0 1px,transparent 1px 4px);border:1px solid var(--faint)}
"""


def _fmt(v: float) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _short(t: str, n: int) -> str:
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def chart_svg(comp: dict) -> str:
    """Render one comparison as a self-contained card with an inline SVG bar chart."""
    e = html.escape
    bars = [b for b in comp.get("bars", []) if b.get("value") is not None]
    if not bars:
        return ""
    better = comp.get("better", "lower")
    dir_word = "Lower is better ◀" if better == "lower" else "Higher is better ▶"
    vmax = max(abs(float(b["value"])) for b in bars) or 1.0
    diverging = any(float(b["value"]) < 0 for b in bars)  # signed figures: zero sits mid-track, bars go both ways
    # geometry: label column, bar track, value column; one row per bar
    w, lab_w, val_w, row_h, gap, pad_top = 480, 200, 54, 17, 6, 6
    track = w - lab_w - val_w - 16
    half = track / 2.0
    zero_x = lab_w + half if diverging else lab_w
    h = pad_top + len(bars) * (row_h + gap) + 6
    rows = []
    for i, b in enumerate(bars):
        y = pad_top + i * (row_h + gap)
        v = float(b["value"])
        span = half if diverging else track
        bw = max(2.0, span * abs(v) / vmax)
        x0 = zero_x - bw if (diverging and v < 0) else zero_x
        role = b.get("role", "ours")
        title = f'{b["label"]}: {_fmt(v)} {comp.get("unit", "")} ({ROLE_WORD.get(role, role)})'
        vx = (x0 - 6) if (diverging and v < 0) else (x0 + bw + 6)
        anchor = ' text-anchor="end"' if (diverging and v < 0) else ""
        rows.append(
            f'<g><title>{e(title)}</title>'
            f'<text class="lbl" x="{lab_w - 8}" y="{y + row_h * 0.72:.1f}" text-anchor="end">{e(_short(b["label"], 34))}</text>'
            f'<rect class="b-{e(role)}" x="{x0:.1f}" y="{y}" width="{bw:.1f}" height="{row_h}" rx="3" ry="3"/>'
            f'<text class="val" x="{vx:.1f}" y="{y + row_h * 0.72:.1f}"{anchor}>{_fmt(v)}</text></g>')
    roles_present = []
    for b in bars:
        if b.get("role") not in roles_present:
            roles_present.append(b.get("role"))
    legend = "".join(f'<span><i class="{e(r)}"></i>{e(ROLE_WORD.get(r, r))}</span>' for r in roles_present)
    unit = comp.get("unit", "").split(" (")[0]
    svg = (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{e(comp["title"])}">'
        f'<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(135)">'
        f'<line class="hatch-line" x1="0" y1="0" x2="0" y2="5"/></pattern></defs>'
        f'<line class="axis" x1="{zero_x:.1f}" y1="{pad_top - 3}" x2="{zero_x:.1f}" y2="{h - 4}"/>'
        + "".join(rows) + "</svg>")
    return (f'<div class="chart"><h4>{e(comp["title"])}</h4>'
            f'<p class="plain">{e(comp.get("plain", ""))}</p>'
            f'<span class="dir">{dir_word} · {e(unit)}</span>'
            f'{svg}<div class="legend">{legend}</div></div>')


def charts_band(comps: list[dict], heading: str) -> str:
    cards = [chart_svg(c) for c in comps]
    cards = [c for c in cards if c]
    if not cards:
        return ""
    return f'<div class="viz"><span class="vk">{html.escape(heading)}</span><div class="viz-grid">{"".join(cards)}</div></div>'
