"""Render a static, read-only status page (for GitHub Pages or any bucket)."""

from __future__ import annotations

import html
from pathlib import Path

from pipeline.atomic import atomic_write_json, atomic_write_text

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transit hunt status</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1d1d1f;background:#fbfaf7}}
h1{{font-size:1.4rem;margin:0 0 .25rem}} .muted{{color:#666}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem;margin:1.25rem 0}}
.stat{{background:#fff;border:1px solid #e3e0d8;border-radius:8px;padding:.75rem}}
.stat b{{display:block;font-size:1.3rem}}
table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{padding:.4rem .5rem;border-bottom:1px solid #eee;text-align:left}}
th{{font-weight:600;background:#f3f1ec}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
pre{{background:#fff;border:1px solid #e3e0d8;border-radius:8px;padding:.75rem;overflow:auto;max-height:24rem;font-size:12px}}
</style></head><body>
<h1>Transit hunt status</h1>
<p class="muted">Updated {updated}. Machine-readable copy: <a href="status.json">status.json</a>.</p>
<div class="stats">{stats}</div>
<h2>Top candidates</h2>
<table><thead><tr><th>TIC</th><th>Label</th><th class="num">Score</th><th class="num">SDE</th>
<th class="num">Period (d)</th><th class="num">Depth (ppm)</th><th>Flags</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Recent log</h2>
<pre>{log}</pre>
</body></html>
"""


def _fmt(value, digits=2):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def build_status(desk, top: int = 50) -> dict:
    progress = desk.continuous.progress()
    found = desk.query_candidates(sort="score", order="desc", limit=top)
    lines = desk.continuous.read_lines(limit=60)
    return {
        "progress": progress,
        "work": desk.store.counts(),
        "candidates_total": found["total"],
        "candidates": found["candidates"],
        "log": [line.get("message") or "" for line in lines],
    }


def render(desk, out_dir: Path, top: int = 50) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = build_status(desk, top=top)
    progress = status["progress"]
    work = status["work"]
    stats = [
        ("Status", progress.get("status")),
        ("Sector", progress.get("sector")),
        ("Stars searched", progress.get("stars_done")),
        ("Candidates", status["candidates_total"]),
        ("Stars / hour", _fmt(progress.get("stars_per_hour"), 1)),
        ("Work done / failed", f"{work['done']} / {work['failed']}"),
    ]
    stat_html = "".join(f"<div class=stat><span class=muted>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>"
                        for k, v in stats)
    rows = "".join(
        "<tr><td>{tic}</td><td>{label}</td><td class=num>{score}</td><td class=num>{sde}</td>"
        "<td class=num>{period}</td><td class=num>{depth}</td><td>{flags}</td></tr>".format(
            tic=html.escape(c["tic_id"]), label=html.escape(c["label"]), score=_fmt(c.get("score"), 3),
            sde=_fmt(c.get("sde"), 1), period=_fmt(c.get("period_days"), 4), depth=_fmt(c.get("depth_ppm"), 0),
            flags=html.escape(", ".join(c.get("hard_flags") or [])) or "-",
        )
        for c in status["candidates"]
    ) or "<tr><td colspan=7 class=muted>No candidates yet.</td></tr>"
    page = PAGE.format(
        updated=html.escape(str(progress.get("updated_at") or "")),
        stats=stat_html,
        rows=rows,
        log=html.escape("\n".join(status["log"])),
    )
    atomic_write_text(out_dir / "index.html", page)
    atomic_write_json(out_dir / "status.json", status)
    return out_dir / "index.html"
