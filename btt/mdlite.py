"""A very small Markdown to HTML converter, enough for the generated model card.

Handles headings, bullet lists (with wrapped continuation lines), tables, block quotes, paragraphs,
**bold**, *italic* and `code`. Everything is escaped first, so card text cannot inject markup.
"""
from __future__ import annotations

import html
import re


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"<em>\1</em>", t)
    return t


def to_html(md: str) -> str:
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{min(n, 4)}>{_inline(ln[n:].strip())}</h{min(n, 4)}>")
            i += 1
        elif ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]] if len(rows) > 1 else []
            out.append("<table><thead><tr>" + "".join(f"<th scope='col'>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
        elif ln.startswith("- "):
            items: list[str] = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")) and lines[i].strip():
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
        else:
            buf = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", ">")):
                buf.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(buf))}</p>")
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model card: Beyond the transcript</title><link rel="stylesheet" href="/static/style.css">
<style>.card{{max-width:760px;margin:0 auto;padding:28px 20px 60px}}.card h1{{font-size:1.5rem}}.card h2{{margin-top:32px}}.card h3{{margin:18px 0 6px}}
.card table{{font-size:.9rem;margin:10px 0}}.card td,.card th{{text-align:left!important}}.card code{{background:rgba(21,34,44,.07);padding:1px 5px;border-radius:3px;font-size:.88em}}
.card blockquote{{margin:12px 0;padding:10px 14px;background:#fff3cd;border-left:4px solid #b58100}}</style></head>
<body><main class="card"><p><a href="/">Back to the demo</a></p>{body}<p class="note">Generated from the training metrics files when you load this page. <a href="/model-card.md">Plain Markdown</a>.</p></main></body></html>
"""
