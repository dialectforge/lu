#!/usr/bin/env python3
"""Build Lu glossary artifacts.

Reads:
  /opt/lu/lu/v4/GLOSSARY.md (source of truth)
  every .rs / .py / .md / .toml / .sh / .service file under the repo

Writes:
  build/Lu_codebase.html  syntax-highlighted browsable codebase with
                          per-line anchors and cross-links from
                          source lines back to glossary entries.
  build/Lu_glossary.pdf   markdown-rendered glossary PDF with external
                          hyperlinks into Lu_codebase.html.

Hard prohibitions (per CLAUDE.md): no npm, no axios, no
CDN-fetch-at-build. Pure Python + system binaries only.

Dependencies (all pure Python from PyPI, installed via
`pip install --break-system-packages --user`):
  pygments      - syntax highlighting
  markdown      - markdown -> HTML for the PDF render
  fpdf2         - HTML -> PDF (no system libs)

Pandoc and WeasyPrint were both ruled out:
  - pandoc requires sudo apt install (not available on this host)
  - WeasyPrint installs but fails at runtime: cannot load libpango

fpdf2 was selected as the pure-Python PDF backend that needs zero
system libraries.

Run:
  python3 scripts/build_glossary.py
"""

from __future__ import annotations

import html as html_lib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
GLOSSARY_MD = REPO / "GLOSSARY.md"

INCLUDE_EXT = {".rs", ".py", ".md", ".toml", ".yml", ".yaml", ".sh",
               ".service", ".sudoers", ".tsv", ".json"}
EXCLUDE_DIRS = {"target", ".git", "__pycache__", "build", "node_modules",
                ".pytest_cache", ".mypy_cache", ".vscode", ".idea",
                "venv", ".venv"}
EXCLUDE_FILES = {"Cargo.lock"}
MAX_FILE_BYTES = 2 * 1024 * 1024  # skip files larger than 2 MiB (e.g. generated JSON)


def discover_files():
    files = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO)
        if any(p in EXCLUDE_DIRS for p in rel.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix.lower() not in INCLUDE_EXT and rel.name.lower() not in {"readme", "license"}:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.append(rel)
    return files


def parse_glossary():
    """Return (entries, raw_md).

    entries: list of dicts:
      name, kind (h2 category), anchor (md heading slug),
      source_file (str, repo-relative), source_line (int)
    """
    raw = GLOSSARY_MD.read_text(encoding="utf-8")
    entries = []
    current_name = None
    current_anchor = None
    current_category = None
    for line in raw.splitlines():
        m_cat = re.match(r"^###\s+\d+\.\s+(.+)", line)
        if m_cat:
            current_category = m_cat.group(1).strip()
            continue
        m_name = re.match(r"^####\s+(.+)", line)
        if m_name:
            current_name = m_name.group(1).strip()
            current_anchor = slugify(current_name)
            continue
        m_src = re.match(r"^- \*\*source\*\*:\s*([^:\s]+):(\d+)", line)
        if m_src and current_name:
            entries.append({
                "name": current_name,
                "anchor": current_anchor,
                "category": current_category,
                "source_file": m_src.group(1).strip(),
                "source_line": int(m_src.group(2)),
            })
    return entries, raw


_SLUG_DROP = re.compile(r"[^a-z0-9\- ]+")


def slugify(text):
    s = text.lower()
    s = _SLUG_DROP.sub("", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


# ---------- HTML build ----------

def build_html(files, entries):
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename, TextLexer
    from pygments.formatters import HtmlFormatter
    from pygments.util import ClassNotFound

    # file index: rel-path-string -> file_n integer
    file_index = {str(rel): n for n, rel in enumerate(files)}

    # glossary -> source: rel-path-string -> list of entries citing that file
    file_refs = {}
    for e in entries:
        file_refs.setdefault(e["source_file"], []).append(e)

    formatter = HtmlFormatter(
        linenos="inline",
        anchorlinenos=True,
        lineanchors="L",   # placeholder; we set per-file below via wrapping
        nowrap=False,
        cssclass="src",
        style="default",
    )
    pygments_css = formatter.get_style_defs(".src")

    page_css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #fff; color: #222; }
    .topbar { background: #1b1f23; color: #fff; padding: 14px 24px; position: sticky; top: 0; z-index: 10; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .topbar h1 { margin: 0; font-size: 18px; font-weight: 600; }
    .topbar .meta { font-size: 12px; color: #9aa0a6; margin-top: 4px; }
    .container { max-width: 1280px; margin: 0 auto; padding: 24px; }
    .toc { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 16px 20px; margin-bottom: 32px; }
    .toc h2 { margin: 0 0 12px 0; font-size: 16px; }
    .toc ul { list-style: none; padding: 0; margin: 0; columns: 3; column-gap: 32px; }
    .toc li { margin: 2px 0; font-size: 13px; }
    .toc a { color: #0969da; text-decoration: none; font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace; }
    .toc a:hover { text-decoration: underline; }
    .file-section { margin-bottom: 48px; }
    .file-header { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px 6px 0 0; padding: 12px 16px; }
    .file-header h2 { margin: 0; font-size: 15px; font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace; color: #24292f; }
    .file-refs { background: #fff8e1; border-left: 3px solid #f9b800; padding: 8px 12px; margin: 0; font-size: 12px; color: #5a4400; }
    .file-refs strong { color: #422e00; }
    .file-refs a { color: #0969da; text-decoration: none; font-family: 'SF Mono', monospace; margin-right: 8px; }
    .file-refs a:hover { text-decoration: underline; }
    .src { background: #f8f9fa; border: 1px solid #d0d7de; border-top: none; border-radius: 0 0 6px 6px; padding: 12px 0; overflow-x: auto; font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace; font-size: 12px; line-height: 1.45; }
    .src pre { margin: 0; padding: 0 16px; }
    .src .linenos { color: #8c959f; user-select: none; padding-right: 12px; }
    .src a.linenos-anchor { color: #8c959f; text-decoration: none; }
    a.back-to-top { float: right; font-size: 11px; color: #57606a; text-decoration: none; }
    a.back-to-top:hover { text-decoration: underline; }
    """

    parts = []
    parts.append("<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>")
    parts.append("<title>Lu Codebase &mdash; browseable source</title>")
    parts.append("<style>")
    parts.append(page_css)
    parts.append(pygments_css)
    parts.append("</style></head><body>")
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append("<div class='topbar' id='top'>")
    parts.append(f"<h1>Lu Codebase &mdash; browseable source</h1>")
    parts.append(f"<div class='meta'>{len(files)} files &middot; {len(entries)} glossary entries &middot; built {built_at}</div>")
    parts.append("</div><div class='container'>")

    # TOC
    parts.append("<div class='toc'><h2>Files</h2><ul>")
    for rel in files:
        n = file_index[str(rel)]
        parts.append(f"<li><a href='#f{n}'>{html_lib.escape(str(rel))}</a></li>")
    parts.append("</ul></div>")

    # Files
    for rel in files:
        n = file_index[str(rel)]
        abs_path = REPO / rel
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"# could not read: {exc}"
        try:
            lexer = get_lexer_for_filename(str(rel), stripnl=False)
        except ClassNotFound:
            lexer = TextLexer(stripnl=False)
        per_file_formatter = HtmlFormatter(
            linenos="inline",
            anchorlinenos=True,
            lineanchors=f"f{n}-line",
            nowrap=False,
            cssclass="src",
            style="default",
        )
        highlighted = highlight(text, lexer, per_file_formatter)

        parts.append(f"<div class='file-section' id='f{n}'>")
        parts.append("<div class='file-header'>")
        parts.append(f"<h2>{html_lib.escape(str(rel))} <a class='back-to-top' href='#top'>&uarr; top</a></h2>")
        parts.append("</div>")

        # Glossary references sidebar for this file
        refs = file_refs.get(str(rel), [])
        if refs:
            ref_links = " ".join(
                f"<a href='#g-{e['anchor']}'>{html_lib.escape(e['name'])} (L{e['source_line']})</a>"
                for e in refs
            )
            parts.append(
                f"<div class='file-refs'><strong>Glossary references &rarr;</strong> {ref_links}</div>"
            )

        parts.append(highlighted)
        parts.append("</div>")

    # Glossary section at bottom (anchors so file-refs can link back)
    parts.append("<div class='file-section' id='glossary'>")
    parts.append("<div class='file-header'><h2>Glossary entries (anchors)</h2></div>")
    parts.append("<div style='padding:16px;background:#fff;border:1px solid #d0d7de;border-top:none;border-radius:0 0 6px 6px;font-size:13px;'>")
    for e in entries:
        n = file_index.get(e["source_file"])
        if n is None:
            target = ""
        else:
            target = f"<a href='#f{n}-line-{e['source_line']}'>{html_lib.escape(e['source_file'])}:{e['source_line']}</a>"
        parts.append(
            f"<div id='g-{e['anchor']}' style='margin-bottom:8px;padding:6px 8px;border-left:3px solid #0969da;background:#f6f8fa;'>"
            f"<strong>{html_lib.escape(e['name'])}</strong> <em>({html_lib.escape(e['category'] or 'uncategorised')})</em> &mdash; source: {target}"
            f"</div>"
        )
    parts.append("</div></div>")

    parts.append("</div></body></html>")
    return "".join(parts)


# ---------- PDF build ----------

def build_pdf(entries, md_raw, html_filename="Lu_codebase.html"):
    from fpdf import FPDF
    import markdown as md_pkg

    # Pre-process: rewrite each `- **source**: file:line` line to include
    # an explicit HTML link target so write_html renders it as a hyperlink
    # into the codebase HTML.
    file_to_n = {}
    # Pull file order from a fresh discover_files call so anchor indices
    # in the PDF match the HTML.
    files = discover_files()
    for n, rel in enumerate(files):
        file_to_n[str(rel)] = n

    def rewrite_source(match):
        path = match.group(1).strip()
        line = int(match.group(2))
        n = file_to_n.get(path)
        if n is None:
            return match.group(0)
        url = f"{html_filename}#f{n}-line-{line}"
        return f"- **source**: [`{path}:{line}`]({url})"

    rewritten = re.sub(
        r"^- \*\*source\*\*:\s*([^:\s]+):(\d+)",
        rewrite_source,
        md_raw,
        flags=re.MULTILINE,
    )
    html_body = md_pkg.markdown(rewritten, extensions=["extra"])

    pdf = FPDF(format="letter")
    pdf.set_auto_page_break(auto=True, margin=16)
    dejavu_dir = "/usr/share/fonts/truetype/dejavu"
    pdf.add_font("DejaVu", "", f"{dejavu_dir}/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", f"{dejavu_dir}/DejaVuSans-Bold.ttf")
    # fpdf2 >= 2.8 raises on any set_font style not explicitly
    # registered; markdown <em>/<strong><em> rendering requests the
    # I and BI faces, so register the oblique variants too
    # (fonts-dejavu-extra provides them).
    pdf.add_font("DejaVu", "I", f"{dejavu_dir}/DejaVuSans-Oblique.ttf")
    pdf.add_font("DejaVu", "BI", f"{dejavu_dir}/DejaVuSans-BoldOblique.ttf")
    pdf.add_font("DejaVuMono", "", f"{dejavu_dir}/DejaVuSansMono.ttf")
    pdf.add_font("DejaVuMono", "B", f"{dejavu_dir}/DejaVuSansMono-Bold.ttf")
    pdf.add_font("DejaVuMono", "I", f"{dejavu_dir}/DejaVuSansMono-Oblique.ttf")
    pdf.add_font("DejaVuMono", "BI", f"{dejavu_dir}/DejaVuSansMono-BoldOblique.ttf")
    pdf.set_font("DejaVu", size=10)
    pdf.add_page()
    # write_html handles <h1>-<h6>, <p>, <ul>, <li>, <a>, <strong>, <em>,
    # <code>, <pre> from fpdf2's HTML subset.
    pdf.write_html(html_body, font_family="DejaVu", pre_code_font="DejaVuMono")
    pdf_path = BUILD / "Lu_glossary.pdf"
    pdf.output(str(pdf_path))
    return pdf_path


# ---------- main ----------

def main():
    BUILD.mkdir(exist_ok=True)

    entries, md_raw = parse_glossary()
    print(f"[build_glossary] parsed {len(entries)} glossary entries from GLOSSARY.md")

    files = discover_files()
    print(f"[build_glossary] discovered {len(files)} source files")

    # Surface entries whose source file isn't in the discovered set
    file_set = {str(f) for f in files}
    missing = [e for e in entries if e["source_file"] not in file_set]
    if missing:
        print("[build_glossary] WARNING: glossary entries reference files not in the build set:")
        for e in missing:
            print(f"  - {e['name']} -> {e['source_file']}")

    html_str = build_html(files, entries)
    html_path = BUILD / "Lu_codebase.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"[build_glossary] wrote {html_path} ({html_path.stat().st_size:,} bytes)")

    pdf_path = build_pdf(entries, md_raw)
    print(f"[build_glossary] wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

    # Self-audit: scan HTML for any external https references
    # (CLAUDE.md hard prohibition: no CDN-fetch-at-build).
    bad = re.findall(
        r"(?:<script[^>]+src=|<link[^>]+href=|@import\s+(?:url\()?)['\"]https?://",
        html_str,
    )
    if bad:
        print("[build_glossary] FATAL: HTML contains external https references:")
        for b in bad[:10]:
            print(f"  {b}")
        sys.exit(1)
    print("[build_glossary] self-audit: 0 external https script/link/@import refs (CDN constraint satisfied)")


if __name__ == "__main__":
    main()
