"""Render an HTML report string to a single-page PDF via Playwright/Chromium.

Optional heavy dependency (playwright + a chromium install). Callers should treat
it as best-effort and fall back to text if it raises.
"""

from __future__ import annotations

from pathlib import Path


def html_to_pdf(html: str, out_path: str, width_px: int = 760) -> str:
    """Render ``html`` to a single tall PDF page at ``out_path``; return the path."""
    from playwright.sync_api import sync_playwright

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width_px, "height": 1200})
        page.set_content(html, wait_until="networkidle")
        height = page.evaluate("() => document.documentElement.scrollHeight")
        page.pdf(
            path=out_path,
            width=f"{width_px}px",
            height=f"{int(height) + 40}px",
            print_background=True,
        )
        browser.close()
    return out_path


def merge_pdfs(paths: list[str], out_path: str) -> str:
    """Concatenate multiple PDFs into one at ``out_path``; return the path."""
    from pypdf import PdfWriter

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for p in paths:
        if Path(p).exists():
            writer.append(p)
    with open(out_path, "wb") as fh:
        writer.write(fh)
    return out_path
