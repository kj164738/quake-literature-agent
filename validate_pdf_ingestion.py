from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quake_agent.pdf_diagnostics import analyze_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PDF text extraction quality before adding papers to the agent.")
    parser.add_argument("pdf", help="Path to a PDF paper.")
    parser.add_argument("--low-text-threshold", type=int, default=80, help="Characters below this number mark a page as suspicious.")
    args = parser.parse_args()

    report = analyze_pdf(args.pdf, low_text_threshold=args.low_text_threshold)
    print(f"PDF: {report.path}")
    print(f"Status: {report.status}")
    print(f"Pages: {report.page_count}")
    print(f"Pages with text: {report.pages_with_text}")
    print(f"Extracted characters: {report.total_characters}")
    print(f"Average characters/page: {report.average_characters_per_page}")
    print(f"Chunks: {report.chunk_count}")
    if report.low_text_pages:
        pages = ", ".join(str(page) for page in report.low_text_pages[:20])
        suffix = " ..." if len(report.low_text_pages) > 20 else ""
        print(f"Low-text pages: {pages}{suffix}")
    else:
        print("Low-text pages: none")

    if report.status in {"empty_pdf", "no_extractable_text", "no_chunks"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
