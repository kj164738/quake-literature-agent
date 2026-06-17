from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quake_agent.document_loader import split_text


@dataclass(frozen=True)
class PdfIngestionReport:
    path: str
    page_count: int
    pages_with_text: int
    total_characters: int
    chunk_count: int
    average_characters_per_page: float
    low_text_pages: list[int]
    status: str


def analyze_pdf(path: str | Path, *, low_text_threshold: int = 80) -> PdfIngestionReport:
    pdf_path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF files requires pypdf. Run: pip install -r requirements.txt") from exc

    reader = PdfReader(str(pdf_path))
    pages = [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)]
    return build_pdf_report(pdf_path, pages, low_text_threshold=low_text_threshold)


def build_pdf_report(
    path: str | Path,
    pages: list[tuple[int, str]],
    *,
    low_text_threshold: int = 80,
) -> PdfIngestionReport:
    page_count = len(pages)
    text_lengths = [(page, len(" ".join(text.split()))) for page, text in pages]
    pages_with_text = sum(1 for _, length in text_lengths if length > 0)
    total_characters = sum(length for _, length in text_lengths)
    chunk_count = sum(len(split_text(text)) for _, text in pages)
    average = total_characters / page_count if page_count else 0.0
    low_text_pages = [
        page for page, length in text_lengths if length < low_text_threshold
    ]
    status = _status(page_count, pages_with_text, total_characters, chunk_count)
    return PdfIngestionReport(
        path=str(path),
        page_count=page_count,
        pages_with_text=pages_with_text,
        total_characters=total_characters,
        chunk_count=chunk_count,
        average_characters_per_page=round(average, 1),
        low_text_pages=low_text_pages,
        status=status,
    )


def _status(page_count: int, pages_with_text: int, total_characters: int, chunk_count: int) -> str:
    if page_count == 0:
        return "empty_pdf"
    if pages_with_text == 0 or total_characters == 0:
        return "no_extractable_text"
    if chunk_count == 0:
        return "no_chunks"
    if pages_with_text < page_count:
        return "partial_text"
    return "ok"
