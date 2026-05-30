from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PaperChunk:
    text: str
    source: str
    page: int | None = None
    chunk_id: int = 0

    @property
    def label(self) -> str:
        if self.page is None:
            return f"{self.source} · chunk {self.chunk_id}"
        return f"{self.source} · page {self.page} · chunk {self.chunk_id}"


def load_documents(paths: Iterable[str | Path]) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.exists() or path.is_dir():
            continue
        text_pages = _read_file(path)
        for page, text in text_pages:
            for chunk_id, chunk in enumerate(split_text(text), start=1):
                chunks.append(
                    PaperChunk(
                        text=chunk,
                        source=path.name,
                        page=page,
                        chunk_id=chunk_id,
                    )
                )
    return chunks


def load_directory(directory: str | Path) -> list[PaperChunk]:
    root = Path(directory)
    if not root.exists():
        return []
    files = [
        path
        for path in root.iterdir()
        if path.suffix.lower() in {".pdf", ".txt", ".md"}
    ]
    return load_documents(sorted(files))


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _read_file(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def _read_pdf(path: Path) -> list[tuple[int | None, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF files requires pypdf. Run: pip install -r requirements.txt") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append((index, page.extract_text() or ""))
    return pages

