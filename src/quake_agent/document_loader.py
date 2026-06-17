from __future__ import annotations

import re
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
    normalized = _normalize_text(text)
    if not normalized:
        return []

    units = _semantic_units(normalized, chunk_size)
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue

        candidate = f"{current}\n\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        prefix = _sentence_overlap(current, overlap)
        with_prefix = f"{prefix}\n\n{unit}" if prefix else unit
        current = with_prefix if len(with_prefix) <= chunk_size else unit

    if current:
        chunks.append(current)
    return chunks


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for raw_paragraph in re.split(r"\n\s*\n+", normalized):
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in raw_paragraph.split("\n")
            if line.strip()
        ]
        paragraph = " ".join(lines).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _semantic_units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [text]

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                hard_overlap = max(0, min(chunk_size // 5, 120, chunk_size - 1))
                units.extend(_hard_split(sentence, chunk_size, overlap=hard_overlap))
    return units


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(paragraph):
        if char in "。！？!?；;":
            sentence = paragraph[start : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = index + 1
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [paragraph]


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def _sentence_overlap(text: str, max_chars: int) -> str:
    sentences = _split_sentences(text.replace("\n", " "))
    if not sentences:
        return ""
    overlap = sentences[-1].strip()
    if len(overlap) <= max_chars:
        return overlap
    return ""


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
