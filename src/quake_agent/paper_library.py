from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UploadedPaper(Protocol):
    name: str

    def getbuffer(self): ...


@dataclass(frozen=True)
class ManagedPaper:
    name: str
    path: Path
    suffix: str
    size_bytes: int
    modified_at: datetime

    @property
    def size_label(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        return f"{self.size_bytes / (1024 * 1024):.1f} MB"


def list_papers(library_dir: str | Path) -> list[ManagedPaper]:
    root = Path(library_dir)
    if not root.exists():
        return []
    papers: list[ManagedPaper] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        papers.append(
            ManagedPaper(
                name=path.name,
                path=path,
                suffix=path.suffix.lower().lstrip(".").upper(),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            )
        )
    return sorted(papers, key=lambda paper: paper.modified_at, reverse=True)


def save_uploaded_papers(uploaded_files: Iterable[UploadedPaper], library_dir: str | Path) -> list[ManagedPaper]:
    root = Path(library_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        if Path(uploaded_file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        filename = safe_filename(uploaded_file.name)
        path = unique_path(root / filename)
        path.write_bytes(bytes(uploaded_file.getbuffer()))
        saved_paths.append(path)
    return [paper for paper in list_papers(root) if paper.path in saved_paths]


def delete_paper(library_dir: str | Path, paper_name: str) -> bool:
    root = Path(library_dir).resolve()
    path = (root / safe_filename(paper_name)).resolve()
    if root not in path.parents or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    return True


def clear_library(library_dir: str | Path) -> None:
    root = Path(library_dir)
    if root.exists():
        shutil.rmtree(root)


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    cleaned_stem = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", stem).strip("._-")
    if not cleaned_stem:
        cleaned_stem = "paper"
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".txt"
    return f"{cleaned_stem[:100]}{suffix}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
