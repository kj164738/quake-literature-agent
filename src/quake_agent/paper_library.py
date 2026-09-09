from __future__ import annotations

import re
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH_BYTES = 60 * 1024 * 1024


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
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
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
    validated = []
    total_bytes = 0
    for uploaded_file in uploaded_files:
        if Path(uploaded_file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        filename = safe_filename(uploaded_file.name)
        buffer = uploaded_file.getbuffer()
        total_bytes += len(buffer)
        if not 0 < len(buffer) <= MAX_UPLOAD_BYTES or total_bytes > MAX_BATCH_BYTES:
            raise ValueError("文件不能为空，单文件上限 20 MB，单批上限 60 MB。")
        content = bytes(buffer)
        if filename.endswith(".pdf"):
            if not content.startswith(b"%PDF"):
                raise ValueError(f"{filename} 不是有效的 PDF 文件。")
        else:
            try:
                decoded = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{filename} 需要 UTF-8 编码。") from exc
            if "\x00" in decoded or not decoded.strip():
                raise ValueError(f"{filename} 不包含有效文本。")
        validated.append((filename, content))
    for filename, content in validated:
        # Publish complete bytes exclusively; concurrent uploads cannot overwrite each other.
        with tempfile.NamedTemporaryFile(dir=root, suffix=".upload", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        try:
            while True:
                path = unique_path(root / filename)
                try:
                    os.link(temp_path, path)
                    saved_paths.append(path)
                    break
                except FileExistsError:
                    continue
        finally:
            temp_path.unlink(missing_ok=True)
    return [paper for paper in list_papers(root) if paper.path in saved_paths]


def delete_paper(library_dir: str | Path, paper_name: str) -> bool:
    root = Path(library_dir).resolve()
    if paper_name != safe_filename(paper_name):
        return False
    original = root / paper_name
    if original.is_symlink():
        return False
    path = original.resolve()
    if root not in path.parents or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    return True


def clear_library(library_dir: str | Path) -> None:
    for paper in list_papers(library_dir):
        delete_paper(library_dir, paper.name)


def safe_filename(filename: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1].strip()
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    cleaned_stem = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", stem).strip("._-")
    if not cleaned_stem:
        cleaned_stem = "paper"
    if cleaned_stem.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        cleaned_stem = "paper_" + cleaned_stem
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
