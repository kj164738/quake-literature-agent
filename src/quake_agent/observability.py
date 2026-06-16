from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quake_agent.agent import AgentAnswer


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_at: str
    question: str
    answer_preview: str
    model_provider: str
    retrieval_provider: str
    arxiv_mode: str
    paper_count: int
    chunk_count: int
    duration_ms: int
    local_result_count: int
    local_top_score: float
    arxiv_result_count: int
    source_count: int
    used_arxiv: bool
    refused: bool
    model_error: str | None
    steps: list[str] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)


def build_run_record(
    *,
    question: str,
    result: AgentAnswer,
    settings,
    paper_count: int,
    chunk_count: int,
    arxiv_mode: str,
) -> RunRecord:
    trace = result.trace
    return RunRecord(
        run_id=str(uuid.uuid4())[:8],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question=question.strip(),
        answer_preview=result.answer.strip().replace("\n", " ")[:220],
        model_provider=settings.active_provider.upper() if settings.has_api_key else "DEMO",
        retrieval_provider=settings.active_embedding_provider.upper(),
        arxiv_mode=arxiv_mode,
        paper_count=paper_count,
        chunk_count=chunk_count,
        duration_ms=trace.duration_ms,
        local_result_count=trace.local_result_count,
        local_top_score=trace.local_top_score,
        arxiv_result_count=trace.arxiv_result_count,
        source_count=trace.source_count,
        used_arxiv=trace.used_arxiv,
        refused=trace.refused,
        model_error=trace.model_error,
        steps=list(result.steps),
        source_labels=[source.label for source in result.sources],
    )


def append_run_record(log_dir: str | Path, record: RunRecord) -> Path:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def load_recent_run_records(log_dir: str | Path, limit: int = 8) -> list[RunRecord]:
    path = Path(log_dir) / "runs.jsonl"
    if not path.exists():
        return []
    rows: list[RunRecord] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_record_from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
    except OSError:
        return []
    return rows[-limit:][::-1]


def _record_from_dict(data: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(data.get("run_id", "")),
        created_at=str(data.get("created_at", "")),
        question=str(data.get("question", "")),
        answer_preview=str(data.get("answer_preview", "")),
        model_provider=str(data.get("model_provider", "")),
        retrieval_provider=str(data.get("retrieval_provider", "")),
        arxiv_mode=str(data.get("arxiv_mode", "")),
        paper_count=int(data.get("paper_count") or 0),
        chunk_count=int(data.get("chunk_count") or 0),
        duration_ms=int(data.get("duration_ms") or 0),
        local_result_count=int(data.get("local_result_count") or 0),
        local_top_score=float(data.get("local_top_score") or 0.0),
        arxiv_result_count=int(data.get("arxiv_result_count") or 0),
        source_count=int(data.get("source_count") or 0),
        used_arxiv=bool(data.get("used_arxiv")),
        refused=bool(data.get("refused")),
        model_error=data.get("model_error"),
        steps=list(data.get("steps") or []),
        source_labels=list(data.get("source_labels") or []),
    )
