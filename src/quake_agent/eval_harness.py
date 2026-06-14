from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quake_agent.agent import AgentAnswer, LiteratureAgent
from quake_agent.arxiv_tool import ArxivPaper, search_arxiv
from quake_agent.document_loader import load_directory
from quake_agent.llm import DemoLLM
from quake_agent.vector_store import LocalKnowledgeBase


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    failures: list[str]
    answer: str
    steps: list[str]
    sources: list[str]


class EvalLLM:
    def invoke(self, prompt: str):
        class Response:
            content = "评测模式回答：系统已根据可用来源生成答案。"

        return Response()


def run_eval_cases(cases: list[dict[str, Any]], sample_dir: str = "sample_papers") -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        chunks = load_directory(sample_dir) if case.get("use_sample_papers", True) else []
        kb = LocalKnowledgeBase(collection_name=f"eval_{case['id']}", use_chroma=False)
        kb.build(chunks)
        agent = LiteratureAgent(
            kb,
            EvalLLM(),
            _arxiv_search if not case.get("live_arxiv") else search_arxiv,
            arxiv_mode=case.get("arxiv_mode", "auto"),
        )
        answer = agent.answer(case["question"])
        results.append(_check_case(case, answer))
    return results


def _arxiv_search(query: str, max_results: int) -> list[ArxivPaper]:
    return [
        ArxivPaper(
            title="Dense Seismic Array Monitoring for Earthquake Detection",
            summary="Dense seismic arrays can improve detection of small earthquakes and support monitoring.",
            url="https://arxiv.org/abs/0000.0000",
            published="2026-01-01",
        )
    ][:max_results]


def _check_case(case: dict[str, Any], answer: AgentAnswer) -> EvalResult:
    failures: list[str] = []
    source_labels = [source.label for source in answer.sources]

    for expected in case.get("expect_steps_contains", []):
        if not any(expected in step for step in answer.steps):
            failures.append(f"missing step: {expected}")

    for expected in case.get("expect_source_label_contains", []):
        if not any(expected in label for label in source_labels):
            failures.append(f"missing source label: {expected}")

    if case.get("expect_refusal") and "不能给出确定结论" not in answer.answer:
        failures.append("expected refusal answer")

    return EvalResult(
        case_id=case["id"],
        passed=not failures,
        failures=failures,
        answer=answer.answer,
        steps=answer.steps,
        sources=source_labels,
    )


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic checks for the literature Agent.")
    parser.add_argument("--cases", default="eval_cases.json", help="Path to eval case JSON file.")
    parser.add_argument("--sample-dir", default="sample_papers", help="Directory with sample documents.")
    args = parser.parse_args()

    results = run_eval_cases(load_cases(args.cases), sample_dir=args.sample_dir)
    passed = sum(1 for result in results if result.passed)
    total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        for failure in result.failures:
            print(f"  - {failure}")
        print(f"  steps: {' | '.join(result.steps)}")
        print(f"  sources: {', '.join(result.sources) if result.sources else 'none'}")

    print(f"\nSummary: {passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)
