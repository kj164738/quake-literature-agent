from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypedDict

from quake_agent.arxiv_tool import ArxivPaper
from quake_agent.document_loader import PaperChunk
from quake_agent.vector_store import LocalKnowledgeBase, SearchResult


@dataclass(frozen=True)
class Source:
    label: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    steps: list[str]
    sources: list[Source]
    trace: "AgentTrace"


@dataclass(frozen=True)
class AgentTrace:
    duration_ms: int = 0
    local_result_count: int = 0
    local_top_score: float = 0.0
    arxiv_result_count: int = 0
    source_count: int = 0
    used_arxiv: bool = False
    refused: bool = False
    model_error: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    local_results: list[SearchResult]
    arxiv_results: list[ArxivPaper]
    steps: list[str]
    answer: str
    sources: list[Source]
    metrics: dict[str, object]


@dataclass
class LiteratureAgent:
    knowledge_base: LocalKnowledgeBase
    llm: object
    arxiv_search: Callable[[str, int], list[ArxivPaper]]
    local_threshold: float = 0.12
    arxiv_mode: str = "auto"

    def answer(self, question: str) -> AgentAnswer:
        graph = self._build_graph()
        start = time.perf_counter()
        try:
            state = graph.invoke({"question": question, "steps": []})
        except Exception as exc:
            if _is_model_call_error(exc):
                answer, step = _model_error_response(exc)
                return AgentAnswer(
                    answer=answer,
                    steps=[step],
                    sources=[],
                    trace=AgentTrace(
                        duration_ms=_elapsed_ms(start),
                        model_error=step,
                    ),
                )
            raise
        trace = _trace_from_state(state, _elapsed_ms(start))
        return AgentAnswer(
            answer=state.get("answer", ""),
            steps=state.get("steps", []),
            sources=state.get("sources", []),
            trace=trace,
        )

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return _FallbackGraph(self)

        graph = StateGraph(AgentState)
        graph.add_node("retrieve_local", self._retrieve_local)
        graph.add_node("search_arxiv", self._search_arxiv)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve_local")
        graph.add_conditional_edges(
            "retrieve_local",
            self._needs_arxiv,
            {True: "search_arxiv", False: "generate"},
        )
        graph.add_edge("search_arxiv", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _retrieve_local(self, state: AgentState) -> AgentState:
        results = self.knowledge_base.search(state["question"], k=4)
        steps = [*state.get("steps", []), "查询了本地论文知识库"]
        top_score = max((result.score for result in results), default=0.0)
        metrics = {
            **state.get("metrics", {}),
            "local_result_count": len(results),
            "local_top_score": top_score,
        }
        return {**state, "local_results": results, "steps": steps, "metrics": metrics}

    def _needs_arxiv(self, state: AgentState) -> bool:
        if self.arxiv_mode == "always":
            return True
        if self.arxiv_mode == "off":
            return False
        results = state.get("local_results", [])
        if not results:
            return True
        return max(result.score for result in results) < self.local_threshold

    def _search_arxiv(self, state: AgentState) -> AgentState:
        papers = self.arxiv_search(state["question"], 3)
        if papers:
            step = "本地资料不足，查询了 arXiv"
        else:
            step = "本地资料不足，arXiv 也没有返回可用结果"
        steps = [*state.get("steps", []), step]
        metrics = {
            **state.get("metrics", {}),
            "arxiv_result_count": len(papers),
            "used_arxiv": True,
        }
        return {**state, "arxiv_results": papers, "steps": steps, "metrics": metrics}

    def _generate(self, state: AgentState) -> AgentState:
        sources = self._collect_sources(state)
        steps = [*state.get("steps", []), "根据可用资料生成回答"]
        if not sources:
            steps = [*steps, "没有找到可靠来源，触发拒答机制"]
            metrics = {
                **state.get("metrics", {}),
                "source_count": 0,
                "refused": True,
            }
            return {
                **state,
                "steps": steps,
                "sources": [],
                "answer": "我没有找到可靠来源来回答这个问题，因此不能给出确定结论。",
                "metrics": metrics,
            }

        prompt = build_prompt(state["question"], sources)
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            answer, step = _model_error_response(exc)
            metrics = {
                **state.get("metrics", {}),
                "source_count": len(sources),
                "model_error": step,
            }
            return {
                **state,
                "steps": [*steps, step],
                "sources": sources,
                "answer": answer,
                "metrics": metrics,
            }
        answer = getattr(response, "content", str(response))
        metrics = {
            **state.get("metrics", {}),
            "source_count": len(sources),
            "refused": False,
        }
        return {**state, "steps": steps, "sources": sources, "answer": answer, "metrics": metrics}

    def _collect_sources(self, state: AgentState) -> list[Source]:
        sources: list[Source] = []
        for result in state.get("local_results", []):
            if result.score < self.local_threshold:
                continue
            sources.append(
                Source(
                    label=result.chunk.label,
                    text=result.chunk.text,
                )
            )
        for paper in state.get("arxiv_results", []):
            sources.append(
                Source(
                    label=f"arXiv · {paper.title} ({paper.published})",
                    text=paper.summary,
                    url=paper.url,
                )
            )
        return sources[:6]


def build_prompt(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(
        f"[{index}] {source.label}\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return f"""你是一个地震与地球物理文献问答助手。
请只根据给定资料回答问题。资料不足时，明确说明“不确定”或“资料不足”。
回答要简洁，并在关键结论后标注来源编号，例如 [1]。

问题：
{question}

资料：
{context}
"""


def _is_model_call_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "insufficient_quota",
            "current quota",
            "rate_limit",
            "429",
            "apiconnectionerror",
            "connection error",
            "timed out",
            "timeout",
        )
    )


def _model_error_response(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    if "insufficient_quota" in message or "current quota" in message:
        return (
            "模型账号当前没有可用额度，无法生成正式回答。请检查账户余额、套餐或付款设置；也可以暂时切换到其他模型。",
            "查询了资料，但模型调用因为额度不足而停止",
        )
    if "rate_limit" in message or "429" in message:
        return (
            "模型服务现在请求过多或被限流，请稍后再试。",
            "查询了资料，但模型服务暂时限制了请求",
        )
    if "apiconnectionerror" in message or "connection error" in message or "timed out" in message or "timeout" in message:
        return (
            "模型服务暂时连接不上，无法生成正式回答。系统已经完成资料检索，请稍后重试或切换模型。",
            "查询了资料，但模型服务暂时连接失败",
        )
    return (
        "模型服务调用失败，无法生成正式回答。系统已经完成资料检索，请检查模型配置后重试。",
        "查询了资料，但模型服务调用失败",
    )


def _trace_from_state(state: AgentState, duration_ms: int) -> AgentTrace:
    metrics = state.get("metrics", {})
    return AgentTrace(
        duration_ms=duration_ms,
        local_result_count=int(metrics.get("local_result_count") or 0),
        local_top_score=round(float(metrics.get("local_top_score") or 0.0), 3),
        arxiv_result_count=int(metrics.get("arxiv_result_count") or 0),
        source_count=int(metrics.get("source_count") or len(state.get("sources", []))),
        used_arxiv=bool(metrics.get("used_arxiv")),
        refused=bool(metrics.get("refused")),
        model_error=metrics.get("model_error") if isinstance(metrics.get("model_error"), str) else None,
    )


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


class _FallbackGraph:
    def __init__(self, agent: LiteratureAgent):
        self.agent = agent

    def invoke(self, state: AgentState) -> AgentState:
        state = self.agent._retrieve_local(state)
        if self.agent._needs_arxiv(state):
            state = self.agent._search_arxiv(state)
        return self.agent._generate(state)
