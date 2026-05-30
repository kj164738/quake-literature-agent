from __future__ import annotations

from dataclasses import dataclass, field
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


class AgentState(TypedDict, total=False):
    question: str
    local_results: list[SearchResult]
    arxiv_results: list[ArxivPaper]
    steps: list[str]
    answer: str
    sources: list[Source]


@dataclass
class LiteratureAgent:
    knowledge_base: LocalKnowledgeBase
    llm: object
    arxiv_search: Callable[[str, int], list[ArxivPaper]]
    local_threshold: float = 0.12

    def answer(self, question: str) -> AgentAnswer:
        graph = self._build_graph()
        state = graph.invoke({"question": question, "steps": []})
        return AgentAnswer(
            answer=state.get("answer", ""),
            steps=state.get("steps", []),
            sources=state.get("sources", []),
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
        return {**state, "local_results": results, "steps": steps}

    def _needs_arxiv(self, state: AgentState) -> bool:
        results = state.get("local_results", [])
        if not results:
            return True
        return max(result.score for result in results) < self.local_threshold

    def _search_arxiv(self, state: AgentState) -> AgentState:
        papers = self.arxiv_search(state["question"], 3)
        steps = [*state.get("steps", []), "本地资料不足，查询了 arXiv"]
        return {**state, "arxiv_results": papers, "steps": steps}

    def _generate(self, state: AgentState) -> AgentState:
        sources = self._collect_sources(state)
        steps = [*state.get("steps", []), "根据可用资料生成回答"]
        if not sources:
            return {
                **state,
                "steps": steps,
                "sources": [],
                "answer": "我没有找到可靠来源来回答这个问题，因此不能给出确定结论。",
            }

        prompt = build_prompt(state["question"], sources)
        response = self.llm.invoke(prompt)
        answer = getattr(response, "content", str(response))
        return {**state, "steps": steps, "sources": sources, "answer": answer}

    def _collect_sources(self, state: AgentState) -> list[Source]:
        sources: list[Source] = []
        for result in state.get("local_results", []):
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


class _FallbackGraph:
    def __init__(self, agent: LiteratureAgent):
        self.agent = agent

    def invoke(self, state: AgentState) -> AgentState:
        state = self.agent._retrieve_local(state)
        if self.agent._needs_arxiv(state):
            state = self.agent._search_arxiv(state)
        return self.agent._generate(state)

