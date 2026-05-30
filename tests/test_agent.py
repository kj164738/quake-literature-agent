from quake_agent.agent import LiteratureAgent
from quake_agent.arxiv_tool import ArxivPaper
from quake_agent.document_loader import PaperChunk
from quake_agent.vector_store import LocalKnowledgeBase


class FakeLLM:
    def invoke(self, prompt: str):
        class Response:
            content = "地震预警需要快速估计震级和震源位置，因为预警窗口很短。[1]"

        return Response()


def build_kb(chunks):
    kb = LocalKnowledgeBase()
    kb.build(chunks)
    return kb


def test_agent_uses_local_sources_when_relevant():
    chunks = [
        PaperChunk(
            text="Earthquake early warning estimates magnitude and source location before strong shaking arrives.",
            source="early-warning.md",
            chunk_id=1,
        )
    ]
    agent = LiteratureAgent(build_kb(chunks), FakeLLM(), lambda query, max_results: [])

    result = agent.answer("Why estimate magnitude and source location for earthquake warning?")

    assert "查询了本地论文知识库" in result.steps
    assert all("arXiv" not in step for step in result.steps)
    assert result.sources[0].label.startswith("early-warning.md")


def test_agent_searches_arxiv_when_local_sources_are_weak():
    chunks = [PaperChunk(text="building code retrofit vulnerability", source="risk.md", chunk_id=1)]

    def fake_arxiv(query, max_results):
        return [
            ArxivPaper(
                title="Seismic Monitoring with Dense Arrays",
                summary="Dense seismic arrays improve event detection.",
                url="https://arxiv.org/abs/0000.0000",
                published="2026-01-01",
            )
        ]

    agent = LiteratureAgent(build_kb(chunks), FakeLLM(), fake_arxiv, local_threshold=0.9)

    result = agent.answer("How do dense seismic arrays improve earthquake detection?")

    assert "本地资料不足，查询了 arXiv" in result.steps
    assert any(source.url == "https://arxiv.org/abs/0000.0000" for source in result.sources)


def test_agent_refuses_when_no_sources_exist():
    agent = LiteratureAgent(build_kb([]), FakeLLM(), lambda query, max_results: [])

    result = agent.answer("What is the exact prediction for tomorrow?")

    assert "不能给出确定结论" in result.answer
    assert result.sources == []
