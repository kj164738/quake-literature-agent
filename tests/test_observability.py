from quake_agent.agent import AgentAnswer, AgentTrace, Source
from quake_agent.observability import append_run_record, build_run_record, load_recent_run_records


class FakeSettings:
    has_api_key = True
    active_provider = "deepseek"
    active_embedding_provider = "local"


def test_run_record_round_trip(tmp_path):
    result = AgentAnswer(
        answer="回答内容",
        steps=["查询了本地论文知识库", "根据可用资料生成回答"],
        sources=[Source(label="paper.md · chunk 1", text="source text")],
        trace=AgentTrace(
            duration_ms=123,
            local_result_count=4,
            local_top_score=0.87,
            arxiv_result_count=1,
            source_count=2,
            used_arxiv=True,
            refused=False,
        ),
    )
    record = build_run_record(
        question="地震预警为什么需要估计震级？",
        result=result,
        settings=FakeSettings(),
        paper_count=3,
        chunk_count=12,
        arxiv_mode="自动",
    )

    append_run_record(tmp_path, record)
    loaded = load_recent_run_records(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].question == "地震预警为什么需要估计震级？"
    assert loaded[0].duration_ms == 123
    assert loaded[0].used_arxiv is True
    assert loaded[0].source_labels == ["paper.md · chunk 1"]


def test_load_recent_run_records_skips_bad_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    assert load_recent_run_records(tmp_path) == []
