from quake_agent.document_loader import PaperChunk, split_text


def test_split_text_keeps_overlap_and_content():
    text = " ".join(f"word{i}" for i in range(220))
    chunks = split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert "word0" in chunks[0]
    assert "word219" in chunks[-1]


def test_split_text_prefers_sentence_boundaries():
    text = "第一句介绍地震预警。" + "第二句说明震级估计。" + "第三句讨论震源定位。"

    chunks = split_text(text, chunk_size=24, overlap=8)

    assert len(chunks) >= 2
    assert all(chunk.endswith(("。", "！", "？")) for chunk in chunks)
    assert "第一句介绍地震预警。" in chunks[0]


def test_split_text_preserves_paragraph_boundaries_when_possible():
    text = "第一段介绍背景。\n\n第二段介绍方法。\n\n第三段介绍结论。"

    chunks = split_text(text, chunk_size=25, overlap=0)

    assert len(chunks) >= 2
    assert "第一段介绍背景。" in chunks[0]
    assert "第二段介绍方法。" in chunks[0]
    assert "第三段介绍结论。" in chunks[-1]


def test_split_text_handles_long_unpunctuated_text_with_small_chunk_size():
    text = "x" * 90

    chunks = split_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_paper_chunk_label_includes_page_when_available():
    chunk = PaperChunk("abc", "paper.pdf", page=2, chunk_id=3)

    assert chunk.label == "paper.pdf · page 2 · chunk 3"
