from quake_agent.document_loader import PaperChunk, split_text


def test_split_text_keeps_overlap_and_content():
    text = " ".join(f"word{i}" for i in range(220))
    chunks = split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert "word0" in chunks[0]
    assert "word219" in chunks[-1]


def test_paper_chunk_label_includes_page_when_available():
    chunk = PaperChunk("abc", "paper.pdf", page=2, chunk_id=3)

    assert chunk.label == "paper.pdf · page 2 · chunk 3"

