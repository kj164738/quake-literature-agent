from quake_agent.document_loader import PaperChunk
from quake_agent.vector_store import LocalKnowledgeBase


def test_keyword_retrieval_ranks_relevant_chunk_first_without_chroma():
    kb = LocalKnowledgeBase(use_chroma=False)
    kb.build(
        [
            PaperChunk(
                text="Earthquake early warning estimates magnitude and source location.",
                source="early-warning.md",
                chunk_id=1,
            ),
            PaperChunk(
                text="Building retrofit and vulnerability reduction reduce seismic risk.",
                source="risk.md",
                chunk_id=1,
            ),
        ]
    )

    results = kb.search("magnitude source location seismic", k=2)

    assert len(results) == 2
    assert results[0].chunk.source == "early-warning.md"
    assert results[0].score > results[1].score


def test_chroma_build_failure_falls_back_to_keyword_retrieval():
    class BrokenEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("embedding service unavailable")

        def embed_query(self, text):
            raise RuntimeError("embedding service unavailable")

    kb = LocalKnowledgeBase(embeddings=BrokenEmbeddings())
    kb.build(
        [
            PaperChunk(
                text="Geophysical inversion estimates source parameters from observations.",
                source="inversion.md",
                chunk_id=1,
            )
        ]
    )

    results = kb.search("source parameters", k=1)

    assert results[0].chunk.source == "inversion.md"
