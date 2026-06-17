import sys
import types

from quake_agent.embeddings import SentenceTransformerEmbeddings, build_embeddings
from quake_agent.vector_store import HashEmbeddings


class FakeSettings:
    active_embedding_provider = "sentence_transformers"
    local_embedding_model = "missing-local-model"
    openai_api_key = None
    openai_embedding_model = "text-embedding-3-small"


def test_sentence_transformer_mode_falls_back_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    embeddings = build_embeddings(FakeSettings())

    assert isinstance(embeddings, HashEmbeddings)


def test_sentence_transformer_embeddings_wrap_model(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            class FakeArray:
                def tolist(self):
                    return [[1.0, 0.0] for _ in texts]

            return FakeArray()

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer

    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embeddings = SentenceTransformerEmbeddings("fake-model")

    assert embeddings.embed_query("地震预警") == [1.0, 0.0]
    assert embeddings.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
