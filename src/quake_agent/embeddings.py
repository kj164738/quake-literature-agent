from __future__ import annotations

from quake_agent.config import Settings
from quake_agent.vector_store import HashEmbeddings


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -r requirements-semantic.txt"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def build_embeddings(settings: Settings):
    if settings.active_embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            return HashEmbeddings()

        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    if settings.active_embedding_provider == "sentence_transformers":
        try:
            return SentenceTransformerEmbeddings(settings.local_embedding_model)
        except Exception:
            return HashEmbeddings()

    return HashEmbeddings()
