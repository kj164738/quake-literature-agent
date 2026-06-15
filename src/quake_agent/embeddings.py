from __future__ import annotations

from quake_agent.config import Settings
from quake_agent.vector_store import HashEmbeddings


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

    return HashEmbeddings()
