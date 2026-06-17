from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str
    openai_api_key: str | None
    openai_model: str
    embedding_provider: str
    openai_embedding_model: str
    local_embedding_model: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    chroma_dir: str
    sample_dir: str
    paper_library_dir: str
    log_dir: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.active_api_key)

    @property
    def active_provider(self) -> str:
        provider = self.model_provider.lower().strip()
        if provider in {"openai", "deepseek"}:
            return provider
        if self.openai_api_key:
            return "openai"
        if self.deepseek_api_key:
            return "deepseek"
        return "demo"

    @property
    def active_api_key(self) -> str | None:
        if self.active_provider == "openai":
            return self.openai_api_key
        if self.active_provider == "deepseek":
            return self.deepseek_api_key
        return None

    @property
    def active_embedding_provider(self) -> str:
        provider = self.embedding_provider.lower().strip()
        if provider == "openai" and self.openai_api_key:
            return "openai"
        if provider in {"sentence_transformers", "sentence-transformers", "local_semantic"}:
            return "sentence_transformers"
        return "local"


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "auto"),
        openai_api_key=_read_key("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        deepseek_api_key=_read_key("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        sample_dir=os.getenv("SAMPLE_PAPER_DIR", "sample_papers"),
        paper_library_dir=os.getenv("PAPER_LIBRARY_DIR", "paper_library"),
        log_dir=os.getenv("LOG_DIR", "logs"),
    )


def _read_key(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip()
    if not value or value.startswith("your_"):
        return None
    return value
