from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from quake_agent.document_loader import PaperChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: PaperChunk
    score: float


class HashEmbeddings:
    """Small local embedding model for demos and tests.

    It avoids external downloads. Chroma can use any object with these two methods.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class LocalKnowledgeBase:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "quake_papers",
        use_chroma: bool = True,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.use_chroma = use_chroma
        self.embeddings = HashEmbeddings()
        self._chunks: list[PaperChunk] = []
        self._chroma = None

    def build(self, chunks: list[PaperChunk]) -> None:
        self._chunks = chunks
        if not chunks:
            self._chroma = None
            return

        if not self.use_chroma:
            self._chroma = None
            return

        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                self._chroma = None
                return

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page or 0,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        ids = [f"{chunk.source}-{chunk.page or 0}-{chunk.chunk_id}" for chunk in chunks]
        self._chroma = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if not query.strip() or not self._chunks:
            return []
        if self._chroma is not None:
            docs_with_scores = self._chroma.similarity_search_with_score(query, k=k)
            results: list[SearchResult] = []
            for doc, distance in docs_with_scores:
                metadata = doc.metadata
                chunk = PaperChunk(
                    text=doc.page_content,
                    source=str(metadata.get("source", "unknown")),
                    page=int(metadata.get("page") or 0) or None,
                    chunk_id=int(metadata.get("chunk_id") or 0),
                )
                normalized_score = 1.0 / (1.0 + max(0.0, float(distance)))
                results.append(SearchResult(chunk=chunk, score=normalized_score))
            return results
        return self._fallback_search(query, k)

    def _fallback_search(self, query: str, k: int) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            terms = set(tokenize(chunk.text))
            overlap = len(query_terms & terms)
            if overlap:
                scored.append(SearchResult(chunk=chunk, score=overlap / max(len(query_terms), 1)))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return words + cjk_chars
