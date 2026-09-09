from __future__ import annotations

import hashlib
import json
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
        embeddings: object | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.use_chroma = use_chroma
        self.embeddings = embeddings or HashEmbeddings()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._chunks: list[PaperChunk] = []
        self._chroma = None
        self.warning: str | None = None
        self.backend = "keyword"

    def build(self, chunks: list[PaperChunk]) -> None:
        self._chunks = list(dict.fromkeys(chunks))
        chunks = self._chunks
        self._chroma = None
        self.warning = None
        self.backend = "keyword"
        if not chunks:
            self._chroma = None
            return

        if not self.use_chroma or isinstance(self.embeddings, HashEmbeddings):
            self._chroma = None
            return

        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                self._chroma = None
                self.warning = "向量依赖不可用，已使用关键词检索。"
                return

        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page or 0,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        ids = [hashlib.sha256(_chunk_key(chunk).encode()).hexdigest() for chunk in chunks]
        identity = [type(self.embeddings).__name__, getattr(self.embeddings, "model", None),
                    getattr(self.embeddings, "model_name", None), ids]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:32]
        # An immutable corpus/model namespace prevents deleted or deselected papers leaking into retrieval.
        collection = f"{self.collection_name[:24]}_{digest}"
        try:
            from chromadb.config import Settings as ChromaSettings
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            store = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection,
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
                persist_directory=self.persist_dir,
            )
            existing = set(store.get(ids=ids)["ids"])
            missing = [i for i, item in enumerate(ids) if item not in existing]
            for start in range(0, len(missing), 256):
                batch = missing[start:start + 256]
                store.add_texts(texts=[texts[i] for i in batch],
                                metadatas=[metadatas[i] for i in batch],
                                ids=[ids[i] for i in batch])
            self._chroma = store
            self.backend = "hybrid"
        except Exception:
            self._chroma = None
            self.warning = "向量索引不可用，已使用关键词检索。"

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if k <= 0 or not query.strip() or not self._chunks:
            return []
        candidate_k = max(k * 4, k)
        keyword_results = self._keyword_search(query, k=candidate_k)
        if self._chroma is not None:
            try:
                vector_results = self._vector_search(query, k=candidate_k)
                return self._merge_results(vector_results, keyword_results, k=k)
            except Exception:
                self.warning = "向量查询失败，本次已使用关键词检索。"
                self.backend = "keyword"
        return keyword_results[:k]

    def _vector_search(self, query: str, k: int) -> list[SearchResult]:
        docs_with_scores = self._chroma.similarity_search_with_score(query, k=k)
        results: list[SearchResult] = []
        allowed = {_chunk_key(chunk) for chunk in self._chunks}
        for doc, distance in docs_with_scores:
            metadata = doc.metadata
            chunk = PaperChunk(
                text=doc.page_content,
                source=str(metadata.get("source", "unknown")),
                page=int(metadata.get("page") or 0) or None,
                chunk_id=int(metadata.get("chunk_id") or 0),
            )
            distance = float(distance)
            if not math.isfinite(distance) or _chunk_key(chunk) not in allowed:
                continue
            normalized_score = max(0.0, min(1.0, 1.0 - distance))
            if normalized_score > 0:
                results.append(SearchResult(chunk=chunk, score=normalized_score))
        return results

    def _keyword_search(self, query: str, k: int) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            terms = set(tokenize(chunk.text))
            overlap = len(query_terms & terms)
            if overlap:
                coverage = overlap / len(query_terms)
                density = overlap / math.sqrt(len(terms) + 1)
                phrase_bonus = 0.15 if query.lower() in chunk.text.lower() else 0.0
                score = min(1.0, 0.75 * coverage + 0.25 * density + phrase_bonus)
                scored.append(SearchResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        k: int,
    ) -> list[SearchResult]:
        chunks: dict[str, PaperChunk] = {}
        scores: dict[str, float] = {}
        seen_vector = set()
        seen_keyword = set()

        for result in vector_results:
            key = _chunk_key(result.chunk)
            chunks[key] = result.chunk
            scores[key] = scores.get(key, 0.0) + result.score * self.vector_weight
            seen_vector.add(key)

        for result in keyword_results:
            key = _chunk_key(result.chunk)
            chunks[key] = result.chunk
            scores[key] = scores.get(key, 0.0) + result.score * self.keyword_weight
            seen_keyword.add(key)

        for key in seen_vector & seen_keyword:
            scores[key] += 0.1

        merged = [
            SearchResult(chunk=chunks[key], score=min(1.0, score))
            for key, score in scores.items()
        ]
        return sorted(merged, key=lambda item: item.score, reverse=True)[:k]


def _chunk_key(chunk: PaperChunk) -> str:
    return json.dumps([chunk.source, chunk.page, chunk.chunk_id, chunk.text], ensure_ascii=False)


def tokenize(text: str) -> list[str]:
    stopwords = {"the", "a", "an", "and", "or", "is", "are", "of", "to", "in", "for",
                 "how", "why", "what", "do", "does", "can", "with", "from", "this", "that"}
    words = [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in stopwords]
    cjk = []
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        cjk.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    cjk = [word for word in cjk if word not in {"什么", "如何", "为什么", "可以", "是否", "需要", "哪些", "一个"}]
    return words + cjk
