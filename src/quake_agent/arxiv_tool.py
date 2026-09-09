from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from defusedxml import ElementTree


@dataclass(frozen=True)
class ArxivPaper:
    title: str
    summary: str
    url: str
    published: str


class ArxivSearchError(RuntimeError):
    pass


_lock = threading.Lock()
_last_request = 0.0
_cache: dict[tuple[str, int], tuple[float, list[ArxivPaper]]] = {}
_terms = {
    "地震预警": "early warning", "震级": "magnitude", "震源": "source location",
    "密集": "dense", "台阵": "seismic array", "小震": "microearthquake",
    "检测": "detection", "监测": "monitoring", "反演": "inversion",
    "机器学习": "machine learning", "深度学习": "deep learning",
    "风险": "risk", "断层": "fault", "地震": "earthquake", "地球物理": "geophysics",
}


def build_arxiv_query(question: str) -> str:
    terms = [value for key, value in _terms.items() if key in question]
    stop = {"how", "why", "what", "the", "and", "or", "do", "does", "is", "are", "can", "for", "with", "of", "in", "to"}
    terms.extend(word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", question.lower()) if word not in stop)
    terms = list(dict.fromkeys(terms))[:12]
    specific = [term for term in terms if term not in {"earthquake", "seismology", "geophysics"}]
    terms = specific or terms
    if not terms:
        raise ArxivSearchError("问题中没有可用于 arXiv 的检索词，请补充英文专业词。")
    topic = " OR ".join(f'all:"{term}"' for term in terms)
    return f'(all:earthquake OR all:seismology OR all:geophysics) AND ({topic})'


def search_arxiv(query: str, max_results: int = 3) -> list[ArxivPaper]:
    global _last_request
    if not query.strip() or not 1 <= max_results <= 10:
        raise ValueError("Invalid arXiv query or result limit")
    search_query = build_arxiv_query(query[:4000])
    key = (search_query, max_results)
    # Serialize requests to respect arXiv's request interval and reuse recent results.
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < 900:
            return list(cached[1])
        time.sleep(max(0, 3 - (time.monotonic() - _last_request)))
        _last_request = time.monotonic()
        params = urlencode({"search_query": search_query, "max_results": max_results,
                            "sortBy": "relevance", "sortOrder": "descending"})
        request = Request("https://export.arxiv.org/api/query?" + params,
                          headers={"User-Agent": "QuakeLiteratureAgent/0.2"})
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Oversized arXiv response")
            papers = parse_arxiv_feed(payload, max_results)
        except Exception as exc:
            raise ArxivSearchError("arXiv 暂时不可用。") from exc
        if len(_cache) >= 64:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (time.monotonic(), papers)
        return list(papers)


def parse_arxiv_feed(payload: bytes, max_results: int) -> list[ArxivPaper]:
    root = ElementTree.fromstring(payload)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    seen = set()
    for entry in root.findall("a:entry", ns):
        url = entry.findtext("a:id", "", ns).replace("http://", "https://", 1)
        if "/api/errors" in url:
            raise ArxivSearchError("arXiv rejected the query")
        title = " ".join(entry.findtext("a:title", "", ns).split())
        summary = " ".join(entry.findtext("a:summary", "", ns).split())
        if not url.startswith("https://arxiv.org/abs/") or not title or not summary or url in seen:
            continue
        seen.add(url)
        papers.append(ArxivPaper(title[:500], summary[:6000], url,
                                  entry.findtext("a:published", "", ns)[:10]))
    return papers[:max_results]
