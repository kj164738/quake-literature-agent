from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArxivPaper:
    title: str
    summary: str
    url: str
    published: str


def search_arxiv(query: str, max_results: int = 3) -> list[ArxivPaper]:
    try:
        import arxiv
    except ImportError:
        return []

    client = arxiv.Client()
    search = arxiv.Search(
        query=f"earthquake OR seismology OR geophysics {query}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: list[ArxivPaper] = []
    try:
        for result in client.results(search):
            papers.append(
                ArxivPaper(
                    title=result.title,
                    summary=" ".join(result.summary.split()),
                    url=result.entry_id,
                    published=result.published.date().isoformat(),
                )
            )
    except Exception:
        return []
    return papers

