from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import time

from quake_agent.agent import AgentAnswer, AgentTrace, LiteratureAgent
from quake_agent.arxiv_tool import search_arxiv
from quake_agent.config import Settings
from quake_agent.document_loader import PaperChunk, load_documents
from quake_agent.embeddings import build_embeddings
from quake_agent.llm import DemoLLM, build_chat_llm
from quake_agent.pdf_worker import load_pdf_bounded
from quake_agent.markdown_view import escape_label, render_answer_html
from quake_agent.vector_store import HashEmbeddings, LocalKnowledgeBase


@dataclass
class Corpus:
    chunks: list[PaperChunk] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    loaded_files: list[str] = field(default_factory=list)


def load_corpus(paths: list[Path]) -> Corpus:
    corpus = Corpus()
    for path in dict.fromkeys(paths):
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("文件不存在或为链接")
            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("文件超过 20 MB")
            chunks = load_pdf_bounded(path) if path.suffix.lower() == ".pdf" else load_documents([path])
            if not chunks:
                corpus.issues.append(f"{path.name}：未提取到文本，扫描版 PDF 需要 OCR。")
                continue
            if len(corpus.chunks) + len(chunks) > 10000:
                corpus.issues.append(f"{path.name}：超出本次 10000 个片段的处理上限。")
                continue
            corpus.chunks.extend(chunks)
            corpus.loaded_files.append(str(path))
        except TimeoutError:
            corpus.issues.append(f"{path.name}：PDF 解析超时，请拆分或重新导出。")
        except Exception:
            corpus.issues.append(f"{path.name}：读取失败，请检查格式、编码、加密状态或文件大小。")
    return corpus


def run_question(settings: Settings, corpus: Corpus, question: str, arxiv_mode: str,
                 *, offline: bool = False, on_progress=None) -> AgentAnswer:
    if not question.strip() or len(question.strip()) > 4000:
        return AgentAnswer("请输入 1 至 4000 字的问题。", [], [], AgentTrace(status="invalid_input"))
    started = time.perf_counter()
    if on_progress:
        on_progress("正在准备检索索引…")
    embeddings = HashEmbeddings() if offline else build_embeddings(settings)
    kb = LocalKnowledgeBase(settings.chroma_dir, embeddings=embeddings)
    kb.build(corpus.chunks)
    fallback = not offline and settings.active_embedding_provider != "local" and isinstance(embeddings, HashEmbeddings)
    if fallback:
        kb.warning = "所选语义模型不可用，已退回关键词检索。"
    llm = DemoLLM() if offline or not settings.has_api_key else build_chat_llm(settings)
    result = LiteratureAgent(kb, llm, search_arxiv, arxiv_mode=arxiv_mode, on_progress=on_progress).answer(question)
    return replace(result, trace=replace(result.trace, duration_ms=int((time.perf_counter() - started) * 1000)))


def export_answer(question: str, result: AgentAnswer) -> str:
    safe_answer = render_answer_html(result.answer, {source.url for source in result.sources if source.url})
    lines = ["# 地震文献研究记录", "", "## 问题", "", escape_label(question), "", "## 回答", "", safe_answer,
             "", f"状态：{result.trace.status}；引用检查：{result.trace.citation_status}",
             "", "## 检索来源", ""]
    for index, source in enumerate(result.sources, 1):
        fence = "`" * max(3, max((len(run) + 1 for run in re.findall(r"`+", source.text)), default=3))
        lines.extend([f"### [{index}] {escape_label(source.label)}", "", fence + "text", source.text, fence, ""])
        if source.url:
            lines.extend([source.url, ""])
    lines.extend(["## 执行记录", "", *[f"- {step}" for step in result.steps]])
    return "\n".join(lines)
