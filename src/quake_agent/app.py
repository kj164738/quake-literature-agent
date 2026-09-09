from __future__ import annotations

import hashlib
from dataclasses import asdict
import json
from pathlib import Path

import streamlit as st

from quake_agent.config import load_settings
from quake_agent.markdown_view import escape_label, render_answer_html
from quake_agent.observability import append_run_record, build_run_record, load_recent_run_records
from quake_agent.paper_library import delete_paper, list_papers, save_uploaded_papers
from quake_agent.service import export_answer, load_corpus, run_question


STATUS = {"answered": "已回答", "demo": "离线预览", "insufficient_evidence": "证据不足",
          "model_error": "模型调用失败", "invalid_input": "问题无效", "unknown": "旧版记录"}
MODES = {"自动补充": "auto", "总是搜索": "always", "仅本地资料": "off"}


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_corpus(signature: tuple[tuple[str, str], ...]):
    return load_corpus([Path(path) for path, _ in signature])


def _signature(paths):
    signature = []
    for path in paths:
        try:
            if not path.is_symlink() and path.stat().st_size <= 20 * 1024 * 1024:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(65536), b""):
                        hasher.update(block)
                digest = hasher.hexdigest()
            else:
                digest = "oversized"
        except OSError:
            digest = "unavailable"
        signature.append((str(path.resolve()), digest))
    return tuple(signature)


def main() -> None:
    st.set_page_config(page_title="地震文献 · 研究工作台", layout="wide")
    _styles()
    settings = load_settings()
    st.session_state.setdefault("research_history", [])
    st.session_state.setdefault("upload_revision", 0)
    papers = list_papers(settings.paper_library_dir)

    with st.sidebar:
        st.markdown("## 研究范围")
        use_samples = st.checkbox("包含示例资料", value=True)
        names = [paper.name for paper in papers]
        selected = st.multiselect("资料库论文", names, default=names)
        mode_label = st.radio("arXiv 搜索", list(MODES), index=0)
        st.divider()
        offline = st.checkbox("离线预览", value=not settings.has_api_key,
                              disabled=not settings.has_api_key)
        save_logs = st.checkbox("保存本地运行记录", value=True)
        st.caption("启用记录将保存问题、回答摘要与来源文件名。")
        st.divider()
        st.caption("回答模型")
        st.write("离线预览" if offline else settings.openai_model if settings.active_provider == "openai" else settings.deepseek_model)
        st.caption("检索配置")
        st.write({"local": "关键词", "openai": "OpenAI 语义", "sentence_transformers": "本地语义"}[settings.active_embedding_provider])
        if st.button("清空当前问答", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.research_history = []
            st.rerun()

    paths = [paper.path for paper in papers if paper.name in selected]
    if use_samples:
        sample_root = Path(settings.sample_dir)
        if sample_root.exists():
            paths = sorted(path for path in sample_root.iterdir()
                           if path.suffix.lower() in {".md", ".txt", ".pdf"}) + paths
    signature = _signature(paths)
    corpus = _cached_corpus(signature)
    st.title("地震文献研究工作台")
    st.caption(f"EARTHQUAKE / GEOPHYSICS     ·     {len(corpus.loaded_files)} 份资料     ·     {len(corpus.chunks)} 个片段")
    question_tab, library_tab, runs_tab = st.tabs(["文献问答", "资料库", "运行记录"])

    with question_tab:
        if corpus.issues:
            with st.expander(f"{len(corpus.issues)} 份资料需要处理", expanded=True):
                for issue in corpus.issues:
                    st.warning(issue)
        if offline:
            st.info("离线预览：仅检索和整理来源，不生成模型结论。")
        if not corpus.chunks:
            st.info("当前没有可用本地资料。" + ("可通过 arXiv 查询论文。" if MODES[mode_label] != "off" else "请添加论文或调整研究范围。"))
        with st.form("research_question"):
            question = st.text_area("研究问题", height=100, max_chars=4000,
                                    placeholder="地震预警系统为什么需要快速估计震级和震源位置？")
            ask = st.form_submit_button("检索并回答", icon=":material/search:", type="primary", width="stretch")
        if ask:
            if not question.strip():
                st.warning("请先输入研究问题。")
            else:
                with st.status("检索资料与检查回答…", expanded=True) as status:
                    try:
                        result = run_question(settings, corpus, question, MODES[mode_label], offline=offline, on_progress=st.write)
                        record = build_run_record(question=question, result=result, settings=settings,
                            paper_count=len(corpus.loaded_files), chunk_count=len(corpus.chunks), arxiv_mode=MODES[mode_label])
                        if offline:
                            from dataclasses import replace
                            record = replace(record, model_provider="DEMO")
                        st.session_state.research_history.append({"question": question, "result": result,
                            "record": record, "scope": signature, "mode": mode_label})
                        st.session_state.research_history = st.session_state.research_history[-20:]
                        for step in result.steps:
                            st.write(step)
                        status.update(label=STATUS[result.trace.status], state="error" if result.trace.model_error else "complete", expanded=False)
                        if save_logs:
                            try:
                                append_run_record(settings.log_dir, record)
                            except OSError:
                                st.warning("回答已保留，但本地记录写入失败。")
                    except Exception:
                        status.update(label="本次运行未完成", state="error")
                        st.error("暂时无法完成运行。请检查本地目录权限或模型配置后重试。")
        history = st.session_state.research_history
        if history:
            latest = history[-1]
            if latest["scope"] != signature:
                st.warning("资料范围已变化。下方回答使用的是提问时的资料。")
            _render_result(latest, "latest")
            if len(history) > 1:
                st.divider()
                st.subheader("本次会话")
                for index, entry in reversed(list(enumerate(history[:-1]))):
                    with st.expander(escape_label(entry["question"][:100])):
                        _render_result(entry, f"history_{index}")
        else:
            st.subheader("当前资料")
            for path in corpus.loaded_files:
                st.markdown(f"- {escape_label(Path(path).name)}")

    with library_tab:
        st.subheader("本地资料库")
        uploads = st.file_uploader("添加 PDF / TXT / MD", type=["pdf", "txt", "md"],
                                  accept_multiple_files=True, key=f"uploads_{st.session_state.upload_revision}")
        st.caption("单文件 20 MB · 单批 60 MB · PDF 最多 500 页 · 文本编码 UTF-8")
        if st.button("保存资料", icon=":material/save:", disabled=not uploads, type="primary"):
            try:
                saved = save_uploaded_papers(uploads, settings.paper_library_dir)
                st.session_state.upload_revision += 1
                st.session_state.library_notice = f"已保存 {len(saved)} 份资料。"
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "保存失败，请检查资料目录权限。")
        if st.session_state.get("library_notice"):
            st.success(st.session_state.pop("library_notice"))
        search = st.text_input("筛选文件", placeholder="文件名")
        visible = [paper for paper in papers if search.casefold() in paper.name.casefold()]
        if not visible:
            st.info("暂无匹配资料。")
        for paper in visible:
            st.divider()
            title, action = st.columns([5, 1])
            title.markdown(f"**{escape_label(paper.name)}**")
            title.caption(f"{paper.suffix} · {paper.size_label} · {paper.modified_at.strftime('%Y-%m-%d %H:%M UTC')}")
            with action:
                if st.button("删除", icon=":material/delete:", key=f"delete_{paper.name}"):
                    st.session_state.pending_delete = paper.name
            if st.session_state.get("pending_delete") == paper.name:
                st.warning(f"将从本地资料库删除：{paper.name}")
                yes, no = st.columns(2)
                if yes.button("确认删除", key=f"confirm_{paper.name}"):
                    try:
                        deleted = delete_paper(settings.paper_library_dir, paper.name)
                        st.session_state.library_notice = "已删除文件。" if deleted else "文件已不存在或不能删除。"
                        st.session_state.pop("pending_delete", None)
                        st.rerun()
                    except OSError:
                        st.error("删除失败，请检查文件是否被占用。")
                if no.button("取消", key=f"cancel_{paper.name}"):
                    st.session_state.pop("pending_delete", None)
                    st.rerun()

    with runs_tab:
        st.subheader("最近运行")
        records = load_recent_run_records(settings.log_dir, limit=50)
        if records:
            st.dataframe([{"时间": r.created_at, "问题": r.question,
                           "状态": STATUS.get(r.status, r.status), "耗时 ms": r.duration_ms,
                           "来源": r.source_count, "模型调用": r.model_calls,
                           "检索": r.retrieval_provider} for r in records],
                         hide_index=True, width="stretch")
            st.download_button("导出运行记录", json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
                               file_name="research-runs.json", mime="application/json", icon=":material/download:")
        else:
            st.info("暂无已保存的运行记录。")


def _render_result(entry, key):
    result = entry["result"]
    trace = result.trace
    st.divider()
    st.subheader(escape_label(entry["question"]))
    st.caption(f"{STATUS[trace.status]} · {trace.duration_ms} ms · {entry['mode']} · {trace.model_calls} 次模型调用")
    st.markdown(render_answer_html(result.answer, {s.url for s in result.sources if s.url}), unsafe_allow_html=True)
    if trace.tool_error:
        st.warning(trace.tool_error)
    for warning in trace.warnings:
        st.warning(warning)
    if trace.citation_status == "valid":
        st.caption("引用编号检查通过。结论与原文的对应关系仍需核对。")
    st.download_button("导出回答与来源", export_answer(entry["question"], result),
                       file_name="research-answer.md", mime="text/markdown", icon=":material/download:", key=f"export_{key}")
    evidence, execution = st.tabs([f"检索来源 ({len(result.sources)})", "执行详情"])
    with evidence:
        for index, source in enumerate(result.sources, 1):
            with st.expander(escape_label(f"[{index}] {source.label}")):
                st.text(source.text)
                if source.url:
                    st.caption("arXiv 摘要")
                    st.link_button("论文页面", source.url)
        if not result.sources:
            st.caption("无可用来源。")
    with execution:
        for index, step in enumerate(result.steps, 1):
            st.write(f"{index}. {step}")
        st.caption(f"实际检索：{trace.retrieval_backend} · 本地候选：{trace.local_result_count} · 最高检索分：{trace.local_top_score:.3f}")


def _styles():
    st.markdown("""<style>
      .stApp {background:#f8faf9;color:#202824;}
      [data-testid="stSidebar"] {background:#edf2ef;border-right:1px solid #d5dfd9;}
      .block-container {max-width:1120px;padding-top:2.2rem;padding-bottom:3rem;}
      h1 {font-size:30px!important;font-weight:650!important;}
      h2,h3 {font-size:21px!important;}
      h1,h2,h3,p,button {letter-spacing:0!important;overflow-wrap:anywhere;}
      .stButton button,.stDownloadButton button {border-radius:6px;min-height:40px;}
      button[kind="primary"],button[kind="primaryFormSubmit"] {background:#176d56;border-color:#176d56;}
      [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #d5dfd9;}
      [data-baseweb="tab"] {font-size:15px;}
      [data-baseweb="tab-highlight"] {background:#176d56;}
      textarea {background:#fff!important;}
      [data-testid="stExpander"] {background:#fff;border-radius:6px;}
      [data-testid="stMetricValue"] {font-size:24px;}
      @media(max-width:640px) {.block-container{padding:4.8rem 1rem 2rem;}h1{font-size:25px!important;} [data-baseweb="tab-list"]{gap:12px;}}
    </style>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
