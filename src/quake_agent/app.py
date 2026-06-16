from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

import streamlit as st

from quake_agent.agent import LiteratureAgent
from quake_agent.arxiv_tool import search_arxiv
from quake_agent.config import load_settings
from quake_agent.document_loader import load_directory, load_documents
from quake_agent.embeddings import build_embeddings
from quake_agent.llm import DemoLLM, MissingApiKeyError, build_chat_llm
from quake_agent.observability import append_run_record, build_run_record, load_recent_run_records
from quake_agent.paper_library import delete_paper, list_papers, save_uploaded_papers
from quake_agent.vector_store import LocalKnowledgeBase


st.set_page_config(page_title="地震文献问答 Agent", page_icon="🌋", layout="wide")


def main() -> None:
    settings = load_settings()
    _inject_styles()
    library_papers = list_papers(settings.paper_library_dir)
    selected_library_names: list[str] = []

    with st.sidebar:
        st.markdown("### 资料入口")
        uploaded_files = st.file_uploader(
            "上传 PDF / TXT / MD 到资料库",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("保存到资料库", use_container_width=True):
            saved_papers = save_uploaded_papers(uploaded_files, settings.paper_library_dir)
            st.success(f"已保存 {len(saved_papers)} 个文件")
            st.rerun()

        use_samples = st.checkbox("使用内置示例资料", value=True)
        if library_papers:
            all_library_names = [paper.name for paper in library_papers]
            selected_library_names = st.multiselect(
                "选择资料库论文",
                all_library_names,
                default=all_library_names,
                help="只有选中的论文会进入本次知识库。",
            )
        else:
            st.caption("资料库还没有保存的论文。")

        arxiv_mode_label = st.radio(
            "外部搜索",
            ["自动", "总是查询 arXiv", "关闭"],
            index=0,
            help="演示时可以选择“总是查询 arXiv”，稳定展示外部工具调用。",
        )
        st.divider()
        st.markdown("### 运行状态")
        if settings.has_api_key:
            st.success(f"已配置 {settings.active_provider.upper()} API Key")
        else:
            st.warning("未配置 API Key，将使用离线演示回答")
        if settings.active_embedding_provider == "openai":
            st.success(f"语义检索：{settings.openai_embedding_model}")
        else:
            st.info("检索：本地混合检索")
        st.caption(f"资料库：{settings.paper_library_dir}")
        st.caption(f"运行日志：{settings.log_dir}/runs.jsonl")
        st.code("streamlit run app.py", language="bash")

    selected_library_paths = [
        paper.path for paper in library_papers if paper.name in selected_library_names
    ]
    chunks = _load_chunks(settings.sample_dir, use_samples, selected_library_paths)
    paper_names = sorted({chunk.source for chunk in chunks})
    recent_records = load_recent_run_records(settings.log_dir)

    _render_header(settings, paper_names, chunks)
    left, right = st.columns([0.36, 0.64], gap="large")

    with left:
        knowledge_tab, management_tab = st.tabs(["知识库", "论文管理"])
        with knowledge_tab:
            _render_knowledge_panel(paper_names, chunks, arxiv_mode_label, settings)
        with management_tab:
            _render_paper_management(library_papers, settings.paper_library_dir)

    result = None
    run_record = None
    with right:
        with st.container(border=True):
            st.markdown("### 提问工作区")
            st.caption("本地论文优先；资料不足时，根据左侧设置决定是否查询 arXiv。")
            question = st.text_area(
                "输入一个和地震、地球物理或论文内容有关的问题",
                value="地震预警系统为什么需要快速估计震级和震源位置？",
                height=118,
            )
            ask = st.button("开始分析", type="primary", disabled=not bool(chunks), use_container_width=True)

        if ask:
            with st.spinner("Agent 正在查资料并生成回答..."):
                kb = LocalKnowledgeBase(
                    settings.chroma_dir,
                    embeddings=build_embeddings(settings),
                )
                kb.build(chunks)
                try:
                    llm = build_chat_llm(settings)
                except MissingApiKeyError:
                    llm = DemoLLM()
                agent = LiteratureAgent(
                    kb,
                    llm,
                    search_arxiv,
                    arxiv_mode=_arxiv_mode(arxiv_mode_label),
                )
                result = agent.answer(question)
                run_record = build_run_record(
                    question=question,
                    result=result,
                    settings=settings,
                    paper_count=len(paper_names),
                    chunk_count=len(chunks),
                    arxiv_mode=arxiv_mode_label,
                )
                log_saved = True
                try:
                    append_run_record(settings.log_dir, run_record)
                except OSError:
                    log_saved = False
                    st.warning("本次回答已完成，但运行日志暂时无法写入。")
                recent_records = load_recent_run_records(settings.log_dir)
                if not log_saved:
                    recent_records = [run_record, *recent_records]

    if result is not None:
        _render_result(result, run_record)
    _render_recent_runs(recent_records)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #182033;
            --muted: #697386;
            --line: #d9dee8;
            --paper: #f7f8fb;
            --panel: #ffffff;
            --accent: #c9432f;
            --accent-dark: #8f2b20;
            --green: #176b52;
            --blue: #245f9f;
            --gold: #b7842b;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(27, 42, 65, .045) 1px, transparent 1px),
                linear-gradient(180deg, rgba(27, 42, 65, .04) 1px, transparent 1px),
                var(--paper);
            background-size: 32px 32px;
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #eef1f6;
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            background:
                radial-gradient(circle at 88% 18%, rgba(201, 67, 47, .13), transparent 24%),
                linear-gradient(135deg, #ffffff 0%, #f3f6f9 64%, #edf2f5 100%);
            border-radius: 8px;
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: 0 18px 40px rgba(32, 44, 62, .08);
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -40px -80px auto;
            width: 240px;
            height: 170px;
            border: 1px solid rgba(36, 95, 159, .18);
            transform: rotate(-14deg);
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 270px;
            gap: 28px;
            align-items: end;
        }

        .hero-label {
            color: var(--accent-dark);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(34px, 4vw, 56px);
            line-height: 1.02;
            font-weight: 700;
            margin: 0 0 12px 0;
        }

        .hero-copy {
            max-width: 760px;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.7;
            margin: 0;
        }

        .workflow-card {
            border: 1px solid #dfe5ee;
            background: rgba(255, 255, 255, .72);
            border-radius: 8px;
            padding: 14px 14px 12px 14px;
        }

        .workflow-title {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 10px;
        }

        .workflow-step {
            display: grid;
            grid-template-columns: 28px 1fr;
            gap: 10px;
            align-items: center;
            padding: 8px 0;
            border-top: 1px solid #e5e9f0;
        }

        .workflow-step:first-of-type {
            border-top: 0;
        }

        .workflow-num {
            display: inline-grid;
            place-items: center;
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: #182033;
            color: #fff;
            font-size: 12px;
            font-weight: 700;
        }

        .workflow-name {
            color: var(--ink);
            font-size: 13px;
            font-weight: 650;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 22px 0;
        }

        .metric {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 82px;
            box-shadow: 0 10px 28px rgba(32, 44, 62, .055);
        }

        .metric-value {
            font-size: 24px;
            font-weight: 760;
            color: var(--ink);
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            margin-top: 3px;
        }

        .tool-panel {
            background: rgba(255, 255, 255, .92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px 20px 20px;
            margin-bottom: 18px;
            box-shadow: 0 14px 34px rgba(32, 44, 62, .07);
        }

        .paper-list {
            display: grid;
            gap: 10px;
            margin-top: 12px;
        }

        .paper-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            border: 1px solid #e0e5ee;
            background: #fbfcfd;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 14px;
        }

        .paper-name {
            word-break: break-word;
        }

        .paper-tag {
            color: var(--blue);
            background: #e9f1fb;
            border-radius: 999px;
            padding: 3px 8px;
            font-size: 12px;
            white-space: nowrap;
        }

        .library-item {
            border: 1px solid #e0e5ee;
            background: #fbfcfd;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0 8px 0;
        }

        .library-name {
            color: var(--ink);
            font-weight: 750;
            word-break: break-word;
            margin-bottom: 6px;
        }

        .library-meta {
            color: var(--muted);
            font-size: 12px;
        }

        .mode-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 4px 6px 4px 0;
            background: #f2f5f8;
            border: 1px solid #dce3eb;
            color: var(--ink);
            font-size: 12px;
        }

        .mode-pill.strong {
            color: var(--green);
            background: #e8f5ef;
            border-color: #cae5d8;
        }

        .section-note {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.65;
            margin: 6px 0 2px 0;
        }

        .stButton > button {
            border-radius: 8px;
            min-height: 42px;
            font-weight: 700;
        }

        .stButton > button[kind="primary"] {
            background: var(--accent);
            border: 1px solid var(--accent);
        }

        .stTextArea textarea {
            border-radius: 8px;
            border-color: #cfd6e2;
            background: #fbfcfe;
        }

        .answer-panel {
            background: #fff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 18px 20px;
            margin-top: 18px;
            box-shadow: 0 14px 34px rgba(32, 44, 62, .07);
        }

        .answer-text {
            font-size: 16px;
            line-height: 1.75;
            color: var(--ink);
        }

        .step-list {
            display: grid;
            gap: 8px;
            margin: 10px 0 18px 0;
        }

        .step-item {
            display: grid;
            grid-template-columns: 30px 1fr;
            gap: 10px;
            align-items: start;
            border: 1px solid #dfe5ee;
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--ink);
        }

        .step-index {
            display: inline-grid;
            place-items: center;
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: #e9f1fb;
            color: var(--blue);
            font-size: 12px;
            font-weight: 800;
        }

        .source-caption {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 10px;
        }

        .trace-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 14px 0 16px 0;
            box-shadow: 0 14px 34px rgba(32, 44, 62, .06);
        }

        .trace-title {
            color: var(--ink);
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .trace-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }

        .trace-card {
            border: 1px solid #e0e5ee;
            border-radius: 8px;
            background: #f8fafc;
            padding: 10px 12px;
        }

        .trace-card b {
            display: block;
            color: var(--ink);
            font-size: 18px;
            margin-bottom: 3px;
        }

        .trace-card span {
            color: var(--muted);
            font-size: 12px;
        }

        .trace-pills {
            margin-top: 12px;
        }

        .trace-pills span {
            display: inline-block;
            border-radius: 999px;
            background: #eef4fb;
            border: 1px solid #dbe7f5;
            color: var(--blue);
            font-size: 12px;
            padding: 5px 10px;
            margin: 0 6px 6px 0;
        }

        @media (max-width: 900px) {
            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .trace-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 20px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(settings, paper_names: list[str], chunks: list) -> None:
    model = settings.active_provider.upper() if settings.has_api_key else "DEMO"
    retrieval = settings.openai_embedding_model if settings.active_embedding_provider == "openai" else "Local hybrid"
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-grid">
                <div>
                    <div class="hero-label">Seismic Literature Agent</div>
                    <h1 class="hero-title">地震文献问答工作台</h1>
                    <p class="hero-copy">
                        面向地震与地球物理资料的专业问答工具。系统会检索本地论文、必要时查询 arXiv，
                        并把回答、操作路径和来源放在同一个工作流里。
                    </p>
                </div>
                <div class="workflow-card">
                    <div class="workflow-title">回答流程</div>
                    <div class="workflow-step"><span class="workflow-num">1</span><span class="workflow-name">检索本地论文</span></div>
                    <div class="workflow-step"><span class="workflow-num">2</span><span class="workflow-name">必要时查询 arXiv</span></div>
                    <div class="workflow-step"><span class="workflow-num">3</span><span class="workflow-name">生成带来源回答</span></div>
                </div>
            </div>
        </section>
        <div class="metric-row">
            <div class="metric"><div class="metric-value">{len(paper_names)}</div><div class="metric-label">当前资料</div></div>
            <div class="metric"><div class="metric-value">{len(chunks)}</div><div class="metric-label">文本片段</div></div>
            <div class="metric"><div class="metric-value">{model}</div><div class="metric-label">回答模型</div></div>
            <div class="metric"><div class="metric-value">{retrieval}</div><div class="metric-label">检索模式</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_knowledge_panel(paper_names: list[str], chunks: list, arxiv_mode_label: str, settings) -> None:
    chunk_counts = Counter(chunk.source for chunk in chunks)
    with st.container(border=True):
        st.markdown("### 当前知识库")
        st.markdown(
            f"""
            <span class="mode-pill strong">{settings.active_provider.upper() if settings.has_api_key else "离线演示"}</span>
            <span class="mode-pill">{settings.active_embedding_provider.upper()} 检索</span>
            <span class="mode-pill">{arxiv_mode_label}</span>
            <p class="section-note">左侧可上传论文，也可以先用内置样例完成演示。</p>
            """,
            unsafe_allow_html=True,
        )
        if paper_names:
            items = "\n".join(
                (
                    '<div class="paper-item">'
                    f'<span class="paper-name">{html.escape(name)}</span>'
                    f'<span class="paper-tag">{chunk_counts[name]} chunks</span>'
                    '</div>'
                )
                for name in paper_names
            )
            st.markdown(f'<div class="paper-list">{items}</div>', unsafe_allow_html=True)
            st.caption(f"共 {len(chunks)} 个文本片段，可用于检索和引用。")
        else:
            st.info("请上传论文，或启用内置示例资料。")


def _render_paper_management(library_papers, library_dir: str) -> None:
    with st.container(border=True):
        st.markdown("### 本地论文管理")
        st.caption("上传后保存到本地资料库，可选择是否纳入本次问答，也可以删除不需要的文件。")
        st.caption(f"目录：{Path(library_dir).name}")
        if not library_papers:
            st.info("还没有保存的论文。先在左侧上传并点击“保存到资料库”。")
            return
        for paper in library_papers:
            st.markdown(
                f"""
                <div class="library-item">
                    <div class="library-name">{html.escape(paper.name)}</div>
                    <div class="library-meta">
                        {paper.suffix} · {paper.size_label} · {paper.modified_at.strftime('%Y-%m-%d %H:%M UTC')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("删除文件", key=f"delete-{paper.name}", use_container_width=True):
                delete_paper(library_dir, paper.name)
                st.rerun()


def _render_result(result, run_record) -> None:
    st.markdown(
        f"""
        <section class="answer-panel">
            <h3>回答</h3>
            <div class="answer-text">{_format_text(result.answer)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if run_record is not None:
        _render_observability(run_record)

    with st.container(border=True):
        st.markdown("### Agent 操作记录")
        steps = "\n".join(
            (
                '<div class="step-item">'
                f'<span class="step-index">{index}</span>'
                f'<span>{html.escape(step)}</span>'
                '</div>'
            )
            for index, step in enumerate(result.steps, start=1)
        )
        st.markdown(f'<div class="step-list">{steps}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 来源")
        st.markdown('<div class="source-caption">用于支撑本次回答的论文片段或外部论文信息。</div>', unsafe_allow_html=True)
        if not result.sources:
            st.info("没有找到可靠来源。")
        for source in result.sources:
            with st.expander(source.label):
                st.write(source.text)
                if source.url:
                    st.link_button("打开来源", source.url)


def _render_observability(record) -> None:
    status = "拒答" if record.refused else "已回答"
    arxiv_status = "触发 arXiv" if record.used_arxiv else "未触发 arXiv"
    model_status = "模型异常" if record.model_error else "模型正常"
    st.markdown(
        f"""
        <section class="trace-panel">
            <div class="trace-title">运行观察</div>
            <div class="trace-grid">
                <div class="trace-card"><b>{record.duration_ms} ms</b><span>总耗时</span></div>
                <div class="trace-card"><b>{record.local_result_count}</b><span>本地命中</span></div>
                <div class="trace-card"><b>{record.local_top_score:.3f}</b><span>最高相关度</span></div>
                <div class="trace-card"><b>{record.source_count}</b><span>引用来源</span></div>
                <div class="trace-card"><b>{record.arxiv_result_count}</b><span>arXiv 返回</span></div>
            </div>
            <div class="trace-pills">
                <span>{html.escape(arxiv_status)}</span>
                <span>{html.escape(status)}</span>
                <span>{html.escape(model_status)}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_runs(records) -> None:
    with st.container(border=True):
        st.markdown("### 最近运行")
        st.caption("这些记录来自本地日志，用来复盘每次问答用了哪些资料和工具。")
        if not records:
            st.info("还没有运行记录。提问一次后，这里会显示最近结果。")
            return
        for record in records[:5]:
            title = f"{record.created_at.replace('T', ' ')} · {record.question[:48]}"
            with st.expander(title):
                st.write(record.answer_preview or "无回答摘要")
                cols = st.columns(5)
                cols[0].metric("耗时", f"{record.duration_ms} ms")
                cols[1].metric("本地命中", record.local_result_count)
                cols[2].metric("arXiv", record.arxiv_result_count)
                cols[3].metric("来源", record.source_count)
                cols[4].metric("资料", record.paper_count)
                if record.steps:
                    st.write(" / ".join(record.steps))
                if record.source_labels:
                    st.caption("来源：" + "；".join(record.source_labels[:4]))


def _format_text(value: str) -> str:
    return html.escape(value).replace("\n", "<br>")


def _load_chunks(sample_dir: str, use_samples: bool, library_paths: list[Path]):
    chunks = []
    if use_samples:
        chunks.extend(load_directory(sample_dir))
    chunks.extend(load_documents(library_paths))
    return chunks


def _arxiv_mode(label: str) -> str:
    if label == "总是查询 arXiv":
        return "always"
    if label == "关闭":
        return "off"
    return "auto"


if __name__ == "__main__":
    main()
