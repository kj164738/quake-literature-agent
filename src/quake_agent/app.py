from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from quake_agent.agent import LiteratureAgent
from quake_agent.arxiv_tool import search_arxiv
from quake_agent.config import load_settings
from quake_agent.document_loader import load_directory, load_documents
from quake_agent.llm import DemoLLM, MissingApiKeyError, build_chat_llm
from quake_agent.vector_store import LocalKnowledgeBase


st.set_page_config(page_title="地震文献问答 Agent", page_icon="🌋", layout="wide")


def main() -> None:
    settings = load_settings()
    st.title("地震/地球物理文献问答 Agent")
    st.caption("上传论文或使用示例资料，向系统提问，并查看它使用了哪些资料和工具。")

    with st.sidebar:
        st.header("论文资料")
        uploaded_files = st.file_uploader(
            "上传 PDF / TXT / MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        use_samples = st.checkbox("使用内置示例资料", value=True)
        st.divider()
        st.header("模型状态")
        if settings.has_api_key:
            st.success(f"已配置 {settings.active_provider.upper()} API Key")
        else:
            st.warning("未配置 API Key，将使用离线演示回答")
        st.code("streamlit run app.py", language="bash")

    chunks = _load_chunks(uploaded_files, settings.sample_dir, use_samples)
    paper_names = sorted({chunk.source for chunk in chunks})

    left, right = st.columns([0.32, 0.68])
    with left:
        st.subheader("当前知识库")
        if paper_names:
            for name in paper_names:
                st.write(f"- {name}")
            st.caption(f"共 {len(chunks)} 个文本片段")
        else:
            st.info("请上传论文，或启用内置示例资料。")

    with right:
        st.subheader("提问")
        question = st.text_area(
            "输入一个和地震、地球物理或论文内容有关的问题",
            value="地震预警系统为什么需要快速估计震级和震源位置？",
            height=110,
        )
        ask = st.button("开始回答", type="primary", disabled=not bool(chunks))

        if ask:
            with st.spinner("Agent 正在查资料并生成回答..."):
                kb = LocalKnowledgeBase(settings.chroma_dir)
                kb.build(chunks)
                try:
                    llm = build_chat_llm(settings)
                except MissingApiKeyError:
                    llm = DemoLLM()
                agent = LiteratureAgent(kb, llm, search_arxiv)
                result = agent.answer(question)

            st.markdown("### 回答")
            st.write(result.answer)

            st.markdown("### Agent 操作记录")
            for step in result.steps:
                st.write(f"- {step}")

            st.markdown("### 来源")
            if not result.sources:
                st.info("没有找到可靠来源。")
            for source in result.sources:
                with st.expander(source.label):
                    st.write(source.text)
                    if source.url:
                        st.link_button("打开来源", source.url)


def _load_chunks(uploaded_files, sample_dir: str, use_samples: bool):
    chunks = []
    if use_samples:
        chunks.extend(load_directory(sample_dir))

    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for uploaded_file in uploaded_files:
                path = Path(tmpdir) / uploaded_file.name
                path.write_bytes(uploaded_file.getbuffer())
                paths.append(path)
            chunks.extend(load_documents(paths))
    return chunks


if __name__ == "__main__":
    main()
