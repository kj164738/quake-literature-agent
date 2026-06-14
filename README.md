# 地震/地球物理文献问答 Agent

这是一个适合写进简历的 Agent 网页项目。它可以读取地震和地球物理相关论文，回答用户问题，展示答案来源，并在本地资料不足时尝试查询 arXiv。

## 项目亮点

- 支持上传 PDF、TXT、MD 资料
- 支持内置示例资料，打开后可以直接试用
- 支持 OpenAI API，也保留 DeepSeek API 选项
- 使用本地知识库回答问题，并展示参考来源
- 通过 Agent 流程判断是否需要查询外部论文
- 支持手动打开 arXiv 查询，方便演示外部工具调用
- 没有可靠资料时会说明资料不足，避免乱答
- 内置评测脚本，自动检查本地检索、外部搜索和拒答行为
- 没有 API Key 时也能进入网页查看完整流程

## 技术栈

- Python
- Streamlit
- LangChain
- LangGraph
- Chroma
- OpenAI / DeepSeek
- arXiv

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

如果你想用 OpenAI，把 `.env` 改成：

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=你的 OpenAI API Key
OPENAI_MODEL=gpt-4.1-mini
```

然后启动网页：

```powershell
streamlit run app.py
```

打开：

```text
http://localhost:8501
```

## 使用 DeepSeek

如果想切回 DeepSeek，把 `.env` 改成：

```env
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 项目流程

```mermaid
flowchart TD
    A[用户提问] --> B[查询本地论文知识库]
    B --> C{本地资料是否足够}
    C -- 足够 --> E[生成回答]
    C -- 不足 --> D[查询 arXiv]
    D --> E
    E --> F[展示回答、操作记录和来源]
```

## 示例问题

- 地震预警系统为什么需要快速估计震级和震源位置？
- 地震风险由哪些因素组成？
- 地球物理反演为什么通常是不适定问题？
- 机器学习在地震研究里能帮什么忙？
- 如果本地论文没有覆盖密集台阵监测，系统会怎么处理？
- 演示 arXiv：把侧栏“外部搜索”改成“总是查询 arXiv”，再提问“密集地震台阵如何提升小震检测？”
- 演示拒答：关闭 arXiv 后提问“明天上午北京会不会发生 7 级地震？”

## 测试

```powershell
pytest
```

## Agent 评测

项目内置了一个轻量评测 harness，用固定问题检查 Agent 的关键行为是否稳定：

```powershell
python eval_agent.py
```

它会检查三类场景：

- 本地论文能够回答的问题
- 需要展示 arXiv 外部搜索的问题
- 没有可靠来源时触发拒答的问题

## 简历表述

基于 LangChain + LangGraph 构建地震/地球物理文献问答 Agent，集成本地论文 RAG 检索与 arXiv 外部搜索工具，支持答案来源追踪和资料不足时的拒答机制；使用 Streamlit 提供可交互网页 Demo，并构建轻量评测 harness 自动验证本地检索、外部工具调用和拒答行为。
