from __future__ import annotations

from quake_agent.config import Settings


class MissingApiKeyError(RuntimeError):
    pass


def build_chat_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    if settings.active_provider == "openai":
        if not settings.openai_api_key:
            raise MissingApiKeyError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
            timeout=35,
            max_retries=1,
            max_tokens=2000,
        )

    if settings.active_provider != "deepseek":
        raise MissingApiKeyError("No model API key is set.")

    if not settings.deepseek_api_key:
        raise MissingApiKeyError("DEEPSEEK_API_KEY is not set.")
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0.2,
        timeout=35,
        max_retries=1,
        max_tokens=2000,
    )


class DemoLLM:
    """Offline responder used when no API key is configured."""

    is_demo = True

    def invoke(self, prompt: str):
        class Response:
            content = (
                "本次为离线预览，已完成资料检索与来源整理，未调用语言模型。"
                "下方列出了候选资料，尚未生成或核实研究结论。"
            )

        return Response()
