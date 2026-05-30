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
    )


class DemoLLM:
    """Offline responder used when no API key is configured."""

    def invoke(self, prompt: str):
        class Response:
            content = (
                "当前没有配置 OpenAI 或 DeepSeek API Key，所以这是离线演示回答。"
                "系统已经完成资料检索与来源整理；配置 API Key 后会生成正式回答。"
            )

        return Response()
