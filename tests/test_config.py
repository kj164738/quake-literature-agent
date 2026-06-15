from quake_agent.config import load_settings


def test_openai_is_selected_when_openai_key_exists(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.active_provider == "openai"
    assert settings.has_api_key


def test_placeholder_keys_are_ignored(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")

    settings = load_settings()

    assert settings.active_provider == "demo"
    assert not settings.has_api_key


def test_embedding_provider_defaults_to_local(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_settings()

    assert settings.active_embedding_provider == "local"


def test_openai_embedding_provider_requires_openai_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = load_settings()

    assert settings.active_embedding_provider == "openai"
