import pytest


@pytest.fixture(autouse=True)
def isolate_model_credentials(monkeypatch):
    # Tests must never inherit paid providers or private data directories from the user's .env.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("MODEL_PROVIDER", "demo")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
