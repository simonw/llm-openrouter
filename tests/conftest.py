import pytest


@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv(
        "OPENROUTER_KEY",
        "sk-...",
    )
