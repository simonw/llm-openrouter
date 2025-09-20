import pytest
import os
import vcr
from models_persister import TruncatedModelsFilesystemPersister

OPENROUTER_KEY = os.getenv("PYTEST_OPENROUTER_KEY", "sk-...")


def pytest_recording_configure(config, vcr):
    vcr.register_persister(TruncatedModelsFilesystemPersister)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "decode_compressed_response": True,
        "persister": TruncatedModelsFilesystemPersister,
    }


@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv("OPENROUTER_KEY", OPENROUTER_KEY)
