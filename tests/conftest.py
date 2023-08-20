import json
import llm
import pytest

DUMMY_MODELS = {
    "data": [
        {
            "id": "openai/gpt-3.5-turbo",
            "pricing": {"prompt": "0.0000015", "completion": "0.000002"},
            "context_length": 4095,
            "per_request_limits": {
                "prompt_tokens": "2871318",
                "completion_tokens": "2153488",
            },
        },
        {
            "id": "anthropic/claude-2",
            "pricing": {"prompt": "0.00001102", "completion": "0.00003268"},
            "context_length": 100000,
            "per_request_limits": {
                "prompt_tokens": "390832",
                "completion_tokens": "131792",
            },
        },
    ]
}


@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    # Write out the models.json file
    (llm.user_dir() / "openrouter_models.json").write_text(
        json.dumps(DUMMY_MODELS), "utf-8"
    )
