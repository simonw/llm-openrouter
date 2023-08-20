from click.testing import CliRunner
from llm.cli import cli
import json
import pytest


@pytest.mark.parametrize("set_key", (False, True))
def test_llm_models(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(json.dumps({"openrouter": "x"}), "utf-8")
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "OpenRouter: openrouter/openai/gpt-3.5-turbo",
        "OpenRouter: openrouter/anthropic/claude-2",
    )
    for fragment in fragments:
        if set_key:
            assert fragment in result.output
        else:
            assert fragment not in result.output
