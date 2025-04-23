import llm
import pytest
from click.testing import CliRunner
from inline_snapshot import snapshot
from llm.cli import cli

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("openrouter/openai/gpt-4o")
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == snapshot("Beaker or Zephyr.")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "content": "Beaker or Zephyr.",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens": 17,
                "total_tokens": 24,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0.0001125,
            },
            "object": "chat.completion.chunk",
            "model": "openai/gpt-4o",
            "created": 1745444094,
        }
    )


@pytest.mark.vcr
def test_llm_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "OpenRouter: openrouter/openai/gpt-3.5-turbo",
        "OpenRouter: openrouter/anthropic/claude-2",
    )
    for fragment in fragments:
        assert fragment in result.output


@pytest.mark.vcr
def test_image_prompt():
    model = llm.get_model("openrouter/anthropic/claude-3.5-sonnet")
    response = model.prompt(
        "Describe image in three words",
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == snapshot("Bright Red Green")
    response_dict = response.response_json
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "content": "Bright Red Green",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens": 1682,
                "total_tokens": 1689,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0.005151,
            },
            "object": "chat.completion.chunk",
            "model": "anthropic/claude-3.5-sonnet",
            "created": 1745444099,
        }
    )
