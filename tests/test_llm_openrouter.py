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
    assert str(response) == snapshot("Pebbles and Skipper.")
    response_dict = dict(response.response_json)
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "content": "Pebbles and Skipper.",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 6,
                "prompt_tokens": 17,
                "total_tokens": 23,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0.0001025,
                "is_byok": False,
            },
            "object": "chat.completion.chunk",
            "model": "openai/gpt-4o",
            "created": 1754441342,
        }
    )


@pytest.mark.vcr
def test_llm_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "OpenRouter: openrouter/openai/gpt-3.5-turbo",
        "OpenRouter: openrouter/anthropic/claude-sonnet-4",
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
    assert str(response) == snapshot("Red green geometric shapes")
    response_dict = response.response_json
    response_dict.pop("id")  # differs between requests
    assert response_dict == snapshot(
        {
            "content": "Red green geometric shapes",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens": 82,
                "total_tokens": 89,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0.000351,
                "is_byok": False,
            },
            "object": "chat.completion.chunk",
            "model": "anthropic/claude-3.5-sonnet",
            "created": 1754441344,
        }
    )


@pytest.mark.vcr
def test_tool_calls():
    model = llm.get_model("openrouter/openai/gpt-4.1-mini")

    def llm_version() -> str:
        "Return the installed version of llm"
        return "0.0+test"

    chain = model.chain(
        "What is the current llm version?",
        tools=[llm_version],
    )

    responses = list(chain.responses())

    responses[0].response_json.pop("id")  # differs between requests
    responses[0].response_json.pop("created")  # differs between requests
    assert responses[0].response_json == snapshot(
        {
            "content": "",
            "role": "assistant",
            "finish_reason": "tool_calls",
            "usage": {
                "completion_tokens": 11,
                "prompt_tokens": 48,
                "total_tokens": 59,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 3.68e-05,
                "is_byok": False,
            },
            "object": "chat.completion.chunk",
            "model": "openai/gpt-4.1-mini",
        }
    )

    responses[1].response_json.pop("id")  # differs between requests
    responses[1].response_json.pop("created")  # differs between requests
    assert responses[1].response_json == snapshot(
        {
            "content": "The current LLM version is 0.0+test.",
            "role": "assistant",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 14,
                "prompt_tokens": 73,
                "total_tokens": 87,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 5.16e-05,
                "is_byok": False,
            },
            "object": "chat.completion.chunk",
            "model": "openai/gpt-4.1-mini",
        }
    )
