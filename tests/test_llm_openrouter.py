from copy import deepcopy

import llm
import pytest
from click.testing import CliRunner
from inline_snapshot import snapshot
from llm.cli import cli
from llm_openrouter import (
    OpenRouterAsyncResponses,
    OpenRouterResponses,
    Shell,
    WebFetch,
    WebSearch,
)

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)


def response_snapshot(response):
    output = deepcopy(response.response_json["output"])
    for item in output:
        item.pop("id", None)
        if "call_id" in item:
            item["call_id"] = "<call_id>"
    usage = deepcopy(response.response_json["usage"])
    usage.pop("cost", None)
    usage.pop("cost_details", None)
    return {
        "object": response.response_json["object"],
        "model": response.response_json["model"],
        "output": output,
        "usage": usage,
    }


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("openrouter/openai/gpt-4o")
    assert isinstance(model, OpenRouterResponses)
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == snapshot("Skipper or Sundance")
    assert response_snapshot(response) == snapshot(
        {
            "object": "response",
            "model": "openai/gpt-4o",
            "output": [
                {
                    "content": [
                        {
                            "annotations": [],
                            "text": "Skipper or Sundance",
                            "type": "output_text",
                            "logprobs": [],
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "usage": {
                "input_tokens": 17,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 21,
                "is_byok": False,
            },
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
    model = llm.get_model("openrouter/openai/gpt-4.1-mini")
    response = model.prompt(
        "Describe image in three words",
        attachments=[llm.Attachment(content=TINY_PNG)],
    )
    assert str(response) == snapshot("Red green blocks")
    assert response_snapshot(response) == snapshot(
        {
            "object": "response",
            "model": "openai/gpt-4.1-mini",
            "output": [
                {
                    "content": [
                        {
                            "annotations": [],
                            "text": "Red green blocks",
                            "type": "output_text",
                            "logprobs": [],
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "usage": {
                "input_tokens": 101,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 105,
                "is_byok": False,
            },
        }
    )


@pytest.mark.vcr
def test_reasoning():
    model = llm.get_model("openrouter/openai/gpt-5-nano")
    response = model.prompt(
        "What is 2 + 2? Reply with the number only.",
        options={"reasoning_effort": "minimal", "max_tokens": 128},
    )

    assert response.text().strip() == "4"
    assert {event.type for event in response.stream_events()} == {
        "reasoning",
        "text",
    }
    assert {
        type(part).__name__
        for message in response.messages()
        for part in message.parts
    } == {"ReasoningPart", "TextPart"}
    reasoning_items = [
        item
        for item in response.response_json["output"]
        if item["type"] == "reasoning"
    ]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"]


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

    assert response_snapshot(responses[0]) == snapshot(
        {
            "object": "response",
            "model": "openai/gpt-4.1-mini",
            "output": [
                {
                    "arguments": "{}",
                    "call_id": "<call_id>",
                    "name": "llm_version",
                    "status": "completed",
                    "type": "function_call",
                }
            ],
            "usage": {
                "input_tokens": 42,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 54,
                "is_byok": False,
            },
        }
    )

    assert response_snapshot(responses[1]) == snapshot(
        {
            "object": "response",
            "model": "openai/gpt-4.1-mini",
            "output": [
                {
                    "content": [
                        {
                            "annotations": [],
                            "text": "The current LLM version is 0.0+test.",
                            "type": "output_text",
                            "logprobs": [],
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "usage": {
                "input_tokens": 65,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 80,
                "is_byok": False,
            },
        }
    )


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_responses_kwargs(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
        reasoning=True,
    )
    response = model.prompt(
        "hello",
        options={
            "provider": '{"order": ["OpenAI"]}',
            "reasoning_effort": "high",
            "reasoning_max_tokens": 512,
            "reasoning_enabled": True,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
        },
    )
    kwargs = model._build_responses_kwargs(response.prompt, stream=True)

    assert kwargs == {
        "reasoning": {
            "summary": "auto",
            "effort": "high",
            "max_tokens": 512,
            "enabled": True,
        },
        "extra_body": {
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
            "provider": {"order": ["OpenAI"]},
        },
    }


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_web_search_server_tool(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    tool = WebSearch(engine="exa", max_results=2, allowed_domains=["example.com"])
    response = model.prompt("search", tools=[tool])

    kwargs = model._build_responses_kwargs(response.prompt, stream=True)

    assert kwargs["tools"] == [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "exa",
                "max_results": 2,
                "allowed_domains": ["example.com"],
            },
        }
    ]
    assert WebSearch in model.supported_server_side_tools


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_web_fetch_server_tool(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    tool = WebFetch(
        engine="openrouter",
        max_uses=1,
        max_content_tokens=2_000,
        allowed_domains=["example.com"],
    )
    response = model.prompt("fetch https://example.com", tools=[tool])

    kwargs = model._build_responses_kwargs(response.prompt, stream=True)

    assert kwargs["tools"] == [
        {
            "type": "openrouter:web_fetch",
            "parameters": {
                "engine": "openrouter",
                "max_uses": 1,
                "max_content_tokens": 2_000,
                "allowed_domains": ["example.com"],
            },
        }
    ]
    assert WebFetch in model.supported_server_side_tools


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_shell_server_tool(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    tool = Shell(
        engine="openrouter",
        environment={"type": "container_auto"},
        sleep_after_seconds=300,
    )
    response = model.prompt("run printf hello", tools=[tool])

    kwargs = model._build_responses_kwargs(response.prompt, stream=True)

    assert kwargs["tools"] == [
        {
            "type": "openrouter:shell",
            "parameters": {
                "engine": "openrouter",
                "environment": {"type": "container_auto"},
                "sleep_after_seconds": 300,
            },
        }
    ]
    assert Shell in model.supported_server_side_tools


@pytest.mark.parametrize(
    ("option", "value"),
    (("stop", "END"), ("logit_bias", {"1": 1}), ("seed", 1)),
)
def test_unsupported_responses_options(option, value):
    model = OpenRouterResponses(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    response = model.prompt("hello", options={option: value})
    with pytest.raises(ValueError, match=option):
        model._build_responses_kwargs(response.prompt, stream=True)
