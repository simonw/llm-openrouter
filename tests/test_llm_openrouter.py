from copy import deepcopy
from types import SimpleNamespace

import llm
import pytest
from click.testing import CliRunner
from inline_snapshot import snapshot
from llm.cli import cli
from llm.parts import Message, TextPart, ToolCallPart, ToolResultPart
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
def test_reasoning_summary_is_only_sent_when_explicit(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
        reasoning=True,
    )

    implicit_response = model.prompt("hello")
    implicit_kwargs = model._finalize_responses_kwargs(
        implicit_response.prompt, stream=True
    )
    assert "reasoning" not in implicit_kwargs
    assert implicit_kwargs["include"] == ["reasoning.encrypted_content"]

    explicit_response = model.prompt(
        "hello", options={"reasoning_summary": "concise"}
    )
    explicit_kwargs = model._finalize_responses_kwargs(
        explicit_response.prompt, stream=True
    )
    assert explicit_kwargs["reasoning"] == {"summary": "concise"}
    assert explicit_kwargs["include"] == ["reasoning.encrypted_content"]

    hidden_response = model.prompt(
        "hello",
        options={"reasoning_summary": "concise"},
        hide_reasoning=True,
    )
    hidden_kwargs = model._finalize_responses_kwargs(
        hidden_response.prompt, stream=True
    )
    assert "reasoning" not in hidden_kwargs


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
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_server_tool_response_items_are_replayed_in_order(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    initial_search_item = {
        "id": "ws_1",
        "type": "openrouter:web_search",
        "status": "in_progress",
        "action": {"type": "search", "query": "pelicans"},
    }
    final_search_item = {
        **initial_search_item,
        "status": "completed",
    }
    messages = [
        Message(role="system", parts=[TextPart("Be concise")]),
        Message(role="user", parts=[TextPart("Research pelicans")]),
        Message(
            role="assistant",
            parts=[
                TextPart("I'll search first."),
                ToolCallPart(
                    name="web_search",
                    arguments=initial_search_item["action"],
                    tool_call_id="ws_1",
                    server_executed=True,
                    provider_metadata={
                        "openrouter": {"response_item": initial_search_item}
                    },
                ),
                TextPart("Search finished. "),
                ToolResultPart(
                    name="web_search",
                    output="completed",
                    tool_call_id="ws_1",
                    server_executed=True,
                    provider_metadata={
                        "openrouter": {"response_item": final_search_item}
                    },
                ),
                TextPart("I'll record it."),
                ToolCallPart(
                    name="record_fact",
                    arguments={"fact": "Pelicans have large bills"},
                    tool_call_id="call_1",
                ),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    name="record_fact",
                    output="stored",
                    tool_call_id="call_1",
                )
            ],
        ),
        Message(role="assistant", parts=[TextPart("Research complete.")]),
        Message(role="user", parts=[TextPart("What did you find?")]),
    ]
    messages = [Message.from_dict(message.to_dict()) for message in messages]
    response = model.prompt(messages=messages)

    items, instructions = model._build_responses_input(response.prompt)

    assert instructions == "Be concise"
    assert items == [
        {"role": "user", "content": "Research pelicans"},
        {"role": "assistant", "content": "I'll search first."},
        final_search_item,
        {"role": "assistant", "content": "Search finished. I'll record it."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "record_fact",
            "arguments": '{"fact": "Pelicans have large bills"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "stored",
        },
        {"role": "assistant", "content": "Research complete."},
        {"role": "user", "content": "What did you find?"},
    ]


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_shell_call_and_output_are_both_replayed(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    shell_call = {
        "id": "sh_1",
        "type": "shell_call",
        "call_id": "shell_call_1",
        "action": {"commands": ["printf hello"]},
    }
    shell_output = {
        "id": "sho_1",
        "type": "shell_call_output",
        "call_id": "shell_call_1",
        "output": [{"stdout": "hello", "stderr": "", "exit_code": 0}],
    }
    messages = [
        Message(role="user", parts=[TextPart("Run a command")]),
        Message(
            role="assistant",
            parts=[
                ToolCallPart(
                    name="shell",
                    arguments=shell_call["action"],
                    tool_call_id="shell_call_1",
                    server_executed=True,
                    provider_metadata={"openrouter": {"response_item": shell_call}},
                ),
                ToolResultPart(
                    name="shell",
                    output="hello",
                    tool_call_id="shell_call_1",
                    server_executed=True,
                    provider_metadata={"openrouter": {"response_item": shell_output}},
                ),
            ],
        ),
        Message(role="user", parts=[TextPart("What was printed?")]),
    ]
    response = model.prompt(messages=messages)

    items, _ = model._build_responses_input(response.prompt)

    assert items == [
        {"role": "user", "content": "Run a command"},
        shell_call,
        shell_output,
        {"role": "user", "content": "What was printed?"},
    ]


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_streamed_server_tool_metadata_is_refreshed(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    initial_item = SimpleNamespace(
        id="wf_1",
        type="openrouter:web_fetch",
        status="in_progress",
        url="https://example.com/",
        content="partial",
    )
    final_item = SimpleNamespace(
        id="wf_1",
        type="openrouter:web_fetch",
        status="completed",
        url="https://example.com/",
        content="complete",
    )
    events = model._server_tool_events(initial_item, message_index=0)

    model._refresh_server_tool_events([final_item], {"wf_1": events})

    metadata_events = [event for event in events if event.provider_metadata]
    assert len(metadata_events) == 1
    assert metadata_events[0].provider_metadata == {
        "openrouter": {
            "response_item": {
                "id": "wf_1",
                "type": "openrouter:web_fetch",
                "status": "completed",
                "url": "https://example.com/",
                "content": "complete",
            }
        }
    }


@pytest.mark.parametrize(
    "model_class", (OpenRouterResponses, OpenRouterAsyncResponses)
)
def test_native_server_tool_response_item_is_preserved(model_class):
    model = model_class(
        model_id="openrouter/test/model",
        model_name="test/model",
        api_base="https://openrouter.ai/api/v1",
    )
    item = SimpleNamespace(
        id="ws_native_1",
        type="web_search_call",
        status="completed",
        action={"type": "search", "query": "pelicans"},
        results=[],
    )

    events = model._server_tool_events(item, message_index=0)

    metadata_events = [event for event in events if event.provider_metadata]
    assert len(metadata_events) == 1
    assert metadata_events[0].provider_metadata == {
        "openrouter": {
            "response_item": {
                "id": "ws_native_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "pelicans"},
                "results": [],
            }
        }
    }


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
