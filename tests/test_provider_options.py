import pytest
from pydantic import ValidationError

from llm_openrouter import (
    OpenRouterAsyncChat,
    OpenRouterAsyncResponses,
    OpenRouterChat,
    OpenRouterResponses,
)


@pytest.fixture(
    params=(
        OpenRouterChat,
        OpenRouterAsyncChat,
        OpenRouterResponses,
        OpenRouterAsyncResponses,
    )
)
def model(request):
    return request.param(model_id="openrouter/test/model")


@pytest.mark.parametrize(
    "provider",
    (
        '["OpenAI"]',
        "[]",
        "null",
        "true",
        "false",
        "42",
        '"OpenAI"',
    ),
)
def test_provider_json_must_be_an_object(model, provider):
    with pytest.raises(ValidationError, match="provider must be a JSON object"):
        model.prompt("hello", options={"provider": provider})


@pytest.mark.parametrize(
    "provider",
    (
        {"order": ["OpenAI"], "allow_fallbacks": False},
        '{"order": ["OpenAI"], "allow_fallbacks": false}',
    ),
)
def test_provider_object_is_preserved_in_request(model, provider):
    response = model.prompt("hello", options={"provider": provider})
    if isinstance(model, (OpenRouterResponses, OpenRouterAsyncResponses)):
        kwargs = model._build_responses_kwargs(response.prompt, stream=False)
    else:
        kwargs = model.build_kwargs(response.prompt, stream=False)
    assert kwargs["extra_body"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
    }


@pytest.mark.parametrize("provider, expected", ((None, None), ({}, {}), ("{}", {})))
def test_provider_optional_or_empty_object(model, provider, expected):
    response = model.prompt("hello", options={"provider": provider})
    assert response.prompt.options.provider == expected


def test_provider_invalid_json_has_clear_error(model):
    with pytest.raises(ValidationError, match="Invalid JSON in provider string"):
        model.prompt("hello", options={"provider": "{"})
