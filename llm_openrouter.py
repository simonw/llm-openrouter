import json
import time
from pathlib import Path
from typing import Optional, Union

import click
import httpx
import llm
from llm.default_plugins.openai_models import (
    AsyncChat,
    AsyncResponses,
    Chat,
    ReasoningEffortEnum,
    Responses,
)
from pydantic import Field, field_validator


def get_openrouter_models(skip_cache=False):
    models = fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=0 if skip_cache else 3600,
    )["data"]
    return models


def get_model_ids(skip_cache=False):
    return [model["id"] for model in get_openrouter_models(skip_cache=skip_cache)]


def get_supports_images(model_definition):
    try:
        return "image" in model_definition["architecture"]["input_modalities"]
    except KeyError:
        return False


def has_parameter(model_definition, parameter):
    try:
        return parameter in model_definition["supported_parameters"]
    except KeyError:
        return False


def build_openrouter_options(base_options):
    class Options(base_options):
        online: Optional[bool] = Field(
            description="Allow the model to search the web using OpenRouter",
            default=None,
        )
        provider: Optional[Union[dict, str]] = Field(
            description=("JSON object to control provider routing"),
            default=None,
        )
        reasoning_effort: Optional[ReasoningEffortEnum] = Field(
            description=(
                'One of "none", "minimal", "low", "medium", "high", "xhigh", '
                'or "max" to control reasoning effort'
            ),
            default=None,
        )
        reasoning_max_tokens: Optional[int] = Field(
            description="Specific token limit to control reasoning effort",
            default=None,
        )
        reasoning_enabled: Optional[bool] = Field(
            description="Set to true to enable reasoning with default parameters",
            default=None,
        )

        @field_validator("provider")
        def validate_provider(cls, provider):
            if provider is None:
                return None

            if isinstance(provider, str):
                try:
                    return json.loads(provider)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in provider string")
            return provider

    return Options


class _mixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Options = build_openrouter_options(self.Options)

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("provider", None)
        kwargs.pop("online", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_max_tokens", None)
        kwargs.pop("reasoning_enabled", None)
        extra_body = {}
        if prompt.options.online:
            extra_body["plugins"] = [{"id": "web"}]
        if prompt.options.provider:
            extra_body["provider"] = prompt.options.provider
        reasoning = {}
        if prompt.options.reasoning_effort:
            reasoning["effort"] = prompt.options.reasoning_effort
        if prompt.options.reasoning_max_tokens:
            reasoning["max_tokens"] = prompt.options.reasoning_max_tokens
        if prompt.options.reasoning_enabled is not None:
            reasoning["enabled"] = prompt.options.reasoning_enabled
        if reasoning:
            extra_body["reasoning"] = reasoning
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _build_responses_kwargs(self, prompt, stream):
        reasoning_effort = prompt.options.reasoning_effort
        reasoning_max_tokens = prompt.options.reasoning_max_tokens
        reasoning_enabled = prompt.options.reasoning_enabled
        online = prompt.options.online
        provider = prompt.options.provider

        kwargs = super()._build_responses_kwargs(prompt, stream)
        for key in (
            "online",
            "provider",
            "reasoning_max_tokens",
            "reasoning_enabled",
        ):
            kwargs.pop(key, None)

        unsupported = [
            key
            for key in ("stop", "logit_bias", "seed")
            if kwargs.pop(key, None) is not None
        ]
        if unsupported:
            raise ValueError(
                "The OpenRouter Responses API does not support these options: {}".format(
                    ", ".join(unsupported)
                )
            )

        reasoning = dict(kwargs.get("reasoning") or {})
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort
        if reasoning_max_tokens is not None:
            reasoning["max_tokens"] = reasoning_max_tokens
        if reasoning_enabled is not None:
            reasoning["enabled"] = reasoning_enabled
        if reasoning:
            kwargs["reasoning"] = reasoning

        if online:
            kwargs.setdefault("tools", []).append({"type": "openrouter:web_search"})

        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        for key in ("frequency_penalty", "presence_penalty"):
            value = kwargs.pop(key, None)
            if value is not None:
                extra_body[key] = value
        if provider:
            extra_body["provider"] = provider
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs


class OpenRouterChat(_mixin, Chat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


class OpenRouterAsyncChat(_mixin, AsyncChat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


class OpenRouterResponses(_mixin, Responses):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    @property
    def supported_server_side_tools(self):
        return (llm.ServerSideTool,)

    def execute(self, prompt, stream, response, conversation=None, key=None):
        if getattr(prompt.options, "chat_completions", None):
            if any(isinstance(tool, llm.ServerSideTool) for tool in prompt.tools):
                raise ValueError(
                    "Server-side tools cannot be used with chat_completions"
                )
            chat = OpenRouterChat(**self._delegate_chat_kwargs())
            yield from chat.execute(prompt, stream, response, conversation, key)
            return
        yield from super().execute(prompt, stream, response, conversation, key)

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


class OpenRouterAsyncResponses(_mixin, AsyncResponses):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    @property
    def supported_server_side_tools(self):
        return (llm.ServerSideTool,)

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        if getattr(prompt.options, "chat_completions", None):
            if any(isinstance(tool, llm.ServerSideTool) for tool in prompt.tools):
                raise ValueError(
                    "Server-side tools cannot be used with chat_completions"
                )
            chat = OpenRouterAsyncChat(**self._delegate_chat_kwargs())
            async for event in chat.execute(
                prompt, stream, response, conversation, key
            ):
                yield event
            return
        async for event in super().execute(
            prompt, stream, response, conversation, key
        ):
            yield event

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the openrouter key is set
    key = llm.get_key("", "openrouter", "OPENROUTER_KEY")
    if not key:
        return
    for model_definition in get_openrouter_models():
        supports_images = get_supports_images(model_definition)
        kwargs = dict(
            model_id="openrouter/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            vision=supports_images,
            reasoning=has_parameter(model_definition, "reasoning"),
            verbosity=has_parameter(model_definition, "verbosity"),
            supports_schema=has_parameter(model_definition, "structured_outputs"),
            supports_tools=has_parameter(model_definition, "tools"),
            api_base="https://openrouter.ai/api/v1",
            headers={
                "HTTP-Referer": "https://llm.datasette.io/",
                "X-OpenRouter-Title": "LLM",
            },
        )
        register(
            OpenRouterResponses(**kwargs),
            OpenRouterAsyncResponses(**kwargs),
        )


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)

    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        # Get the file's modification time
        mod_time = path.stat().st_mtime
        # Check if it's more than the cache_timeout old
        if time.time() - mod_time < cache_timeout:
            # If not, load the file
            with open(path, "r") as file:
                return json.load(file)

    # Try to download the data
    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()  # This will raise an HTTPError if the request fails

        # If successful, write to the file
        with open(path, "w") as file:
            json.dump(response.json(), file)

        return response.json()
    except httpx.HTTPError:
        # If there's an existing file, load it
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            # If not, raise an error
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def openrouter():
        "Commands relating to the llm-openrouter plugin"

    @openrouter.command()
    @click.option("--free", is_flag=True, help="List free models")
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(free, json_):
        "List of OpenRouter models"
        if free:
            all_models = [
                model
                for model in get_openrouter_models()
                if model["id"].endswith(":free")
            ]
        else:
            all_models = get_openrouter_models()
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            # Custom format
            for model in all_models:
                bits = []
                bits.append(f"- id: {model['id']}")
                bits.append(f"  name: {model['name']}")
                bits.append(f"  context_length: {model['context_length']:,}")
                architecture = model.get("architecture", None)
                if architecture:
                    bits.append("  architecture:")
                    for key, value in architecture.items():
                        bits.append(
                            "    "
                            + key
                            + ": "
                            + (value if isinstance(value, str) else json.dumps(value))
                        )
                bits.append(
                    f"  supports_schema: {has_parameter(model, 'structured_outputs')}"
                )
                bits.append(f"  supports_tools: {has_parameter(model, 'tools')}")
                pricing = format_pricing(model["pricing"])
                if pricing:
                    bits.append("  pricing: " + pricing)
                click.echo("\n".join(bits) + "\n")

    @openrouter.command()
    def refresh():
        "Refresh the list of available OpenRouter models"
        before = set(get_model_ids())
        after = set(get_model_ids(skip_cache=True))
        added = after - before
        removed = before - after
        if added:
            click.echo(
                f"Added models: {', '.join('openrouter/' + m for m in added)}",
                err=True,
            )
        if removed:
            click.echo(
                f"Removed models: {', '.join('openrouter/' + m for m in removed)}",
                err=True,
            )
        else:
            click.echo("No changes", err=True)

    @openrouter.command()
    @click.option("--key", help="Key to inspect")
    def key(key):
        "View information and rate limits for the current key"
        key = llm.get_key(key, "openrouter", "OPENROUTER_KEY")
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json()["data"], indent=2))


def format_price(key, price_str):
    """Format a price value with appropriate scaling and no trailing zeros."""
    price = float(price_str)

    if price == 0:
        return None

    # Determine scale based on magnitude
    if price < 0.0001:
        scale = 1000000
        suffix = "/M"
    elif price < 0.001:
        scale = 1000
        suffix = "/K"
    elif price < 1:
        scale = 1000
        suffix = "/K"
    else:
        scale = 1
        suffix = ""

    # Scale the price
    scaled_price = price * scale

    # Format without trailing zeros
    # Convert to string and remove trailing .0
    price_str = (
        f"{scaled_price:.10f}".rstrip("0").rstrip(".")
        if "." in f"{scaled_price:.10f}"
        else f"{scaled_price:.0f}"
    )

    return f"{key} ${price_str}{suffix}"


def format_pricing(pricing_dict):
    formatted_parts = []
    for key, value in pricing_dict.items():
        formatted_price = format_price(key, value)
        if formatted_price:
            formatted_parts.append(formatted_price)
    return ", ".join(formatted_parts)
