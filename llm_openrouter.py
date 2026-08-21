import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Union

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
from llm.parts import StreamEvent
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


def _validate_integer(name, value, minimum, maximum=None):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validate_domains(name, domains):
    if domains is None:
        return None
    if not isinstance(domains, list):
        raise TypeError(f"{name} must be a list")
    if any(not isinstance(domain, str) or not domain for domain in domains):
        raise TypeError(f"{name} entries must be non-empty strings")
    return list(domains)


class WebSearch(llm.ServerSideTool):
    """Search the web using OpenRouter's hosted web search tool."""

    name = "web_search"
    _engines = frozenset(
        {"auto", "native", "exa", "firecrawl", "parallel", "perplexity"}
    )
    _search_context_sizes = frozenset({"low", "medium", "high"})

    def __init__(
        self,
        engine: Literal[
            "auto", "native", "exa", "firecrawl", "parallel", "perplexity"
        ]
        | None = None,
        max_results: int | None = None,
        max_uses: int | None = None,
        max_total_results: int | None = None,
        search_context_size: Literal["low", "medium", "high"] | None = None,
        max_characters: int | None = None,
        user_location: dict | None = None,
        allowed_domains: list[str] | None = None,
        excluded_domains: list[str] | None = None,
    ):
        super().__init__()
        if engine is not None and engine not in self._engines:
            raise ValueError(
                "engine must be one of: auto, native, exa, firecrawl, parallel "
                "or perplexity"
            )
        if search_context_size is not None and (
            search_context_size not in self._search_context_sizes
        ):
            raise ValueError("search_context_size must be one of: low, medium or high")
        _validate_integer("max_results", max_results, minimum=1, maximum=25)
        _validate_integer("max_uses", max_uses, minimum=1)
        _validate_integer("max_total_results", max_total_results, minimum=1)
        _validate_integer(
            "max_characters", max_characters, minimum=1, maximum=100_000
        )
        if user_location is not None and not isinstance(user_location, dict):
            raise TypeError("user_location must be a dictionary")
        self.engine = engine
        self.max_results = max_results
        self.max_uses = max_uses
        self.max_total_results = max_total_results
        self.search_context_size = search_context_size
        self.max_characters = max_characters
        self.user_location = dict(user_location) if user_location is not None else None
        self.allowed_domains = _validate_domains("allowed_domains", allowed_domains)
        self.excluded_domains = _validate_domains(
            "excluded_domains", excluded_domains
        )

    def tool_spec(self, model):
        parameters = {}
        for key in (
            "engine",
            "max_results",
            "max_uses",
            "max_total_results",
            "search_context_size",
            "max_characters",
            "user_location",
            "allowed_domains",
            "excluded_domains",
        ):
            value = getattr(self, key)
            if value is not None:
                parameters[key] = value
        spec = {"type": "openrouter:web_search"}
        if parameters:
            spec["parameters"] = parameters
        return spec


class WebFetch(llm.ServerSideTool):
    """Fetch and extract web page content using OpenRouter."""

    name = "web_fetch"
    _engines = frozenset(
        {"auto", "native", "exa", "openrouter", "firecrawl", "parallel"}
    )

    def __init__(
        self,
        engine: Literal[
            "auto", "native", "exa", "openrouter", "firecrawl", "parallel"
        ]
        | None = None,
        max_uses: int | None = None,
        max_content_tokens: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ):
        super().__init__()
        if engine is not None and engine not in self._engines:
            raise ValueError(
                "engine must be one of: auto, native, exa, openrouter, firecrawl "
                "or parallel"
            )
        _validate_integer("max_uses", max_uses, minimum=1)
        _validate_integer("max_content_tokens", max_content_tokens, minimum=1)
        self.engine = engine
        self.max_uses = max_uses
        self.max_content_tokens = max_content_tokens
        self.allowed_domains = _validate_domains("allowed_domains", allowed_domains)
        self.blocked_domains = _validate_domains("blocked_domains", blocked_domains)

    def tool_spec(self, model):
        parameters = {}
        for key in (
            "engine",
            "max_uses",
            "max_content_tokens",
            "allowed_domains",
            "blocked_domains",
        ):
            value = getattr(self, key)
            if value is not None:
                parameters[key] = value
        spec = {"type": "openrouter:web_fetch"}
        if parameters:
            spec["parameters"] = parameters
        return spec


class Shell(llm.ServerSideTool):
    """Run commands in an OpenRouter-hosted sandbox."""

    name = "shell"
    _engines = frozenset({"auto", "openrouter"})
    _environment_types = frozenset({"container_auto", "container_reference"})

    def __init__(
        self,
        engine: Literal["auto", "openrouter"] | None = None,
        environment: dict | None = None,
        sleep_after_seconds: int | None = None,
    ):
        super().__init__()
        if engine is not None and engine not in self._engines:
            raise ValueError("engine must be auto or openrouter")
        if environment is not None:
            if not isinstance(environment, dict):
                raise TypeError("environment must be a dictionary")
            environment = dict(environment)
            environment_type = environment.get("type")
            if environment_type not in self._environment_types:
                raise ValueError(
                    "environment type must be container_auto or container_reference"
                )
            if environment_type == "container_reference":
                container_id = environment.get("container_id")
                if not isinstance(container_id, str) or not container_id:
                    raise ValueError(
                        "container_reference environment requires a container_id"
                    )
        _validate_integer(
            "sleep_after_seconds",
            sleep_after_seconds,
            minimum=0,
            maximum=2_592_000,
        )
        self.engine = engine
        self.environment = environment
        self.sleep_after_seconds = sleep_after_seconds

    def tool_spec(self, model):
        parameters = {}
        for key in ("engine", "environment", "sleep_after_seconds"):
            value = getattr(self, key)
            if value is not None:
                parameters[key] = value
        spec = {"type": "openrouter:shell"}
        if parameters:
            spec["parameters"] = parameters
        return spec


def _response_item_dict(item):
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True, warnings=False)
    if isinstance(item, dict):
        return dict(item)
    return {
        key: value
        for key, value in vars(item).items()
        if value is not None and not key.startswith("_")
    }


def _openrouter_response_item(part):
    if not getattr(part, "server_executed", False):
        return None
    provider_metadata = getattr(part, "provider_metadata", None) or {}
    openrouter_metadata = provider_metadata.get("openrouter") or {}
    response_item = openrouter_metadata.get("response_item")
    if response_item is None:
        return None
    return response_item


def _response_item_key(item):
    item_type = item.get("type")
    if item_type is None:
        return None
    if item.get("id") is not None:
        return (item_type, "id", item["id"])
    if item.get("call_id") is not None:
        return (item_type, "call_id", item["call_id"])
    return None


class _PromptMessagesProxy:
    def __init__(self, prompt, messages):
        self._prompt = prompt
        self.messages = messages

    def __getattr__(self, name):
        return getattr(self._prompt, name)


class _mixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Options = build_openrouter_options(self.Options)

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("provider", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_max_tokens", None)
        kwargs.pop("reasoning_enabled", None)
        extra_body = {}
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
        provider = prompt.options.provider

        kwargs = super()._build_responses_kwargs(prompt, stream)
        for key in (
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

    def _build_responses_input(self, prompt, image_detail=None):
        """Replay raw OpenRouter server-tool items in conversation history."""
        from llm.parts import Message

        base_builder = super()._build_responses_input
        messages = prompt.messages
        if not any(
            _openrouter_response_item(part) is not None
            for message in messages
            for part in message.parts
        ):
            return base_builder(prompt, image_detail=image_detail)

        items = []
        instructions = None
        response_item_indexes = {}

        def append_ordinary_parts(message, parts):
            # Delegate ordinary text, attachments and local tools back to LLM
            # in segments so raw server-tool items retain their exact position.
            nonlocal instructions
            if not parts:
                return
            segment = Message(
                role=message.role,
                parts=parts,
                provider_metadata=message.provider_metadata,
            )
            segment_items, segment_instructions = base_builder(
                _PromptMessagesProxy(prompt, [segment]),
                image_detail=image_detail,
            )
            items.extend(segment_items)
            if segment_instructions is not None:
                instructions = segment_instructions

        for message in messages:
            ordinary_parts = []
            for part in message.parts:
                response_item = _openrouter_response_item(part)
                if response_item is None:
                    ordinary_parts.append(part)
                    continue
                response_item = deepcopy(_response_item_dict(response_item))
                response_item_key = _response_item_key(response_item)

                if (
                    response_item_key is not None
                    and response_item_key in response_item_indexes
                ):
                    items[response_item_indexes[response_item_key]] = response_item
                    continue

                if ordinary_parts:
                    append_ordinary_parts(message, ordinary_parts)
                    ordinary_parts = []

                if response_item_key is not None:
                    response_item_indexes[response_item_key] = len(items)
                items.append(response_item)

            append_ordinary_parts(message, ordinary_parts)

        return items, instructions

    def _server_tool_events(self, item, message_index):
        events = super()._server_tool_events(item, message_index)
        if events:
            response_item = _response_item_dict(item)
            events[0].provider_metadata = {
                **(events[0].provider_metadata or {}),
                "openrouter": {"response_item": response_item},
            }
            return events

        item_type = getattr(item, "type", None)
        if item_type not in (
            "openrouter:web_search",
            "openrouter:web_fetch",
            "openrouter:shell",
            "shell_call",
            "shell_call_output",
        ):
            return []
        response_item = _response_item_dict(item)
        if item_type == "shell_call":
            call_id = response_item.get("call_id") or response_item.get("id")
            return [
                StreamEvent(
                    type="tool_call_name",
                    chunk="shell",
                    tool_call_id=call_id,
                    server_executed=True,
                    provider_metadata={
                        "openrouter": {"response_item": response_item}
                    },
                    message_index=message_index,
                ),
                StreamEvent(
                    type="tool_call_args",
                    chunk=json.dumps(response_item.get("action") or {}),
                    tool_call_id=call_id,
                    server_executed=True,
                    message_index=message_index,
                ),
            ]
        if item_type == "shell_call_output":
            call_id = response_item.get("call_id") or response_item.get("id")
            return [
                StreamEvent(
                    type="tool_result",
                    chunk=json.dumps(response_item.get("output") or []),
                    tool_call_id=call_id,
                    server_executed=True,
                    tool_name="shell",
                    provider_metadata={
                        "openrouter": {"response_item": response_item}
                    },
                    message_index=message_index,
                )
            ]
        if item_type == "openrouter:shell":
            call_id = response_item.get("call_id") or response_item.get("id")
            action = response_item.get("action") or {}
            output = response_item.get("output")
            result = (
                json.dumps(output)
                if output is not None
                else (response_item.get("status") or "completed")
            )
            return [
                StreamEvent(
                    type="tool_call_name",
                    chunk="shell",
                    tool_call_id=call_id,
                    server_executed=True,
                    provider_metadata={
                        "openrouter": {"response_item": response_item}
                    },
                    message_index=message_index,
                ),
                StreamEvent(
                    type="tool_call_args",
                    chunk=json.dumps(action),
                    tool_call_id=call_id,
                    server_executed=True,
                    message_index=message_index,
                ),
                StreamEvent(
                    type="tool_result",
                    chunk=result,
                    tool_call_id=call_id,
                    server_executed=True,
                    tool_name="shell",
                    message_index=message_index,
                ),
            ]
        if item_type == "openrouter:web_fetch":
            item_id = response_item.get("id")
            result = {
                key: value
                for key, value in response_item.items()
                if key not in ("type", "id")
            }
            return [
                StreamEvent(
                    type="tool_call_name",
                    chunk="web_fetch",
                    tool_call_id=item_id,
                    server_executed=True,
                    provider_metadata={
                        "openrouter": {"response_item": response_item}
                    },
                    message_index=message_index,
                ),
                StreamEvent(
                    type="tool_call_args",
                    chunk=json.dumps({"url": response_item.get("url")}),
                    tool_call_id=item_id,
                    server_executed=True,
                    message_index=message_index,
                ),
                StreamEvent(
                    type="tool_result",
                    chunk=json.dumps(result),
                    tool_call_id=item_id,
                    server_executed=True,
                    tool_name="web_fetch",
                    message_index=message_index,
                ),
            ]
        item_id = response_item.get("id")
        action = response_item.get("action") or {}
        return [
            StreamEvent(
                type="tool_call_name",
                chunk="web_search",
                tool_call_id=item_id,
                server_executed=True,
                provider_metadata={
                    "openrouter": {"response_item": response_item}
                },
                message_index=message_index,
            ),
            StreamEvent(
                type="tool_call_args",
                chunk=json.dumps(action),
                tool_call_id=item_id,
                server_executed=True,
                message_index=message_index,
            ),
            StreamEvent(
                type="tool_result",
                chunk=response_item.get("status") or "completed",
                tool_call_id=item_id,
                server_executed=True,
                tool_name="web_search",
                message_index=message_index,
            ),
        ]

    def _refresh_server_tool_events(self, output, done_events):
        super()._refresh_server_tool_events(output, done_events)
        for item in output or []:
            item_id = getattr(item, "id", None)
            prior_events = done_events.get(item_id)
            if not prior_events:
                continue
            response_item = _response_item_dict(item)
            for event in prior_events:
                provider_metadata = event.provider_metadata or {}
                openrouter_metadata = provider_metadata.get("openrouter") or {}
                if "response_item" in openrouter_metadata:
                    openrouter_metadata["response_item"] = deepcopy(response_item)


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
        return (WebSearch, WebFetch, Shell, llm.ServerSideTool)

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
        return (WebSearch, WebFetch, Shell, llm.ServerSideTool)

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
