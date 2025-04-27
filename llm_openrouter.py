import click
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional, Union, List, Dict, Any
import json
import time
import httpx


def get_openrouter_models():
    models = fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]
    schema_supporting_ids = {
        model["id"]
        for model in fetch_cached_json(
            url="https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs",
            path=llm.user_dir() / "openrouter_models_structured_outputs.json",
            cache_timeout=3600,
        )["data"]
    }
    # Annotate models with their schema support
    for model in models:
        model["supports_schema"] = model["id"] in schema_supporting_ids
    return models


class _mixin:
    class Options(Chat.Options):
        online: Optional[bool] = Field(
            description="Use relevant search results from Exa",
            default=None,
        )
        provider: Optional[Union[dict, str]] = Field(
            description=("JSON object to control provider routing"),
            default=None,
        )
        cache_prompt: Optional[bool] = Field(
            description="Whether to cache the user prompt for future use (for supported providers like Anthropic and Gemini)",
            default=False,
        )
        cache_system: Optional[bool] = Field(
            description="Whether to cache the system prompt for future use (for supported providers like Anthropic and Gemini)",
            default=False,
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

    def __init__(self, model_id, **kwargs):
        # Initialize with standard headers
        headers = kwargs.get("headers", {})
        if not headers:
            headers = {"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"}
        kwargs["headers"] = headers
        
        # Check if it's an Anthropic model and add the appropriate header
        if "claude" in model_id.lower():
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        
        super().__init__(model_id, **kwargs)
    
    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("provider", None)
        kwargs.pop("online", None)
        kwargs.pop("cache_prompt", None)
        kwargs.pop("cache_system", None)
        
        extra_body = {}
        if prompt.options.online:
            extra_body["plugins"] = [{"id": "web"}]
        if prompt.options.provider:
            extra_body["provider"] = prompt.options.provider
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        return kwargs
    
    def _is_anthropic_model(self, prompt):
        """Check if the model is from Anthropic or if provider routing to Anthropic is specified"""
        provider = prompt.options.provider
        if provider and isinstance(provider, dict):
            return provider.get("name") == "anthropic"
        return "claude" in self.model_id.lower()
    
    def _is_gemini_model(self, prompt):
        """Check if the model is from Google Gemini or if provider routing to Gemini is specified"""
        provider = prompt.options.provider
        if provider and isinstance(provider, dict):
            return provider.get("name") == "google"
        return "gemini" in self.model_id.lower()

    def build_messages(self, prompt, conversation):
        messages = []
        current_system = None
        cache_control_count = 0
        max_cache_control_blocks = 2  # Similar to Claude's limit
        
        # Get previous responses
        if conversation is not None:
            for prev_response in conversation.responses:
                # System messages
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    if prompt.options.cache_system is not False and cache_control_count < max_cache_control_blocks:
                        # For Anthropic and Gemini, we need to format system messages with cache_control
                        if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                            system_content = [
                                {
                                    "type": "text",
                                    "text": prev_response.prompt.system,
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                            messages.append({"role": "system", "content": system_content})
                            cache_control_count += 1
                        else:
                            # For other models, use the standard format
                            messages.append({"role": "system", "content": prev_response.prompt.system})
                    else:
                        # Standard format without caching
                        messages.append({"role": "system", "content": prev_response.prompt.system})
                    current_system = prev_response.prompt.system
                
                # User messages with attachments
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        text_content = {"type": "text", "text": prev_response.prompt.prompt}
                        if prompt.options.cache_prompt is not False and cache_control_count < max_cache_control_blocks:
                            if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                                text_content["cache_control"] = {"type": "ephemeral"}
                                cache_control_count += 1
                        attachment_message.append(text_content)
                    
                    for attachment in prev_response.attachments:
                        attachment_message.append(self._format_attachment(attachment))
                    
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    # User messages without attachments
                    if prompt.options.cache_prompt is not False and cache_control_count < max_cache_control_blocks:
                        if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prev_response.prompt.prompt,
                                        "cache_control": {"type": "ephemeral"}
                                    }
                                ]
                            })
                            cache_control_count += 1
                        else:
                            # Standard format for other models
                            messages.append({"role": "user", "content": prev_response.prompt.prompt})
                    else:
                        # Standard format without caching
                        messages.append({"role": "user", "content": prev_response.prompt.prompt})
                
                # Assistant messages
                messages.append({"role": "assistant", "content": prev_response.text_or_raise()})
        
        # Current system message
        if prompt.system and prompt.system != current_system:
            if prompt.options.cache_system is not False and cache_control_count < max_cache_control_blocks:
                if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                    system_content = [
                        {
                            "type": "text",
                            "text": prompt.system,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    messages.append({"role": "system", "content": system_content})
                    cache_control_count += 1
                else:
                    # Standard format for other models
                    messages.append({"role": "system", "content": prompt.system})
            else:
                # Standard format without caching
                messages.append({"role": "system", "content": prompt.system})
        
        # Current user message
        if not prompt.attachments:
            if prompt.options.cache_prompt is not False and cache_control_count < max_cache_control_blocks:
                if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.prompt or "",
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    })
                else:
                    # Standard format for other models
                    messages.append({"role": "user", "content": prompt.prompt or ""})
            else:
                # Standard format without caching
                messages.append({"role": "user", "content": prompt.prompt or ""})
        else:
            # Handle attachments
            attachment_message = []
            if prompt.prompt:
                text_content = {"type": "text", "text": prompt.prompt}
                if prompt.options.cache_prompt is not False and cache_control_count < max_cache_control_blocks:
                    if self._is_anthropic_model(prompt) or self._is_gemini_model(prompt):
                        text_content["cache_control"] = {"type": "ephemeral"}
                attachment_message.append(text_content)
            
            for attachment in prompt.attachments:
                attachment_message.append(self._format_attachment(attachment))
            
            messages.append({"role": "user", "content": attachment_message})
        
        return messages
    
    def _format_attachment(self, attachment):
        """Format an attachment for the OpenRouter API"""
        media_type = attachment.resolve_type()
        
        if media_type.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": attachment.base64_content(),
                }
            }
        elif media_type == "application/pdf":
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": attachment.base64_content(),
                }
            }
        else:
            # Generic attachment
            return {
                "type": "file",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": attachment.base64_content(),
                }
            }


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
            supports_schema=model_definition["supports_schema"],
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(
            OpenRouterChat(**kwargs),
            OpenRouterAsyncChat(**kwargs),
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


def get_supports_images(model_definition):
    try:
        # e.g. `text->text` or `text+image->text`
        modality = model_definition["architecture"]["modality"]

        input_modalities = modality.split("->")[0].split("+")
        return "image" in input_modalities
    except Exception:
        return False


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
                bits.append(f"  supports_schema: {model['supports_schema']}")
                pricing = format_pricing(model["pricing"])
                if pricing:
                    bits.append("  pricing: " + pricing)
                click.echo("\n".join(bits) + "\n")

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
