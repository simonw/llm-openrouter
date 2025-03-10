import click
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional, Union
import json
import time
import httpx


def get_openrouter_models():
    return fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]


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

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("provider", None)
        kwargs.pop("online", None)
        extra_body = {}
        if prompt.options.online:
            extra_body["plugins"] = [{"id": "web"}]
        if prompt.options.provider:
            extra_body["provider"] = prompt.options.provider
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
            supports_schema=True,
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
    def models(free):
        "JSON list of OpenRouter models"
        if free:
            all_models = [
                model
                for model in get_openrouter_models()
                if model["id"].endswith(":free")
            ]
        else:
            all_models = get_openrouter_models()
        click.echo(json.dumps(all_models, indent=2))

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
