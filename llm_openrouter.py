import click
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional, Union, Dict, Any, List
import json
import time
import httpx
import sqlite_utils


# --- Database Setup for Saved Configurations ---

def get_db_path() -> Path:
    """Returns the path to the SQLite database for saved configs."""
    dir_path = llm.user_dir() / "openrouter_saved_configs"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / "logs.db"

def get_db() -> sqlite_utils.Database:
    """Returns a sqlite_utils Database instance."""
    return sqlite_utils.Database(get_db_path())

def _ensure_table_exists(db: sqlite_utils.Database):
    """Ensures the saved_configs table exists."""
    if "saved_configs" not in db.table_names():
        db["saved_configs"].create(
            {
                "alias": str,
                "base_model_id": str,
                "options": str, # JSON serialized options
                "created_at": str,
            },
            pk="alias",
        )

def save_config(alias: str, base_model_id: str, options: Dict[str, Any]):
    """Saves a configuration alias to the database."""
    db = get_db()
    _ensure_table_exists(db)
    db["saved_configs"].insert(
        {
            "alias": alias,
            "base_model_id": base_model_id,
            "options": json.dumps(options),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        replace=True,
    )

def get_config(alias: str) -> Optional[Dict[str, Any]]:
    """Retrieves a saved configuration by alias."""
    db = get_db()
    _ensure_table_exists(db)
    try:
        row = db["saved_configs"].get(alias)
        return {
            "alias": row["alias"],
            "base_model_id": row["base_model_id"],
            "options": json.loads(row["options"]),
            "created_at": row["created_at"],
        }
    except sqlite_utils.db.NotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def list_configs() -> List[Dict[str, Any]]:
    """Lists all saved configurations."""
    db = get_db()
    _ensure_table_exists(db)
    configs = []
    for row in db["saved_configs"].rows:
        try:
            configs.append({
                "alias": row["alias"],
                "base_model_id": row["base_model_id"],
                "options": json.loads(row["options"]),
                "created_at": row["created_at"],
            })
        except json.JSONDecodeError:
            continue
    return configs

def remove_config(alias: str):
    """Removes a saved configuration by alias."""
    db = get_db()
    _ensure_table_exists(db)
    table = db["saved_configs"]
    if table.count_where("alias = ?", [alias]) == 0:
        raise sqlite_utils.db.NotFoundError(f"Alias '{alias}' not found.")
    table.delete(alias)


def get_openrouter_models():
    models = fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]
    schema_supporting_ids = {
        model["id"]
        for model in fetch_cached_json(
            url="http://openrouter.ai/api/v1/models?supported_parameters=structured_outputs",
            path=llm.user_dir() / "openrouter_models_structured_outputs.json",
            cache_timeout=3600,
        )["data"]
    }
    # Annotate models with their schema support
    for model in models:
        model["supports_schema"] = model["id"] in schema_supporting_ids
    return models


class _mixin:
    _saved_options: Optional[Dict[str, Any]] = None
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

    def __init__(self, *args, saved_options: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_options = saved_options or {}

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        combined_options = self._saved_options.copy() if self._saved_options else {}
        if prompt.options.online is not None:
            combined_options['online'] = prompt.options.online
        if prompt.options.provider is not None:
            combined_options['provider'] = prompt.options.provider
            
        extra_body = {}
        if combined_options.get("online"):
            extra_body["plugins"] = [{"id": "web"}]
        if combined_options.get("provider"):
            provider_val = combined_options["provider"]
            if isinstance(provider_val, str):
                try:
                    provider_val = json.loads(provider_val)
                except json.JSONDecodeError:
                    provider_val = None

            if isinstance(provider_val, dict):
                extra_body["provider"] = provider_val

        kwargs.pop("provider", None)
        kwargs.pop("online", None)

        if extra_body:
            existing_extra = kwargs.get("extra_body", {})
            kwargs["extra_body"] = {**existing_extra, **extra_body}
            
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

    registered_base_models = {}

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
        
        chat_model = OpenRouterChat(**kwargs)
        async_chat_model = OpenRouterAsyncChat(**kwargs)
        register(
            chat_model,
            async_chat_model,
        )
        registered_base_models[kwargs["model_id"]] = (chat_model, async_chat_model)
    
    saved_configs = list_configs()
    
    for config in saved_configs:
        alias = config["alias"]
        base_model_id = config["base_model_id"]
        saved_options = config["options"]

        if base_model_id in registered_base_models:
            base_chat, base_async_chat = registered_base_models[base_model_id]

            try:
                alias_chat_model = OpenRouterChat(
                    model_id=base_chat.model_id,
                    model_name=base_chat.model_name,
                    vision=getattr(base_chat,'vision', False),
                    supports_schema=getattr(base_chat, 'supports_schema', False),
                    api_base=base_chat.api_base,
                    headers=base_chat.headers.copy(),
                    saved_options=saved_options
                )
                alias_async_chat_model = OpenRouterAsyncChat(
                     model_id=base_async_chat.model_id,
                     model_name=base_async_chat.model_name,
                     vision=getattr(base_async_chat, 'vision', False),
                     supports_schema=getattr(base_async_chat, 'supports_schema', False),
                     api_base=base_async_chat.api_base,
                     headers=base_async_chat.headers.copy(),
                     saved_options=saved_options
                )

                alias_chat_model.model_id = alias
                alias_async_chat_model.model_id = alias

                register(
                    alias_chat_model,
                    alias_async_chat_model
                )
            except Exception:
                continue


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
                    processed_values = []
                    for value in architecture.values():
                        if isinstance(value, list):
                            # Join list items into a comma-separated string
                            processed_values.append(", ".join(str(item) for item in value if item))
                        elif value: # Check if value is truthy (not None, empty string, etc.)
                            processed_values.append(str(value)) # Ensure it's a string
                    if processed_values: # Only add if there are values to show
                        bits.append(f"  architecture: {' '.join(processed_values)}")
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

    # New commands for saved configurations
    @openrouter.command(name="save")
    @click.argument("base_model_id")
    @click.option("--name", "-n", required=True, help="Alias name for this configuration")
    @click.option(
        "-o",
        "--option",
        "options",
        multiple=True,
        nargs=2,
        metavar="KEY VALUE",
        help="Set an option (e.g., -o online true, -o provider '{\"order\":[\"Groq\"]}')",
    )
    def save_config_command(base_model_id, name, options):
        """
        Save a model configuration with specific options under an alias.

        Example:

        llm openrouter save openrouter/mistralai/mistral-7b-instruct \\
            --name mistral-groq \\
            -o provider '{"order":["Groq"],"allow_fallbacks":false}'
        """
        parsed_options = {}
        for key, value in options:
            if value.lower() == 'true':
                parsed_value = True
            elif value.lower() == 'false':
                parsed_value = False
            else:
                parsed_value = value
            parsed_options[key] = parsed_value

        try:
            save_config(name, base_model_id, parsed_options)
            click.echo(f"Configuration '{name}' saved for model '{base_model_id}' with options: {json.dumps(parsed_options)}")
        except Exception as e:
            raise click.ClickException(f"Error saving configuration: {e}")

    @openrouter.command(name="list-saved")
    def list_saved_configs_command():
        """List all saved OpenRouter model configuration aliases."""
        try:
            configs = list_configs()
            if not configs:
                click.echo("No saved configurations found.")
                return

            click.echo("Saved OpenRouter configurations:")
            for config in configs:
                click.echo(f"- Alias: {config['alias']}")
                click.echo(f"  Base Model: {config['base_model_id']}")
                click.echo(f"  Options: {json.dumps(config['options'])}")
                click.echo(f"  Created: {config['created_at']}")
                click.echo()
        except Exception as e:
            raise click.ClickException(f"Error listing configurations: {e}")

    @openrouter.command(name="remove-saved")
    @click.argument("alias")
    def remove_saved_config_command(alias):
        """Remove a saved OpenRouter model configuration alias."""
        try:
            remove_config(alias)
            click.echo(f"Configuration alias '{alias}' removed.")
        except sqlite_utils.db.NotFoundError:
            raise click.ClickException(f"Configuration alias '{alias}' not found.")
        except Exception as e:
            raise click.ClickException(f"Error removing configuration '{alias}': {e}")


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