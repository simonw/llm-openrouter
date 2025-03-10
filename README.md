# llm-openrouter

[![PyPI](https://img.shields.io/pypi/v/llm-openrouter.svg)](https://pypi.org/project/llm-openrouter/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-openrouter?include_prereleases&label=changelog)](https://github.com/simonw/llm-openrouter/releases)
[![Tests](https://github.com/simonw/llm-openrouter/workflows/Test/badge.svg)](https://github.com/simonw/llm-openrouter/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-openrouter/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models hosted by [OpenRouter](https://openrouter.ai/)

## Installation

First, [install the LLM command-line utility](https://llm.datasette.io/en/stable/setup.html).

Now install this plugin in the same environment as LLM.
```bash
llm install llm-openrouter
```

## Configuration

You will need an API key from OpenRouter. You can [obtain one here](https://openrouter.ai/keys).

You can set that as an environment variable called `LLM_OPENROUTER_KEY`, or add it to the `llm` set of saved keys using:

```bash
llm keys set openrouter
```
```
Enter key: <paste key here>
```

## Usage

To list available models, run:
```bash
llm models list
```
You should see a list that looks something like this:
```
OpenRouter: openrouter/openai/gpt-3.5-turbo
OpenRouter: openrouter/anthropic/claude-2
OpenRouter: openrouter/meta-llama/llama-2-70b-chat
...
```
To run a prompt against a model, pass its full model ID to the `-m` option, like this:
```bash
llm -m openrouter/anthropic/claude-2 "Five spooky names for a pet tarantula"
```
You can set a shorter alias for a model using the `llm aliases` command like so:
```bash
llm aliases set claude openrouter/anthropic/claude-2
```
Now you can prompt Claude using:
```bash
cat llm_openrouter.py | llm -m claude -s 'write some pytest tests for this'
```

Images are supported too, for some models:
```bash
llm -m openrouter/anthropic/claude-3.5-sonnet 'describe this image' -a https://static.simonwillison.net/static/2024/pelicans.jpg
llm -m openrouter/anthropic/claude-3-haiku 'extract text' -a page.png
```

### Listing models

The `llm models -q openrouter` command will display all available models, or you can use this command to see more detailed JSON:

```bash
llm openrouter models
```
Output starts like this:
```json
[
  {
    "id": "microsoft/phi-4-multimodal-instruct",
    "name": "Microsoft: Phi 4 Multimodal Instruct",
    "created": 1741396284,
    "description": "Phi-4 Multimodal Instruct is a versatile...",
    "context_length": 131072,
    "architecture": {
      "modality": "text+image->text",
      "tokenizer": "Other",
      "instruct_type": null
    },
    "pricing": {
      "prompt": "0.00000007",
      "completion": "0.00000014",
      "image": "0.0002476",
      "request": "0",
      "input_cache_read": "0",
      "input_cache_write": "0",
      "web_search": "0",
      "internal_reasoning": "0"
    },
    "top_provider": {
      "context_length": 131072,
      "max_completion_tokens": null,
      "is_moderated": false
    },
    "per_request_limits": null
  }
```

### Incorporating search results from Exa

OpenRouter have [a partnership](https://openrouter.ai/docs/features/web-search) with [Exa](https://exa.ai/) where prompts through _any_ supported model can be augmented with relevant search results from the Exa index - a form of RAG.

Enable this feature using the `-o plugins_web 1` option:

```bash
llm -m openrouter/mistralai/mistral-small -o plugins_web 1 'key events on march 1st 2025'
```
Consult the OpenRouter documentation for [current pricing](https://openrouter.ai/docs/features/web-search#pricing).

### Information about your key

The `llm openrouter key-info` command shows you information about your current API key, including rate limits:

```bash
llm openrouter key-info
```
Example output:
```json
{
  "label": "sk-or-v1-0fa...240",
  "limit": null,
  "usage": 0.65017511,
  "limit_remaining": null,
  "is_free_tier": false,
  "rate_limit": {
    "requests": 40,
    "interval": "10s"
  }
}
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-openrouter
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```
To add new recordings and snapshots, run:
```bash
pytest --record-mode=once --inline-snapshot=create
```