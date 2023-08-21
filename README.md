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

You can set that as an environment variable called `OPENROUTER_KEY`, or add it to the `llm` set of saved keys using:

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
## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-openrouter
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
