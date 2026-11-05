from click.testing import CliRunner
import httpx
import json
from llm.cli import cli
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    "before, after, expected_lines",
    [
        pytest.param(
            ["test/kept"],
            ["test/kept", "test/added"],
            {"Added models: openrouter/test/added"},
            id="added-only",
        ),
        pytest.param(
            ["test/kept", "test/removed"],
            ["test/kept"],
            {"Removed models: openrouter/test/removed"},
            id="removed-only",
        ),
        pytest.param(
            ["test/kept", "test/removed"],
            ["test/kept", "test/added"],
            {
                "Added models: openrouter/test/added",
                "Removed models: openrouter/test/removed",
            },
            id="added-and-removed",
        ),
        pytest.param(
            ["test/first", "test/second"],
            ["test/second", "test/first"],
            {"No changes"},
            id="unchanged",
        ),
    ],
)
def test_cli_refresh_reports_changes(
    user_path, monkeypatch, before, after, expected_lines
):
    cache_path = Path(str(user_path)) / "openrouter_models.json"
    cache_path.write_text(
        json.dumps({"data": [{"id": model_id} for model_id in before]}),
        encoding="utf-8",
    )
    refreshed = {"data": [{"id": model_id} for model_id in after]}
    requests = []

    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=refreshed)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        result = CliRunner().invoke(cli, ["openrouter", "refresh"])

    assert result.exit_code == 0, result.exception
    assert result.stdout == ""
    assert set(result.stderr.splitlines()) == expected_lines
    assert len(result.stderr.splitlines()) == len(expected_lines)
    assert requests == ["https://openrouter.ai/api/v1/models"]
    assert json.loads(cache_path.read_text(encoding="utf-8")) == refreshed
