from click.testing import CliRunner
import httpx
import json
from llm.cli import cli
import os
from pathlib import Path
import pytest

from llm_openrouter import DownloadError, fetch_cached_json

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHED_DATA = {"data": [{"id": "test/cached-model"}]}


@pytest.mark.parametrize(
    "status, content",
    [
        (200, b"<html>Temporary upstream error</html>"),
        (200, b'{"data": ['),
        (503, b'{"error": "unavailable"}'),
    ],
)
def test_failed_download_preserves_cached_models(
    tmp_path, monkeypatch, status, content
):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CACHED_DATA), encoding="utf-8")
    os.utime(path, (1, 1))
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime

    def handle(request):
        assert str(request.url) == MODELS_URL
        return httpx.Response(status, content=content)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        assert fetch_cached_json(MODELS_URL, path, 3600) == CACHED_DATA

    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime == original_mtime


def test_invalid_json_without_cache_does_not_create_empty_file(tmp_path, monkeypatch):
    path = tmp_path / "models.json"

    def handle(request):
        return httpx.Response(200, content=b'{"data": [')

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        with pytest.raises(DownloadError, match="no cache is available"):
            fetch_cached_json(MODELS_URL, path, 3600)

    assert not path.exists()


def test_successful_download_updates_cached_models(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CACHED_DATA), encoding="utf-8")
    os.utime(path, (1, 1))
    new_data = {"data": [{"id": "test/new-model"}]}

    def handle(request):
        return httpx.Response(200, json=new_data)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        assert fetch_cached_json(MODELS_URL, path, 3600) == new_data

    assert json.loads(path.read_text()) == new_data


@pytest.mark.parametrize(
    "cache_timeout, status, expected_data, expected_requests",
    [
        (3600, 200, CACHED_DATA, []),
        (0, 200, {"data": [{"id": "test/new-model"}]}, [MODELS_URL]),
        (0, 503, CACHED_DATA, [MODELS_URL]),
    ],
)
def test_fresh_cache_and_forced_refresh(
    tmp_path, monkeypatch, cache_timeout, status, expected_data, expected_requests
):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CACHED_DATA), encoding="utf-8")
    requests = []

    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(status, json={"data": [{"id": "test/new-model"}]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        assert fetch_cached_json(MODELS_URL, path, cache_timeout) == expected_data

    assert requests == expected_requests
    assert json.loads(path.read_text()) == expected_data


def test_cli_models_remains_usable_after_invalid_json(user_path, monkeypatch):
    path = Path(str(user_path)) / "openrouter_models.json"
    path.write_text(json.dumps(CACHED_DATA), encoding="utf-8")
    os.utime(path, (1, 1))
    original_bytes = path.read_bytes()
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if len(requests) == 1:
            return httpx.Response(200, text="<html>Temporary upstream error</html>")
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_openrouter.httpx.get", client.get)
        runner = CliRunner()
        for _ in range(2):
            result = runner.invoke(cli, ["openrouter", "models", "--json"])
            assert result.exit_code == 0, result.exception
            assert json.loads(result.output) == CACHED_DATA["data"]

    assert requests == [MODELS_URL, MODELS_URL]
    assert path.read_bytes() == original_bytes
