"""Regression tests for outbound-network diagnostics (2026-10-23 reliability audit).

Covers BUG-001 / BUG-007:
1. OpenAlex's own 502 "Bad Gateway" is a retryable upstream error, not a proxy-policy block.
2. A policy-proxy rejection is classified as such, is not retried, and names the configured proxy.
3. Upstream response bodies are not echoed to API clients (403), but OpenAlex's own JSON error
   message is surfaced for invalid requests (400).
4. Proxy hints appear only when the request is actually routed through an environment proxy.
5. Ollama failures through a proxy carry the same actionable hint.
"""

from __future__ import annotations

import io
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ideagen.api.main import app
from ideagen.core.network import looks_like_proxy_denial, proxy_for_url, proxy_hint
from ideagen.ingestion.search import OpenAlexSearchError, OpenAlexSearchProvider
from ideagen.llm.exceptions import ProviderFailureError
from ideagen.llm.ollama_provider import OllamaLLMProvider

PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy")


@pytest.fixture
def no_proxy_env(monkeypatch):
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sandbox_proxy_env(no_proxy_env, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@127.0.0.1:57643")
    monkeypatch.setenv("HTTP_PROXY", "http://user:secret@127.0.0.1:57643")


def _http_error(code: int, body: str, reason: str = "Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.openalex.org/works", code, reason, Message(), io.BytesIO(body.encode())
    )


def test_openalex_bad_gateway_is_retried_and_not_a_proxy_block(no_proxy_env):
    provider = OpenAlexSearchProvider(email="", raise_on_error=True, max_retries=3)
    errors = [_http_error(502, "<html><title>502 Bad Gateway</title></html>", "Bad Gateway") for _ in range(3)]
    with patch("urllib.request.urlopen", side_effect=errors) as mock_urlopen, patch("time.sleep"):
        with pytest.raises(OpenAlexSearchError) as exc:
            provider.search("retrieval augmented generation")
    assert mock_urlopen.call_count == 3
    assert exc.value.error_category == "service_unavailable"
    assert exc.value.is_proxy_block is False


def test_policy_proxy_denial_is_classified_not_retried_and_names_proxy(sandbox_proxy_env):
    provider = OpenAlexSearchProvider(email="", raise_on_error=True, max_retries=3)
    body = "Request to GET /works on api.openalex.org not allowed by policy"
    with patch("urllib.request.urlopen", side_effect=_http_error(502, body)) as mock_urlopen, patch("time.sleep"):
        with pytest.raises(OpenAlexSearchError) as exc:
            provider.search("zero-day attack detection")
    assert mock_urlopen.call_count == 1
    assert exc.value.error_category == "proxy_policy_block"
    assert "not allowed by policy" in exc.value.message
    assert "http://127.0.0.1:57643" in exc.value.message
    # Proxy credentials are never surfaced.
    assert "secret" not in exc.value.message


def test_search_api_does_not_echo_upstream_403_body(no_proxy_env):
    client = TestClient(app)
    upstream = _http_error(403, "internal-token=abc123 debug stack trace", "Forbidden")
    with patch("urllib.request.urlopen", side_effect=upstream):
        resp = client.get("/api/v1/search/openalex", params={"query": "rag"})
    assert resp.status_code == 403
    assert "abc123" not in resp.json()["detail"]


def test_search_api_surfaces_openalex_validation_message(no_proxy_env):
    client = TestClient(app)
    upstream = _http_error(400, '{"error": "Invalid query parameters error.", "message": "filter is not valid"}')
    with patch("urllib.request.urlopen", side_effect=upstream):
        resp = client.get("/api/v1/search/openalex", params={"query": "rag"})
    assert resp.status_code == 400
    assert "filter is not valid" in resp.json()["detail"]


def test_proxy_hint_only_when_routed_through_proxy(sandbox_proxy_env, monkeypatch):
    assert proxy_for_url("https://api.openalex.org/works") == "http://127.0.0.1:57643"
    assert "127.0.0.1:57643" in proxy_hint("http://localhost:11434/api/chat")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert proxy_for_url("http://localhost:11434/api/chat") is None
    assert proxy_hint("http://localhost:11434/api/chat") == ""


def test_proxy_hint_empty_without_proxy(no_proxy_env):
    assert proxy_for_url("https://api.openalex.org/works") is None
    assert proxy_hint("https://api.openalex.org/works") == ""


def test_looks_like_proxy_denial():
    assert looks_like_proxy_denial("Port 11434 not allowed for HTTP")
    assert looks_like_proxy_denial("Request to GET /works on api.openalex.org not allowed by policy")
    assert not looks_like_proxy_denial("502 Bad Gateway")
    assert not looks_like_proxy_denial("Gateway Timeout from upstream proxy server")


def test_ollama_error_through_proxy_is_actionable(sandbox_proxy_env):
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model_name="qwen3:8b")
    response = MagicMock(status_code=403, text="Port 11434 not allowed for HTTP\n")
    with patch("ideagen.llm.ollama_provider.requests.post", return_value=response):
        from pydantic import BaseModel

        class Out(BaseModel):
            value: str

        with pytest.raises(ProviderFailureError) as exc:
            provider.generate_structured("prompt", Out)
    message = str(exc.value)
    assert "HTTP 403" in message and "Port 11434 not allowed" in message
    assert "127.0.0.1:57643" in message and "NO_PROXY" in message


def test_ollama_missing_model_suggests_pull(no_proxy_env):
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model_name="qwen3:8b")
    response = MagicMock(status_code=404, text='{"error":"model \\"qwen3:8b\\" not found, try pulling it first"}')
    with patch("ideagen.llm.ollama_provider.requests.post", return_value=response):
        from pydantic import BaseModel

        class Out(BaseModel):
            value: str

        with pytest.raises(ProviderFailureError, match="ollama pull qwen3:8b"):
            provider.generate_structured("prompt", Out)
