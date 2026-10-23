"""Comprehensive mocked regression tests for OpenAlex search endpoint failures (F-01).

Covers:
1. Successful OpenAlex search with representative paper results.
2. Correct forwarding of query, publication dates, and result limit.
3. Upstream HTTP 403 from OpenAlex (distinguishable from proxy block).
4. Intermediary / proxy policy block (e.g. 'not allowed by policy').
5. Upstream HTTP 429 rate limit exceeded.
6. Upstream timeout (504) and 5xx server failure (502).
7. Malformed / unexpected upstream responses (invalid JSON).
"""

import io
import json
import socket
import urllib.error
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from ideagen.api.main import app
from ideagen.ingestion.search import OpenAlexSearchProvider, OpenAlexSearchError


client = TestClient(app)

MOCK_CYBERSECURITY_WORK = {
    "id": "https://openalex.org/W4402437224",
    "doi": "https://doi.org/10.1371/journal.pone.0308469",
    "title": "An intrusion detection model to detect zero-day attacks in unseen data using machine learning",
    "publication_year": 2024,
    "publication_date": "2024-08-09",
    "cited_by_count": 14,
    "abstract_inverted_index": {
        "Intrusion": [0],
        "detection": [1],
        "systems": [2],
        "face": [3],
        "zero-day": [4],
        "attacks.": [5],
    },
    "authorships": [
        {"author": {"display_name": "Alice Security"}},
        {"author": {"display_name": "Bob Analyst"}},
    ],
    "open_access": {
        "is_oa": True,
        "oa_status": "gold",
        "oa_url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0308469&type=printable",
    },
    "best_oa_location": {
        "pdf_url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0308469&type=printable",
        "landing_page_url": "https://doi.org/10.1371/journal.pone.0308469",
        "is_oa": True,
    },
    "primary_location": {
        "pdf_url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0308469&type=printable",
        "landing_page_url": "https://doi.org/10.1371/journal.pone.0308469",
        "is_oa": True,
    },
}


def _make_mock_response(status=200, data=None):
    mock_resp = MagicMock()
    mock_resp.status = status
    payload = json.dumps(data or {}).encode("utf-8")
    mock_resp.read.return_value = payload
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_search_openalex_success():
    """1. Successful OpenAlex search with representative cybersecurity paper results."""
    mock_resp = _make_mock_response(200, {"meta": {"count": 1}, "results": [MOCK_CYBERSECURITY_WORK]})
    with patch("urllib.request.urlopen", return_value=mock_resp):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day attack detection machine learning intrusion detection", "limit": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "OPENALEX"
    assert data["total_count"] == 1
    assert len(data["papers"]) == 1

    paper = data["papers"][0]
    assert paper["id"] == "W4402437224"
    assert paper["title"] == "An intrusion detection model to detect zero-day attacks in unseen data using machine learning"
    assert paper["publication_year"] == 2024
    assert paper["doi"] == "10.1371/journal.pone.0308469"
    assert paper["is_oa"] is True
    assert paper["authors"] == ["Alice Security", "Bob Analyst"]


def test_search_openalex_forwarding_dates_and_limit():
    """2. Correct forwarding of query, publication dates, and result limit."""
    mock_resp = _make_mock_response(200, {"results": [MOCK_CYBERSECURITY_WORK]})
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        response = client.get(
            "/api/v1/search/openalex",
            params={
                "query": "intrusion detection",
                "limit": 25,
                "start_date": "2023-01-01",
                "end_date": "2025-12-31",
            },
        )

    assert response.status_code == 200
    called_req = mock_urlopen.call_args[0][0]
    assert "per_page=25" in called_req.full_url
    assert "from_publication_date%3A2023-01-01" in called_req.full_url or "from_publication_date:2023-01-01" in called_req.full_url
    assert "to_publication_date%3A2025-12-31" in called_req.full_url or "to_publication_date:2025-12-31" in called_req.full_url


def test_search_openalex_proxy_policy_block():
    """3. Intermediary / proxy policy block is accurately reported as proxy block."""
    proxy_error = urllib.error.HTTPError(
        url="https://api.openalex.org/works",
        code=403,
        msg="Forbidden",
        hdrs=MagicMock(),
        fp=io.BytesIO(b"Request to GET /works on api.openalex.org not allowed by policy\n"),
    )
    with patch("urllib.request.urlopen", side_effect=proxy_error):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 502
    data = response.json()
    assert "Network proxy or policy blocked request to api.openalex.org" in data["detail"]
    assert "not allowed by policy" in data["detail"]


def test_search_openalex_upstream_403_forbidden():
    """4. Genuine upstream OpenAlex HTTP 403 Forbidden returns 403 status with actionable detail."""
    upstream_error = urllib.error.HTTPError(
        url="https://api.openalex.org/works",
        code=403,
        msg="Forbidden",
        hdrs=MagicMock(),
        fp=io.BytesIO(b'{"error": "Forbidden", "message": "API key required or cloudflare challenge"}'),
    )
    with patch("urllib.request.urlopen", side_effect=upstream_error):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 403
    data = response.json()
    assert "OpenAlex access forbidden (HTTP 403)" in data["detail"]


def test_search_openalex_upstream_429_rate_limited():
    """5. Upstream HTTP 429 rate limit returns 429 with clear retry advice."""
    rate_error = urllib.error.HTTPError(
        url="https://api.openalex.org/works",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "1"},
        fp=io.BytesIO(b"Rate limit exceeded"),
    )
    with patch("urllib.request.urlopen", side_effect=rate_error), \
         patch("time.sleep"):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 429
    data = response.json()
    assert "rate limit" in data["detail"].lower()


def test_search_openalex_upstream_timeout_504():
    """6. Upstream timeout returns 504 Gateway Timeout."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")), \
         patch("time.sleep"):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 504
    data = response.json()
    assert "timed out" in data["detail"].lower()


def test_search_openalex_upstream_503_unavailable():
    """7. Upstream 5xx returns 502 with service temporarily unavailable."""
    server_error = urllib.error.HTTPError(
        url="https://api.openalex.org/works",
        code=503,
        msg="Service Unavailable",
        hdrs=MagicMock(),
        fp=io.BytesIO(b"OpenAlex down for maintenance"),
    )
    with patch("urllib.request.urlopen", side_effect=server_error), \
         patch("time.sleep"):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 502
    data = response.json()
    assert "temporarily unavailable" in data["detail"].lower()


def test_search_openalex_malformed_json_502():
    """8. Malformed or invalid upstream JSON response returns 502."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"<html><title>Bad Gateway</title></html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        response = client.get(
            "/api/v1/search/openalex",
            params={"query": "zero-day"},
        )

    assert response.status_code == 502
    data = response.json()
    assert "invalid or malformed response" in data["detail"].lower()
