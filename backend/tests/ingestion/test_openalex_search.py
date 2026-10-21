"""Focused mocked unit tests for OpenAlexSearchProvider (Stage 1).

Verifies:
1. Keyword search against OpenAlex /works API parsing title, authors, date, DOI, abstract, citations, OA, and PDF URL.
2. OpenAlex work ID normalization across full URL and short ID formats.
3. Publication-date filtering (from_publication_date, to_publication_date).
4. Direct DOI search filtering.
5. Polite pool mailto and User-Agent configuration when email is set vs unset.
6. HTTP 429 rate limit backoff and retry handling.
7. Graceful degradation on network/API failure.
"""

import json
import urllib.error
import urllib.parse
from unittest.mock import patch, MagicMock
import pytest

from ideagen.ingestion.search import OpenAlexSearchProvider, PaperSearchResult
from ideagen.core.config import Settings


MOCK_WORK_JSON = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.48550/arxiv.2005.11401",
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "publication_year": 2020,
    "publication_date": "2020-05-22",
    "cited_by_count": 1540,
    "abstract_inverted_index": {
        "We": [0],
        "explore": [1],
        "RAG": [2],
        "models": [3],
        "combining": [4],
        "parametric": [5],
        "and": [6],
        "non-parametric": [7],
        "memory.": [8]
    },
    "authorships": [
        {"author": {"display_name": "Patrick Lewis"}},
        {"author": {"display_name": "Ethan Perez"}}
    ],
    "open_access": {
        "is_oa": True,
        "oa_status": "green",
        "oa_url": "https://arxiv.org/abs/2005.11401"
    },
    "best_oa_location": {
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf",
        "landing_page_url": "https://arxiv.org/abs/2005.11401",
        "is_oa": True
    },
    "primary_location": {
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf",
        "landing_page_url": "https://arxiv.org/abs/2005.11401",
        "is_oa": True
    }
}


def _make_mock_response(status=200, data=None):
    mock_resp = MagicMock()
    mock_resp.status = status
    payload = json.dumps(data or {}).encode("utf-8")
    mock_resp.read.return_value = payload
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_openalex_search_success_with_all_fields():
    provider = OpenAlexSearchProvider(email="researcher@example.com")

    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        results = provider.search("retrieval augmented generation", limit=5)

    assert len(results) == 1
    p = results[0]

    assert p.id == "W2741809807"
    assert p.source == "OPENALEX"
    assert p.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert p.authors == ["Patrick Lewis", "Ethan Perez"]
    assert p.publication_year == 2020
    assert p.publication_date == "2020-05-22"
    assert p.abstract == "We explore RAG models combining parametric and non-parametric memory."
    assert p.doi == "10.48550/arxiv.2005.11401"
    assert p.citation_count == 1540
    assert p.is_oa is True
    assert p.pdf_url == "https://arxiv.org/pdf/2005.11401.pdf"
    assert p.url == "https://arxiv.org/abs/2005.11401"
    assert p.raw_metadata == MOCK_WORK_JSON

    # Check request headers
    called_req = mock_urlopen.call_args[0][0]
    assert called_req.get_header("User-agent") == "IdeaGen (mailto:researcher@example.com)"
    parsed_url = urllib.parse.urlparse(called_req.full_url)
    qs = urllib.parse.parse_qs(parsed_url.query)
    assert qs["search"] == ["retrieval augmented generation"]
    assert qs["per_page"] == ["5"]
    assert qs["mailto"] == ["researcher@example.com"]


def test_openalex_search_publication_date_filtering():
    provider = OpenAlexSearchProvider(email=None)

    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        results = provider.search(
            "retrieval",
            limit=10,
            start_date="2020-01-01",
            end_date="2023-12-31"
        )

    assert len(results) == 1
    called_req = mock_urlopen.call_args[0][0]
    parsed_url = urllib.parse.urlparse(called_req.full_url)
    qs = urllib.parse.parse_qs(parsed_url.query)
    assert "filter" in qs
    filters = qs["filter"][0].split(",")
    assert "from_publication_date:2020-01-01" in filters
    assert "to_publication_date:2023-12-31" in filters


def test_openalex_search_by_openalex_id_full_and_short():
    provider = OpenAlexSearchProvider()

    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    # 1. Full URL
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider.search("https://openalex.org/W2741809807")
        called_req = mock_urlopen.call_args[0][0]
        parsed_url = urllib.parse.urlparse(called_req.full_url)
        qs = urllib.parse.parse_qs(parsed_url.query)
        assert qs.get("filter") == ["openalex:W2741809807"]
        assert "search" not in qs

    # 2. Short ID
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider.search("w2741809807")
        called_req = mock_urlopen.call_args[0][0]
        parsed_url = urllib.parse.urlparse(called_req.full_url)
        qs = urllib.parse.parse_qs(parsed_url.query)
        assert qs.get("filter") == ["openalex:W2741809807"]
        assert "search" not in qs


def test_openalex_search_by_doi():
    provider = OpenAlexSearchProvider()
    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider.search("https://doi.org/10.48550/arxiv.2005.11401")
        called_req = mock_urlopen.call_args[0][0]
        parsed_url = urllib.parse.urlparse(called_req.full_url)
        qs = urllib.parse.parse_qs(parsed_url.query)
        assert qs.get("filter") == ["doi:10.48550/arxiv.2005.11401"]
        assert "search" not in qs


@pytest.mark.parametrize(
    "query",
    [
        "Wasserstein GAN",
        "What makes retrieval-augmented generation faithful",
        "W2741809807 citation analysis",
        "llama 10.5 benchmark",
        "10.5 percent hallucination rate",
        "doi.org guidelines for RAG",
        "2023",
    ],
)
def test_openalex_free_text_queries_are_not_misrouted(query):
    provider = OpenAlexSearchProvider()
    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider.search(query)
        called_req = mock_urlopen.call_args[0][0]
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(called_req.full_url).query)
        assert qs.get("search") == [query]
        assert "filter" not in qs


@pytest.mark.parametrize(
    "query,expected_filter",
    [
        ("10.48550/arXiv.2005.11401", "doi:10.48550/arxiv.2005.11401"),
        ("doi:10.1145/3442188.3445922", "doi:10.1145/3442188.3445922"),
        ("https://api.openalex.org/works/W2741809807", "openalex:W2741809807"),
        ("  W2741809807  ", "openalex:W2741809807"),
    ],
)
def test_openalex_identifier_queries_still_route_to_filters(query, expected_filter):
    provider = OpenAlexSearchProvider()
    mock_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        provider.search(query)
        called_req = mock_urlopen.call_args[0][0]
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(called_req.full_url).query)
        assert qs.get("filter") == [expected_filter]
        assert "search" not in qs


def test_openalex_search_unset_email_configuration():
    # When email is unset in settings
    with patch("ideagen.ingestion.search.get_settings") as mock_settings:
        mock_settings.return_value = Settings(openalex_email=None)
        provider = OpenAlexSearchProvider()

    assert provider.email is None
    mock_resp = _make_mock_response(200, {"results": []})
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        results = provider.search("test")

    assert results == []
    called_req = mock_urlopen.call_args[0][0]
    assert called_req.get_header("User-agent") == "IdeaGen"
    parsed_url = urllib.parse.urlparse(called_req.full_url)
    qs = urllib.parse.parse_qs(parsed_url.query)
    assert "mailto" not in qs


def test_openalex_search_rate_limit_retry():
    provider = OpenAlexSearchProvider(email="test@example.com", max_retries=3)

    err_429 = urllib.error.HTTPError(
        url="https://api.openalex.org/works",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "1"},
        fp=None,
    )
    success_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    with patch("urllib.request.urlopen", side_effect=[err_429, success_resp]) as mock_urlopen, \
         patch("time.sleep") as mock_sleep:
        results = provider.search("rag")

    assert len(results) == 1
    assert mock_urlopen.call_count == 2
    mock_sleep.assert_called_with(1.0)


def test_openalex_search_empty_query_and_network_error():
    provider = OpenAlexSearchProvider()

    # Empty query returns [] immediately
    assert provider.search("") == []
    assert provider.search("   ") == []

    # Network error returns [] gracefully
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")):
        assert provider.search("rag") == []


def test_openalex_retry_after_is_capped():
    from ideagen.ingestion.search import MAX_RETRY_DELAY_SECONDS

    provider = OpenAlexSearchProvider(max_retries=2)
    err_429 = urllib.error.HTTPError(
        url="https://api.openalex.org/works", code=429, msg="Too Many Requests",
        hdrs={"Retry-After": "3600"}, fp=None,
    )
    success_resp = _make_mock_response(200, {"results": [MOCK_WORK_JSON]})

    with patch("urllib.request.urlopen", side_effect=[err_429, success_resp]), \
         patch("time.sleep") as mock_sleep:
        results = provider.search("rag")

    assert len(results) == 1
    mock_sleep.assert_called_once_with(MAX_RETRY_DELAY_SECONDS)


def test_openalex_final_attempt_does_not_sleep():
    provider = OpenAlexSearchProvider(max_retries=1)
    err_429 = urllib.error.HTTPError(
        url="https://api.openalex.org/works", code=429, msg="Too Many Requests",
        hdrs={"Retry-After": "5"}, fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=[err_429]), patch("time.sleep") as mock_sleep:
        assert provider.search("rag") == []
    mock_sleep.assert_not_called()
