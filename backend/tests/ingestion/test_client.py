import pytest
import urllib.error
from unittest.mock import patch, MagicMock
from ideagen.ingestion.client import OpenAlexClient, OpenAlexRateLimitError
from ideagen.ingestion.models import IngestionConfig

@pytest.fixture
def config():
    return IngestionConfig(
        query_filters={"has_abstract": "true"},
        polite_email="test@example.com"
    )

@patch("urllib.request.urlopen")
@patch("time.sleep")
def test_client_success(mock_sleep, mock_urlopen, config):
    client = OpenAlexClient(config)
    
    # Mock response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"results": [{"id": "W1"}]}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp
    
    data = client._make_request("http://test.url")
    assert data == {"results": [{"id": "W1"}]}

@patch("urllib.request.urlopen")
@patch("time.sleep")
def test_client_retry_on_429(mock_sleep, mock_urlopen, config):
    client = OpenAlexClient(config)
    
    # First 2 calls raise 429, 3rd succeeds
    err_429 = urllib.error.HTTPError(
        url="http://test.url", code=429, msg="Too Many Requests", 
        hdrs={}, fp=None
    )
    
    mock_success = MagicMock()
    mock_success.status = 200
    mock_success.read.return_value = b'{"results": []}'
    mock_success.__enter__.return_value = mock_success
    
    mock_urlopen.side_effect = [err_429, err_429, mock_success]
    
    data = client._make_request("http://test.url")
    assert data == {"results": []}
    assert mock_urlopen.call_count == 3
    # Check backoff delays with jitter
    delay1 = mock_sleep.call_args_list[0][0][0]
    delay2 = mock_sleep.call_args_list[1][0][0]
    assert 1.0 <= delay1 <= 1.5
    assert 2.0 <= delay2 <= 2.5

@patch("urllib.request.urlopen")
@patch("time.sleep")
def test_client_exhausts_retries(mock_sleep, mock_urlopen, config):
    client = OpenAlexClient(config)
    
    err_503 = urllib.error.HTTPError(
        url="http://test.url", code=503, msg="Service Unavailable", 
        hdrs={}, fp=None
    )
    mock_urlopen.side_effect = err_503
    
    with pytest.raises(OpenAlexRateLimitError, match="Max retries exceeded"):
        client._make_request("http://test.url")
        
    assert mock_urlopen.call_count == 5

@patch("urllib.request.urlopen")
@patch("time.sleep")
def test_client_retry_after_header(mock_sleep, mock_urlopen, config):
    client = OpenAlexClient(config)
    
    err_429 = urllib.error.HTTPError(
        url="http://test.url", code=429, msg="Too Many Requests", 
        hdrs={"Retry-After": "42"}, fp=None
    )
    
    mock_success = MagicMock()
    mock_success.status = 200
    mock_success.read.return_value = b'{"results": []}'
    mock_success.__enter__.return_value = mock_success
    
    mock_urlopen.side_effect = [err_429, mock_success]
    
    client._make_request("http://test.url")
    assert mock_sleep.call_args_list[0][0][0] == 42.0

@patch("ideagen.ingestion.client.OpenAlexClient._make_request")
def test_fetch_works_pagination(mock_make_request, config):
    client = OpenAlexClient(config)
    
    # 3 pages
    mock_make_request.side_effect = [
        {"meta": {"next_cursor": "c1"}, "results": [{"id": "W1"}]},
        {"meta": {"next_cursor": "c2"}, "results": [{"id": "W2"}]},
        {"meta": {"next_cursor": "c2"}, "results": []}, # no next cursor update / empty results
    ]
    
    results = list(client.fetch_works())
    assert len(results) == 2
    assert results[0]["id"] == "W1"
    assert results[1]["id"] == "W2"
    assert mock_make_request.call_count == 3
