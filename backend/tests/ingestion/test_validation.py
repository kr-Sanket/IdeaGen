import pytest
from unittest.mock import patch, MagicMock
from ideagen.ingestion.validation import ValidationHarness, PREDEFINED_CANDIDATES, BASELINE_RESULTS

def test_baselines_preserved():
    assert "A" in BASELINE_RESULTS
    assert "B" in BASELINE_RESULTS
    assert BASELINE_RESULTS["B"]["filtered_count"] == 66400

@patch("ideagen.ingestion.client.OpenAlexClient._make_request")
def test_validation_harness_evaluate(mock_make_request):
    harness = ValidationHarness("test@example.com")
    
    # Mock sequence: 
    # 1. raw_count req -> count=100
    # 2. full_count req -> count=50, results=[sample1, sample2]
    # 3-8. batch reqs (6 batches) -> count=10
    
    mock_make_request.side_effect = [
        {"meta": {"count": 100}, "results": []},  # raw
        {"meta": {"count": 50}, "results": [      # full
            {"id": "W1", "title": "T1", "doi": "D1", "type": "article", "abstract_inverted_index": {"A": [0]}},
            {"id": "W2", "title": "T2", "doi": None, "type": "preprint", "abstract_inverted_index": {}}
        ]},
        {"meta": {"count": 10}, "results": []}, # batch 0
        {"meta": {"count": 10}, "results": []}, # 2025 Q1
        {"meta": {"count": 10}, "results": []}, # 2025 Q2
        {"meta": {"count": 10}, "results": []}, # 2025 Q3
        {"meta": {"count": 10}, "results": []}, # 2025 Q4
        {"meta": {"count": 0}, "results": []},  # 2026 H1
    ]
    
    report = harness.evaluate_candidate("TEST", {"test": "val"})
    
    assert report["query_id"] == "TEST"
    assert report["raw_count"] == 100
    assert report["filtered_count"] == 50
    assert report["temporal_counts"]["Batch 0"] == 10
    assert report["temporal_counts"]["2026 H1"] == 0
    assert len(report["samples"]) == 2
    assert report["metadata_quality"]["has_doi"] == 1
    assert report["metadata_quality"]["is_preprint"] == 1
    assert report["metadata_quality"]["has_abstract"] == 1
    assert len(report["errors"]) == 0
    assert mock_make_request.call_count == 8

@patch("ideagen.ingestion.client.OpenAlexClient._make_request")
def test_validation_harness_error_handling(mock_make_request):
    harness = ValidationHarness("test@example.com")
    
    mock_make_request.side_effect = Exception("API completely failed")
    report = harness.evaluate_candidate("ERR", {"test": "val"})
    
    assert report["raw_count"] == -1
    assert len(report["errors"]) == 1
    assert "API completely failed" in report["errors"][0]
