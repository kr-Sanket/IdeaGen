import pytest
import tempfile
import json
import os
from unittest.mock import MagicMock
from ideagen.ingestion.sources import (
    OpenAlexAPISource, 
    OpenAlexSnapshotSource, 
    ManualUploadSource,
    SourceType
)
from ideagen.ingestion.parser import OpenAlexResponseParser
from ideagen.ingestion.client import OpenAlexClient

@pytest.fixture
def sample_openalex_json():
    return {
        "id": "https://openalex.org/W1",
        "title": "API Paper",
        "publication_date": "2020-01-01",
        "doi": "https://doi.org/10.123/api",
        "abstract_inverted_index": {"Test": [0]}
    }

@pytest.fixture
def mock_client(sample_openalex_json):
    client = MagicMock(spec=OpenAlexClient)
    client.fetch_works.return_value = iter([sample_openalex_json])
    return client

def test_openalex_api_source(mock_client):
    parser = OpenAlexResponseParser()
    source = OpenAlexAPISource(client=mock_client, parser=parser)
    
    papers = list(source.fetch_papers())
    
    assert len(papers) == 1
    p = papers[0]
    assert p.source_type == SourceType.OPENALEX_API
    assert p.source_id == "W1"
    assert p.doi == "10.123/api"
    assert p.title == "API Paper"

def test_openalex_snapshot_source(sample_openalex_json):
    sample_openalex_json["title"] = "Snapshot Paper"
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json.dumps(sample_openalex_json) + "\n")
        temp_path = f.name
        
    try:
        parser = OpenAlexResponseParser()
        source = OpenAlexSnapshotSource(snapshot_path=temp_path, parser=parser)
        
        papers = list(source.fetch_papers())
        assert len(papers) == 1
        p = papers[0]
        assert p.source_type == SourceType.OPENALEX_SNAPSHOT
        assert p.source_id == "W1" # Canonical ID matching API
        assert p.doi == "10.123/api"
        assert p.title == "Snapshot Paper"
    finally:
        os.remove(temp_path)


def test_manual_upload_source():
    manual_json = {
        "id": "manual-1",
        "title": "Manual Paper",
        "publication_date": "2020-01-01",
        # Missing DOI and abstract to test optional fields
    }
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json.dumps(manual_json) + "\n")
        temp_path = f.name
        
    try:
        source = ManualUploadSource(fixture_path=temp_path)
        papers = list(source.fetch_papers())
        
        assert len(papers) == 1
        p = papers[0]
        assert p.source_type == SourceType.MANUAL_UPLOAD
        assert p.source_id == "manual-1"
        assert p.doi is None
        assert p.title == "Manual Paper"
        assert p.abstract == ""
        assert p.raw_metadata == manual_json
    finally:
        os.remove(temp_path)
