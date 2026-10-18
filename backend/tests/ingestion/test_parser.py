import pytest
from datetime import date
from ideagen.ingestion.parser import (
    OpenAlexResponseParser, 
    BatchAssigner, 
    PaperNormalizer
)

def test_reconstruct_abstract():
    aii = {
        "We": [0, 5],
        "study": [1],
        "generating": [2],
        "abstractive": [3],
        "summaries": [4]
    }
    abstract = OpenAlexResponseParser.reconstruct_abstract(aii)
    assert abstract == "We study generating abstractive summaries We"
    
def test_reconstruct_abstract_empty():
    assert OpenAlexResponseParser.reconstruct_abstract({}) == ""
    assert OpenAlexResponseParser.reconstruct_abstract(None) == ""
    assert OpenAlexResponseParser.reconstruct_abstract({"bad": "data"}) == ""

def test_batch_assigner_valid_dates():
    b1 = BatchAssigner.assign_batch(date(2019, 1, 1))
    assert b1.id == "Batch 0"
    b2 = BatchAssigner.assign_batch(date(2025, 2, 15))
    assert b2.id == "2025 Q1"
    b3 = BatchAssigner.assign_batch(date(2026, 6, 30))
    assert b3.id == "2026 H1"

def test_batch_assigner_invalid_dates():
    with pytest.raises(ValueError, match="outside all defined temporal batches"):
        BatchAssigner.assign_batch(date(2018, 12, 31))
    with pytest.raises(ValueError, match="outside all defined temporal batches"):
        BatchAssigner.assign_batch(date(2026, 7, 1))

def test_doi_normalization():
    assert PaperNormalizer.normalize_doi("https://doi.org/10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi("http://doi.org/10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi("doi:10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi(" 10.123/abc ") == "10.123/abc"
    assert PaperNormalizer.normalize_doi(None) is None

def test_parse_work_valid():
    work = {
        "id": "W123",
        "doi": "https://doi.org/10.1",
        "title": "Test Paper",
        "publication_date": "2020-05-15",
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [{"author": {"id": "A1"}}],
        "referenced_works": ["W2", "W3"]
    }
    paper = OpenAlexResponseParser.parse_work(work)
    assert paper.source_id == "W123"
    assert paper.source_type == "OPENALEX_API"
    assert paper.doi == "10.1"
    assert paper.title == "Test Paper"
    assert paper.batch_id == "Batch 0"
    assert paper.abstract == "Hello world"
    assert len(paper.authors) == 1
    assert paper.referenced_works == ["W2", "W3"]
    assert paper.raw_metadata == work

def test_parse_work_missing_fields():
    with pytest.raises(ValueError, match="Missing OpenAlex ID"):
        OpenAlexResponseParser.parse_work({})
    with pytest.raises(ValueError, match="Missing title"):
        OpenAlexResponseParser.parse_work({"id": "W1"})
    with pytest.raises(ValueError, match="Missing publication date"):
        OpenAlexResponseParser.parse_work({"id": "W1", "title": "T"})
