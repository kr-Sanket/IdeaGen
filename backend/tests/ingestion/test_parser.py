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

def test_batch_assigner_dynamic_quarter_fallback():
    # Dates outside historical batches dynamically map to YYYY-QX
    b_pre = BatchAssigner.assign_batch(date(2018, 12, 31))
    assert b_pre.id == "2018-Q4"
    assert b_pre.start_date == date(2018, 10, 1)
    assert b_pre.end_date == date(2018, 12, 31)

    b_post = BatchAssigner.assign_batch(date(2026, 7, 1))
    assert b_post.id == "2026-Q3"
    assert b_post.start_date == date(2026, 7, 1)
    assert b_post.end_date == date(2026, 9, 30)

    b_2015 = BatchAssigner.assign_batch(date(2015, 4, 15))
    assert b_2015.id == "2015-Q2"
    assert b_2015.start_date == date(2015, 4, 1)
    assert b_2015.end_date == date(2015, 6, 30)

def test_doi_normalization():
    assert PaperNormalizer.normalize_doi("https://doi.org/10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi("http://doi.org/10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi("doi:10.123/abc") == "10.123/abc"
    assert PaperNormalizer.normalize_doi(" 10.123/abc ") == "10.123/abc"
    assert PaperNormalizer.normalize_doi(None) is None

def test_openalex_id_normalization():
    assert PaperNormalizer.normalize_openalex_id("https://openalex.org/W123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("http://openalex.org/W123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("openalex.org/W123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("https://api.openalex.org/works/W123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("W123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("w123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("123") == "W123"
    assert PaperNormalizer.normalize_openalex_id("https://openalex.org/w987654") == "W987654"
    assert PaperNormalizer.normalize_openalex_id(None) is None
    assert PaperNormalizer.normalize_openalex_id("   ") is None

def test_parse_work_valid():
    work = {
        "id": "https://openalex.org/W4285719001",
        "doi": "https://doi.org/10.1",
        "title": "Test Paper",
        "publication_date": "2020-05-15",
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [{"author": {"id": "A1"}}],
        "referenced_works": ["W2", "W3"]
    }
    paper = OpenAlexResponseParser.parse_work(work)
    assert paper.source_id == "W4285719001"
    assert paper.source_type == "OPENALEX_API"
    assert paper.doi == "10.1"
    assert paper.title == "Test Paper"
    assert paper.batch_id == "Batch 0"
    assert paper.abstract == "Hello world"
    assert len(paper.authors) == 1
    assert paper.referenced_works == ["W2", "W3"]
    assert paper.raw_metadata == work


def test_parse_work_id_variants():
    base = {
        "title": "Test",
        "publication_date": "2022-01-01",
    }
    # Short ID
    p1 = OpenAlexResponseParser.parse_work({**base, "id": "W999"})
    assert p1.source_id == "W999"

    # Lowercase short ID
    p2 = OpenAlexResponseParser.parse_work({**base, "id": "w999"})
    assert p2.source_id == "W999"

    # API URL
    p3 = OpenAlexResponseParser.parse_work({**base, "id": "https://api.openalex.org/works/W999"})
    assert p3.source_id == "W999"

    # Numeric ID string
    p4 = OpenAlexResponseParser.parse_work({**base, "id": "999"})
    assert p4.source_id == "W999"

    # Fallback to stripped raw string (unmodified) if unusual format
    p5 = OpenAlexResponseParser.parse_work({**base, "id": " custom-source-id "})
    assert p5.source_id == "custom-source-id"


def test_normalize_openalex_id_rejects_free_text():
    for text in ("Wasserstein GAN", "What makes RAG faithful", "W12a", "Word embeddings", "10.1234/abc"):
        assert PaperNormalizer.normalize_openalex_id(text) is None


def test_canonical_openalex_source_id_strips_prefix_and_preserves_case():
    assert PaperNormalizer.canonical_openalex_source_id("https://openalex.org/w42") == "W42"
    assert PaperNormalizer.canonical_openalex_source_id("https://openalex.org/W_FIXTURE_1") == "W_FIXTURE_1"
    assert PaperNormalizer.canonical_openalex_source_id(" custom-source-id ") == "custom-source-id"
    assert PaperNormalizer.canonical_openalex_source_id("   ") is None


def test_parse_work_missing_fields():
    with pytest.raises(ValueError, match="Missing OpenAlex ID"):
        OpenAlexResponseParser.parse_work({})
    with pytest.raises(ValueError, match="Missing OpenAlex ID"):
        OpenAlexResponseParser.parse_work({"id": "   "})
    with pytest.raises(ValueError, match="Missing title"):
        OpenAlexResponseParser.parse_work({"id": "W1"})
    with pytest.raises(ValueError, match="Missing publication date"):
        OpenAlexResponseParser.parse_work({"id": "W1", "title": "T"})
