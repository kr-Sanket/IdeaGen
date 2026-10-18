import pytest
from datetime import date
from ideagen.ingestion.pipeline import DefaultDuplicateDetector
from ideagen.ingestion.models import PaperCreateDTO, SourceType

class DummyPersistenceLayer:
    def __init__(self):
        self.saved_papers = []
        
    def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
        return any(p.source_type == source_type and p.source_id == source_id for p in self.saved_papers)
        
    def paper_exists_by_doi(self, doi: str) -> bool:
        return doi in [p.doi for p in self.saved_papers if p.doi]

def test_api_vs_snapshot_w123():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    api_paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W123",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    snapshot_paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_SNAPSHOT,
        source_id="W123",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(api_paper)
    
    # Should be recognized as duplicate because it's the same OpenAlex work
    assert detector.is_duplicate(snapshot_paper, persistence) is True

def test_api_vs_api_w456():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    api_paper_123 = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W123",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    api_paper_456 = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W456",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(api_paper_123)
    
    # Not a duplicate
    assert detector.is_duplicate(api_paper_456, persistence) is False

def test_same_doi_across_api_snapshot():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    api_paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W1",
        doi="10.123/test",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    snapshot_paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_SNAPSHOT,
        source_id="W2",
        doi="10.123/test",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(api_paper)
    assert detector.is_duplicate(snapshot_paper, persistence) is True

def test_same_doi_across_manual_uploads():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    m1 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M1",
        doi="10.123/test",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    m2 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M2",
        doi="10.123/test",
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(m1)
    assert detector.is_duplicate(m2, persistence) is True

def test_missing_doi_distinct():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    m1 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M1",
        doi=None,
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    m2 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M2",
        doi=None,
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(m1)
    assert detector.is_duplicate(m2, persistence) is False

def test_repeated_manual_source_id():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    m1 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M1",
        doi=None,
        title="Test",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    m2 = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="M1",
        doi=None,
        title="Test Uploaded Again",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(m1)
    assert detector.is_duplicate(m2, persistence) is True

def test_distinct_manual_source_ids():
    # Covered by missing_doi_distinct test
    pass

def test_preprint_without_doi_vs_published_doi():
    persistence = DummyPersistenceLayer()
    detector = DefaultDuplicateDetector()
    
    preprint = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W1", # Preprint OpenAlex ID
        doi=None,
        title="Preprint Title",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    published = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W2", # Published OpenAlex ID
        doi="10.123/published",
        title="Published Title",
        publication_date=date(2020, 6, 1),
        batch_id="Batch 0",
        abstract="",
        raw_metadata={}
    )
    
    persistence.saved_papers.append(preprint)
    # Should NOT be duplicate because IDs differ and preprint has no DOI to match
    assert detector.is_duplicate(published, persistence) is False
