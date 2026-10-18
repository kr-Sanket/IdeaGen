import pytest
from ideagen.ingestion.pipeline import IngestionService, DefaultDuplicateDetector
from ideagen.ingestion.models import IngestionConfig, PaperCreateDTO, SourceType
from ideagen.ingestion.sources import PaperSource
from unittest.mock import MagicMock

class DummyPersistenceLayer:
    def __init__(self):
        self.saved_papers = []
        
    def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
        return any(p.source_type == source_type and p.source_id == source_id for p in self.saved_papers)
        
    def paper_exists_by_doi(self, doi: str) -> bool:
        return doi in [p.doi for p in self.saved_papers if p.doi]
        
    def save_paper(self, paper: PaperCreateDTO) -> None:
        self.saved_papers.append(paper)

class DummySource:
    def __init__(self, papers):
        self.papers = papers
    def fetch_papers(self):
        yield from self.papers

@pytest.fixture
def config():
    return IngestionConfig(
        query_filters={"test": "filter"},
        polite_email="test@example.com"
    )

def test_pipeline_success_and_filtering(config):
    # Create test papers
    from datetime import date
    base_paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W1",
        doi="10.1/1",
        title="Valid",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Hello",
        raw_metadata={}
    )
    dup_id = base_paper.model_copy()
    dup_id.title = "Dup ID"
    
    dup_doi = base_paper.model_copy(update={"source_id": "W3", "title": "Dup DOI"})
    
    no_abs = base_paper.model_copy(update={"source_id": "W4", "abstract": ""})
    
    source = DummySource([base_paper, dup_id, dup_doi, no_abs])
    
    persistence = DummyPersistenceLayer()
    service = IngestionService(config, persistence, source)
    
    stats = service.run_ingestion()
    
    assert stats["fetched"] == 4
    assert stats["saved"] == 1
    assert stats["duplicates"] == 2
    assert stats["rejected_no_abstract"] == 1
    assert stats["errors"] == 0
    
    assert len(persistence.saved_papers) == 1
    assert persistence.saved_papers[0].source_id == "W1"
