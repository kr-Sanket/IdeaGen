import logging
import uuid
from typing import Protocol, Optional
from .models import PaperCreateDTO, IngestionConfig
from .sources import PaperSource

logger = logging.getLogger(__name__)

class PersistenceLayer(Protocol):
    def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
        ...
        
    def paper_exists_by_doi(self, doi: str) -> bool:
        ...
        
    def save_paper(self, paper: PaperCreateDTO) -> uuid.UUID:
        ...
        
    def save_paper_document(self, doc: 'ideagen.ingestion.models.PaperDocumentCreateDTO') -> None:
        ...
        
class DuplicateDetectionInterface(Protocol):
    def is_duplicate(self, paper: PaperCreateDTO, persistence: PersistenceLayer) -> bool:
        ...

class DefaultDuplicateDetector:
    def is_duplicate(self, paper: PaperCreateDTO, persistence: PersistenceLayer) -> bool:
        # 1. Exact source type and source ID match
        if persistence.paper_exists_by_source_id(paper.source_type, paper.source_id):
            return True
            
        # 2. Cross-source OpenAlex ID match
        from .models import SourceType
        if paper.source_type in (SourceType.OPENALEX_API, SourceType.OPENALEX_SNAPSHOT):
            other_source = (
                SourceType.OPENALEX_SNAPSHOT 
                if paper.source_type == SourceType.OPENALEX_API 
                else SourceType.OPENALEX_API
            )
            if persistence.paper_exists_by_source_id(other_source, paper.source_id):
                return True
                
        # 3. DOI match
        if paper.doi and persistence.paper_exists_by_doi(paper.doi):
            return True
            
        return False

class IngestionService:
    def __init__(
        self, 
        config: IngestionConfig, 
        persistence: PersistenceLayer,
        source: PaperSource,
        duplicate_detector: Optional[DuplicateDetectionInterface] = None
    ):
        self.config = config
        self.persistence = persistence
        self.source = source
        self.duplicate_detector = duplicate_detector or DefaultDuplicateDetector()
        
    def run_ingestion(self) -> dict:
        stats = {
            "fetched": 0,
            "saved": 0,
            "duplicates": 0,
            "rejected_no_abstract": 0,
            "rejected_out_of_bounds": 0,
            "errors": 0
        }
        
        logger.info(f"Starting ingestion from source...")
        
        try:
            for paper in self.source.fetch_papers():
                stats["fetched"] += 1
                
                try:
                    if not paper.abstract.strip():
                        logger.debug(f"Rejecting {paper.source_id}: No abstract")
                        stats["rejected_no_abstract"] += 1
                        continue
                        
                    if self.duplicate_detector.is_duplicate(paper, self.persistence):
                        logger.debug(f"Skipping duplicate {paper.source_id}")
                        stats["duplicates"] += 1
                        continue
                        
                    self.persistence.save_paper(paper)
                    stats["saved"] += 1
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing {paper.source_id}: {e}")
                    stats["errors"] += 1
        except Exception as e:
            logger.error(f"Source fetch error: {e}")
            stats["errors"] += 1
                
        logger.info(f"Ingestion completed. Stats: {stats}")
        return stats
