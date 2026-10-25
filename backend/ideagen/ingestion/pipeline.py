import logging
import uuid
from typing import Any, Optional, Protocol
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
        
    def save_paper_document(self, doc: 'ideagen.ingestion.models.PaperDocumentCreateDTO') -> uuid.UUID:
        ...

    def document_exists_by_hash(self, content_hash: str) -> bool:
        ...

    def save_passages(self, passages: list['ideagen.ingestion.models.PassageCreateDTO']) -> None:
        ...

    def delete_paper(self, paper_id: uuid.UUID) -> bool:
        ...

    def transaction(self) -> Any:
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

    def resolve_duplicate_paper_id(self, paper: PaperCreateDTO, persistence: Any) -> Optional[uuid.UUID]:
        """Safely and unambiguously resolves the existing paper UUID for a duplicate paper.

        Returns existing UUID if uniquely identifiable by source ID or DOI.
        If ambiguous (conflicting matches or multiple matches), returns None.
        """
        matched_ids = set()

        # 1. Query by canonical source ID
        from .models import SourceType
        st_val = paper.source_type.value if hasattr(paper.source_type, "value") else str(paper.source_type)
        if hasattr(persistence, "get_paper_id_by_source_id"):
            id_by_source = persistence.get_paper_id_by_source_id(st_val, paper.source_id)
            if id_by_source:
                matched_ids.add(id_by_source)

            # Cross-source OpenAlex ID
            if paper.source_type in (
                SourceType.OPENALEX_API,
                SourceType.OPENALEX_SNAPSHOT,
                SourceType.OPENALEX_API.value,
                SourceType.OPENALEX_SNAPSHOT.value,
            ):
                other_source = (
                    SourceType.OPENALEX_SNAPSHOT.value
                    if st_val == SourceType.OPENALEX_API.value
                    else SourceType.OPENALEX_API.value
                )
                id_by_other = persistence.get_paper_id_by_source_id(other_source, paper.source_id)
                if id_by_other:
                    matched_ids.add(id_by_other)

        # 2. Query by DOI
        if paper.doi and hasattr(persistence, "get_paper_id_by_doi"):
            id_by_doi = persistence.get_paper_id_by_doi(paper.doi)
            if id_by_doi:
                matched_ids.add(id_by_doi)

        # 3. Fallback for in-memory persistence with saved_papers
        if not matched_ids and hasattr(persistence, "saved_papers"):
            for p in getattr(persistence, "saved_papers", []):
                p_st = p.source_type.value if hasattr(p.source_type, "value") else str(p.source_type)
                p_id = getattr(p, "id", None)
                if not p_id:
                    continue
                if p.source_id == paper.source_id and (
                    p_st == st_val or (p_st in ("OPENALEX_API", "OPENALEX_SNAPSHOT") and st_val in ("OPENALEX_API", "OPENALEX_SNAPSHOT"))
                ):
                    matched_ids.add(p_id)
                if paper.doi and getattr(p, "doi", None) and p.doi == paper.doi:
                    matched_ids.add(p_id)

        if len(matched_ids) == 1:
            return next(iter(matched_ids))
        return None

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
