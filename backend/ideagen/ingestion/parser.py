from datetime import date
from typing import Optional, Dict, Any
from .models import PaperCreateDTO, TemporalBatch

BATCHES = [
    TemporalBatch(id="Batch 0", start_date=date(2019, 1, 1), end_date=date(2024, 12, 31)),
    TemporalBatch(id="2025 Q1", start_date=date(2025, 1, 1), end_date=date(2025, 3, 31)),
    TemporalBatch(id="2025 Q2", start_date=date(2025, 4, 1), end_date=date(2025, 6, 30)),
    TemporalBatch(id="2025 Q3", start_date=date(2025, 7, 1), end_date=date(2025, 9, 30)),
    TemporalBatch(id="2025 Q4", start_date=date(2025, 10, 1), end_date=date(2025, 12, 31)),
    TemporalBatch(id="2026 H1", start_date=date(2026, 1, 1), end_date=date(2026, 6, 30)),
]

class BatchAssigner:
    @staticmethod
    def assign_batch(pub_date: date) -> TemporalBatch:
        for b in BATCHES:
            if b.start_date <= pub_date <= b.end_date:
                return b
        raise ValueError(f"Date {pub_date} falls outside all defined temporal batches.")

class PaperNormalizer:
    @staticmethod
    def normalize_doi(raw_doi: Optional[str]) -> Optional[str]:
        if not raw_doi:
            return None
        doi = raw_doi.lower().strip()
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "", 1)
        elif doi.startswith("http://doi.org/"):
            doi = doi.replace("http://doi.org/", "", 1)
        elif doi.startswith("doi:"):
            doi = doi.replace("doi:", "", 1)
        return doi.strip()

class OpenAlexResponseParser:
    @staticmethod
    def reconstruct_abstract(inverted_index: Optional[Dict[str, list]]) -> str:
        if not inverted_index:
            return ""
        
        word_positions = []
        for word, positions in inverted_index.items():
            if not isinstance(positions, list):
                continue
            for pos in positions:
                if isinstance(pos, int):
                    word_positions.append((pos, word))
        
        word_positions.sort()
        return " ".join(word for _, word in word_positions)

    @staticmethod
    def parse_work(work: Dict[str, Any], source_type: str = "OPENALEX_API") -> PaperCreateDTO:
        source_id = work.get("id")
        if not source_id:
            raise ValueError("Missing OpenAlex ID")
            
        raw_doi = work.get("doi")
        doi = PaperNormalizer.normalize_doi(raw_doi)
        
        title = work.get("title")
        if not title:
            raise ValueError("Missing title")
            
        pub_date_str = work.get("publication_date")
        if not pub_date_str:
            raise ValueError("Missing publication date")
        pub_date = date.fromisoformat(pub_date_str)
        
        batch = BatchAssigner.assign_batch(pub_date)
        
        abstract_index = work.get("abstract_inverted_index")
        abstract = OpenAlexResponseParser.reconstruct_abstract(abstract_index)
        
        authors = work.get("authorships", [])
        referenced_works = work.get("referenced_works", [])
        
        return PaperCreateDTO(
            source_type=source_type,
            source_id=source_id,
            doi=doi,
            title=title,
            publication_date=pub_date,
            batch_id=batch.id,
            abstract=abstract,
            authors=authors,
            referenced_works=referenced_works,
            raw_metadata=work
        )
