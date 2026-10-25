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
        # 1. Preserve established historical batch mappings
        for b in BATCHES:
            if b.start_date <= pub_date <= b.end_date:
                return b

        # 2. For newly handled dates outside established fixed batches, compute calendar quarter (YYYY-QX)
        quarter = (pub_date.month - 1) // 3 + 1
        start_month = (quarter - 1) * 3 + 1
        start_date = date(pub_date.year, start_month, 1)
        if quarter == 1:
            end_date = date(pub_date.year, 3, 31)
        elif quarter == 2:
            end_date = date(pub_date.year, 6, 30)
        elif quarter == 3:
            end_date = date(pub_date.year, 9, 30)
        else:
            end_date = date(pub_date.year, 12, 31)

        return TemporalBatch(
            id=f"{pub_date.year}-Q{quarter}",
            start_date=start_date,
            end_date=end_date,
        )

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

    @staticmethod
    def normalize_openalex_id(raw_id: Optional[str]) -> Optional[str]:
        """Normalizes OpenAlex work identifiers (URLs or short strings) to canonical 'W...' format.

        Returns None for anything that is not an OpenAlex work identifier, so callers
        never mistake free text (e.g. "Wasserstein GAN") for an ID.
        """
        cleaned = PaperNormalizer.strip_openalex_prefix(raw_id)
        if not cleaned:
            return None
        if cleaned.upper().startswith("W") and cleaned[1:].isdigit():
            return f"W{cleaned[1:]}"
        if cleaned.isdigit():
            return f"W{cleaned}"
        return None

    @staticmethod
    def strip_openalex_prefix(raw_id: Optional[str]) -> Optional[str]:
        """Removes a leading OpenAlex URL prefix and surrounding whitespace; None if empty."""
        if not raw_id:
            return None
        cleaned = raw_id.strip()
        for prefix in (
            "https://openalex.org/",
            "http://openalex.org/",
            "https://api.openalex.org/works/",
            "http://api.openalex.org/works/",
            "openalex.org/",
        ):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):]
                break
        return cleaned.strip() or None

    @staticmethod
    def canonical_openalex_source_id(raw_id: Optional[str]) -> Optional[str]:
        """Storage form of an OpenAlex source id: canonical 'W<digits>' when it is a work id,
        otherwise the value with any OpenAlex URL prefix removed (case preserved)."""
        return PaperNormalizer.normalize_openalex_id(raw_id) or PaperNormalizer.strip_openalex_prefix(raw_id)

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
    def parse_work(work: Dict[str, Any], source_type: Any = "OPENALEX_API") -> PaperCreateDTO:
        raw_source_id = work.get("id")
        if not raw_source_id:
            raise ValueError("Missing OpenAlex ID")

        source_id = PaperNormalizer.canonical_openalex_source_id(str(raw_source_id))
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
        
        from .models import SourceType
        if isinstance(source_type, str):
            try:
                st = SourceType(source_type)
            except ValueError:
                st = SourceType.OPENALEX_API
        else:
            st = source_type

        return PaperCreateDTO(
            source_type=st,
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
