from typing import Protocol, Iterator
from .models import PaperCreateDTO, SourceType
from .client import OpenAlexClient
from .parser import OpenAlexResponseParser

class PaperSource(Protocol):
    """Abstraction for acquiring research papers from any source."""
    def fetch_papers(self) -> Iterator[PaperCreateDTO]:
        ...

class OpenAlexAPISource:
    """Source implementation fetching papers from the live OpenAlex API."""
    def __init__(self, client: OpenAlexClient, parser: OpenAlexResponseParser):
        self.client = client
        self.parser = parser
        
    def fetch_papers(self) -> Iterator[PaperCreateDTO]:
        for raw_work in self.client.fetch_works():
            yield self.parser.parse_work(raw_work, source_type=SourceType.OPENALEX_API)

class OpenAlexSnapshotSource:
    """Source implementation fetching papers from a local OpenAlex snapshot fixture."""
    def __init__(self, snapshot_path: str, parser: OpenAlexResponseParser):
        self.snapshot_path = snapshot_path
        self.parser = parser
        
    def fetch_papers(self) -> Iterator[PaperCreateDTO]:
        import json
        with open(self.snapshot_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                raw_work = json.loads(line)
                yield self.parser.parse_work(raw_work, source_type=SourceType.OPENALEX_SNAPSHOT)

class ManualUploadSource:
    """Source implementation for manually uploaded papers/fixtures."""
    def __init__(self, fixture_path: str):
        self.fixture_path = fixture_path
        
    def fetch_papers(self) -> Iterator[PaperCreateDTO]:
        import json
        from datetime import date
        from .models import TemporalBatch
        from .parser import BatchAssigner
        
        with open(self.fixture_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                raw_data = json.loads(line)
                
                pub_date = date.fromisoformat(raw_data["publication_date"])
                batch = BatchAssigner.assign_batch(pub_date)
                
                yield PaperCreateDTO(
                    source_type=SourceType.MANUAL_UPLOAD,
                    source_id=raw_data["id"],
                    doi=raw_data.get("doi"),
                    title=raw_data["title"],
                    publication_date=pub_date,
                    batch_id=batch.id,
                    abstract=raw_data.get("abstract", ""),
                    authors=raw_data.get("authors", []),
                    referenced_works=raw_data.get("referenced_works", []),
                    raw_metadata=raw_data
                )
