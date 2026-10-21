from datetime import date, datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field

import enum

class SourceType(str, enum.Enum):
    OPENALEX_API = "OPENALEX_API"
    OPENALEX_SNAPSHOT = "OPENALEX_SNAPSHOT"
    MANUAL_UPLOAD = "MANUAL_UPLOAD"

class TemporalBatch(BaseModel):
    id: str
    start_date: date
    end_date: date

class PaperCreateDTO(BaseModel):
    source_type: SourceType
    source_id: str
    doi: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[date] = None
    batch_id: str = "manual"
    abstract: str = ""
    authors: List[Dict[str, Any]] = Field(default_factory=list)
    referenced_works: List[str] = Field(default_factory=list)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestionConfig(BaseModel):
    query_filters: Dict[str, str] = Field(description="OpenAlex query filters (e.g. title_and_abstract.search)")
    polite_email: Optional[str] = None
    base_url: str = "https://api.openalex.org/works"
    per_page: int = 50

class PaperDocumentCreateDTO(BaseModel):
    paper_id: UUID
    full_text: str
    extraction_method: str
    extraction_version: str
    extracted_at: datetime
    content_hash: Optional[str] = None
    original_filename: Optional[str] = None
    acquisition_timestamp: Optional[datetime] = None
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)

class PassageCreateDTO(BaseModel):
    paper_id: UUID
    document_id: UUID
    passage_index: int
    page_start: int
    page_end: int
    section: Optional[str] = None
    text: str

class ExtractedPage(BaseModel):
    page_number: int
    text: str
    blocks: List[str] = Field(default_factory=list)

class ExtractedSection(BaseModel):
    name: Optional[str] = None
    text: str
    page_start: int
    page_end: int

class PDFExtractionResult(BaseModel):
    title: Optional[str] = None
    full_text: str
    pages: List[ExtractedPage] = Field(default_factory=list)
    sections: List[ExtractedSection] = Field(default_factory=list)
    content_hash: str
    original_filename: str
    extraction_method: str = "pymupdf"
    extraction_version: str
    extracted_at: datetime
