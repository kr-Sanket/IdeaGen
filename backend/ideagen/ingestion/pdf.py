"""Real PDF Ingestion Service using PyMuPDF.

Performs validation, SHA-256 provenance calculation, metadata & title extraction,
page boundary preservation, rule-based section detection, text cleaning,
and deterministic passage/chunk creation with PostgreSQL persistence.
"""

import hashlib
import re
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import uuid
import pymupdf

from ideagen.ingestion.models import (
    PaperCreateDTO,
    PaperDocumentCreateDTO,
    PassageCreateDTO,
    ExtractedPage,
    ExtractedSection,
    PDFExtractionResult,
    SourceType,
)
from ideagen.ingestion.pipeline import PersistenceLayer
from .text_cleaner import TextCleaner
from .section_detector import SectionDetector
from .passage_chunker import PassageChunker, BlockUnit


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails or is unreadable."""
    pass


class DuplicateDocumentError(ValueError):
    """Raised when an identical PDF content hash has already been ingested."""
    pass


class ManualPDFIngestionService:
    """Production PDF ingestion service for scientific literature."""

    def __init__(self, persistence: Optional[PersistenceLayer] = None):
        self.persistence = persistence
        self.extraction_method = "pymupdf"
        self.extraction_version = pymupdf.__version__
        self.min_text_threshold = 500  # Conservative threshold to prevent empty documents
        self.chunker = PassageChunker()

    def validate_file(self, pdf_path: Path) -> None:
        """Validates that the file exists and is a valid file path."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

    def compute_sha256(self, pdf_path: Path) -> str:
        """Computes deterministic SHA-256 content hash of PDF file bytes."""
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_content_duplicate(self, pdf_path: Path) -> bool:
        """Checks if a document with the same SHA-256 content hash already exists in persistence."""
        if not self.persistence or not hasattr(self.persistence, "document_exists_by_hash"):
            return False
        content_hash = self.compute_sha256(pdf_path)
        return self.persistence.document_exists_by_hash(content_hash)

    def detect_title(self, doc: pymupdf.Document) -> Optional[str]:
        """Detects document title when detectable. Does not fabricate missing metadata."""
        # 1. Check embedded PDF metadata
        meta_title = doc.metadata.get("title") if doc.metadata else None
        if meta_title:
            cleaned = meta_title.strip()
            # Verify not generic or file extension
            if (
                len(cleaned) > 5
                and len(cleaned) < 300
                and not cleaned.lower().endswith(".pdf")
                and cleaned.lower() not in ("untitled", "microsoft word", "latex")
            ):
                return cleaned

        # 2. Inspect first page blocks/spans for prominent heading text
        if len(doc) == 0:
            return None

        first_page = doc[0]
        try:
            page_dict = first_page.get_text("dict")
            all_spans = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            span_size = span.get("size", 0.0)
                            if span_text:
                                all_spans.append((span_size, span_text))

            if all_spans:
                sizes = [s[0] for s in all_spans]
                max_size = max(sizes)
                avg_size = sum(sizes) / len(sizes)
                # A title must be prominently larger than body text by at least 2pt and >= 12pt
                if max_size >= avg_size + 2.0 and max_size >= 12.0:
                    title_parts = [s[1] for s in all_spans if abs(s[0] - max_size) < 0.5]
                    candidate_title = " ".join(title_parts).strip()
                    if (
                        5 <= len(candidate_title) <= 300
                        and not candidate_title.lower().startswith("arxiv:")
                        and not candidate_title.lower().startswith("page ")
                    ):
                        return candidate_title
        except Exception:
            pass

        return None

    def extract_text(self, pdf_path: Path) -> str:
        """Extracts complete cleaned text from PDF with minimum threshold validation."""
        self.validate_file(pdf_path)
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF file: {e}")

        if not doc.is_pdf:
            doc.close()
            raise PDFExtractionError("File is not a valid PDF document.")

        text_parts = []
        try:
            for page in doc:
                text_parts.append(page.get_text("text"))
        except Exception as e:
            doc.close()
            raise PDFExtractionError(f"Error extracting text: {e}")
        finally:
            doc.close()

        raw_full = "\n\n".join(text_parts)
        cleaned_text = TextCleaner.clean_text(raw_full)

        if len(cleaned_text) < self.min_text_threshold:
            raise PDFExtractionError(
                f"Extracted text length ({len(cleaned_text)}) is below the minimum threshold ({self.min_text_threshold}). "
                "Document may be empty, image-based, or unreadable."
            )

        return cleaned_text

    def extract_structured(self, pdf_path: Path) -> Tuple[PDFExtractionResult, List[BlockUnit]]:
        """Extracts structured document with page preservation, section detection, and block units."""
        self.validate_file(pdf_path)
        content_hash = self.compute_sha256(pdf_path)

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF file: {e}")

        if not doc.is_pdf:
            doc.close()
            raise PDFExtractionError("File is not a valid PDF document.")

        pages: List[ExtractedPage] = []
        block_units: List[BlockUnit] = []
        sections_map: Dict[str, Dict[str, Any]] = {}
        current_section: Optional[str] = None
        detected_title: Optional[str] = None

        try:
            detected_title = self.detect_title(doc)

            for page_idx, page in enumerate(doc, start=1):
                raw_blocks = page.get_text("blocks")
                page_cleaned_blocks = []

                for b in raw_blocks:
                    if len(b) >= 5 and b[4]:  # block text
                        raw_block_text = b[4].strip()
                        if not raw_block_text:
                            continue

                        # Process lines within the block, splitting on any detected section heading
                        lines = [l.strip() for l in raw_block_text.split("\n") if l.strip()]
                        if not lines:
                            continue

                        current_lines = []

                        for line in lines:
                            detected_heading = SectionDetector.detect_heading(line)
                            if detected_heading:
                                # Emit any accumulated lines under the previous section
                                if current_lines:
                                    cleaned_sub = TextCleaner.clean_block("\n".join(current_lines))
                                    if cleaned_sub:
                                        page_cleaned_blocks.append(cleaned_sub)
                                        block_units.append(
                                            BlockUnit(
                                                text=cleaned_sub,
                                                page_number=page_idx,
                                                section=current_section,
                                            )
                                        )
                                        if current_section:
                                            sections_map[current_section]["texts"].append(cleaned_sub)
                                            sections_map[current_section]["page_end"] = page_idx
                                    current_lines = []

                                current_section = detected_heading
                                page_cleaned_blocks.append(line)
                                if current_section not in sections_map:
                                    sections_map[current_section] = {
                                        "name": current_section,
                                        "texts": [],
                                        "page_start": page_idx,
                                        "page_end": page_idx,
                                    }
                            else:
                                current_lines.append(line)

                        if current_lines:
                            cleaned_sub = TextCleaner.clean_block("\n".join(current_lines))
                            if cleaned_sub:
                                page_cleaned_blocks.append(cleaned_sub)
                                block_units.append(
                                    BlockUnit(
                                        text=cleaned_sub,
                                        page_number=page_idx,
                                        section=current_section,
                                    )
                                )
                                if current_section:
                                    sections_map[current_section]["texts"].append(cleaned_sub)
                                    sections_map[current_section]["page_end"] = page_idx

                page_text = "\n\n".join(page_cleaned_blocks)
                pages.append(
                    ExtractedPage(
                        page_number=page_idx,
                        text=page_text,
                        blocks=page_cleaned_blocks,
                    )
                )

        except Exception as e:
            doc.close()
            raise PDFExtractionError(f"Error extracting structured PDF text: {e}")
        finally:
            doc.close()

        full_doc_text = "\n\n".join(p.text for p in pages if p.text)
        if len(full_doc_text) < self.min_text_threshold:
            raise PDFExtractionError(
                f"Extracted text length ({len(full_doc_text)}) is below the minimum threshold ({self.min_text_threshold}). "
                "Document may be empty, image-based, or unreadable."
            )

        extracted_sections = [
            ExtractedSection(
                name=info["name"],
                text="\n\n".join(info["texts"]),
                page_start=info["page_start"],
                page_end=info["page_end"],
            )
            for info in sections_map.values()
        ]

        result = PDFExtractionResult(
            title=detected_title,
            full_text=full_doc_text,
            pages=pages,
            sections=extracted_sections,
            content_hash=content_hash,
            original_filename=pdf_path.name,
            extraction_method=self.extraction_method,
            extraction_version=self.extraction_version,
            extracted_at=datetime.now(timezone.utc),
        )

        return result, block_units

    def ingest_pdf(
        self,
        pdf_path: Path,
        title: Optional[str] = None,
        publication_date: Optional[date] = None,
        doi: Optional[str] = None,
        authors: Optional[List[Dict[str, Any]]] = None,
        source_id: Optional[str] = None,
        check_duplicates: bool = False,
    ) -> Tuple[PaperCreateDTO, PaperDocumentCreateDTO]:
        """Runs the complete ingestion pipeline with validation, provenance, passage chunking, and persistence."""
        self.validate_file(pdf_path)
        content_hash = self.compute_sha256(pdf_path)

        # 1. Duplicate detection by SHA-256 content hash
        if check_duplicates and self.persistence and hasattr(self.persistence, "document_exists_by_hash"):
            if self.persistence.document_exists_by_hash(content_hash):
                raise DuplicateDocumentError(
                    f"Duplicate document detected (IntegrityError: content hash {content_hash} already exists)"
                )

        # 2. Extract structured representation
        extraction_result, block_units = self.extract_structured(pdf_path)

        # 3. Determine title and metadata
        final_title = title or extraction_result.title
        final_source_id = source_id or f"manual_{content_hash[:16]}"
        now_ts = datetime.now(timezone.utc)

        # Determine abstract if present in detected sections
        abstract_text = ""
        for sec in extraction_result.sections:
            if sec.name == "Abstract":
                abstract_text = sec.text
                break
        if not abstract_text:
            # First 500 chars as abstract preview
            abstract_text = extraction_result.full_text[:500]

        paper_dto = PaperCreateDTO(
            source_type=SourceType.MANUAL_UPLOAD,
            source_id=final_source_id,
            doi=doi,
            title=final_title,
            publication_date=publication_date,
            batch_id="manual",
            abstract=abstract_text,
            authors=authors or [],
            referenced_works=[],
            raw_metadata={
                "original_filename": pdf_path.name,
                "content_hash": content_hash,
                "detected_title": extraction_result.title,
                "num_pages": len(extraction_result.pages),
                "num_sections": len(extraction_result.sections),
            },
        )

        if not self.persistence:
            # Offline extraction without persistence
            doc_dto = PaperDocumentCreateDTO(
                paper_id=uuid.uuid4(),
                full_text=extraction_result.full_text,
                extraction_method=self.extraction_method,
                extraction_version=self.extraction_version,
                extracted_at=extraction_result.extracted_at,
                content_hash=content_hash,
                original_filename=pdf_path.name,
                acquisition_timestamp=now_ts,
                provenance_metadata={
                    "source_type": SourceType.MANUAL_UPLOAD.value,
                    "source_id": final_source_id,
                    "content_hash": content_hash,
                    "original_filename": pdf_path.name,
                    "extraction_version": self.extraction_version,
                    "num_pages": len(extraction_result.pages),
                },
            )
            return paper_dto, doc_dto

        # 4. Persist Paper
        paper_id = self.persistence.save_paper(paper_dto)
        if not paper_id:
            raise RuntimeError("Failed to retrieve paper ID after saving.")

        # 5. Persist PaperDocument
        doc_dto = PaperDocumentCreateDTO(
            paper_id=paper_id,
            full_text=extraction_result.full_text,
            extraction_method=self.extraction_method,
            extraction_version=self.extraction_version,
            extracted_at=extraction_result.extracted_at,
            content_hash=content_hash,
            original_filename=pdf_path.name,
            acquisition_timestamp=now_ts,
            provenance_metadata={
                "source_type": SourceType.MANUAL_UPLOAD.value,
                "source_id": final_source_id,
                "content_hash": content_hash,
                "original_filename": pdf_path.name,
                "extraction_version": self.extraction_version,
                "num_pages": len(extraction_result.pages),
            },
        )
        saved_doc_id = self.persistence.save_paper_document(doc_dto)
        doc_id = saved_doc_id or getattr(doc_dto, "id", uuid.uuid4())

        # 6. Create and persist deterministic passages
        passages = self.chunker.create_passages(
            blocks=block_units,
            paper_id=paper_id,
            document_id=doc_id,
        )
        if hasattr(self.persistence, "save_passages"):
            self.persistence.save_passages(passages)

        return paper_dto, doc_dto
