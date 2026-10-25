"""Mocked offline tests for IdeaGen Stage 2: Open-Access PDF Ingestion.

Covers:
- Successful PDF download, extraction, passage creation, and persistence
- PDF signature validation (%PDF-) and rejection of HTML responses
- Missing PDF URL, 403, 404, timeout, network failure, and oversized files
- Guaranteed temporary-file cleanup on success and all failure modes
- Duplicate document content hash detection
- Metadata-only NO_PDF persistence and exclusion from full-text pipeline processing
- Canonical OpenAlex ID normalization during raw-record parsing
- Backward compatibility of ManualPDFIngestionService
"""

import io
import os
import uuid
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest
import pymupdf

from ideagen.ingestion.models import (
    PaperCreateDTO,
    PaperDocumentCreateDTO,
    PassageCreateDTO,
    SourceType,
)
from ideagen.ingestion.downloader import (
    OpenAccessPDFDownloader,
    PDFDownloadError,
    PDFUnavailableError,
    OversizedPDFError,
    InvalidPDFSignatureError,
    SafeRedirectHandler,
    SSRFProtectionError,
)
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
)
from ideagen.ingestion.pdf import (
    ManualPDFIngestionService,
    DuplicateDocumentError,
)
from ideagen.ingestion.parser import OpenAlexResponseParser
from tests.conftest import PatchedUrlopenOpener


def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    """Generates valid, parseable in-memory PDF bytes with > 500 characters of text."""
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [title, "", "Abstract"]
    for i in range(25):
        lines.append(f"Abstract sentence {i}: {body}")
    lines.extend(["", "1. Introduction"])
    for i in range(25):
        lines.append(f"Introduction sentence {i}: {body}")
    page.insert_text((50, 50), "\n".join(lines))
    data = doc.tobytes()
    doc.close()
    return data


class DummyMockResponse:
    """Simulates an urllib response stream."""
    def __init__(self, data: bytes, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.bio = io.BytesIO(data)
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}
        self.status = status

    def read(self, amt: int = -1) -> bytes:
        return self.bio.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bio.close()



class MockPersistenceLayer:
    """In-memory persistence layer simulating database operations."""
    def __init__(self):
        self.saved_papers: List[PaperCreateDTO] = []
        self.saved_documents: List[PaperDocumentCreateDTO] = []
        self.saved_passages: List[PassageCreateDTO] = []

    def transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def _tx():
            snap_papers = list(self.saved_papers)
            snap_docs = list(self.saved_documents)
            snap_passages = list(self.saved_passages)
            try:
                yield self
            except Exception:
                self.saved_papers = snap_papers
                self.saved_documents = snap_docs
                self.saved_passages = snap_passages
                raise

        return _tx()

    def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
        return any(p.source_type == source_type and p.source_id == source_id for p in self.saved_papers)

    def paper_exists_by_doi(self, doi: str) -> bool:
        if not doi:
            return False
        return any(p.doi == doi for p in self.saved_papers if p.doi)

    def document_exists_by_hash(self, content_hash: str) -> bool:
        if not content_hash:
            return False
        return any(d.content_hash == content_hash for d in self.saved_documents)

    def save_paper(self, paper: PaperCreateDTO) -> uuid.UUID:
        paper_id = getattr(paper, "id", None) or uuid.uuid4()
        paper_obj = paper.model_copy()
        object.__setattr__(paper_obj, "id", paper_id)
        self.saved_papers.append(paper_obj)
        return paper_id

    def save_paper_document(self, doc: PaperDocumentCreateDTO) -> uuid.UUID:
        doc_id = getattr(doc, "id", None) or uuid.uuid4()
        doc_obj = doc.model_copy()
        object.__setattr__(doc_obj, "id", doc_id)
        self.saved_documents.append(doc_obj)
        return doc_id

    def save_passages(self, passages: List[PassageCreateDTO]) -> None:
        self.saved_passages.extend(passages)

    def get_passages_by_paper(self, paper_id: uuid.UUID) -> List[PassageCreateDTO]:
        return [p for p in self.saved_passages if p.paper_id == paper_id]

    def delete_paper(self, paper_id: uuid.UUID) -> bool:
        orig_len = len(self.saved_papers)
        self.saved_papers = [p for p in self.saved_papers if getattr(p, "id", None) != paper_id]
        self.saved_documents = [d for d in self.saved_documents if d.paper_id != paper_id]
        self.saved_passages = [ps for ps in self.saved_passages if ps.paper_id != paper_id]
        return len(self.saved_papers) < orig_len



@pytest.fixture
def sample_work_dict() -> Dict[str, Any]:
    return {
        "id": "https://openalex.org/W4285719001",
        "doi": "https://doi.org/10.1234/test.2025",
        "title": "Quantum Computing Advances",
        "publication_date": "2025-02-15",
        "abstract_inverted_index": {
            "Quantum": [0],
            "computing": [1],
            "advances": [2],
            "rapidly.": [3],
        },
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
        ],
        "best_oa_location": {
            "pdf_url": "https://example.org/papers/w4285719001.pdf",
            "is_oa": True,
        },
        "primary_location": {
            "pdf_url": "https://backup.example.org/w4285719001.pdf",
        },
    }


def test_successful_pdf_download_and_ingestion(monkeypatch, sample_work_dict):
    """Verifies that accessible PDFs are downloaded, parsed, chunked, and saved with full text."""
    pdf_bytes = create_sample_pdf_bytes("Quantum Computing Advances")
    persistence = MockPersistenceLayer()

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.source_id == "W4285719001"
    assert result.doi == "10.1234/test.2025"
    assert result.batch_id == "2025 Q1"
    assert result.passage_count > 0

    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 1
    assert len(persistence.saved_passages) == result.passage_count

    saved_paper = persistence.saved_papers[0]
    assert saved_paper.source_id == "W4285719001"
    assert saved_paper.raw_metadata["pdf_status"] == "ACCESSIBLE"
    assert saved_paper.raw_metadata["has_pdf"] is True


def test_pdf_signature_validation_and_html_rejection(monkeypatch, sample_work_dict):
    """Verifies rejection of HTML responses / paywall landing pages with fallback to NO_PDF."""
    html_bytes = b"<!DOCTYPE html><html><head><title>Access Denied</title></head><body>Login required</body></html>"
    persistence = MockPersistenceLayer()

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(html_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    downloader = OpenAccessPDFDownloader(opener=PatchedUrlopenOpener())
    # Downloader itself directly raises InvalidPDFSignatureError
    with pytest.raises(InvalidPDFSignatureError, match="HTML landing page"):
        with downloader.download_pdf("https://example.org/paywall.pdf"):
            pass

    # When run through OpenAlexPaperIngestionService, it safely falls back to NO_PDF
    service = OpenAlexPaperIngestionService(persistence=persistence, downloader=downloader)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.NO_PDF
    assert result.has_pdf is False
    assert result.document_id is None
    assert result.passage_count == 0

    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 0
    assert len(persistence.saved_passages) == 0
    assert persistence.saved_papers[0].raw_metadata["pdf_status"] == "NO_PDF"


def test_invalid_binary_signature_rejected(monkeypatch):
    """Verifies non-PDF binary data (e.g. random bytes) is rejected."""
    junk_bytes = b"RANDOM_NON_PDF_BINARY_DATA_HEADER" * 10
    downloader = OpenAccessPDFDownloader(opener=PatchedUrlopenOpener())

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(junk_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(InvalidPDFSignatureError, match="File lacks valid '%PDF-' header signature"):
        with downloader.download_pdf("https://example.org/junk.pdf"):
            pass


def test_missing_pdf_url(sample_work_dict):
    """Verifies paper without any PDF URL is saved as NO_PDF."""
    persistence = MockPersistenceLayer()
    sample_work_dict["best_oa_location"] = None
    sample_work_dict["primary_location"] = None

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.NO_PDF
    assert result.has_pdf is False
    assert result.error_message == "NO_PDF_URL"

    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 0
    assert len(persistence.saved_passages) == 0
    assert persistence.saved_papers[0].raw_metadata["pdf_status"] == "NO_PDF"


def test_http_403_and_404_errors(monkeypatch, sample_work_dict):
    """Verifies HTTP 403 / 404 lead to NO_PDF metadata-only record."""
    persistence = MockPersistenceLayer()

    def mock_urlopen_404(req, timeout=30.0):
        raise urllib.error.HTTPError(
            url="https://example.org/missing.pdf",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_404)

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.NO_PDF
    assert result.has_pdf is False
    assert "HTTP 404" in result.error_message
    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 0


def test_network_timeout_and_error(monkeypatch, sample_work_dict):
    """Verifies network timeouts and connection errors fall back to NO_PDF."""
    persistence = MockPersistenceLayer()

    def mock_urlopen_timeout(req, timeout=30.0):
        raise TimeoutError("Connection timed out")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_timeout)

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.NO_PDF
    assert result.has_pdf is False
    assert "Timeout" in result.error_message
    assert len(persistence.saved_documents) == 0


def test_oversized_pdf_rejection(monkeypatch, sample_work_dict):
    """Verifies Content-Length header and streaming size limits enforce the 50 MB bound."""
    persistence = MockPersistenceLayer()
    downloader = OpenAccessPDFDownloader(max_size_bytes=1024, opener=PatchedUrlopenOpener())  # Small 1KB limit for test

    # Case 1: Content-Length header exceeds limit
    def mock_urlopen_header(req, timeout=30.0):
        return DummyMockResponse(b"%PDF-1.4...", headers={"Content-Length": "2048"})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_header)

    with pytest.raises(OversizedPDFError, match="Content-Length"):
        with downloader.download_pdf("https://example.org/large.pdf"):
            pass

    # Case 2: Streaming byte accumulation exceeds limit (no Content-Length header)
    big_pdf = b"%PDF-" + b"x" * 2000

    def mock_urlopen_stream(req, timeout=30.0):
        return DummyMockResponse(big_pdf, headers={})

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_stream)

    with pytest.raises(OversizedPDFError, match="Downloaded bytes exceeded limit"):
        with downloader.download_pdf("https://example.org/large_stream.pdf"):
            pass

    # When run through service, falls back to NO_PDF
    service = OpenAlexPaperIngestionService(persistence=persistence, downloader=downloader)
    result = service.ingest_work(sample_work_dict)
    assert result.status == IngestionStatus.NO_PDF
    assert "exceeded limit" in result.error_message


def test_temporary_file_cleanup_on_success_and_error(monkeypatch):
    """Verifies that temporary files are deleted after download completes or fails."""
    downloader = OpenAccessPDFDownloader(opener=PatchedUrlopenOpener())
    pdf_bytes = create_sample_pdf_bytes("Temp File Test")
    captured_paths = []

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # Success case: file exists inside block, cleaned up outside
    with downloader.download_pdf("https://example.org/test.pdf") as temp_p:
        captured_paths.append(temp_p)
        assert temp_p.exists()
        assert temp_p.stat().st_size > 0

    assert not captured_paths[0].exists()

    # Exception case: error occurs during processing inside the context manager
    error_path = None
    try:
        with downloader.download_pdf("https://example.org/test.pdf") as temp_p2:
            error_path = temp_p2
            assert temp_p2.exists()
            raise RuntimeError("Extraction failed!")
    except RuntimeError:
        pass

    assert error_path is not None
    assert not error_path.exists()


def test_duplicate_pdf_content(monkeypatch, sample_work_dict):
    """Verifies duplicate document content hash raises DuplicateDocumentError and prevents records."""
    pdf_bytes = create_sample_pdf_bytes("Duplicate Content")
    persistence = MockPersistenceLayer()

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    service = OpenAlexPaperIngestionService(persistence=persistence)

    # First ingestion succeeds
    res1 = service.ingest_work(sample_work_dict)
    assert res1.status == IngestionStatus.INGESTED

    # Different work with identical PDF content
    different_work = dict(sample_work_dict)
    different_work["id"] = "https://openalex.org/W9999999999"
    different_work["doi"] = "10.9999/different"

    with pytest.raises(DuplicateDocumentError, match="Duplicate document detected"):
        service.ingest_work(different_work, check_duplicates=True)

    # Check that second paper was NOT partially created
    assert len(persistence.saved_papers) == 1


def test_metadata_only_no_pdf_exclusion_from_full_text(sample_work_dict):
    """Verifies that NO_PDF records are excluded from full-text passage retrieval."""
    persistence = MockPersistenceLayer()
    sample_work_dict["best_oa_location"] = None
    sample_work_dict["primary_location"] = None

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    assert result.status == IngestionStatus.NO_PDF
    assert result.paper_id is not None

    # Verify query for passages returns empty list
    passages = persistence.get_passages_by_paper(result.paper_id)
    assert passages == []
    assert len(persistence.saved_documents) == 0


def test_url_scheme_validation():
    """Verifies non-HTTP(S) schemes are blocked."""
    downloader = OpenAccessPDFDownloader()

    with pytest.raises(PDFDownloadError, match="Disallowed URL scheme"):
        downloader.validate_url("file:///etc/passwd")

    with pytest.raises(PDFDownloadError, match="Disallowed URL scheme"):
        downloader.validate_url("ftp://example.org/doc.pdf")

    with pytest.raises(PDFDownloadError, match="Empty or invalid URL"):
        downloader.validate_url("")


def test_manual_pdf_ingestion_service_backward_compatibility(tmp_path):
    """Verifies ManualPDFIngestionService continues to work identically for manual uploads."""
    pdf_bytes = create_sample_pdf_bytes("Manual Upload Test")
    pdf_file = tmp_path / "manual_test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    service = ManualPDFIngestionService()
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=pdf_file,
        title="Custom Manual Title",
        publication_date=date(2023, 5, 10),
    )

    assert paper_dto.source_type == SourceType.MANUAL_UPLOAD
    assert paper_dto.batch_id == "manual"
    assert paper_dto.title == "Custom Manual Title"
    assert paper_dto.publication_date == date(2023, 5, 10)
    assert doc_dto.full_text != ""


def test_safe_redirect_handler_allowed_public_redirect():
    """Verifies that legitimate public HTTP/HTTPS redirects are permitted."""
    handler = SafeRedirectHandler()
    req = urllib.request.Request("https://openalex.org/work/W123.pdf", headers={"User-Agent": "IdeaGen (mailto:author@example.com)"})

    new_req = handler.redirect_request(
        req=req,
        fp=None,
        code=302,
        msg="Found",
        headers={},
        newurl="https://cdn.example.org/papers/W123.pdf",
    )

    assert new_req is not None
    assert new_req.full_url == "https://cdn.example.org/papers/W123.pdf"
    # User-Agent mailto must be stripped across different origins
    assert new_req.headers["User-agent"] == "IdeaGen"


def test_safe_redirect_handler_blocks_private_and_loopback_destinations():
    """Verifies that redirects targeting loopback, private, or alternative numeric IPs are blocked before connecting."""
    base_req = urllib.request.Request("https://example.org/paper.pdf")

    prohibited_destinations = [
        "http://127.0.0.1/admin.pdf",
        "http://127.0.0.2:8080/secret.pdf",
        "http://localhost:8000/internal",
        "http://[::1]/internal.pdf",
        "http://2130706433/num.pdf",          # 127.0.0.1 in decimal integer
        "http://0x7f000001/hex.pdf",          # 127.0.0.1 in hex
        "http://0177.0.0.1/octal.pdf",        # 127.0.0.1 in dotted octal
        "http://[::ffff:127.0.0.1]/v6.pdf",   # IPv6-mapped IPv4 loopback
        "http://10.0.0.1/corp.pdf",           # Private RFC 1918
        "http://172.16.5.1/corp.pdf",         # Private RFC 1918
        "http://192.168.1.1/router.pdf",      # Private RFC 1918
        "http://0.0.0.0/any.pdf",             # Unspecified
    ]

    for dest in prohibited_destinations:
        handler = SafeRedirectHandler()
        with pytest.raises(SSRFProtectionError, match="Blocked connection"):
            handler.redirect_request(
                req=base_req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl=dest,
            )


def test_safe_redirect_handler_blocks_cloud_metadata():
    """Verifies that redirects targeting cloud metadata addresses are blocked."""
    base_req = urllib.request.Request("https://example.org/paper.pdf")

    metadata_destinations = [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
    ]

    for dest in metadata_destinations:
        handler = SafeRedirectHandler()
        with pytest.raises(SSRFProtectionError, match="Blocked connection"):
            handler.redirect_request(
                req=base_req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl=dest,
            )


def test_safe_redirect_handler_blocks_prohibited_schemes():
    """Verifies that redirects attempting scheme changes to file:// or ftp:// are blocked."""
    handler = SafeRedirectHandler()
    base_req = urllib.request.Request("https://example.org/paper.pdf")

    with pytest.raises(PDFDownloadError, match="Disallowed URL scheme"):
        handler.redirect_request(
            req=base_req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="file:///etc/passwd",
        )

    with pytest.raises(PDFDownloadError, match="Disallowed URL scheme"):
        handler.redirect_request(
            req=base_req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="ftp://example.org/paper.pdf",
        )


def test_dishonest_content_length_header_detected(monkeypatch):
    """Verifies that an undersized Content-Length header cannot bypass the 50 MB bound."""
    downloader = OpenAccessPDFDownloader(max_size_bytes=1024, opener=PatchedUrlopenOpener())

    # Remote server claims file is only 50 bytes, but streams 2000 bytes
    dishonest_body = b"%PDF-" + b"0" * 2000
    mock_resp = DummyMockResponse(dishonest_body, headers={"Content-Length": "50"})

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: mock_resp)

    with pytest.raises(OversizedPDFError, match="Downloaded bytes exceeded limit"):
        with downloader.download_pdf("https://example.org/dishonest.pdf"):
            pass


def test_mid_persistence_failure_rollback_and_error_reporting(monkeypatch, sample_work_dict):
    """Verifies that when document persistence fails after paper has been saved, paper is rolled back and reported as ERROR."""
    pdf_bytes = create_sample_pdf_bytes("Rollback Test")
    persistence = MockPersistenceLayer()

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # Monkeypatch save_paper_document to simulate database failure
    def mock_save_document_fail(doc_dto):
        raise RuntimeError("Database disk full while saving document")

    persistence.save_paper_document = mock_save_document_fail

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    # 1. Result status must be distinctly ERROR (never NO_PDF)
    assert result.status == IngestionStatus.ERROR
    assert result.has_pdf is True  # Download was successful
    assert "Database disk full" in result.error_message

    # 2. Paper record must be rolled back to avoid orphaned / inconsistent records
    assert len(persistence.saved_papers) == 0
    assert len(persistence.saved_documents) == 0
    assert len(persistence.saved_passages) == 0


def test_mid_persistence_failure_without_rollback_support(monkeypatch, sample_work_dict):
    """Verifies that if persistence layer lacks delete_paper, no duplicate save_paper is attempted and status is ERROR."""
    pdf_bytes = create_sample_pdf_bytes("No Rollback Test")

    class MockPersistenceWithoutDelete:
        def __init__(self):
            self.saved_papers = []
            self.saved_documents = []
            self.saved_passages = []

        def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
            return any(p.source_type == source_type and p.source_id == source_id for p in self.saved_papers)

        def paper_exists_by_doi(self, doi: str) -> bool:
            return False

        def document_exists_by_hash(self, content_hash: str) -> bool:
            return False

        def save_paper(self, paper: PaperCreateDTO) -> uuid.UUID:
            paper_id = getattr(paper, "id", None) or uuid.uuid4()
            paper_obj = paper.model_copy()
            object.__setattr__(paper_obj, "id", paper_id)
            self.saved_papers.append(paper_obj)
            return paper_id

        def save_paper_document(self, doc: PaperDocumentCreateDTO) -> uuid.UUID:
            raise RuntimeError("Foreign key constraint violation")

        def save_passages(self, passages: List[PassageCreateDTO]) -> None:
            pass

    persistence = MockPersistenceWithoutDelete()
    assert not hasattr(persistence, "delete_paper")

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    service = OpenAlexPaperIngestionService(persistence=persistence)
    result = service.ingest_work(sample_work_dict)

    # Status must be ERROR, not NO_PDF
    assert result.status == IngestionStatus.ERROR
    assert result.has_pdf is True
    assert "Foreign key constraint violation" in result.error_message

    # Exactly 1 paper was saved, save_paper was NOT called a second time
    assert len(persistence.saved_papers) == 1


def test_repeated_ingestion_idempotency(monkeypatch, sample_work_dict):
    """Verifies that calling ingest_work repeatedly on the same paper is idempotent and returns SKIPPED_DUPLICATE."""
    pdf_bytes = create_sample_pdf_bytes("Idempotency Test")
    persistence = MockPersistenceLayer()

    def mock_urlopen(req, timeout=30.0):
        return DummyMockResponse(pdf_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    service = OpenAlexPaperIngestionService(persistence=persistence)

    # First ingestion succeeds
    res1 = service.ingest_work(sample_work_dict, check_duplicates=True)
    assert res1.status == IngestionStatus.INGESTED
    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 1
    orig_passage_count = len(persistence.saved_passages)

    # Second ingestion with exact same work dict
    res2 = service.ingest_work(sample_work_dict, check_duplicates=True)
    assert res2.status == IngestionStatus.SKIPPED_DUPLICATE
    assert res2.has_pdf is False

    # Persistence counts must remain identical (no duplicate rows)
    assert len(persistence.saved_papers) == 1
    assert len(persistence.saved_documents) == 1
    assert len(persistence.saved_passages) == orig_passage_count


# ==============================================================================
# Task A Tests: Connection-Level SSRF Protection, DNS Fail-Closed, & IP Pinning
# ==============================================================================

import socket
from ideagen.ingestion.downloader import (
    ssrf_safe_create_connection,
    validate_host,
    validate_url,
)


def test_connection_destination_ip_pinning_and_rebinding_blocked(monkeypatch):
    """Verifies that connection-time DNS rebinding to metadata or private IP is blocked."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        # Simulates a domain that passed syntactic checks but resolves to cloud metadata at connection time
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    with pytest.raises(SSRFProtectionError, match="restricted IP: 169.254.169.254"):
        ssrf_safe_create_connection(("rebind.attacker.org", 80))


def test_connection_dns_failure_fails_closed(monkeypatch):
    """Verifies that DNS failure (gaierror) fails closed and raises SSRFProtectionError."""
    def mock_getaddrinfo_fail(host, port, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo_fail)

    with pytest.raises(SSRFProtectionError, match="DNS resolution failed for host"):
        ssrf_safe_create_connection(("nonexistent-domain.org", 443))


def test_connection_multiple_dns_answers_with_prohibited_ip_rejected(monkeypatch):
    """Verifies that if DNS returns dual-homed answers (one public, one private), all answers are validated and blocked."""
    def mock_getaddrinfo_dual(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo_dual)

    with pytest.raises(SSRFProtectionError, match="restricted IP: 10.0.0.1"):
        ssrf_safe_create_connection(("dual-homed.publisher.org", 80))


def test_connection_public_ip_destination_succeeds(monkeypatch):
    """Verifies that a valid public IP passes validation and successfully establishes socket connection."""
    connected_addr = None

    class MockSocket:
        def settimeout(self, timeout):
            pass
        def bind(self, addr):
            pass
        def connect(self, sa):
            nonlocal connected_addr
            connected_addr = sa
        def close(self):
            pass

    def mock_getaddrinfo_pub(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo_pub)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: MockSocket())

    sock = ssrf_safe_create_connection(("example.com", 443))
    assert sock is not None
    assert connected_addr == ("93.184.216.34", 443)


def test_streamed_oversized_response_without_content_length(monkeypatch):
    """Verifies that streaming byte limit triggers even if Content-Length header is omitted."""
    downloader = OpenAccessPDFDownloader(max_size_bytes=100, opener=PatchedUrlopenOpener())
    # Stream 150 bytes with no Content-Length
    stream_data = b"%PDF-" + b"x" * 150
    mock_resp = DummyMockResponse(stream_data, headers={})

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: mock_resp)

    with pytest.raises(OversizedPDFError, match="Downloaded bytes exceeded limit"):
        with downloader.download_pdf("https://example.org/stream_no_cl.pdf"):
            pass


# ==============================================================================
# Task B Tests: Atomic Persistence Rollback with Real SQLAlchemy Session
# ==============================================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage
from ideagen.db.persistence import SQLAlchemyPersistenceLayer


@pytest.fixture
def sqlite_persistence():
    """Provides a fresh in-memory SQLite database session and persistence layer."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    persistence = SQLAlchemyPersistenceLayer(session)
    yield persistence
    session.close()


def test_sqlalchemy_atomic_persistence_success(monkeypatch, sample_work_dict, sqlite_persistence):
    """Verifies successful atomic persistence of Paper, PaperDocument, and Passages in SQLite."""
    pdf_bytes = create_sample_pdf_bytes("Atomic Persistence Success")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: DummyMockResponse(pdf_bytes))

    service = OpenAlexPaperIngestionService(persistence=sqlite_persistence)
    res = service.ingest_work(sample_work_dict)

    assert res.status == IngestionStatus.INGESTED
    assert res.has_pdf is True
    assert res.paper_id is not None

    session = sqlite_persistence.session
    saved_paper = session.query(Paper).filter(Paper.id == res.paper_id).first()
    assert saved_paper is not None
    assert saved_paper.source_id == "W4285719001"

    saved_doc = session.query(PaperDocument).filter(PaperDocument.paper_id == res.paper_id).first()
    assert saved_doc is not None

    passages = session.query(PaperPassage).filter(PaperPassage.paper_id == res.paper_id).all()
    assert len(passages) == res.passage_count
    assert len(passages) > 0


def test_sqlalchemy_atomic_persistence_rollback_on_passage_failure(monkeypatch, sample_work_dict, sqlite_persistence):
    """Verifies that if passage persistence fails, paper and document are rolled back atomically (0 rows)."""
    pdf_bytes = create_sample_pdf_bytes("Passage Failure Rollback")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: DummyMockResponse(pdf_bytes))

    # Force save_passages to raise an exception
    orig_save_passages = sqlite_persistence.save_passages
    def mock_fail_passages(passage_dtos, commit=None):
        raise RuntimeError("Database write error during passages commit")

    sqlite_persistence.save_passages = mock_fail_passages

    service = OpenAlexPaperIngestionService(persistence=sqlite_persistence)
    res = service.ingest_work(sample_work_dict)

    # Status must be ERROR, has_pdf=True
    assert res.status == IngestionStatus.ERROR
    assert res.has_pdf is True
    assert res.paper_id is None
    assert "Database write error during passages commit" in res.error_message

    # Entire transaction was rolled back: 0 papers, 0 documents, 0 passages in DB!
    session = sqlite_persistence.session
    assert session.query(Paper).count() == 0
    assert session.query(PaperDocument).count() == 0
    assert session.query(PaperPassage).count() == 0


def test_sqlalchemy_atomic_persistence_rollback_on_document_failure(monkeypatch, sample_work_dict, sqlite_persistence):
    """Verifies that if document persistence fails, paper is rolled back atomically (0 rows)."""
    pdf_bytes = create_sample_pdf_bytes("Doc Failure Rollback")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: DummyMockResponse(pdf_bytes))

    def mock_fail_document(doc_dto, commit=None):
        raise RuntimeError("Constraint error saving document")

    sqlite_persistence.save_paper_document = mock_fail_document

    service = OpenAlexPaperIngestionService(persistence=sqlite_persistence)
    res = service.ingest_work(sample_work_dict)

    assert res.status == IngestionStatus.ERROR
    assert res.has_pdf is True
    assert res.paper_id is None
    assert "Constraint error saving document" in res.error_message

    session = sqlite_persistence.session
    assert session.query(Paper).count() == 0
    assert session.query(PaperDocument).count() == 0
    assert session.query(PaperPassage).count() == 0


def test_manual_pdf_ingestion_atomic_rollback(tmp_path, sqlite_persistence):
    """Verifies that ManualPDFIngestionService rolls back atomically on passage persistence failure."""
    pdf_bytes = create_sample_pdf_bytes("Manual PDF Rollback Test")
    pdf_file = tmp_path / "manual_test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    # Force save_passages to fail
    def mock_fail_passages(passage_dtos, commit=None):
        raise RuntimeError("Passage disk error")

    sqlite_persistence.save_passages = mock_fail_passages

    manual_service = ManualPDFIngestionService(persistence=sqlite_persistence)

    with pytest.raises(RuntimeError, match="Passage disk error"):
        manual_service.ingest_pdf(pdf_file)

    session = sqlite_persistence.session
    assert session.query(Paper).count() == 0
    assert session.query(PaperDocument).count() == 0
    assert session.query(PaperPassage).count() == 0


def test_sqlalchemy_persistence_standalone_backward_compatibility(sqlite_persistence):
    """Verifies that standalone save_paper/save_document without transaction() still commits for legacy callers."""
    paper_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="standalone_001",
        doi=None,
        title="Standalone Paper",
        publication_date=date(2025, 1, 1),
        batch_id="manual",
        abstract="Standalone abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    paper_id = sqlite_persistence.save_paper(paper_dto)
    assert paper_id is not None

    session = sqlite_persistence.session
    saved = session.query(Paper).filter(Paper.id == paper_id).first()
    assert saved is not None
    assert saved.title == "Standalone Paper"


# ==============================================================================
# SEC-01, SEC-02, and TEST-01 Regression Tests
# ==============================================================================

def test_validate_host_normalizes_trailing_dots():
    """Verifies that validate_host rejects prohibited destinations with trailing dots."""
    prohibited_trailing_dots = [
        "metadata.google.internal.",
        "localhost.",
        "instance-data.",
        "metadata.",
        "worker.local.",
        "api.internal.",
        "gateway.lan.",
        "home.home.arpa.",
        "127.0.0.1.",
        "10.0.0.1.",
        "169.254.169.254.",
    ]
    for host in prohibited_trailing_dots:
        with pytest.raises(SSRFProtectionError):
            validate_host(host)

    # validate_url with trailing dot URL
    with pytest.raises(SSRFProtectionError):
        validate_url("http://metadata.google.internal./computeMetadata/v1/")

    with pytest.raises(SSRFProtectionError):
        validate_url("http://cluster.internal./secrets.pdf")


def test_open_request_custom_opener_dispatch_without_urlopen_mock(monkeypatch):
    """Verifies that _open_request dispatches through custom opener and invokes SSRFSafe handler."""
    # Ensure urlopen is NOT monkeypatched so custom opener chain is exercised
    assert getattr(urllib.request.urlopen, "__name__", "") == "urlopen"

    def mock_getaddrinfo_rebind(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo_rebind)

    downloader = OpenAccessPDFDownloader()
    req = urllib.request.Request("http://rebind.attacker.org/paper.pdf")

    # _open_request must run build_opener -> SSRFSafeHTTPHandler -> ssrf_safe_create_connection
    with pytest.raises(SSRFProtectionError, match="restricted IP: 169.254.169.254"):
        downloader._open_request(req)


def test_downloader_ignores_environment_proxies(monkeypatch):
    """Verifies that ProxyHandler({}) prevents environment proxies from altering requests."""
    # Set proxy variables pointing to a local mock proxy address
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:59999")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:59999")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:59999")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:59999")

    resolved_hosts = []

    def mock_getaddrinfo_capture(host, port, *args, **kwargs):
        resolved_hosts.append(host)
        # Fail closed with SSRFProtectionError to abort connection without real network
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo_capture)

    downloader = OpenAccessPDFDownloader()
    req = urllib.request.Request("http://openaccess.example.org/doc.pdf")

    with pytest.raises(SSRFProtectionError):
        downloader._open_request(req)

    # 1. Host in request must NOT be rewritten to proxy host
    assert req.host == "openaccess.example.org"

    # 2. DNS resolution must resolve the target host, NEVER the proxy (127.0.0.1)
    assert "openaccess.example.org" in resolved_hosts
    assert "127.0.0.1" not in resolved_hosts
