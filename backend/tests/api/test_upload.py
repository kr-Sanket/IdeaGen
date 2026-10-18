import os
from fastapi.testclient import TestClient
from ideagen.api.main import app
import io
import pytest

client = TestClient(app)

def create_mock_pdf_content(valid=True):
    # Just creating some dummy PDF-like bytes or empty bytes for invalid
    if valid:
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n%%EOF"
    else:
        return b"This is just some text, not a PDF."

def test_upload_single_pdf(monkeypatch):
    # Mock ManualPDFIngestionService.extract_text to avoid PyMuPDF dependency on dummy bytes
    def mock_extract(*args, **kwargs):
        return "This is a mock extracted text from a valid PDF." * 50
    
    from ideagen.ingestion.pdf import ManualPDFIngestionService
    monkeypatch.setattr(ManualPDFIngestionService, "extract_text", mock_extract)
    
    file_obj = io.BytesIO(create_mock_pdf_content(valid=True))
    files = [("files", ("test1.pdf", file_obj, "application/pdf"))]
    
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "UPLOAD"
    assert data["count"] == 1
    assert data["papers"][0]["title"] == "test1.pdf"
    assert "This is a mock extracted text" in data["papers"][0]["abstract"]

def test_upload_multiple_pdfs(monkeypatch):
    def mock_extract(*args, **kwargs):
        return "Mock extracted text." * 50
    
    from ideagen.ingestion.pdf import ManualPDFIngestionService
    monkeypatch.setattr(ManualPDFIngestionService, "extract_text", mock_extract)
    
    files = [
        ("files", ("test1.pdf", io.BytesIO(b"data1"), "application/pdf")),
        ("files", ("test2.pdf", io.BytesIO(b"data2"), "application/pdf")),
        ("files", ("test3.pdf", io.BytesIO(b"data3"), "application/pdf"))
    ]
    
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 3

def test_upload_limit():
    files = []
    for i in range(21):
        files.append(("files", (f"test{i}.pdf", io.BytesIO(b"data"), "application/pdf")))
        
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 400
    assert "Maximum 20 files" in res.json()["detail"]

def test_upload_invalid_file(monkeypatch):
    # Test non-PDF extension
    files = [
        ("files", ("test.txt", io.BytesIO(b"not a pdf"), "text/plain"))
    ]
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 400
    assert "No readable PDFs" in res.json()["detail"]

def test_upload_malformed_pdf(monkeypatch):
    # PyMuPDF should raise PDFExtractionError for malformed bytes
    file_obj = io.BytesIO(create_mock_pdf_content(valid=False))
    files = [("files", ("malformed.pdf", file_obj, "application/pdf"))]
    
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 400
    assert "No readable PDFs found." in res.json()["detail"]
