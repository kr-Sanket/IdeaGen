"""Shared test seams.

Vector index: discovery tests persist FAISS indexes to a per-test temp directory, never to
the repository's data/indexes.

PDF downloads: production code always uses the SSRF-safe opener. Tests that fake HTTP by
monkeypatching ``urllib.request.urlopen`` inject ``PatchedUrlopenOpener`` through the
downloader's explicit ``opener`` parameter instead. Downloaders built implicitly by
``OpenAlexPaperIngestionService`` get it automatically via the autouse fixture below.
"""
import urllib.request

import pytest

from ideagen.ingestion import openalex_ingestion
from ideagen.ingestion.downloader import OpenAccessPDFDownloader, PDFDownloadError


class PatchedUrlopenOpener:
    """Opener that routes requests to the test's monkeypatched ``urllib.request.urlopen``.

    Fails closed if the test did not patch urlopen, so no test can reach the network.
    """

    def open(self, req, timeout=None):
        if getattr(urllib.request.urlopen, "__name__", "") == "urlopen":
            raise PDFDownloadError("Network access is disabled in tests: patch urllib.request.urlopen")
        return urllib.request.urlopen(req, timeout=timeout)


@pytest.fixture(autouse=True)
def _isolate_vector_index_dir(monkeypatch, tmp_path):
    """Discovery without a cutoff persists its FAISS index under settings.index_dir;
    keep tests from writing into the repository's data/indexes directory."""
    monkeypatch.setenv("IDEAGEN_INDEX_DIR", str(tmp_path / "indexes"))


@pytest.fixture(autouse=True)
def _ingestion_service_uses_test_pdf_opener(monkeypatch):
    def _default_downloader(*args, **kwargs):
        kwargs.setdefault("opener", PatchedUrlopenOpener())
        return OpenAccessPDFDownloader(*args, **kwargs)

    monkeypatch.setattr(openalex_ingestion, "OpenAccessPDFDownloader", _default_downloader)
