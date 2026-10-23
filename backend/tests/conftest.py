"""Shared test seams.

Vector index: discovery tests persist FAISS indexes to a per-test temp directory, never to
the repository's data/indexes.

PDF downloads: production code always uses the SSRF-safe opener. Tests that fake HTTP by
monkeypatching ``urllib.request.urlopen`` inject ``PatchedUrlopenOpener`` through the
downloader's explicit ``opener`` parameter instead. Downloaders built implicitly by
``OpenAlexPaperIngestionService`` get it automatically via the autouse fixture below.
"""
import os
import urllib.request
from pathlib import Path

import pytest

from ideagen.ingestion import openalex_ingestion
from ideagen.ingestion.downloader import OpenAccessPDFDownloader, PDFDownloadError

_PROJECT_DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()


def _sqlite_file_inside_project_data(db_url: str | None) -> Path | None:
    """Returns the database path when ``db_url`` is a SQLite file under the repository's data/ dir."""
    if not db_url or not db_url.startswith("sqlite") or ":memory:" in db_url:
        return None
    raw_path = db_url.split(":///", 1)[-1].split("?", 1)[0]
    if not raw_path:
        return None
    db_path = Path(raw_path).expanduser().resolve()
    return db_path if _PROJECT_DATA_DIR in db_path.parents else None


def pytest_configure(config):
    """Refuse to run against a real project database.

    Several fixtures connect to IDEAGEN_DATABASE_URL (or the backend .env) and run DDL and test
    transactions there, and TestClient startup hooks touch the global session factory. Pointing
    them at e.g. data/demo/*.db would write test data into the demo database.
    """
    from ideagen.core.config import get_settings

    for source, url in (
        ("IDEAGEN_DATABASE_URL", os.environ.get("IDEAGEN_DATABASE_URL")),
        ("backend settings (.env)", get_settings().database_url),
    ):
        db_path = _sqlite_file_inside_project_data(url)
        if db_path is not None:
            pytest.exit(
                f"Refusing to run tests: {source} points at the project database {db_path}. "
                "Unset IDEAGEN_DATABASE_URL (or use a temporary database) before running pytest.",
                returncode=4,
            )


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
