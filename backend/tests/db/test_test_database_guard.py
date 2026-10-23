"""The test suite must refuse to run against databases under the repository's data/ directory."""

from pathlib import Path

from tests.conftest import _PROJECT_DATA_DIR, _sqlite_file_inside_project_data


def test_guard_detects_demo_database_urls():
    demo = _PROJECT_DATA_DIR / "demo" / "ideagen_demo_20261022.db"
    assert _sqlite_file_inside_project_data(f"sqlite:///{demo}") == demo
    # Relative URL as written in the Makefile / shell from the backend directory.
    relative = Path("..") / "data" / "demo" / "ideagen_demo_20261022.db"
    backend_dir = Path(__file__).resolve().parents[2]
    import os

    cwd = os.getcwd()
    try:
        os.chdir(backend_dir)
        assert _sqlite_file_inside_project_data(f"sqlite:///{relative}") == demo
    finally:
        os.chdir(cwd)


def test_guard_allows_temporary_and_non_sqlite_databases(tmp_path):
    assert _sqlite_file_inside_project_data(None) is None
    assert _sqlite_file_inside_project_data("sqlite:///:memory:") is None
    assert _sqlite_file_inside_project_data(f"sqlite:///{tmp_path / 'x.db'}") is None
    assert _sqlite_file_inside_project_data("postgresql+psycopg2://u:p@localhost/ideagen_test") is None
