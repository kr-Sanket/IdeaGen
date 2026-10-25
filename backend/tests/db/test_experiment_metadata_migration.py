"""Migration e7b3c9a14f52: experiment_metadata table on Alembic-managed databases.

Runs the real Alembic chain against throwaway SQLite files. PostgreSQL behaviour is not
exercised here.
"""
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from ideagen.db.models import Base, ExperimentMetadataRecord

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


@pytest.fixture
def alembic_sqlite(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'alembic_test.db'}"
    monkeypatch.setenv("IDEAGEN_DATABASE_URL", url)
    return Config(str(ALEMBIC_INI)), sa.create_engine(url)


def _columns(engine):
    return {c["name"]: c for c in sa.inspect(engine).get_columns("experiment_metadata")}


def test_upgrade_creates_table_matching_model_and_downgrade_removes_it(alembic_sqlite):
    config, engine = alembic_sqlite

    command.upgrade(config, "head")
    cols = _columns(engine)
    assert set(cols) == {"key", "value"}
    assert cols["value"]["nullable"] is False
    assert sa.inspect(engine).get_pk_constraint("experiment_metadata")["constrained_columns"] == ["key"]
    model_cols = {c.name for c in ExperimentMetadataRecord.__table__.columns}
    assert set(cols) == model_cols

    command.downgrade(config, "d5e8a71b2c40")
    assert not sa.inspect(engine).has_table("experiment_metadata")

    command.upgrade(config, "head")
    assert sa.inspect(engine).has_table("experiment_metadata")


def test_upgrade_keeps_existing_table_created_by_create_all(alembic_sqlite):
    config, engine = alembic_sqlite
    command.upgrade(config, "d5e8a71b2c40")
    ExperimentMetadataRecord.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(ExperimentMetadataRecord.__table__.insert().values(key="configuration_fingerprint", value="{}"))

    command.upgrade(config, "head")

    with engine.connect() as conn:
        rows = conn.execute(sa.select(ExperimentMetadataRecord.__table__)).all()
    assert [tuple(r) for r in rows] == [("configuration_fingerprint", "{}")]


def test_upgrade_refuses_unexpected_existing_table_shape(alembic_sqlite):
    config, engine = alembic_sqlite
    command.upgrade(config, "d5e8a71b2c40")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE experiment_metadata (key VARCHAR(128) PRIMARY KEY, payload TEXT)"))

    with pytest.raises(RuntimeError, match="unexpected columns"):
        command.upgrade(config, "head")
