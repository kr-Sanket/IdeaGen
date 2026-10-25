"""add experiment_metadata key-value table

The table backs ExperimentMetadataRecord (db/models.py). It was previously created only
by Base.metadata.create_all in the experiment runner, but the API now reads it
(run metadata, configured temporal cutoff), so Alembic-managed databases need it too.

Revision ID: e7b3c9a14f52
Revises: d5e8a71b2c40
Create Date: 2026-10-21 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7b3c9a14f52'
down_revision: Union[str, Sequence[str], None] = 'd5e8a71b2c40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table('experiment_metadata'):
        # Created earlier by create_all: keep the existing data, but refuse to proceed
        # if its shape differs from the model instead of silently diverging.
        columns = {c['name'] for c in inspector.get_columns('experiment_metadata')}
        if columns != {'key', 'value'}:
            raise RuntimeError(
                "Cannot apply migration e7b3c9a14f52: existing experiment_metadata table has "
                f"unexpected columns {sorted(columns)}; expected ['key', 'value']."
            )
        return

    op.create_table(
        'experiment_metadata',
        sa.Column('key', sa.String(length=128), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('key'),
    )


def downgrade() -> None:
    op.drop_table('experiment_metadata')
