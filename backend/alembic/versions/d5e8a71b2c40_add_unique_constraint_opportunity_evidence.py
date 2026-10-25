"""add unique constraint to opportunity evidence table

Revision ID: d5e8a71b2c40
Revises: b4e67f1a9c23
Create Date: 2026-10-20 21:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e8a71b2c40'
down_revision: Union[str, Sequence[str], None] = 'b4e67f1a9c23'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Explicitly check for pre-existing duplicate (opportunity_id, paper_id) pairs
    # Do not silently delete or drop data
    bind = op.get_bind()
    dup_query = sa.text(
        "SELECT opportunity_id, paper_id, COUNT(*) as c "
        "FROM opportunity_evidence "
        "GROUP BY opportunity_id, paper_id "
        "HAVING COUNT(*) > 1"
    )
    dup_rows = bind.execute(dup_query).fetchall()
    if dup_rows:
        duplicates_repr = [
            f"(opportunity_id={row[0]}, paper_id={row[1]}, count={row[2]})"
            for row in dup_rows
        ]
        raise RuntimeError(
            f"Cannot apply migration d5e8a71b2c40: found {len(dup_rows)} duplicate "
            f"(opportunity_id, paper_id) pairs in opportunity_evidence table. "
            f"Duplicate records must be resolved explicitly before applying the unique constraint: "
            f"{', '.join(duplicates_repr)}"
        )

    # 2. Add the unique constraint using batch_alter_table for SQLite and PostgreSQL compatibility
    with op.batch_alter_table('opportunity_evidence') as batch_op:
        batch_op.create_unique_constraint(
            'uq_opportunity_evidence_opp_paper',
            ['opportunity_id', 'paper_id']
        )


def downgrade() -> None:
    with op.batch_alter_table('opportunity_evidence') as batch_op:
        batch_op.drop_constraint('uq_opportunity_evidence_opp_paper', type_='unique')
