"""add opportunity bridge fields to opportunities table

Revision ID: b4e67f1a9c23
Revises: a95f51a7e212
Create Date: 2026-10-19 12:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4e67f1a9c23'
down_revision: Union[str, Sequence[str], None] = 'a95f51a7e212'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('opportunities') as batch_op:
        batch_op.add_column(sa.Column('research_opportunity_id', sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column('gap_id', sa.Uuid(), nullable=True))
        batch_op.create_index('ix_opportunities_research_opportunity_id', ['research_opportunity_id'], unique=True)
        batch_op.create_index('ix_opportunities_gap_id', ['gap_id'], unique=False)
        batch_op.create_foreign_key(
            'fk_opportunities_research_opportunity_id_research_opportunities',
            'research_opportunities',
            ['research_opportunity_id'], ['id'],
            ondelete='SET NULL'
        )
        batch_op.create_foreign_key(
            'fk_opportunities_gap_id_research_gaps',
            'research_gaps',
            ['gap_id'], ['id'],
            ondelete='SET NULL'
        )


def downgrade() -> None:
    with op.batch_alter_table('opportunities') as batch_op:
        batch_op.drop_constraint('fk_opportunities_gap_id_research_gaps', type_='foreignkey')
        batch_op.drop_constraint('fk_opportunities_research_opportunity_id_research_opportunities', type_='foreignkey')
        batch_op.drop_index('ix_opportunities_gap_id')
        batch_op.drop_index('ix_opportunities_research_opportunity_id')
        batch_op.drop_column('gap_id')
        batch_op.drop_column('research_opportunity_id')
