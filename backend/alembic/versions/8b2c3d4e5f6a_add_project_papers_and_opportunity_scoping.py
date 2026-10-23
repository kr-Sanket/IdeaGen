"""add project_papers join table and project_id to opportunities

Revision ID: 8b2c3d4e5f6a
Revises: 7a1b2c3d4e5f
Create Date: 2026-10-22 17:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = '8b2c3d4e5f6a'
down_revision: Union[str, Sequence[str], None] = '7a1b2c3d4e5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    uuid_type = UUID(as_uuid=True).with_variant(sa.CHAR(36), 'sqlite')

    # 1. Create project_papers association table
    if not inspector.has_table('project_papers'):
        op.create_table(
            'project_papers',
            sa.Column('project_id', sa.String(length=64), nullable=False),
            sa.Column('paper_id', uuid_type, nullable=False),
            sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('project_id', 'paper_id'),
        )
        op.create_index('ix_project_papers_project_id', 'project_papers', ['project_id'])
        op.create_index('ix_project_papers_paper_id', 'project_papers', ['paper_id'])

        # Backfill all existing papers to the default 'rag-benchmark' project
        bind.execute(
            sa.text(
                "INSERT OR IGNORE INTO project_papers (project_id, paper_id, added_at) "
                "SELECT 'rag-benchmark', id, CURRENT_TIMESTAMP FROM papers"
            )
        )

    # 2. Add project_id column to opportunities (lifecycle table)
    opp_cols = {c['name'] for c in inspector.get_columns('opportunities')}
    if 'project_id' not in opp_cols:
        op.add_column('opportunities', sa.Column('project_id', sa.String(length=64), nullable=True))
        op.create_index('ix_opportunities_project_id', 'opportunities', ['project_id'])
        bind.execute(
            sa.text("UPDATE opportunities SET project_id = 'rag-benchmark' WHERE project_id IS NULL")
        )


def downgrade() -> None:
    op.drop_table('project_papers')
    # SQLite does not easily drop columns without full copy; ignore for downgrade or drop index
    op.drop_index('ix_opportunities_project_id', table_name='opportunities')
