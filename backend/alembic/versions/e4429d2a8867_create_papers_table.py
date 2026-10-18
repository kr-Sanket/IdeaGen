"""create papers table

Revision ID: e4429d2a8867
Revises: 
Create Date: 2026-10-04 17:33:12.874149

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4429d2a8867'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum type first
    from sqlalchemy.dialects.postgresql import ENUM
    source_type_enum = ENUM('OPENALEX_API', 'OPENALEX_SNAPSHOT', 'MANUAL_UPLOAD', name='source_type_enum', create_type=False)
    source_type_enum.create(op.get_bind(), checkfirst=True)
    
    op.create_table(
        'papers',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('source_type', source_type_enum, nullable=False),
        sa.Column('source_id', sa.String(), nullable=False),
        sa.Column('doi', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('publication_date', sa.Date(), nullable=False),
        sa.Column('batch_id', sa.String(), nullable=False),
        sa.Column('abstract', sa.Text(), nullable=False),
        sa.Column('authors', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('referenced_works', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('raw_metadata', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doi')
    )
    op.create_index(op.f('ix_papers_batch_id'), 'papers', ['batch_id'], unique=False)
    op.create_index(op.f('ix_papers_source_id'), 'papers', ['source_id'], unique=False)
    op.create_index('idx_paper_openalex_identity', 'papers', ['source_id'], unique=True, postgresql_where=sa.text("source_type IN ('OPENALEX_API', 'OPENALEX_SNAPSHOT')"))
    op.create_index('idx_paper_manual_identity', 'papers', ['source_id'], unique=True, postgresql_where=sa.text("source_type = 'MANUAL_UPLOAD'"))

def downgrade() -> None:
    op.drop_index('idx_paper_manual_identity', table_name='papers')
    op.drop_index('idx_paper_openalex_identity', table_name='papers')
    op.drop_index(op.f('ix_papers_source_id'), table_name='papers')
    op.drop_index(op.f('ix_papers_batch_id'), table_name='papers')
    op.drop_table('papers')
    
    from sqlalchemy.dialects.postgresql import ENUM
    source_type_enum = ENUM('OPENALEX_API', 'OPENALEX_SNAPSHOT', 'MANUAL_UPLOAD', name='source_type_enum', create_type=False)
    source_type_enum.drop(op.get_bind(), checkfirst=True)
