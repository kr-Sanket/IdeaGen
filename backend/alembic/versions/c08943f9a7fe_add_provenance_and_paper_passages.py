"""add_provenance_and_paper_passages

Revision ID: c08943f9a7fe
Revises: 86e8cbcb3375
Create Date: 2026-10-06 12:53:07.692170

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c08943f9a7fe'
down_revision: Union[str, Sequence[str], None] = '86e8cbcb3375'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add provenance columns and paper_passages table."""
    # 1. Allow nullable paper title when not detectable
    op.alter_column('papers', 'title', existing_type=sa.String(), nullable=True)

    # 2. Add provenance columns to paper_documents
    op.add_column('paper_documents', sa.Column('content_hash', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_paper_documents_content_hash'), 'paper_documents', ['content_hash'], unique=False)
    op.add_column('paper_documents', sa.Column('original_filename', sa.String(), nullable=True))
    op.add_column('paper_documents', sa.Column('acquisition_timestamp', sa.DateTime(timezone=True), nullable=True))
    op.add_column('paper_documents', sa.Column('provenance_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # 3. Create paper_passages table
    op.create_table(
        'paper_passages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('paper_id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('passage_index', sa.Integer(), nullable=False),
        sa.Column('page_start', sa.Integer(), nullable=False),
        sa.Column('page_end', sa.Integer(), nullable=False),
        sa.Column('section', sa.String(), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['paper_documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_paper_passages_paper_id'), 'paper_passages', ['paper_id'], unique=False)
    op.create_index(op.f('ix_paper_passages_document_id'), 'paper_passages', ['document_id'], unique=False)
    op.create_index('idx_passage_doc_order', 'paper_passages', ['document_id', 'passage_index'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_passage_doc_order', table_name='paper_passages')
    op.drop_index(op.f('ix_paper_passages_document_id'), table_name='paper_passages')
    op.drop_index(op.f('ix_paper_passages_paper_id'), table_name='paper_passages')
    op.drop_table('paper_passages')

    op.drop_index(op.f('ix_paper_documents_content_hash'), table_name='paper_documents')
    op.drop_column('paper_documents', 'provenance_metadata')
    op.drop_column('paper_documents', 'acquisition_timestamp')
    op.drop_column('paper_documents', 'original_filename')
    op.drop_column('paper_documents', 'content_hash')

    op.alter_column('papers', 'title', existing_type=sa.String(), nullable=False)
