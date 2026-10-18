"""create paper_documents table

Revision ID: 2c0c9697b8f1
Revises: e4429d2a8867
Create Date: 2026-10-04 18:20:16.865213

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2c0c9697b8f1'
down_revision: Union[str, Sequence[str], None] = 'e4429d2a8867'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'paper_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('paper_id', sa.UUID(), nullable=False),
        sa.Column('full_text', sa.Text(), nullable=False),
        sa.Column('extraction_method', sa.String(), nullable=False),
        sa.Column('extraction_version', sa.String(), nullable=False),
        sa.Column('extracted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_paper_documents_paper_id'), 'paper_documents', ['paper_id'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_paper_documents_paper_id'), table_name='paper_documents')
    op.drop_table('paper_documents')
