"""make publication_date nullable

Revision ID: f414e41fe9d7
Revises: 2c0c9697b8f1
Create Date: 2026-10-04 18:30:03.169464

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f414e41fe9d7'
down_revision: Union[str, Sequence[str], None] = '2c0c9697b8f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('papers', 'publication_date',
               existing_type=sa.Date(),
               nullable=True)

def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('papers', 'publication_date',
               existing_type=sa.Date(),
               nullable=False)
