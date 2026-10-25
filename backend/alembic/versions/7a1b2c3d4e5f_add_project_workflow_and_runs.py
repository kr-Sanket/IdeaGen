"""add projects, paper selections, research runs and project scoping

Revision ID: 7a1b2c3d4e5f
Revises: e7b3c9a14f52
Create Date: 2026-10-22 14:30:00.000000

"""
from typing import Sequence, Union
from datetime import date
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '7a1b2c3d4e5f'
down_revision: Union[str, Sequence[str], None] = 'e7b3c9a14f52'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    json_type = JSONB().with_variant(sa.JSON(), 'sqlite')

    # 1. Projects table
    if not inspector.has_table('projects'):
        op.create_table(
            'projects',
            sa.Column('id', sa.String(length=64), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('topic', sa.Text(), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('temporal_cutoff', sa.Date(), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='ACTIVE'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )

    # Pre-populate default benchmark project if missing
    project_table = sa.table(
        'projects',
        sa.column('id', sa.String),
        sa.column('name', sa.String),
        sa.column('topic', sa.Text),
        sa.column('description', sa.Text),
        sa.column('temporal_cutoff', sa.Date),
        sa.column('status', sa.String),
    )
    existing_proj = bind.execute(sa.text("SELECT id FROM projects WHERE id = 'rag-benchmark'")).fetchone()
    if not existing_proj:
        bind.execute(
            project_table.insert().values(
                id='rag-benchmark',
                name='Retrieval-Augmented Generation (RAG)',
                topic='Evidence-grounded discovery, gap analysis, and temporal evolution of RAG architectures',
                description='Core benchmark project analyzing pre-2021 foundation papers against 2023–2026 evaluation literature.',
                temporal_cutoff=date(2020, 12, 31),
                status='ACTIVE',
            )
        )

    # 2. Project Paper Selections table
    if not inspector.has_table('project_paper_selections'):
        op.create_table(
            'project_paper_selections',
            sa.Column('id', sa.String(length=64), nullable=False),
            sa.Column('project_id', sa.String(length=64), nullable=False),
            sa.Column('openalex_id', sa.String(length=64), nullable=False),
            sa.Column('doi', sa.String(length=255), nullable=True),
            sa.Column('title', sa.Text(), nullable=False),
            sa.Column('authors', json_type, nullable=False),
            sa.Column('publication_date', sa.String(length=32), nullable=True),
            sa.Column('pdf_url', sa.Text(), nullable=True),
            sa.Column('landing_page_url', sa.Text(), nullable=True),
            sa.Column('citation_count', sa.Integer(), nullable=True),
            sa.Column('abstract', sa.Text(), nullable=True),
            sa.Column('selected_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_project_paper_selections_project_id', 'project_paper_selections', ['project_id'])
        op.create_index('ix_project_paper_selections_openalex_id', 'project_paper_selections', ['openalex_id'])

    # 3. Research Runs table
    if not inspector.has_table('research_runs'):
        op.create_table(
            'research_runs',
            sa.Column('id', sa.String(length=64), nullable=False),
            sa.Column('project_id', sa.String(length=64), nullable=False),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='PENDING'),
            sa.Column('current_stage', sa.String(length=64), nullable=False, server_default='INGESTION'),
            sa.Column('planned_stages', json_type, nullable=False),
            sa.Column('stage_states', json_type, nullable=False),
            sa.Column('summary', json_type, nullable=True),
            sa.Column('logs', json_type, nullable=False),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_research_runs_project_id', 'research_runs', ['project_id'])
        op.create_index('ix_research_runs_status', 'research_runs', ['status'])

    # 4. Research Run Papers table
    if not inspector.has_table('research_run_papers'):
        op.create_table(
            'research_run_papers',
            sa.Column('id', sa.String(length=64), nullable=False),
            sa.Column('run_id', sa.String(length=64), nullable=False),
            sa.Column('openalex_id', sa.String(length=64), nullable=False),
            sa.Column('doi', sa.String(length=255), nullable=True),
            sa.Column('title', sa.Text(), nullable=False),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='PENDING'),
            sa.Column('paper_id', sa.String(length=64), nullable=True),
            sa.Column('passage_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('statement_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(['run_id'], ['research_runs.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_research_run_papers_run_id', 'research_run_papers', ['run_id'])

    # 5. Scoping columns on research_gaps and research_opportunities
    gap_cols = {c['name'] for c in inspector.get_columns('research_gaps')}
    if 'project_id' not in gap_cols:
        op.add_column('research_gaps', sa.Column('project_id', sa.String(length=64), nullable=True))
        op.create_index('ix_research_gaps_project_id', 'research_gaps', ['project_id'])
        bind.execute(sa.text("UPDATE research_gaps SET project_id = 'rag-benchmark' WHERE project_id IS NULL"))

    opp_cols = {c['name'] for c in inspector.get_columns('research_opportunities')}
    if 'project_id' not in opp_cols:
        op.add_column('research_opportunities', sa.Column('project_id', sa.String(length=64), nullable=True))
        op.create_index('ix_research_opportunities_project_id', 'research_opportunities', ['project_id'])
        bind.execute(sa.text("UPDATE research_opportunities SET project_id = 'rag-benchmark' WHERE project_id IS NULL"))


def downgrade() -> None:
    op.drop_table('research_run_papers')
    op.drop_table('research_runs')
    op.drop_table('project_paper_selections')
    op.drop_table('projects')
