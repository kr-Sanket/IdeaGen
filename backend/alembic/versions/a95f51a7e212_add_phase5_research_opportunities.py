"""add phase 5 research opportunities tables

Revision ID: a95f51a7e212
Revises: 651ce2a4eb86
Create Date: 2026-10-06 15:39:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a95f51a7e212'
down_revision: Union[str, Sequence[str], None] = '651ce2a4eb86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'research_opportunities',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gap_id', sa.Uuid(), nullable=False),
        sa.Column('research_problem', sa.Text(), nullable=False),
        sa.Column('identified_gap', sa.Text(), nullable=False),
        sa.Column('proposed_research_direction', sa.Text(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('technical_approach', sa.Text(), nullable=False),
        sa.Column('expected_contribution', sa.Text(), nullable=False),
        sa.Column('feasibility', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('novelty_assessment', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('supporting_statement_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('supporting_paper_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('prior_art_statement_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=64), nullable=False),
        sa.Column('generation_model', sa.String(length=255), nullable=False),
        sa.Column('validation_model', sa.String(length=255), nullable=False),
        sa.Column('embedding_model', sa.String(length=255), nullable=False),
        sa.Column('refinement_iteration_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('pipeline_version', sa.String(length=32), nullable=False, server_default='v1.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['gap_id'], ['research_gaps.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_research_opportunities_gap_id'), 'research_opportunities', ['gap_id'], unique=False)
    op.create_index(op.f('ix_research_opportunities_status'), 'research_opportunities', ['status'], unique=False)

    op.create_table(
        'prior_art_assessments',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('opportunity_id', sa.Uuid(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('assessment', sa.String(length=64), nullable=False),
        sa.Column('overlapping_statement_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('overlapping_paper_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('differentiation', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['opportunity_id'], ['research_opportunities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prior_art_assessments_opportunity_id'), 'prior_art_assessments', ['opportunity_id'], unique=False)

    op.create_table(
        'opportunity_refinement_iterations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('opportunity_id', sa.Uuid(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('original_direction', sa.Text(), nullable=False),
        sa.Column('critique', sa.Text(), nullable=False),
        sa.Column('revised_direction', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['opportunity_id'], ['research_opportunities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_opportunity_refinement_iterations_opportunity_id'), 'opportunity_refinement_iterations', ['opportunity_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_opportunity_refinement_iterations_opportunity_id'), table_name='opportunity_refinement_iterations')
    op.drop_table('opportunity_refinement_iterations')
    op.drop_index(op.f('ix_prior_art_assessments_opportunity_id'), table_name='prior_art_assessments')
    op.drop_table('prior_art_assessments')
    op.drop_index(op.f('ix_research_opportunities_status'), table_name='research_opportunities')
    op.drop_index(op.f('ix_research_opportunities_gap_id'), table_name='research_opportunities')
    op.drop_table('research_opportunities')
