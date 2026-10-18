"""Thin orchestration layer connecting LLM classification to deterministic state transitions.

This module does NOT contain transition logic, database persistence, or LLM calls.
It delegates classification to ``RelationshipClassifier`` and state application to
``OpportunityTransitionService``.  The deterministic state machine in
``state_machine.py`` remains the sole authority for status transitions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassificationInput,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService


@dataclass(frozen=True)
class ClassificationTransitionResult:
    """Typed result of a classify-and-apply operation."""

    opportunity_id: uuid.UUID
    paper_id: uuid.UUID
    relationship: OpportunityRelationship
    previous_status: OpportunityStatus
    new_status: OpportunityStatus
    state_changed: bool
    evidence_id: Optional[uuid.UUID]
    history_id: Optional[uuid.UUID]
    classifier_confidence: float
    classifier_rationale: str
    prompt_version: str


class OpportunityClassificationIntegration:
    """Orchestrates classification followed by deterministic transition.

    Flow:
        1. Load bounded opportunity context (title, problem_statement, description).
        2. Load bounded paper context (title, abstract).
        3. Construct ``RelationshipClassificationInput``.
        4. Call ``RelationshipClassifier.classify()``.
        5. Receive validated ``RelationshipClassification``.
        6. Extract ``classification.relationship``.
        7. Pass relationship to ``OpportunityTransitionService.apply_relationship()``.
        8. Return typed ``ClassificationTransitionResult``.

    This layer does NOT:
        - calculate the next status
        - modify Opportunity.status
        - interpret relationship semantics
        - duplicate transition rules
        - write Evidence directly
        - write History directly
        - call an LLM provider directly
    """

    def __init__(
        self,
        classifier: RelationshipClassifier,
        transition_service: OpportunityTransitionService,
        session: Session,
    ) -> None:
        self._classifier = classifier
        self._transition_service = transition_service
        self._session = session

    def classify_and_apply(
        self,
        opportunity_id: uuid.UUID,
        paper_id: uuid.UUID,
        evidence_excerpt: Optional[str] = None,
    ) -> ClassificationTransitionResult:
        """Classify the paper-opportunity relationship and apply the transition.

        If classification fails for any reason (provider failure, malformed
        response, schema validation, missing input), the error propagates
        immediately.  No transition, evidence, or history is created.

        Args:
            opportunity_id: Target opportunity UUID.
            paper_id: Source paper UUID.
            evidence_excerpt: Optional bounded excerpt from the paper.

        Returns:
            A ``ClassificationTransitionResult`` describing the outcome.

        Raises:
            ValueError: If opportunity or paper not found.
            ProviderFailureError: If the LLM provider fails.
            MalformedResponseError: If the LLM returns invalid data.
            MissingInputError: If required classifier inputs are missing.
        """
        # ── 1. Load bounded opportunity context ──────────────────────────
        opportunity = self._session.execute(
            select(Opportunity).where(Opportunity.id == opportunity_id)
        ).scalar_one_or_none()

        if opportunity is None:
            raise ValueError(f"Opportunity {opportunity_id} not found.")

        # ── 2. Load bounded paper context ────────────────────────────────
        paper = self._session.execute(
            select(Paper).where(Paper.id == paper_id)
        ).scalar_one_or_none()

        if paper is None:
            raise ValueError(f"Paper {paper_id} not found.")

        previous_status = OpportunityStatus(opportunity.status)

        # ── 3. Construct bounded classifier input ────────────────────────
        classifier_input = RelationshipClassificationInput(
            opportunity_title=opportunity.title,
            opportunity_problem_statement=opportunity.problem_statement,
            opportunity_description=opportunity.description,
            paper_title=paper.title,
            paper_abstract=paper.abstract,
            paper_excerpt=evidence_excerpt,
        )

        # ── 4–5. Classify (errors propagate, no transition on failure) ───
        classification = self._classifier.classify(classifier_input)

        # ── 6–7. Delegate transition to existing service ─────────────────
        updated_opportunity = self._transition_service.apply_relationship(
            opportunity_id=opportunity_id,
            paper_id=paper_id,
            relationship=classification.relationship,
            evidence_excerpt=classification.evidence_excerpt,
        )

        new_status = OpportunityStatus(updated_opportunity.status)
        state_changed = new_status != previous_status

        # ── Resolve evidence and history identifiers ─────────────────────
        evidence_id: Optional[uuid.UUID] = None
        history_id: Optional[uuid.UUID] = None

        # Evidence is always created by the transition service
        from ideagen.db.models import OpportunityEvidence, OpportunityHistory

        evidence_row = self._session.execute(
            select(OpportunityEvidence).where(
                OpportunityEvidence.opportunity_id == opportunity_id,
                OpportunityEvidence.paper_id == paper_id,
            )
        ).scalar_one_or_none()
        if evidence_row is not None:
            evidence_id = evidence_row.id

        # History is only created when status changes
        if state_changed:
            history_row = self._session.execute(
                select(OpportunityHistory)
                .where(OpportunityHistory.opportunity_id == opportunity_id)
                .order_by(OpportunityHistory.created_at.desc())
            ).scalars().first()
            if history_row is not None:
                history_id = history_row.id

        # ── 8. Return typed result ───────────────────────────────────────
        return ClassificationTransitionResult(
            opportunity_id=opportunity_id,
            paper_id=paper_id,
            relationship=classification.relationship,
            previous_status=previous_status,
            new_status=new_status,
            state_changed=state_changed,
            evidence_id=evidence_id,
            history_id=history_id,
            classifier_confidence=classification.confidence,
            classifier_rationale=classification.rationale,
            prompt_version=classification.prompt_version,
        )
