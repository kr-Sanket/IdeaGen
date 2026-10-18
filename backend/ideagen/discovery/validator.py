"""AI Gap Validator with strict evidence integrity checking (Phase 4).

Validates candidate research gaps using the LLM infrastructure and enforces
rigorous evidence integrity so every claim is traceable to real stored statements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set
import uuid

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from .models import (
    GapCandidate,
    GapValidationResult,
    ValidationVerdict,
    GapStatus,
    ResearchGapCreateDTO,
    StatementEvidenceDTO,
    EvidenceRole,
)

logger = logging.getLogger(__name__)


VALIDATION_PROMPT = """You are a rigorous scientific reviewer evaluating whether a proposed research gap is substantiated by empirical evidence.

CANDIDATE RESEARCH GAP:
Statement: {gap_statement}
Research Problem: {research_problem}
Why Insufficient: {why_insufficient}

PROVIDED EVIDENCE STATEMENTS:
{evidence_block}

RULES:
1. Determine if the evidence genuinely substantiates the candidate gap (VALID) or if it is ungrounded / unsupported / trivial (REJECT).
2. Ground all reasoning in the provided statements.
3. Every cited statement ID in supporting_evidence_statement_ids or counter_evidence_statement_ids MUST match an ID listed above.
4. Output strictly conforming to the JSON schema.
"""


class GapValidator:
    """Validates candidate research gaps using an LLM and verifies evidence integrity."""

    def __init__(self, provider: LLMProvider, min_valid_evidence: int = 1):
        self.provider = provider
        self.min_valid_evidence = min_valid_evidence

    def validate_candidate(
        self,
        candidate: GapCandidate,
        statements_by_id: Dict[uuid.UUID, Any],
    ) -> ResearchGapCreateDTO:
        """Validates a candidate gap, enforces evidence integrity, and produces a persistable DTO."""
        # 1. Format evidence statements
        evidence_lines = []
        valid_evidence_pool: Set[uuid.UUID] = set()

        for s_id in candidate.supporting_statement_ids:
            if s_id in statements_by_id:
                s = statements_by_id[s_id]
                valid_evidence_pool.add(s_id)
                p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
                st_type = s.statement_type if hasattr(s, "statement_type") else s["statement_type"]
                text = s.statement_text if hasattr(s, "statement_text") else s["statement_text"]
                sec = s.section if hasattr(s, "section") else s.get("section")
                evidence_lines.append(
                    f"- [ID: {s_id} | Paper: {p_id} | Type: {st_type} | Section: {sec or 'N/A'}] {text}"
                )

        if not evidence_lines:
            # Candidate has zero valid statements
            return ResearchGapCreateDTO(
                gap_statement=candidate.gap_statement,
                research_problem=candidate.research_problem,
                status=GapStatus.REJECTED,
                supporting_statement_ids=[],
                supporting_paper_ids=[],
                evidence_count=0,
                limitations=candidate.relevant_limitations,
                future_work=candidate.relevant_future_work,
                validation_reasoning="Candidate rejected: no supporting statements found in database.",
                validation_confidence=0.0,
                evidence_items=[],
                metadata={"rejection_reason": "EMPTY_EVIDENCE"},
            )

        prompt = VALIDATION_PROMPT.format(
            gap_statement=candidate.gap_statement,
            research_problem=candidate.research_problem,
            why_insufficient=candidate.why_existing_work_insufficient,
            evidence_block="\n".join(evidence_lines),
        )

        # 2. Invoke LLM for validation
        try:
            val_result: GapValidationResult = self.provider.generate_structured(
                prompt=prompt,
                response_schema=GapValidationResult,
            )
        except (ProviderFailureError, MalformedResponseError, Exception) as e:
            logger.error(f"Gap validation failed during model execution: {e}")
            return ResearchGapCreateDTO(
                gap_statement=candidate.gap_statement,
                research_problem=candidate.research_problem,
                status=GapStatus.REJECTED,
                supporting_statement_ids=candidate.supporting_statement_ids,
                supporting_paper_ids=candidate.supporting_paper_ids,
                evidence_count=len(candidate.supporting_statement_ids),
                limitations=candidate.relevant_limitations,
                future_work=candidate.relevant_future_work,
                validation_reasoning=f"Validation failed due to provider error: {str(e)}",
                validation_confidence=0.0,
                evidence_items=[],
                metadata={"rejection_reason": "MODEL_INFERENCE_ERROR"},
            )

        # 3. Evidence Integrity Verification
        verified_stmt_ids: List[uuid.UUID] = []
        for raw_id in val_result.supporting_evidence_statement_ids:
            try:
                parsed_id = uuid.UUID(str(raw_id).strip())
                if parsed_id in valid_evidence_pool:
                    if parsed_id not in verified_stmt_ids:
                        verified_stmt_ids.append(parsed_id)
            except ValueError:
                continue

        # If LLM didn't cite specific IDs or cited invalid ones, retain all candidate pool if verdict was VALID
        if not verified_stmt_ids and val_result.verdict == ValidationVerdict.VALID:
            verified_stmt_ids = list(valid_evidence_pool)

        # Check evidence count integrity
        is_supported = (
            val_result.verdict == ValidationVerdict.VALID
            and len(verified_stmt_ids) >= self.min_valid_evidence
        )

        status = GapStatus.VALIDATED if is_supported else GapStatus.REJECTED
        reasoning = val_result.reasoning
        if val_result.verdict == ValidationVerdict.VALID and not is_supported:
            reasoning = "Model voted VALID but cited statements could not be verified in candidate evidence pool."

        # Build verified paper list and evidence items
        verified_paper_ids: Set[uuid.UUID] = set()
        evidence_items: List[StatementEvidenceDTO] = []

        for s_id in verified_stmt_ids:
            s = statements_by_id[s_id]
            p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
            if isinstance(p_id, str):
                p_id = uuid.UUID(p_id)
            verified_paper_ids.add(p_id)

            evidence_items.append(
                StatementEvidenceDTO(
                    statement_id=s_id,
                    paper_id=p_id,
                    statement_type=str(s.statement_type if hasattr(s, "statement_type") else s["statement_type"]),
                    statement_text=str(s.statement_text if hasattr(s, "statement_text") else s["statement_text"]),
                    section=s.section if hasattr(s, "section") else s.get("section"),
                    page_start=s.page_start if hasattr(s, "page_start") else s.get("page_start"),
                    page_end=s.page_end if hasattr(s, "page_end") else s.get("page_end"),
                    evidence_role=EvidenceRole.SUPPORTING,
                )
            )

        return ResearchGapCreateDTO(
            gap_statement=candidate.gap_statement,
            research_problem=candidate.research_problem,
            status=status,
            supporting_statement_ids=verified_stmt_ids,
            supporting_paper_ids=list(verified_paper_ids),
            evidence_count=len(verified_stmt_ids),
            limitations=candidate.relevant_limitations,
            future_work=candidate.relevant_future_work,
            validation_reasoning=reasoning,
            validation_confidence=val_result.confidence,
            evidence_items=evidence_items,
            metadata={
                "signals": candidate.signals,
                "model_verdict": val_result.verdict.value,
            },
        )
