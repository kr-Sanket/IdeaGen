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

PROVIDED EVIDENCE STATEMENTS & ORIGINAL SOURCE PASSAGES:
{evidence_block}

RULES:
1. Determine if the evidence genuinely substantiates the candidate gap (VALID) or if it is ungrounded / unsupported / trivial (REJECT).
2. Ground all reasoning in the provided statements and their original source passages.
3. Every cited statement ID in supporting_evidence_statement_ids or counter_evidence_statement_ids MUST match an ID listed above.
4. If a statement's source passage is unavailable or contradicts the claimed limitation, evaluate it critically.
5. Output strictly conforming to the JSON schema.
"""

MAX_PASSAGE_EXCERPT_CHARS = 800


def _resolve_passage_text(
    statement: Any,
    passages_by_id: Optional[Dict[uuid.UUID, Any]] = None,
    max_chars: int = MAX_PASSAGE_EXCERPT_CHARS,
) -> str:
    """Resolves and bounds the original passage text for a research statement.

    Returns an explicit unavailable indicator if passage is missing, unlinked, or invalid.
    Never fabricates passage text.
    """
    raw_pid = getattr(statement, "passage_id", None) if hasattr(statement, "passage_id") else (
        statement.get("passage_id") if isinstance(statement, dict) else None
    )

    if raw_pid is None:
        return "[Source passage unavailable - no linked passage]"

    try:
        if isinstance(raw_pid, str):
            passage_uuid = uuid.UUID(raw_pid.strip())
        elif isinstance(raw_pid, uuid.UUID):
            passage_uuid = raw_pid
        else:
            return "[Source passage unavailable - invalid passage ID]"
    except (ValueError, AttributeError):
        return "[Source passage unavailable - invalid passage ID]"

    # Resolve passage object from mapping or directly from statement relationship
    passage_obj = None
    if passages_by_id and passage_uuid in passages_by_id:
        passage_obj = passages_by_id[passage_uuid]
    elif getattr(statement, "passage", None) is not None:
        passage_obj = getattr(statement, "passage")
    elif isinstance(statement, dict) and statement.get("passage") is not None:
        passage_obj = statement.get("passage")

    raw_text = None
    if passage_obj is not None:
        if isinstance(passage_obj, str):
            raw_text = passage_obj
        elif hasattr(passage_obj, "text"):
            raw_text = passage_obj.text
        elif isinstance(passage_obj, dict):
            raw_text = passage_obj.get("text")

    if not raw_text:
        # Fallback to direct attribute on statement if provided
        if hasattr(statement, "passage_text") and statement.passage_text:
            raw_text = statement.passage_text
        elif isinstance(statement, dict) and statement.get("passage_text"):
            raw_text = statement.get("passage_text")

    if not raw_text or not str(raw_text).strip():
        return "[Source passage unavailable - passage not found in database]"

    clean_text = " ".join(str(raw_text).strip().split())
    if len(clean_text) > max_chars:
        return clean_text[:max_chars].rstrip() + "..."
    return clean_text


class GapValidator:
    """Validates candidate research gaps using an LLM and verifies evidence integrity."""

    def __init__(self, provider: LLMProvider, min_valid_evidence: int = 1):
        self.provider = provider
        self.min_valid_evidence = min_valid_evidence

    def validate_candidate(
        self,
        candidate: GapCandidate,
        statements_by_id: Dict[uuid.UUID, Any],
        passages_by_id: Optional[Dict[uuid.UUID, Any]] = None,
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
                passage_text = _resolve_passage_text(s, passages_by_id=passages_by_id)

                evidence_lines.append(
                    f"- [ID: {s_id} | Paper: {p_id} | Type: {st_type} | Section: {sec or 'N/A'}]\n"
                    f"  Statement: \"{text}\"\n"
                    f"  Original Source Passage: {passage_text}"
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
                else:
                    logger.warning(
                        f"GapValidator: Cited statement ID {parsed_id} not in candidate evidence pool."
                    )
            except ValueError:
                logger.warning(
                    f"GapValidator: Unparseable statement ID cited: {raw_id!r}"
                )
                continue

        rejection_reason_code = None
        if val_result.verdict == ValidationVerdict.VALID:
            if not val_result.supporting_evidence_statement_ids:
                rejection_reason_code = "EMPTY_EVIDENCE_CITATIONS"
                reasoning = (
                    "Candidate rejected: Model returned VALID verdict but provided no evidence statement citations."
                )
                logger.warning(
                    f"Candidate gap rejected: Model verdict was VALID for gap '{candidate.gap_statement[:80]}' "
                    f"but provided an empty list of evidence statement IDs."
                )
            elif not verified_stmt_ids:
                rejection_reason_code = "INVALID_OR_FABRICATED_CITATIONS"
                reasoning = (
                    "Candidate rejected: Model returned VALID verdict but none of the cited statement IDs "
                    "could be verified in the candidate evidence pool."
                )
                logger.warning(
                    f"Candidate gap rejected: Model verdict was VALID for gap '{candidate.gap_statement[:80]}' "
                    f"but cited statement IDs ({val_result.supporting_evidence_statement_ids}) "
                    f"could not be verified in candidate evidence pool ({list(valid_evidence_pool)})."
                )
            elif len(verified_stmt_ids) < self.min_valid_evidence:
                rejection_reason_code = "INSUFFICIENT_EVIDENCE_COUNT"
                reasoning = (
                    f"Candidate rejected: Verified evidence count ({len(verified_stmt_ids)}) "
                    f"is below minimum required ({self.min_valid_evidence})."
                )
                logger.warning(
                    f"Candidate gap rejected: Model verdict was VALID for gap '{candidate.gap_statement[:80]}' "
                    f"but verified evidence count ({len(verified_stmt_ids)}) is below minimum ({self.min_valid_evidence})."
                )
            else:
                reasoning = val_result.reasoning
        else:
            # Model explicitly rejected the candidate
            reasoning = val_result.reasoning

        is_supported = (
            val_result.verdict == ValidationVerdict.VALID
            and len(verified_stmt_ids) >= self.min_valid_evidence
        )

        status = GapStatus.VALIDATED if is_supported else GapStatus.REJECTED

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

        metadata = {
            "signals": candidate.signals,
            "model_verdict": val_result.verdict.value,
        }
        if rejection_reason_code:
            metadata["rejection_reason"] = rejection_reason_code

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
            metadata=metadata,
        )
