"""AI Gap Validator with strict evidence integrity checking (Phase 4).

Validates candidate research gaps using the LLM infrastructure and enforces
rigorous evidence integrity so every claim is traceable to real stored statements.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set
import uuid

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from .models import (
    GapCandidate,
    GapValidationResult,
    GapCategory,
    ValidationVerdict,
    GapStatus,
    ResearchGapCreateDTO,
    StatementEvidenceDTO,
    EvidenceRole,
)

logger = logging.getLogger(__name__)

# Patterns describing why superseded, external, or traditional baseline paradigms were replaced
BACKGROUND_MOTIVATION_PATTERNS = [
    re.compile(r"\b(?:traditional|conventional|centralized|classical|standard)\s+(?:learning|methods?|approaches?|systems?|models?)\s+(?:requires?|relies?|suffers?|poses?|raises?)\b", re.IGNORECASE),
    re.compile(r"\b(?:privacy\s+risks\s+and\s+storage\s+demands\s+of\s+centralized|storing\s+all\s+data\s+centrally)\b", re.IGNORECASE),
    re.compile(r"\b(?:motivation\s+for|in\s+order\s+to\s+avoid|to\s+alleviate\s+the\s+need\s+for)\s+(?:centralized|traditional)\b", re.IGNORECASE),
]

# Patterns describing narrow single-study experimental setups or author disclaimers
STUDY_SPECIFIC_DISCLAIMER_PATTERNS = [
    re.compile(r"\b(?:simulated\s+environment|simulation\s+(?:setting|framework|data)|limited\s+to\s+a\s+simulated|real-world\s+hospital\s+data,\s+which\s+may\s+not\s+fully\s+represent|single\s+hospital\s+data|specific\s+hospital\s+data|cohort\s+size\s+is\s+limited)\b", re.IGNORECASE),
]

# Patterns describing known high-level domain challenges that require specific sub-problem formulation
BROAD_CHALLENGE_PATTERNS = [
    re.compile(r"\b(?:assumption\s+of\s+(?:independent\s+and\s+identically\s+distributed|iid)\s+data|fl\s+faces\s+challenges\s+such\s+as\s+communication\s+cost,\s+energy\s+consumption)\b", re.IGNORECASE),
]


VALIDATION_PROMPT = """You are a rigorous scientific reviewer evaluating whether a proposed research gap is substantiated by empirical evidence.

CANDIDATE RESEARCH GAP:
Statement: {gap_statement}
Research Problem: {research_problem}
Why Insufficient: {why_insufficient}

PROVIDED EVIDENCE STATEMENTS & ORIGINAL SOURCE PASSAGES:
{evidence_block}

RULES:
1. Classify the candidate into one of four scientific categories (gap_category) and decide the validation verdict:
   - BACKGROUND_MOTIVATION (verdict: REJECT): The candidate describes introductory rationale, standard historical motivation for established paradigms, or textbook facts (e.g. why centralized data storage creates privacy risks, motivating federated learning) rather than an open scientific question. Reject as background knowledge.
   - BROAD_CHALLENGE (verdict: VALID, needs_refinement: true): The candidate identifies a legitimate real-world challenge or complexity (e.g. non-IID data distribution in federated learning), but the claim is broad and requires formulating a specific sub-problem, setting, or research question. Retain as a candidate requiring refinement.
   - STUDY_LIMITATION (verdict: VALID, needs_refinement: false): The candidate describes an ordinary procedural constraint or author disclaimer of a single experimental setup (e.g. evaluation limited to a simulated environment or specific hospital cohort). Do not claim field-wide novelty without cross-paper evidence.
   - EVIDENCE_GROUNDED_GAP (verdict: VALID, needs_refinement: false): The candidate represents a concrete, plausible, unresolved scientific gap or bottleneck with direct passage-level evidentiary backing.
2. Ground all reasoning strictly in the provided statements and their original source passages. Do not extrapolate beyond the text.
3. Every cited statement ID in supporting_evidence_statement_ids or counter_evidence_statement_ids MUST match an ID listed in the evidence block above.
4. If a statement's source passage is unavailable or contradicts the claimed limitation, evaluate it critically.
5. Provide a non-empty confidence score between 0.0 and 1.0 representing your confidence in the verdict and classification.
6. Output strictly conforming to the JSON schema.
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


DEFAULT_MAX_EVIDENCE_PER_CANDIDATE = 3


class GapValidator:
    """Validates candidate research gaps using an LLM and verifies evidence integrity."""

    def __init__(
        self,
        provider: LLMProvider,
        min_valid_evidence: int = 1,
        max_evidence_per_candidate: int = DEFAULT_MAX_EVIDENCE_PER_CANDIDATE,
    ):
        self.provider = provider
        self.min_valid_evidence = min_valid_evidence
        self.max_evidence_per_candidate = max_evidence_per_candidate

    def validate_candidate(
        self,
        candidate: GapCandidate,
        statements_by_id: Dict[uuid.UUID, Any],
        passages_by_id: Optional[Dict[uuid.UUID, Any]] = None,
    ) -> ResearchGapCreateDTO:
        """Validates a candidate gap, enforces evidence integrity, and produces a persistable DTO."""
        # 1. Format evidence statements
        available_statement_ids = [
            s_id for s_id in candidate.supporting_statement_ids if s_id in statements_by_id
        ]
        if not available_statement_ids:
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

        # Bound evidence statements for validation prompt
        if self.max_evidence_per_candidate and len(available_statement_ids) > self.max_evidence_per_candidate:
            bounded_stmt_ids = available_statement_ids[:self.max_evidence_per_candidate]
        else:
            bounded_stmt_ids = available_statement_ids

        evidence_lines = []
        valid_evidence_pool: Set[uuid.UUID] = set(bounded_stmt_ids)
        seen_passage_provenance: Dict[str, uuid.UUID] = {}

        for s_id in bounded_stmt_ids:
            s = statements_by_id[s_id]
            p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
            st_type = s.statement_type if hasattr(s, "statement_type") else s["statement_type"]
            text = s.statement_text if hasattr(s, "statement_text") else s["statement_text"]
            sec = s.section if hasattr(s, "section") else s.get("section")
            raw_pid = getattr(s, "passage_id", None) if hasattr(s, "passage_id") else (
                s.get("passage_id") if isinstance(s, dict) else None
            )

            passage_key = str(raw_pid).strip() if raw_pid else f"stmt_{s_id}"
            if passage_key in seen_passage_provenance:
                first_s_id = seen_passage_provenance[passage_key]
                passage_text = f"[Same source passage as statement {first_s_id} above]"
            else:
                seen_passage_provenance[passage_key] = s_id
                passage_text = _resolve_passage_text(s, passages_by_id=passages_by_id)

            evidence_lines.append(
                f"- [ID: {s_id} | Paper: {p_id} | Type: {st_type} | Section: {sec or 'N/A'}]\n"
                f"  Statement: \"{text}\"\n"
                f"  Original Source Passage: {passage_text}"
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
        except ProviderFailureError:
            # Propagate infrastructure/provider failures (timeouts, network errors, 5xx)
            # rather than masking them as scientific domain-level rejections.
            raise
        except (MalformedResponseError, Exception) as e:
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

        # Build verified paper list
        verified_paper_ids: Set[uuid.UUID] = set()
        for s_id in verified_stmt_ids:
            s = statements_by_id[s_id]
            p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
            if isinstance(p_id, str):
                p_id = uuid.UUID(p_id)
            verified_paper_ids.add(p_id)

        return self._resolve_validation_results(
            candidate=candidate,
            val_result=val_result,
            verified_stmt_ids=verified_stmt_ids,
            verified_paper_ids=verified_paper_ids,
            statements_by_id=statements_by_id,
            bounded_stmt_ids=bounded_stmt_ids,
        )

    def _infer_gap_category(
        self,
        candidate: GapCandidate,
        val_result: GapValidationResult,
        verified_stmt_ids: List[uuid.UUID],
        statements_by_id: Dict[uuid.UUID, Any],
        verified_paper_ids: Set[uuid.UUID],
    ) -> GapCategory:
        """Contextually classifies candidate into scientific taxonomy if omitted by model."""
        reasoning_lower = (val_result.reasoning or "").lower()
        if "background motivation" in reasoning_lower or "textbook" in reasoning_lower or "established knowledge" in reasoning_lower:
            return GapCategory.BACKGROUND_MOTIVATION
        if "broad challenge" in reasoning_lower or "needs refinement" in reasoning_lower:
            return GapCategory.BROAD_CHALLENGE
        if "study-specific" in reasoning_lower or "single-study" in reasoning_lower or "single study" in reasoning_lower:
            return GapCategory.STUDY_LIMITATION

        # Pattern analyzer signal checks
        if candidate.signals.get("potential_background_motivation"):
            return GapCategory.BACKGROUND_MOTIVATION
        if candidate.signals.get("potential_study_specific_limitation") and len(verified_paper_ids) <= 1:
            return GapCategory.STUDY_LIMITATION

        stmt_text = candidate.research_problem or candidate.gap_statement
        combined_text = f"{candidate.gap_statement} {candidate.research_problem} {candidate.why_existing_work_insufficient}".lower()

        # Check for background paradigm motivation
        for pat in BACKGROUND_MOTIVATION_PATTERNS:
            if pat.search(stmt_text) or pat.search(candidate.why_existing_work_insufficient):
                return GapCategory.BACKGROUND_MOTIVATION

        # Check for study-specific disclaimer in single-paper candidates
        if len(verified_paper_ids) <= 1:
            for pat in STUDY_SPECIFIC_DISCLAIMER_PATTERNS:
                if pat.search(combined_text):
                    return GapCategory.STUDY_LIMITATION

        # Check for broad unrefined challenge
        for pat in BROAD_CHALLENGE_PATTERNS:
            if pat.search(stmt_text) and not candidate.relevant_future_work:
                return GapCategory.BROAD_CHALLENGE

        return GapCategory.EVIDENCE_GROUNDED_GAP

    def _resolve_validation_results(
        self,
        candidate: GapCandidate,
        val_result: GapValidationResult,
        verified_stmt_ids: List[uuid.UUID],
        verified_paper_ids: Set[uuid.UUID],
        statements_by_id: Dict[uuid.UUID, Any],
        bounded_stmt_ids: List[uuid.UUID],
    ) -> ResearchGapCreateDTO:
        rejection_reason_code = None
        assigned_category = val_result.gap_category
        needs_refinement = getattr(val_result, "needs_refinement", False)

        if assigned_category is None:
            assigned_category = self._infer_gap_category(
                candidate=candidate,
                val_result=val_result,
                verified_stmt_ids=verified_stmt_ids,
                statements_by_id=statements_by_id,
                verified_paper_ids=verified_paper_ids,
            )

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
                    f"could not be verified in candidate evidence pool."
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
            elif assigned_category == GapCategory.BACKGROUND_MOTIVATION:
                rejection_reason_code = "BACKGROUND_MOTIVATION"
                reasoning = (
                    f"Candidate rejected: Identified as introductory background motivation or established baseline rationale "
                    f"rather than an unaddressed research gap. ({val_result.reasoning})"
                )
            else:
                reasoning = val_result.reasoning
        else:
            reasoning = val_result.reasoning

        # Precision gating based on category and citation verification
        if rejection_reason_code or val_result.verdict == ValidationVerdict.REJECT or assigned_category == GapCategory.BACKGROUND_MOTIVATION:
            status = GapStatus.REJECTED
        else:
            # Category B: Broad challenge requiring human refinement -> Retained as CANDIDATE
            if assigned_category == GapCategory.BROAD_CHALLENGE or needs_refinement:
                status = GapStatus.CANDIDATE
                needs_refinement = True
            # Category C: Single-study limitation -> Retained as CANDIDATE with single-study scope
            elif assigned_category == GapCategory.STUDY_LIMITATION and len(verified_paper_ids) <= 1:
                status = GapStatus.CANDIDATE
            # Category D: Evidence-grounded research gap -> VALIDATED
            else:
                status = GapStatus.VALIDATED

        # Build evidence items
        evidence_items: List[StatementEvidenceDTO] = []
        if status != GapStatus.REJECTED or (verified_stmt_ids and not rejection_reason_code):
            for s_id in verified_stmt_ids:
                s = statements_by_id[s_id]
                p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
                if isinstance(p_id, str):
                    p_id = uuid.UUID(p_id)

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

        # Resolve model confidence score without fabricating certainty
        if val_result.confidence is not None:
            try:
                conf_val = max(0.0, min(1.0, float(val_result.confidence)))
            except (ValueError, TypeError):
                conf_val = None
        else:
            conf_val = None

        metadata = {
            "signals": candidate.signals,
            "model_verdict": val_result.verdict.value,
            "gap_category": assigned_category.value if assigned_category else None,
            "needs_refinement": needs_refinement,
        }
        if assigned_category == GapCategory.STUDY_LIMITATION and len(verified_paper_ids) <= 1:
            metadata["single_study_scope"] = True
        if rejection_reason_code:
            metadata["rejection_reason"] = rejection_reason_code
        if len(candidate.supporting_statement_ids) > len(bounded_stmt_ids):
            metadata["total_candidate_evidence_count"] = len(candidate.supporting_statement_ids)
            metadata["evaluated_evidence_count"] = len(bounded_stmt_ids)

        return ResearchGapCreateDTO(
            gap_statement=candidate.gap_statement,
            research_problem=candidate.research_problem,
            status=status,
            supporting_statement_ids=verified_stmt_ids if status != GapStatus.REJECTED or not rejection_reason_code else [],
            supporting_paper_ids=list(verified_paper_ids) if status != GapStatus.REJECTED or not rejection_reason_code else [],
            evidence_count=len(verified_stmt_ids) if status != GapStatus.REJECTED or not rejection_reason_code else 0,
            limitations=candidate.relevant_limitations,
            future_work=candidate.relevant_future_work,
            validation_reasoning=reasoning,
            validation_confidence=conf_val,
            evidence_items=evidence_items,
            metadata=metadata,
        )
