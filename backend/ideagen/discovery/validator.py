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
from .pattern_analyzer import is_background_motivation_claim

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

# Patterns describing scale/scope divergence
SCALE_UTILITY_PATTERNS = [
    re.compile(r"\b(?:utility-scale|grid-scale|transmission-level|transmission\s+grid|megawatt|MW)\b", re.IGNORECASE),
]
SCALE_RESIDENTIAL_PATTERNS = [
    re.compile(r"\b(?:residential|microgrids?|behind-the-meter|rooftop|distribution-level|low-voltage)\b", re.IGNORECASE),
]

# Patterns identifying introductory or background sections
INTRODUCTORY_SECTION_PATTERNS = [
    re.compile(r"\b(?:intro|introduction|background|related\s+work|literature\s+review|overview)\b", re.IGNORECASE),
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
   - BACKGROUND_MOTIVATION (verdict: REJECT): The candidate describes introductory rationale, domain importance, standard historical motivation for established paradigms, or textbook facts (e.g. why forecasting net load is required to safeguard operations, or why centralized storage creates privacy risks) rather than an open scientific question. Reject as background knowledge unless an explicit technical deficit/bottleneck is substantiated.
   - BROAD_CHALLENGE (verdict: VALID, needs_refinement: true): The candidate identifies a legitimate real-world challenge or complexity (e.g. non-IID data distribution in federated learning), but the claim is broad and requires formulating a specific sub-problem, setting, or research question. Retain as a candidate requiring refinement.
   - STUDY_LIMITATION (verdict: VALID, needs_refinement: false): The candidate describes an ordinary procedural constraint or author disclaimer of a single experimental setup (e.g. evaluation limited to a simulated environment or specific hospital cohort). Do not claim field-wide novelty without cross-paper evidence.
   - EVIDENCE_GROUNDED_GAP (verdict: VALID, needs_refinement: false): The candidate represents a concrete, plausible, unresolved scientific gap or bottleneck with direct passage-level evidentiary backing.
2. Ground all reasoning strictly in the provided statements and their original source passages. Do not extrapolate beyond the text.
3. Citations MUST select from the evidence aliases (e.g. "E1", "E2") or exact statement IDs provided in the evidence block above:
   - A VALID verdict REQUIRES at least one supporting evidence reference in supporting_evidence_statement_ids that directly supports the candidate claim.
   - If you cannot identify sufficient supporting evidence from the provided evidence block, you MUST NOT return VALID (return REJECT).
   - If there are no counter-evidence statements, return counter_evidence_statement_ids as an empty list [].
   - supporting_evidence_statement_ids and counter_evidence_statement_ids MUST be included in your output.
   - Do NOT invent evidence aliases or statement IDs.
4. If a statement's source passage is unavailable or contradicts the claimed limitation, evaluate it critically:
   - Check if an introductory limitation describing prior literature was addressed by the source paper itself. If the corpus evidence does not resolve whether it was solved, set needs_refinement to true.
   - Check scale and scope fidelity: do not extrapolate utility-scale findings to residential microgrids (or vice-versa) without evidence of transferability; set needs_refinement to true if scale diverges.
5. Provide a non-empty confidence score between 0.0 and 1.0 representing your confidence in the verdict and classification.
6. Output strictly conforming to the JSON schema.
"""

MAX_PASSAGE_EXCERPT_CHARS = 800


def resolve_evidence_citation(
    raw_citation: Any,
    alias_to_uuid: Dict[str, uuid.UUID],
    valid_evidence_pool: Set[uuid.UUID],
) -> Optional[uuid.UUID]:
    """Resolves an evidence citation (alias like 'E1', '[E1]' or raw UUID) to a verified statement UUID.

    Accepts only citations that map directly to statements available in the candidate's evidence pool:
    - Stable local aliases: 'E1', 'e1', '[E1]' present in alias_to_uuid.
    - Exact UUIDs present in valid_evidence_pool.

    Returns:
        The matching statement UUID, or None if unmapped, foreign, or invalid.
    """
    if raw_citation is None:
        return None
    raw_str = str(raw_citation).strip()
    if not raw_str:
        return None

    cleaned = raw_str.strip("[]'\" \t\r\n")

    # 1. Check alias mapping (case-insensitive)
    if cleaned in alias_to_uuid:
        return alias_to_uuid[cleaned]
    if cleaned.upper() in alias_to_uuid:
        return alias_to_uuid[cleaned.upper()]

    # 2. Check exact UUID string
    try:
        parsed_uuid = uuid.UUID(cleaned)
        if parsed_uuid in valid_evidence_pool:
            return parsed_uuid
    except (ValueError, AttributeError):
        pass

    return None


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
        alias_to_uuid: Dict[str, uuid.UUID] = {}
        seen_passage_provenance: Dict[str, uuid.UUID] = {}

        for idx, s_id in enumerate(bounded_stmt_ids, start=1):
            alias = f"E{idx}"
            alias_to_uuid[alias] = s_id
            alias_to_uuid[alias.lower()] = s_id

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
                f"- [{alias}] [ID: {s_id} | Paper: {p_id} | Type: {st_type} | Section: {sec or 'N/A'}]\n"
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
        for raw_id in (val_result.supporting_evidence_statement_ids or []):
            resolved_id = resolve_evidence_citation(
                raw_citation=raw_id,
                alias_to_uuid=alias_to_uuid,
                valid_evidence_pool=valid_evidence_pool,
            )
            if resolved_id is not None:
                if resolved_id not in verified_stmt_ids:
                    verified_stmt_ids.append(resolved_id)
            else:
                logger.warning(
                    f"GapValidator: Cited statement reference {raw_id!r} could not be resolved in candidate evidence pool."
                )

        verified_counter_stmt_ids: List[uuid.UUID] = []
        for raw_id in (val_result.counter_evidence_statement_ids or []):
            resolved_counter_id = resolve_evidence_citation(
                raw_citation=raw_id,
                alias_to_uuid=alias_to_uuid,
                valid_evidence_pool=valid_evidence_pool,
            )
            if resolved_counter_id is not None and resolved_counter_id not in verified_counter_stmt_ids:
                verified_counter_stmt_ids.append(resolved_counter_id)

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
        if is_background_motivation_claim(stmt_text) or is_background_motivation_claim(candidate.why_existing_work_insufficient):
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

        # 1. Distinct-passage analysis using passage_id
        verified_passage_ids: Set[uuid.UUID] = set()
        for s_id in verified_stmt_ids:
            s = statements_by_id[s_id]
            raw_pid = getattr(s, "passage_id", None) if hasattr(s, "passage_id") else (
                s.get("passage_id") if isinstance(s, dict) else None
            )
            if raw_pid:
                try:
                    if isinstance(raw_pid, str):
                        verified_passage_ids.add(uuid.UUID(raw_pid.strip()))
                    elif isinstance(raw_pid, uuid.UUID):
                        verified_passage_ids.add(raw_pid)
                except (ValueError, AttributeError):
                    pass

        distinct_passage_count = len(verified_passage_ids)
        is_same_passage_evidence = (
            distinct_passage_count <= 1
            and len(verified_stmt_ids) > 1
            and len(verified_passage_ids) > 0
        )

        # 2. Check for background motivation in candidate or evidence
        cand_problem = candidate.research_problem or candidate.gap_statement or ""
        cand_is_bg = is_background_motivation_claim(cand_problem) or is_background_motivation_claim(candidate.gap_statement)

        all_ev_is_bg = False
        if verified_stmt_ids:
            all_ev_is_bg = True
            for s_id in verified_stmt_ids:
                s = statements_by_id[s_id]
                txt = s.statement_text if hasattr(s, "statement_text") else s.get("statement_text", "")
                if not is_background_motivation_claim(str(txt)):
                    all_ev_is_bg = False
                    break

        if cand_is_bg or all_ev_is_bg or candidate.signals.get("potential_background_motivation"):
            assigned_category = GapCategory.BACKGROUND_MOTIVATION

        if assigned_category is None:
            assigned_category = self._infer_gap_category(
                candidate=candidate,
                val_result=val_result,
                verified_stmt_ids=verified_stmt_ids,
                statements_by_id=statements_by_id,
                verified_paper_ids=verified_paper_ids,
            )

        # 3. Check scale/scope divergence
        cand_target_residential = bool(re.search(r"\bresidential\b", candidate.gap_statement, re.IGNORECASE))
        cand_target_utility = any(pat.search(candidate.gap_statement) for pat in SCALE_UTILITY_PATTERNS)

        ev_texts = []
        for s_id in verified_stmt_ids:
            s = statements_by_id[s_id]
            txt = s.statement_text if hasattr(s, "statement_text") else s.get("statement_text", "")
            ev_texts.append(str(txt))
        combined_ev = " ".join(ev_texts).lower()

        ev_has_utility = any(pat.search(combined_ev) for pat in SCALE_UTILITY_PATTERNS)
        ev_has_residential = bool(re.search(r"\bresidential\b", combined_ev, re.IGNORECASE))

        scale_divergence = None
        if ev_has_utility and not ev_has_residential and cand_target_residential:
            scale_divergence = "utility_scale_evidence_generalized_to_residential_scope"
        elif ev_has_residential and not ev_has_utility and cand_target_utility:
            scale_divergence = "residential_evidence_generalized_to_utility_scope"

        # 4. Check introductory limitation from single paper
        all_introductory = False
        if len(verified_paper_ids) <= 1 and verified_stmt_ids:
            all_introductory = True
            for s_id in verified_stmt_ids:
                s = statements_by_id[s_id]
                sec = s.section if hasattr(s, "section") else s.get("section", "")
                sec_str = str(sec or "").strip()
                if not any(pat.search(sec_str) for pat in INTRODUCTORY_SECTION_PATTERNS):
                    all_introductory = False
                    break

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
            # Deterministic safeguards: scale divergence and single-paper introductory limitations
            if scale_divergence:
                status = GapStatus.CANDIDATE
                needs_refinement = True
                reasoning = f"{reasoning} [Scope note: Evidence evaluates utility-scale systems, while candidate claims residential microgrid scope. Retained as CANDIDATE requiring researcher refinement to establish cross-scale transferability.]"
            elif all_introductory and len(verified_paper_ids) <= 1:
                status = GapStatus.CANDIDATE
                needs_refinement = True
                reasoning = f"{reasoning} [Evidence note: Limitation is cited exclusively from introductory/related-work text of a single paper without conclusion or cross-paper corroboration. Uncertainty flagged for researcher review to verify resolution status.]"
            # Category B: Broad challenge requiring human refinement -> Retained as CANDIDATE
            elif assigned_category == GapCategory.BROAD_CHALLENGE or needs_refinement:
                status = GapStatus.CANDIDATE
                needs_refinement = True
            # Category C: Single-study limitation -> Retained as CANDIDATE with single-study scope
            elif assigned_category == GapCategory.STUDY_LIMITATION and len(verified_paper_ids) <= 1:
                status = GapStatus.CANDIDATE
            # Category D: Evidence-grounded research gap -> VALIDATED
            else:
                status = GapStatus.VALIDATED

        # Build evidence items with passage_id
        evidence_items: List[StatementEvidenceDTO] = []
        if status != GapStatus.REJECTED or (verified_stmt_ids and not rejection_reason_code):
            for s_id in verified_stmt_ids:
                s = statements_by_id[s_id]
                p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
                if isinstance(p_id, str):
                    p_id = uuid.UUID(p_id)

                raw_pid = getattr(s, "passage_id", None) if hasattr(s, "passage_id") else (
                    s.get("passage_id") if isinstance(s, dict) else None
                )
                passage_uuid = None
                if raw_pid:
                    try:
                        if isinstance(raw_pid, str):
                            passage_uuid = uuid.UUID(raw_pid.strip())
                        elif isinstance(raw_pid, uuid.UUID):
                            passage_uuid = raw_pid
                    except (ValueError, AttributeError):
                        passage_uuid = None

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
                        passage_id=passage_uuid,
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
            "distinct_passage_count": distinct_passage_count,
            "distinct_paper_count": len(verified_paper_ids),
        }
        if is_same_passage_evidence:
            metadata["single_passage_evidence"] = True
        if scale_divergence:
            metadata["scale_scope_divergence"] = scale_divergence
        if all_introductory and len(verified_paper_ids) <= 1:
            metadata["introductory_limitation_single_paper"] = True
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
