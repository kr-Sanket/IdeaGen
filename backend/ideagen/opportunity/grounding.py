"""Evidence Grounding Validation (Phase 6 / Checkpoint 8).

Verifies that evidence excerpts referenced during opportunity evolution are strictly
grounded in actual stored paper passages. Prevents hallucinated or fabricated LLM
claims from altering opportunity lifecycle states.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional, Sequence, Union

from ideagen.opportunity.models import OpportunityRelationship

logger = logging.getLogger(__name__)

# Explicit default threshold for approximate excerpt matching (0.85 = 85% token sequence similarity)
DEFAULT_GROUNDING_THRESHOLD: float = 0.85

SUBSTANTIVE_RELATIONSHIPS = {
    OpportunityRelationship.SUPPORTS,
    OpportunityRelationship.ADDRESSES,
    OpportunityRelationship.CONTRADICTS,
    OpportunityRelationship.EXTENDS,
}


class UngroundedEvidenceError(ValueError):
    """Raised when claimed evidence excerpt is not grounded in source paper passages."""
    pass


def normalize_text(text: str) -> str:
    """Normalizes text for grounding comparison by standardizing quotes, dashes, and whitespace."""
    if not text:
        return ""
    # Standardize typographic quotes and dashes
    t = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    # Collapse all whitespace and lowercase
    return re.sub(r"\s+", " ", t).strip().lower()


def is_excerpt_grounded(
    excerpt: Optional[str],
    passages: Sequence[Union[str, any]],
    threshold: float = DEFAULT_GROUNDING_THRESHOLD,
) -> bool:
    """Verifies whether an evidence excerpt is grounded in source passages.

    Strategy:
    1. Exact normalized substring matching against any source passage.
    2. If exact substring is not found, token sliding-window similarity against the threshold.
    """
    if not excerpt or not str(excerpt).strip():
        return False

    raw_texts = []
    for p in passages:
        if hasattr(p, "text"):
            txt = getattr(p, "text", "")
        else:
            txt = str(p)
        if txt and txt.strip():
            raw_texts.append(txt)

    if not raw_texts:
        return False

    norm_excerpt = normalize_text(excerpt)
    if not norm_excerpt:
        return False

    norm_passages = [normalize_text(t) for t in raw_texts]

    # 1. Exact normalized substring matching
    for norm_p in norm_passages:
        if norm_excerpt in norm_p:
            return True

    # 2. Approximate matching via sliding window over passage tokens
    excerpt_tokens = norm_excerpt.split()
    k = len(excerpt_tokens)
    if k < 2:
        # Single token: requires exact substring match to prevent single-word false positives
        return False

    for norm_p in norm_passages:
        p_tokens = norm_p.split()
        if len(p_tokens) < k:
            ratio = difflib.SequenceMatcher(None, norm_excerpt, norm_p).ratio()
            if ratio >= threshold:
                return True
            continue

        # Slide window of size k-1, k, k+1 over passage tokens
        for w_len in (k, max(1, k - 1), k + 1):
            if w_len > len(p_tokens):
                continue
            for i in range(len(p_tokens) - w_len + 1):
                window_str = " ".join(p_tokens[i : i + w_len])
                ratio = difflib.SequenceMatcher(None, norm_excerpt, window_str).ratio()
                if ratio >= threshold:
                    return True

    return False


def validate_evidence_grounding(
    excerpt: Optional[str],
    passages: Sequence[Union[str, any]],
    relationship: OpportunityRelationship,
    threshold: float = DEFAULT_GROUNDING_THRESHOLD,
    paper_id: Optional[any] = None,
) -> None:
    """Validates that substantive evidence relationships are strictly grounded in stored passages.

    Raises UngroundedEvidenceError if excerpt is missing or unsupported by passages.
    """
    if relationship not in SUBSTANTIVE_RELATIONSHIPS:
        # UNRELATED does not require passage grounding
        return

    if not is_excerpt_grounded(excerpt, passages, threshold=threshold):
        paper_info = f" for paper {paper_id}" if paper_id else ""
        raise UngroundedEvidenceError(
            f"Evidence excerpt is not grounded in stored passages{paper_info} "
            f"for relationship '{relationship.value}'. Excerpt: {excerpt!r}"
        )
