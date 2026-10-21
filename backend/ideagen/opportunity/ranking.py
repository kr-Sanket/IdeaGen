"""Candidate Opportunity Ranking Service (TF-IDF & FIFO).

Provides in-process, deterministic candidate opportunity ranking for incremental
lifecycle evolution, ordering eligible opportunities by relevance to incoming papers
under a candidate budget cap.
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from ideagen.db.models import Opportunity


def _get_opportunity_effective_creation_date(opp: Opportunity) -> datetime.date:
    """Resolves an opportunity's effective creation date deterministically."""
    opp_dt = opp.created_at
    if opp_dt is None and getattr(opp, "research_opportunity", None):
        opp_dt = opp.research_opportunity.created_at

    if opp_dt is not None:
        return opp_dt.date() if isinstance(opp_dt, datetime.datetime) else opp_dt
    return datetime.date.min


@runtime_checkable
class CandidateRanker(Protocol):
    """Protocol defining the candidate opportunity ranking interface."""

    def rank(
        self,
        query_text: str,
        candidates: Sequence[Opportunity],
        max_candidates: Optional[int] = None,
    ) -> List[Opportunity]:
        """Ranks candidate opportunities against query text.

        Args:
            query_text: Incoming paper query text (e.g. title + abstract).
            candidates: Sequence of candidate opportunities to rank.
            max_candidates: Optional maximum count of candidates to return.

        Returns:
            List of ranked Opportunity instances.
        """
        ...


class FIFOCandidateRanker:
    """Ranks candidate opportunities strictly in chronological order matching existing production behavior."""

    def rank(
        self,
        query_text: str,
        candidates: Sequence[Opportunity],
        max_candidates: Optional[int] = None,
    ) -> List[Opportunity]:
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                _get_opportunity_effective_creation_date(c),
                str(c.id),
            ),
        )

        if max_candidates is not None and max_candidates > 0:
            return sorted_candidates[:max_candidates]
        return sorted_candidates


class TFIDFCandidateRanker:
    """Ranks candidate opportunities using TF-IDF cosine similarity against incoming paper text.

    Constructs document vectors over candidate opportunity text (title + problem_statement + description)
    and query paper text.
    Breaks ties deterministically by (effective_creation_date.asc(), opportunity_id.asc()).
    """

    def __init__(
        self,
        stop_words: str = "english",
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.stop_words = stop_words
        self.ngram_range = ngram_range

    def rank(
        self,
        query_text: str,
        candidates: Sequence[Opportunity],
        max_candidates: Optional[int] = None,
    ) -> List[Opportunity]:
        if not candidates:
            return []

        clean_query = query_text.strip() if query_text else ""
        candidate_texts = [
            f"{c.title or ''} {c.problem_statement or ''} {c.description or ''}".strip()
            for c in candidates
        ]

        # If query or all candidate texts are empty, fall back to deterministic FIFO order
        if not clean_query or not any(candidate_texts):
            sorted_candidates = sorted(
                candidates,
                key=lambda c: (
                    _get_opportunity_effective_creation_date(c),
                    str(c.id),
                ),
            )
            if max_candidates is not None and max_candidates > 0:
                return sorted_candidates[:max_candidates]
            return sorted_candidates

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(candidate_texts + [clean_query])
            cand_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(cand_vectors, query_vector).flatten()
        except ValueError:
            # Fallback if vocabulary is empty (e.g. all stop-words)
            similarities = [0.0] * len(candidates)

        # Deterministic sorting: (-similarity, effective_creation_date, id)
        scored = list(zip(similarities, candidates))
        scored.sort(
            key=lambda item: (
                -float(item[0]),
                _get_opportunity_effective_creation_date(item[1]),
                str(item[1].id),
            )
        )

        ranked = [c for _, c in scored]
        if max_candidates is not None and max_candidates > 0:
            return ranked[:max_candidates]
        return ranked


def get_candidate_ranker(strategy: str = "tfidf") -> CandidateRanker:
    """Returns the candidate ranker instance corresponding to strategy name."""
    s = (strategy or "tfidf").strip().lower()
    if s == "fifo":
        return FIFOCandidateRanker()
    elif s == "tfidf":
        return TFIDFCandidateRanker()
    else:
        raise ValueError(f"Unsupported candidate ranking strategy: '{strategy}'. Supported: 'tfidf', 'fifo'.")
