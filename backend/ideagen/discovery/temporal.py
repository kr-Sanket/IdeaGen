"""Temporal filtering and leakage prevention utilities for gap discovery (Phase 3).

Ensures that papers and statements published after an evaluation cutoff date cannot
influence research-gap discovery, clustering, candidate generation, or validation.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def normalize_cutoff_date(
    cutoff_date: Optional[Union[datetime.date, datetime.datetime, str]],
) -> Optional[datetime.date]:
    """Normalizes an optional cutoff date to a datetime.date object.

    Handles datetime.date, datetime.datetime, and ISO-8601 string formats.
    Raises ValueError on invalid string format or empty string.
    Raises TypeError on unsupported types.
    """
    if cutoff_date is None:
        return None

    if isinstance(cutoff_date, datetime.datetime):
        return cutoff_date.date()

    if isinstance(cutoff_date, datetime.date):
        return cutoff_date

    if isinstance(cutoff_date, str):
        cleaned = cutoff_date.strip()
        if not cleaned:
            raise ValueError("Cutoff date string cannot be empty or whitespace.")
        date_str = cleaned.split("T")[0].split(" ")[0]
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid cutoff date format: {cutoff_date!r}. Expected ISO format YYYY-MM-DD."
            ) from e

    raise TypeError(
        f"Unsupported cutoff date type: {type(cutoff_date).__name__}. Expected date, datetime, or ISO string."
    )


def extract_paper_publication_date(paper: Any) -> Optional[datetime.date]:
    """Extracts and normalizes the publication date from a paper model or dictionary.

    Returns None if publication_date is missing, null, or unparseable.
    Never fabricates publication dates.
    """
    if paper is None:
        return None

    raw_date = getattr(paper, "publication_date", None) if hasattr(paper, "publication_date") else (
        paper.get("publication_date") if isinstance(paper, dict) else None
    )

    if raw_date is None:
        return None

    if isinstance(raw_date, datetime.datetime):
        return raw_date.date()

    if isinstance(raw_date, datetime.date):
        return raw_date

    if isinstance(raw_date, str):
        cleaned = raw_date.strip()
        if not cleaned:
            return None
        date_str = cleaned.split("T")[0].split(" ")[0]
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            logger.warning(f"Unparseable paper publication date string: {raw_date!r}")
            return None

    return None


def is_paper_eligible(paper: Any, cutoff_date: datetime.date) -> bool:
    """Evaluates whether a paper is eligible under a temporal cutoff date.

    A paper is eligible ONLY IF its publication date is known and <= cutoff_date.
    Undated or future papers are strictly ineligible.
    """
    if cutoff_date is None:
        return True

    pub_date = extract_paper_publication_date(paper)
    if pub_date is None:
        return False

    return pub_date <= cutoff_date


def filter_statements_by_cutoff(
    statements: List[Any],
    cutoff_date: Optional[Union[datetime.date, datetime.datetime, str]],
    papers_by_id: Optional[dict] = None,
) -> List[Any]:
    """Filters research statements to only those belonging to papers published on or before cutoff_date.

    If cutoff_date is None, returns the statements unchanged.
    """
    norm_cutoff = normalize_cutoff_date(cutoff_date)
    if norm_cutoff is None:
        return statements

    eligible_statements = []
    for s in statements:
        # Check attached paper object
        paper = getattr(s, "paper", None) if hasattr(s, "paper") else (
            s.get("paper") if isinstance(s, dict) else None
        )

        # Or lookup paper from papers_by_id
        if paper is None and papers_by_id:
            paper_id = getattr(s, "paper_id", None) if hasattr(s, "paper_id") else (
                s.get("paper_id") if isinstance(s, dict) else None
            )
            paper = papers_by_id.get(paper_id)

        if paper is not None:
            if is_paper_eligible(paper, norm_cutoff):
                eligible_statements.append(s)
            else:
                logger.debug(
                    f"Statement {getattr(s, 'id', None)} excluded by cutoff {norm_cutoff} "
                    f"(paper publication date: {extract_paper_publication_date(paper)})"
                )
        else:
            # If statement has direct publication_date metadata (e.g. in test mock)
            stmt_date = getattr(s, "publication_date", None) if hasattr(s, "publication_date") else (
                s.get("publication_date") if isinstance(s, dict) else None
            )
            if stmt_date is not None:
                norm_stmt_date = normalize_cutoff_date(stmt_date)
                if norm_stmt_date is not None and norm_stmt_date <= norm_cutoff:
                    eligible_statements.append(s)
                else:
                    logger.debug(f"Statement excluded by publication_date {stmt_date} > {norm_cutoff}")
            else:
                # Undated statements cannot be verified as historical; exclude in temporal runs
                logger.warning(
                    f"Statement {getattr(s, 'id', None)} lacks associated paper/publication_date; "
                    "excluding from temporally constrained discovery run."
                )

    return eligible_statements
