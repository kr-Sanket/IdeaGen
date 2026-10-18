"""Database and persistence layer."""

from .models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    PriorArtAssessmentRecord,
    OpportunityRefinementRecord,
)
from .persistence import SQLAlchemyPersistenceLayer
from .session import get_db, get_engine, get_session_factory

__all__ = [
    "Base",
    "Paper",
    "PaperDocument",
    "PaperPassage",
    "ResearchKnowledge",
    "ResearchStatement",
    "ResearchGap",
    "GapEvidence",
    "Opportunity",
    "OpportunityEvidence",
    "OpportunityHistory",
    "ResearchOpportunityRecord",
    "PriorArtAssessmentRecord",
    "OpportunityRefinementRecord",
    "SQLAlchemyPersistenceLayer",
    "engine",
    "SessionLocal",
    "get_db",
    "get_engine",
    "get_session_factory",
]


def __getattr__(name: str):
    """Provide lazy access to engine and SessionLocal from session module."""
    if name in ("engine", "SessionLocal"):
        from . import session
        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

