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
from .session import engine, SessionLocal, get_db

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
]

