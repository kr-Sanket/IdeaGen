"""Phase 5 Research Opportunity Generation & Prior-Art Validation Package."""

from ideagen.generation.models import (
    OpportunityStatus,
    FeasibilityAssessment,
    NoveltyAssessment,
    PriorArtAssessment,
    ResearchOpportunityInputDTO,
    CandidateOpportunity,
    RefinementCritique,
    ResearchOpportunityDTO,
)
from ideagen.generation.validator import (
    EvidenceIntegrityValidator,
    sanitize_novelty_language,
)
from ideagen.generation.generator import OpportunityGenerator
from ideagen.generation.prior_art import PriorArtRetriever, PriorArtValidator
from ideagen.generation.refinement import IterativeRefinementEngine
from ideagen.generation.persistence import OpportunityPersistenceLayer
from ideagen.generation.pipeline import ResearchOpportunityPipeline

__all__ = [
    "OpportunityStatus",
    "FeasibilityAssessment",
    "NoveltyAssessment",
    "PriorArtAssessment",
    "ResearchOpportunityInputDTO",
    "CandidateOpportunity",
    "RefinementCritique",
    "ResearchOpportunityDTO",
    "EvidenceIntegrityValidator",
    "sanitize_novelty_language",
    "OpportunityGenerator",
    "PriorArtRetriever",
    "PriorArtValidator",
    "IterativeRefinementEngine",
    "OpportunityPersistenceLayer",
    "ResearchOpportunityPipeline",
]
