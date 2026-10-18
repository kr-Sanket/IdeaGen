from pydantic import BaseModel
from typing import List

class BaselineDefinition(BaseModel):
    name: str
    target_component: str
    description: str
    metrics: List[str]
    is_implemented: bool

def get_baselines() -> List[BaselineDefinition]:
    return [
        BaselineDefinition(
            name="TF-IDF Discovery",
            target_component="OpportunityDiscoveryService",
            description="Lexical similarity clustering without semantic research-gap normalization.",
            metrics=["pairwise_precision", "pairwise_recall"],
            is_implemented=False
        ),
        BaselineDefinition(
            name="BM25 Retrieval",
            target_component="GroundedOpportunityService",
            description="Standard BM25 lexical retrieval against entire document text.",
            metrics=["top_k_recall"],
            is_implemented=False
        ),
        BaselineDefinition(
            name="Static Opportunity",
            target_component="OpportunityEvolutionService",
            description="Assessing the gap once at the end of the timeline without tracking intermediate states.",
            metrics=["temporal_state_accuracy"],
            is_implemented=False
        ),
        BaselineDefinition(
            name="Unconstrained Generation",
            target_component="ExplanationGenerator",
            description="Explanation generation without explicit evidence-reference validation.",
            metrics=["citation_validity_rate"],
            is_implemented=False
        )
    ]
