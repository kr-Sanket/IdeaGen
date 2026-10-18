from typing import Optional
from pydantic import BaseModel, Field

from ideagen.opportunity.models import OpportunityRelationship
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import MissingInputError

RELATIONSHIP_CLASSIFIER_PROMPT_VERSION = "v1"

RELATIONSHIP_CLASSIFIER_PROMPT = """You are an expert research assistant. Your task is to analyze a research paper against a known research opportunity and determine their relationship.

Definitions:
- SUPPORTS: The paper provides evidence that strengthens the existence or importance of the unresolved research opportunity. It does not substantially solve the opportunity.
- ADDRESSES: The paper directly tackles the problem represented by the opportunity. It provides a method/result that reduces the unresolved aspect. This does NOT mean the opportunity is automatically RESOLVED.
- CONTRADICTS: The paper provides evidence that conflicts with a premise, claim, or assumption underlying the opportunity.
- EXTENDS: The paper builds on, broadens, or redirects the research direction in a way related to the opportunity. EXTENDS does not automatically mean the opportunity is solved.
- UNRELATED: The paper is not materially relevant to the opportunity.

Opportunity Context:
Title: {opportunity_title}
Problem Statement: {opportunity_problem_statement}
Description: {opportunity_description}

Paper Context:
Title: {paper_title}
Abstract: {paper_abstract}
Excerpt: {paper_excerpt}

Instructions:
1. Compare the Paper Context against the Opportunity Context.
2. Determine the relationship strictly from the provided Definitions.
3. Determine a confidence score between 0.0 and 1.0.
4. Provide a brief rationale.
5. Identify a concise evidence_excerpt from the provided Paper Context that supports your classification (if applicable, else null). Do NOT fabricate evidence or add text outside the provided context.
"""

class RelationshipClassificationInput(BaseModel):
    opportunity_title: str
    opportunity_problem_statement: str
    opportunity_description: str
    paper_title: str
    paper_abstract: str
    paper_excerpt: Optional[str] = None

class RelationshipClassification(BaseModel):
    relationship: OpportunityRelationship
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_excerpt: Optional[str] = None
    prompt_version: str = RELATIONSHIP_CLASSIFIER_PROMPT_VERSION

class RelationshipClassifier:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def classify(self, input_data: RelationshipClassificationInput) -> RelationshipClassification:
        if not input_data.opportunity_title or not input_data.paper_title:
            raise MissingInputError("Missing required opportunity or paper title.")

        prompt = RELATIONSHIP_CLASSIFIER_PROMPT.format(
            opportunity_title=input_data.opportunity_title,
            opportunity_problem_statement=input_data.opportunity_problem_statement,
            opportunity_description=input_data.opportunity_description,
            paper_title=input_data.paper_title,
            paper_abstract=input_data.paper_abstract,
            paper_excerpt=input_data.paper_excerpt or "N/A"
        )
        
        result = self.provider.generate_structured(prompt, RelationshipClassification)
        result.prompt_version = RELATIONSHIP_CLASSIFIER_PROMPT_VERSION
        return result
