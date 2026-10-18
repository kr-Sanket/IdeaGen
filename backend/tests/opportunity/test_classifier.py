import pytest
from pydantic import ValidationError

from ideagen.opportunity.classifier import RelationshipClassifier, RelationshipClassificationInput, RELATIONSHIP_CLASSIFIER_PROMPT_VERSION, RELATIONSHIP_CLASSIFIER_PROMPT
from ideagen.opportunity.models import OpportunityRelationship
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError, MissingInputError
from tests.opportunity.fake_provider import FakeLLMProvider

@pytest.fixture
def valid_input():
    return RelationshipClassificationInput(
        opportunity_title="Opp 1",
        opportunity_problem_statement="Problem 1",
        opportunity_description="Desc 1",
        paper_title="Paper 1",
        paper_abstract="Abstract 1",
        paper_excerpt="Excerpt 1"
    )

def test_classifier_supports(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": 0.9,
        "rationale": "Reasoning",
        "evidence_excerpt": "Excerpt 1"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.SUPPORTS
    assert result.confidence == 0.9
    assert result.evidence_excerpt == "Excerpt 1"
    assert result.prompt_version == RELATIONSHIP_CLASSIFIER_PROMPT_VERSION

def test_classifier_addresses(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "ADDRESSES",
        "confidence": 0.8,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.ADDRESSES

def test_classifier_contradicts(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "CONTRADICTS",
        "confidence": 0.99,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.CONTRADICTS

def test_classifier_extends(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "EXTENDS",
        "confidence": 0.7,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.EXTENDS

def test_classifier_unrelated(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "UNRELATED",
        "confidence": 0.1,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.UNRELATED

def test_unknown_relationship_rejected(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "UNKNOWN_REL",
        "confidence": 0.5,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    with pytest.raises(MalformedResponseError):
        classifier.classify(valid_input)

def test_confidence_out_of_bounds(valid_input):
    provider_low = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": -0.1,
        "rationale": "Reasoning"
    })
    with pytest.raises(MalformedResponseError):
        RelationshipClassifier(provider_low).classify(valid_input)

    provider_high = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": 1.1,
        "rationale": "Reasoning"
    })
    with pytest.raises(MalformedResponseError):
        RelationshipClassifier(provider_high).classify(valid_input)

def test_missing_required_fields(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": 0.5
    }) # Missing rationale
    with pytest.raises(MalformedResponseError):
        RelationshipClassifier(provider).classify(valid_input)
        
    provider2 = FakeLLMProvider(response={
        "confidence": 0.5,
        "rationale": "R"
    }) # Missing relationship
    with pytest.raises(MalformedResponseError):
        RelationshipClassifier(provider2).classify(valid_input)

def test_provider_failure(valid_input):
    provider = FakeLLMProvider(should_fail=True)
    classifier = RelationshipClassifier(provider)
    with pytest.raises(ProviderFailureError):
        classifier.classify(valid_input)
        
def test_malformed_response_error(valid_input):
    provider = FakeLLMProvider(should_malform=True)
    classifier = RelationshipClassifier(provider)
    with pytest.raises(MalformedResponseError):
        classifier.classify(valid_input)

def test_missing_input():
    provider = FakeLLMProvider(response={"relationship": "SUPPORTS", "confidence": 0.9, "rationale": "R"})
    classifier = RelationshipClassifier(provider)
    with pytest.raises(MissingInputError):
        classifier.classify(RelationshipClassificationInput(
            opportunity_title="",
            opportunity_problem_statement="Problem 1",
            opportunity_description="Desc 1",
            paper_title="Paper 1",
            paper_abstract="Abstract 1"
        ))

def test_prompt_content(valid_input):
    provider = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": 0.9,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    classifier.classify(valid_input)
    
    assert "SUPPORTS:" in provider.last_prompt
    assert "ADDRESSES:" in provider.last_prompt
    assert "CONTRADICTS:" in provider.last_prompt
    assert "EXTENDS:" in provider.last_prompt
    assert "UNRELATED:" in provider.last_prompt
    
    assert valid_input.opportunity_title in provider.last_prompt
    assert valid_input.paper_title in provider.last_prompt

# Ensure classifier has no db session
def test_classifier_does_not_require_db(valid_input):
    # Pass a None session just to be sure if it accidentally requires one.
    # We didn't even pass a session, it only takes a provider!
    provider = FakeLLMProvider(response={
        "relationship": "SUPPORTS",
        "confidence": 0.9,
        "rationale": "Reasoning"
    })
    classifier = RelationshipClassifier(provider)
    result = classifier.classify(valid_input)
    assert result.relationship == OpportunityRelationship.SUPPORTS
