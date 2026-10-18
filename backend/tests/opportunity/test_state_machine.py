import pytest
from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from ideagen.opportunity.state_machine import transition

def test_transition_unrelated():
    for status in OpportunityStatus:
        assert transition(status, OpportunityRelationship.UNRELATED) == status

def test_transition_supports():
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.SUPPORTS) == OpportunityStatus.STRENGTHENED
    assert transition(OpportunityStatus.STRENGTHENED, OpportunityRelationship.SUPPORTS) == OpportunityStatus.STRENGTHENED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.SUPPORTS) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.RESOLVED, OpportunityRelationship.SUPPORTS) == OpportunityStatus.RESOLVED

def test_transition_addresses():
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.ADDRESSES) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.STRENGTHENED, OpportunityRelationship.ADDRESSES) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.ADDRESSES) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.RESOLVED, OpportunityRelationship.ADDRESSES) == OpportunityStatus.RESOLVED

def test_transition_contradicts():
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.CONTRADICTS) == OpportunityStatus.OPEN
    assert transition(OpportunityStatus.STRENGTHENED, OpportunityRelationship.CONTRADICTS) == OpportunityStatus.OPEN
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.CONTRADICTS) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.RESOLVED, OpportunityRelationship.CONTRADICTS) == OpportunityStatus.OPEN

def test_transition_extends():
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.EXTENDS) == OpportunityStatus.OPEN
    assert transition(OpportunityStatus.STRENGTHENED, OpportunityRelationship.EXTENDS) == OpportunityStatus.STRENGTHENED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.EXTENDS) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.RESOLVED, OpportunityRelationship.EXTENDS) == OpportunityStatus.RESOLVED

def test_no_single_addresses_creates_resolved():
    for status in OpportunityStatus:
        if status != OpportunityStatus.RESOLVED:
            assert transition(status, OpportunityRelationship.ADDRESSES) != OpportunityStatus.RESOLVED
