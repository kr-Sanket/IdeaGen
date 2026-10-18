from .models import OpportunityStatus, OpportunityRelationship

def transition(current_status: OpportunityStatus, relationship: OpportunityRelationship) -> OpportunityStatus:
    """
    Pure Python, deterministic state transition function.
    """
    if relationship == OpportunityRelationship.UNRELATED:
        return current_status
        
    if relationship == OpportunityRelationship.SUPPORTS:
        if current_status == OpportunityStatus.OPEN:
            return OpportunityStatus.STRENGTHENED
        if current_status == OpportunityStatus.STRENGTHENED:
            return OpportunityStatus.STRENGTHENED
        if current_status == OpportunityStatus.PARTIALLY_ADDRESSED:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.RESOLVED:
            return OpportunityStatus.RESOLVED
            
    if relationship == OpportunityRelationship.ADDRESSES:
        if current_status == OpportunityStatus.OPEN:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.STRENGTHENED:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.PARTIALLY_ADDRESSED:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.RESOLVED:
            return OpportunityStatus.RESOLVED
            
    if relationship == OpportunityRelationship.CONTRADICTS:
        if current_status == OpportunityStatus.OPEN:
            return OpportunityStatus.OPEN
        if current_status == OpportunityStatus.STRENGTHENED:
            return OpportunityStatus.OPEN
        if current_status == OpportunityStatus.PARTIALLY_ADDRESSED:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.RESOLVED:
            return OpportunityStatus.OPEN
            
    if relationship == OpportunityRelationship.EXTENDS:
        if current_status == OpportunityStatus.OPEN:
            return OpportunityStatus.OPEN
        if current_status == OpportunityStatus.STRENGTHENED:
            return OpportunityStatus.STRENGTHENED
        if current_status == OpportunityStatus.PARTIALLY_ADDRESSED:
            return OpportunityStatus.PARTIALLY_ADDRESSED
        if current_status == OpportunityStatus.RESOLVED:
            return OpportunityStatus.RESOLVED

    # Fallback to current status if an unexpected state/relationship occurs
    return current_status
