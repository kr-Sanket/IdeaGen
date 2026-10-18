import uuid
import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session

from ideagen.db.models import Opportunity, Paper
from ideagen.opportunity.models import OpportunityStatus
from ideagen.opportunity.discovery import OpportunityCandidate, OpportunityDiscoveryService
from ideagen.rag.explanation import GroundedOpportunityService, GroundedExplanation
from ideagen.opportunity.temporal import OpportunityEvolutionService, TemporalBatch, OpportunityEvolutionResult

@dataclass
class EndToEndOpportunityResult:
    opportunity_id: uuid.UUID
    discovery_cluster_id: str
    supporting_papers: List[uuid.UUID]
    grounded_explanation: GroundedExplanation
    temporal_events: List[OpportunityEvolutionResult]
    initial_status: OpportunityStatus
    final_status: OpportunityStatus

class IdeaGenOrchestrator:
    """Thin orchestration service for the end-to-end controlled demonstration."""
    
    def __init__(
        self,
        discovery_service: OpportunityDiscoveryService,
        grounding_service: GroundedOpportunityService,
        evolution_service: OpportunityEvolutionService
    ):
        self.discovery = discovery_service
        self.grounding = grounding_service
        self.evolution = evolution_service
        
    def discover_and_validate(
        self,
        db: Session,
        signals, # List of ideagen.opportunity.discovery.ResearchGapSignal
        candidate_selection_fn # Callable that takes List[OpportunityCandidate] and returns OpportunityCandidate
    ) -> Tuple[uuid.UUID, OpportunityCandidate]:
        """Runs discovery and simulates a researcher validating a candidate into an Opportunity."""
        # 1. Discover candidates
        candidates = self.discovery.discover_opportunities(signals)
        if not candidates:
            raise ValueError("No candidates discovered")
            
        # 2. Select controlled candidate (representing human review)
        selected_candidate = candidate_selection_fn(candidates)
        if not selected_candidate:
            raise ValueError("No candidate selected by validation function")
            
        # 3. Validate and persist as Opportunity
        opp = Opportunity(
            id=uuid.uuid4(),
            title=selected_candidate.title,
            problem_statement=selected_candidate.problem_statement,
            description=selected_candidate.description,
            status=OpportunityStatus.OPEN.value,
            confidence=selected_candidate.confidence
        )
        db.add(opp)
        db.commit()
        
        return opp.id, selected_candidate
        
    def ground_evidence(
        self,
        candidate: OpportunityCandidate
    ) -> GroundedExplanation:
        """Grounds the selected candidate with verifiable evidence."""
        return self.grounding.ground(candidate)
        
    def evolve_temporally(
        self,
        db: Session,
        opportunity_id: uuid.UUID,
        batches: List[TemporalBatch],
        papers: List[Paper]
    ) -> List[OpportunityEvolutionResult]:
        """Evaluates historical opportunity against chronological literature batches."""
        # Use replay to ensure chronological ordering and isolation
        return self.evolution.replay(db, opportunity_id, batches, papers)
        
    def run_controlled_demo(
        self,
        db: Session,
        signals, 
        candidate_selection_fn,
        candidate_for_grounding: OpportunityCandidate, # Passed separately if we want to isolate or reuse the selected one
        batches: List[TemporalBatch],
        papers: List[Paper]
    ) -> EndToEndOpportunityResult:
        """Convenience method combining discovery, validation, grounding, and evolution."""
        # In a real system, these would happen asynchronously, but for demo we combine them.
        opp_id, candidate = self.discover_and_validate(db, signals, candidate_selection_fn)
        
        explanation = self.ground_evidence(candidate_for_grounding or candidate)
        
        # Determine initial state directly from DB
        opp = db.query(Opportunity).filter(Opportunity.id == opp_id).first()
        initial_status = OpportunityStatus(opp.status)
        
        # Evolve over time
        evolution_results = self.evolve_temporally(db, opp_id, batches, papers)
        
        # Determine final state
        db.refresh(opp)
        final_status = OpportunityStatus(opp.status)
        
        return EndToEndOpportunityResult(
            opportunity_id=opp_id,
            discovery_cluster_id=candidate.cluster_id,
            supporting_papers=candidate.supporting_paper_ids,
            grounded_explanation=explanation,
            temporal_events=evolution_results,
            initial_status=initial_status,
            final_status=final_status
        )
