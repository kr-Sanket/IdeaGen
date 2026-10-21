import uuid
import datetime
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.db.models import Base, Paper, Opportunity
from ideagen.ingestion.models import SourceType
from ideagen.opportunity.discovery import ResearchGapSignal, OpportunityDiscoveryService
from ideagen.retrieval.evidence import EvidenceResult
from ideagen.rag.explanation import GroundedOpportunityService, GroundedClaim, GroundedExplanation
from ideagen.opportunity.temporal import TemporalBatch, OpportunityEvolutionService
from ideagen.opportunity.classifier import RelationshipClassifier, RelationshipClassification
from ideagen.opportunity.models import OpportunityRelationship
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.pipeline.end_to_end import IdeaGenOrchestrator
from ideagen.api.schema import DemoScenarioResponse

class FakeGapEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping
    def embed(self, texts):
        return [self.mapping.get(t, [0.0]*3) for t in texts]

class FakeRetriever:
    def __init__(self, results):
        self.results = results
    def retrieve(self, query, top_k=5):
        return self.results

class FakeExplanationGenerator:
    def __init__(self, claims, ev_refs):
        self.claims = claims
        self.ev_refs = ev_refs
    def generate(self, candidate, evidence):
        return GroundedExplanation(
            summary="This opportunity highlights a critical gap in controlling Hallucinations in RAG.", 
            claims=self.claims, 
            unsupported_claims=[], 
            evidence_refs=self.ev_refs, 
            generation_metadata={}
        )

class ControlledProvider:
    def __init__(self, mapping):
        self.mapping = mapping
    def generate_structured(self, prompt, schema):
        rel = OpportunityRelationship.UNRELATED
        for k, v in self.mapping.items():
            if k in prompt:
                rel = v
                break
        return RelationshipClassification(
            relationship=rel,
            confidence=0.9,
            rationale="Deterministic evaluation based on real paper abstract.",
            evidence_excerpt="This paper directly relates to the identified RAG hallucination gap.",
            prompt_version="v1"
        )

def _describe_future_paper_outcome(p_future, evolution_results) -> str:
    """States what actually happened to the latest demo paper, derived from recorded events."""
    if p_future is None:
        return "No paper beyond batch b1 in this demonstration corpus."
    evaluated = any(
        ev.paper_id == p_future.id for er in evolution_results for ev in er.events
    )
    window = "demonstration batch b2 (2025-01-01 to 2025-12-31)"
    if evaluated:
        return f"Not excluded by date: this paper falls inside {window} and an evolution event was recorded."
    return (
        f"Not excluded by date: this paper falls inside {window}, but no evolution event was "
        "recorded (for example, its scripted classification was rejected by evidence grounding)."
    )


def build_demo_scenario(custom_raw_papers=None) -> DemoScenarioResponse:
    """Executes the Phase 14 End-to-End Demo scenario on a fresh in-memory database using real data."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        if custom_raw_papers is not None:
            raw_papers = custom_raw_papers
        else:
            corpus_path = os.path.join(os.path.dirname(__file__), "../../../docs/prototype_corpus_v1.json")
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
            raw_papers = corpus_data.get("papers", [])
            
        papers = []
        
        # We will split them artificially by temporal group
        b0 = TemporalBatch("b0", datetime.date(2023, 1, 1), datetime.date(2023, 12, 31), 0)
        b1 = TemporalBatch("b1", datetime.date(2024, 1, 1), datetime.date(2024, 12, 31), 1)
        
        # Let's map real papers to roles
        # First 3 in Group A -> gap papers
        # Rest in Group B -> mapped to ADDRESSES, SUPPORTS, UNRELATED
        
        gap_papers = []
        newer_papers = []
        for p in raw_papers:
            if p.get("temporal_group") == "Group A (Baseline)":
                gap_papers.append(p)
            else:
                newer_papers.append(p)
                
        # If not enough in A, take some from B
        while len(gap_papers) < 3 and newer_papers:
            gap_papers.append(newer_papers.pop(0))
            
        def make_paper(raw, batch):
            dt = datetime.datetime.strptime(raw["publication_date"], "%Y-%m-%d").date()
            
            # Reconstruct abstract if needed
            abstract = raw.get("abstract", "")
            if not abstract and "abstract_inverted_index" in raw:
                abstract_words = {}
                for word, positions in raw.get("abstract_inverted_index", {}).items():
                    for pos in positions:
                        abstract_words[pos] = word
                abstract = " ".join([abstract_words[i] for i in sorted(abstract_words.keys())])
            
            p = Paper(
                source_type=SourceType.OPENALEX_API.value,
                source_id=raw["id"],
                title=raw["title"],
                publication_date=dt,
                batch_id=batch.batch_id,
                abstract=abstract,
                authors=[],
                referenced_works=[],
                raw_metadata=raw,
            )
            session.add(p)
            return p

        # Insert Group A
        p_gaps = []
        for raw in gap_papers:
            p_gaps.append(make_paper(raw, b0))
        papers.extend(p_gaps)
        
        provider_map = {}
        for i, raw in enumerate(newer_papers[:6]):
            p = make_paper(raw, b1)
            papers.append(p)
            # Map some to specific relationships
            if i % 3 == 0:
                provider_map[p.title] = OpportunityRelationship.ADDRESSES
            elif i % 3 == 1:
                provider_map[p.title] = OpportunityRelationship.SUPPORTS
            else:
                provider_map[p.title] = OpportunityRelationship.UNRELATED
        
        p_future = None
        if len(newer_papers) > 6:
            b2 = TemporalBatch("b2", datetime.date(2025, 1, 1), datetime.date(2025, 12, 31), 2)
            p_future = make_paper(newer_papers[6], b2)
            papers.append(p_future)
            provider_map[p_future.title] = OpportunityRelationship.ADDRESSES
            
        session.commit()

        def make_signal(paper_id, statement):
            return ResearchGapSignal(
                id=uuid.uuid4(),
                paper_id=paper_id,
                statement=statement,
                normalized_statement=statement,
                confidence=0.9
            )

        gap_statement = f"Research highlights critical limitations and unaddressed challenges in the domains covered by these papers."
        signals = []
        for p_g in p_gaps:
            signals.append(make_signal(p_g.id, gap_statement))

        embedder_map = {
            gap_statement: [1.0, 0.0, 0.0]
        }
        discovery = OpportunityDiscoveryService(
            embedder=FakeGapEmbedder(embedder_map),
            similarity_threshold=0.8,
            min_cluster_size=2
        )

        evidence = [EvidenceResult(
            paper_id=p_gaps[0].id, title=p_gaps[0].title, publication_date=p_gaps[0].publication_date,
            source_type="openalex", source_id=p_gaps[0].source_id, evidence_text=p_gaps[0].abstract[:150] + "...",
            evidence_type="ABSTRACT", score=1.0, rank=1, provenance="DB"
        )]
        grounding = GroundedOpportunityService(
            retriever=FakeRetriever(evidence),
            generator=FakeExplanationGenerator(
                claims=[GroundedClaim("Current systems exhibit limitations as identified in recent literature.", [p_gaps[0].source_id])],
                ev_refs=[p_gaps[0].source_id]
            )
        )

        classifier = RelationshipClassifier(ControlledProvider(provider_map))
        transition = OpportunityTransitionService(session)
        integration = OpportunityClassificationIntegration(classifier, transition, session)
        evolution = OpportunityEvolutionService(integration)

        orchestrator = IdeaGenOrchestrator(discovery, grounding, evolution)

        opp_id, candidate = orchestrator.discover_and_validate(session, signals, lambda c: c[0])
        explanation = orchestrator.ground_evidence(candidate)
        
        # Evolve using active batches
        batches = [b0, b1]
        if p_future:
            batches.append(b2)
            
        evolution_results = orchestrator.evolve_temporally(session, opp_id, batches, papers)
        
        opp = session.query(Opportunity).filter(Opportunity.id == opp_id).first()
        
        timeline_batches = []
        timeline_batches.append({
            "batch_id": "b0",
            "name": "Phase 1: Discovery (Group A)",
            "events": [
                {
                    "relationship": "DISCOVERED",
                    "paper_title": f"{len(candidate.supporting_paper_ids)} Baseline Papers",
                    "paper_date": None,
                    "status_change": "None -> OPEN",
                    "excerpt": "Demonstration gap constructed from the scripted baseline cluster."
                }
            ]
        })

        for er in evolution_results:
            batch_events = []
            for ev in er.events:
                p_obj = session.query(Paper).filter(Paper.id == ev.paper_id).first()
                batch_events.append({
                    "relationship": ev.relationship.value,
                    "paper_title": p_obj.title if p_obj else str(ev.paper_id),
                    "paper_date": p_obj.publication_date if p_obj else None,
                    "status_change": f"{ev.previous_status.value} -> {ev.new_status.value}",
                    "excerpt": "Scripted demonstration classification (not model output)"
                })
            
            if batch_events:
                b_name = f"Phase 2: Evolution (Batch {er.events[0].batch_id})"
                timeline_batches.append({
                    "batch_id": er.events[0].batch_id,
                    "name": b_name,
                    "events": batch_events
                })

        return DemoScenarioResponse(
            opportunity_id=str(opp_id),
            title=candidate.title,
            problem_statement=candidate.problem_statement,
            initial_status="OPEN",
            final_status=opp.status,
            discovery={
                "cluster_size": candidate.cluster_size,
                "papers": [{"id": str(p.id), "title": p.title, "date": p.publication_date} for p in papers if p.id in candidate.supporting_paper_ids]
            },
            grounding={
                "claims": [{"text": c.text, "refs": c.evidence_refs} for c in explanation.claims],
                "evidence": [
                    {
                        "source_id": e.source_id,
                        "title": e.title,
                        "text": e.evidence_text
                    } for e in evidence
                ]
            },
            timeline=timeline_batches,
            leakage_protection={
                "future_paper": p_future.title if p_future else "None",
                "future_date": p_future.publication_date if p_future else None,
                "result": _describe_future_paper_outcome(p_future, evolution_results),
            }
        )

    finally:
        session.close()
