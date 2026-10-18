# System Architecture

## High-Level System Workflow

The intended pipeline:

```
OpenAlex scientific literature
        │
        ▼
Paper ingestion
        │
        ▼
Paper processing / cleaning
        │
        ▼
LLM / Transformer structured extraction
        │
        ▼
Research knowledge representation
        │
        ▼
Embeddings
        │
        ▼
Opportunity discovery
        │
        ▼
Opportunity object
        │
        ▼
New paper batches arrive
        │
        ▼
Semantic candidate retrieval
        │
        ▼
LLM relationship classification
        │
        ▼
Deterministic opportunity state transition
        │
        ▼
Evidence-grounded explanation
        │
        ▼
Evaluation / dashboard
```

---

## Technology Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| Backend | Python + FastAPI | **IMPLEMENTED** (Phase 10 Demo Adapter) |
| Database | PostgreSQL | **DESIGN** — not yet connected |
| Vector index | FAISS (initial) | **DESIGN** — not yet implemented |
| Whole-paper embeddings | SPECTER2 | **DESIGN** — not yet implemented |
| Statement embeddings | Sentence-transformer style model | **DESIGN** — not yet implemented |
| Clustering | HDBSCAN | **DESIGN** — not yet implemented |
| LLM | Provider abstraction, initially compatible with Claude | **DESIGN** — not yet implemented |
| Frontend | React + Vite | **IMPLEMENTED** (Phase 10 Demo UI) |
| Python target | Python 3.11+ | **IMPLEMENTED** — `pyproject.toml` specifies `>=3.11` |

---

## Core Architectural Principle: LLM Classifies, Code Transitions

This is a **NON-NEGOTIABLE** architecture principle.

### LLM Responsibilities

- Structured information extraction
- Relationship classification
- Evidence interpretation
- Explanation generation

### Deterministic Application Responsibilities

- Validate structured outputs
- Apply state-transition rules
- Update opportunity status
- Maintain history
- Enforce evidence constraints
- Prevent invalid state transitions

**The LLM must never directly write `opportunity.status`.**

---

## Evidence-Grounded RAG

RAG is used for evidence retrieval and explanation. It is **not** the primary research novelty, but a mechanism for evidence grounding, provenance, traceability, reproducible explanations, and protection against unsupported claims.

### RAG Principles

- For an opportunity transition, the explanation should use the actual evidence papers responsible for the transition.
- The system enforces: **"No citation, no claim."**
- Explanation output must not cite papers outside the evidence set.

### Core RAG Components

- `EvidenceResult`: Typed representation of retrieved evidence, containing strictly validated IDs, provenance, text, and ranking scores.
- `EvidenceRetriever`: The core protocol for sourcing literature evidence. PostgreSQL acts as the definitive source of truth while derived/local/FAISS indexes fulfill vector search.
- `GroundedExplanation`: Typed representation containing validated `GroundedClaim` instances and isolated `UnsupportedClaim` instances.
- `GroundedOpportunityService`: The orchestration layer enforcing evidence validation logic ("No fabricated citations allowed") wrapping retrieval and generation.

### RAG Validation Requirements

The system should validate that:

1. At least one valid evidence source exists.
2. Cited paper IDs belong to the permitted evidence set.
3. Unsupported claims are rejected or flagged.

The explanation must be traceable to the triggering evidence.

### Relationship to Opportunity Lifecycle
The RAG pipeline generates grounded explanations for an `OpportunityCandidate` independently of lifecycle execution. Discovery creates Candidates -> RAG evaluates and grounds Candidates with Explanations -> Validated Candidate transitions to persistent `Opportunity` -> deterministic temporal rules apply. RAG does **not** modify lifecycle status or create history records.

### RAG Implementation Principle

Do **not** introduce LangChain or LlamaIndex as mandatory architecture.

The initial RAG implementation uses a thin/custom retrieval layer.

Additional frameworks can only be introduced if there is a demonstrated engineering/research reason.

---

## Architectural Principles

1. **PostgreSQL is the source of truth.** FAISS is derived.
2. **LLM classifies, code transitions.** The LLM must never directly set opportunity state.
3. **Temporal leakage must be prevented.** Future literature must not leak into historical analysis.
4. **Evidence must remain traceable.** Every claim links to source papers.
5. **No hardcoded literature data.** Literature records must come from ingestion, not source code.
6. **Provider-specific logic must remain isolated.** LLM provider abstraction is required.
7. **Structured outputs must be validated.** LLM output is validated before acceptance.
8. **Embedding versions must be tracked.** Model name, version, and generation metadata are stored.
9. **Modules should have typed interfaces.**
10. **Implementation should be incremental and testable.**
11. **Avoid unnecessary enterprise complexity.**
12. **Research methodology changes require explicit architectural review.**

---

## Data Flow Summary

| Stage | Input | Output | Actor |
|-------|-------|--------|-------|
| Ingestion | OpenAlex API | Raw paper records | Application code |
| Processing | Raw records | Normalized paper records | Application code |
| Extraction | Normalized papers | Structured extracted information | LLM (validated by code) |
| Embedding | Papers + extracted statements | SPECTER2 + sentence-transformer vectors | Transformer models |
| Discovery | Embedded statements | Candidate opportunities | Clustering + LLM synthesis |
| Lifecycle | New paper batches + opportunities | State transitions + evidence | LLM classification → deterministic code |
| Explanation | Transition evidence | Evidence-grounded explanations | LLM (validated by code) |
| Evaluation | System outputs | Quality metrics | Evaluation framework |
| Orchestration | Fixture papers + models | End-to-End Timeline Demo | Integration pipeline orchestrator |

### End-to-End Controlled Pipeline
The `IdeaGenOrchestrator` orchestrates these stages into an isolated pipeline:
1. Ingests mock fixtures into PostgreSQL
2. Discovers clusters from signals via `OpportunityDiscoveryService`
3. Validates and generates `Opportunity` DB records
4. Generates grounded explanations securely via `GroundedOpportunityService`
5. Fast-forwards temporally across paper batches via `OpportunityEvolutionService`
6. Evaluates deterministic timeline reconstruction from History/Evidence traces

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Pipeline design | **AUTHORITATIVE DESIGN** |
| Technology stack | **AUTHORITATIVE DESIGN** |
| Architectural principles | **AUTHORITATIVE DESIGN** |
| RAG design principles | **AUTHORITATIVE DESIGN** |
| "LLM classifies, code transitions" | **AUTHORITATIVE DESIGN — NON-NEGOTIABLE** |
| Current implementation | **Phase 10 demo dashboard complete** |
