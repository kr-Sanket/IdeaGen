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
| Backend | Python + FastAPI | **IMPLEMENTED** (Full API & experiment pipeline) |
| Database | PostgreSQL / SQLite | **IMPLEMENTED** (PostgreSQL schema via Alembic; SQLite for isolated runs/tests) |
| Vector index | FAISS | **IMPLEMENTED** (`backend/ideagen/discovery/vector_index.py`) |
| Whole-paper embeddings | SPECTER2 | **DESIGN** — planned whole-paper representation option |
| Statement embeddings | SentenceTransformer, Ollama (`nomic-embed-text`), Gemini | **IMPLEMENTED** (`backend/ideagen/discovery/embeddings.py`) |
| Clustering | Agglomerative Clustering (scikit-learn) | **IMPLEMENTED** (Cosine distance threshold; HDBSCAN was initial proposal) |
| LLM | Provider abstraction (Ollama `qwen3:8b`, Gemini `gemini-2.5-flash`, Claude) | **IMPLEMENTED** (`backend/ideagen/llm/`) |
| Frontend | React 19 + Vite 8 + TypeScript | **IMPLEMENTED** (Read-only explorer UI; light-theme redesign pending) |
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

## Current Architectural Scope & Limitations

1. **Authentication & Multi-Tenant Isolation:**
   - User authentication, JWT sessions, and role-based access control are **not implemented**.
   - Multi-tenant workspaces are **not implemented**. The API exposes open endpoints (`allow_origins=["*"]`).
   - Workspace-oriented statistics (`/api/v1/stats`) report global database aggregates, not isolated user workspace data.
   - The frontend is an exploratory viewing interface; it does not provide user accounts or workspace management.

2. **Database & Execution Environment:**
   - PostgreSQL is the production store with 11 Alembic migrations; automated CI suites execute against SQLite (`create_all`).
   - PostgreSQL-specific behaviors (concurrency locks, transactional savepoints) require a running PostgreSQL instance and have not been tested in SQLite-only runs.
   - Long-running pipeline operations are synchronous; asynchronous job queues (e.g. Celery/Redis) are not implemented.

3. **Empirical Evaluation Bounds:**
   - Automated test suites verify software correctness, schema conformance, and deterministic pipeline mechanics.
   - LLM evaluations (e.g., relationship classifier prompt v2) are exploratory prototype evaluations conducted on small datasets (e.g., $N=25$). They do not constitute evidence of broad scientific accuracy across open-domain literature.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Pipeline design | **AUTHORITATIVE DESIGN** |
| Technology stack | **AUTHORITATIVE DESIGN** |
| Architectural principles | **AUTHORITATIVE DESIGN** |
| RAG design principles | **AUTHORITATIVE DESIGN** |
| "LLM classifies, code transitions" | **AUTHORITATIVE DESIGN — NON-NEGOTIABLE** |
| Current implementation | **Backend pipeline & read-only explorer UI implemented** (see `docs/CURRENT_STATUS.md`) |
