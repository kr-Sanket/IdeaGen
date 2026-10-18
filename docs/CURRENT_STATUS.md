# Current Status

> **Last updated:** 2026-10-04

---

## Current Phase

**Phase 3** — Dockerized PostgreSQL Persistence Foundation implemented. Paper schema and Alembic migrations created. Awaiting rate-limit recovery for full corpus ingestion.

---

## Completed

| Item | Evidence |
|------|----------|
| Phase 0 — Backend Foundation | `backend/ideagen/core/config.py`, `backend/pyproject.toml`, 13 passing tests |
| Python package scaffold (`ideagen`) | `backend/ideagen/__init__.py` (v0.1.0) |
| Pydantic-settings configuration | `Settings` class with `IDEAGEN_` env prefix |
| Environment variable template | `backend/.env.example` |
| Data directory scaffold | `data/{raw,processed,indexes}/` with `.gitkeep` files |
| `.gitignore` | Covers Python, Node.js, data, IDE, OS files |
| Backend README | `backend/README.md` with setup instructions |
| Phase 0 completion record | `docs/06_STATE_AND_SESSION.md` |
| Documentation foundation | 13 design docs + `AGENTS.md` created from authoritative specification |
| Phase 1 — Ingestion components | `backend/ideagen/ingestion/` (client, parser, pipeline, models) and 12 passing tests |
| Phase 2 — Validation Harness | `backend/ideagen/ingestion/validation.py` and 3 passing tests |
| Phase 2.5 — Multi-Source Architecture | `backend/ideagen/ingestion/sources.py`, generic `PaperCreateDTO`, and 4 passing tests |
| Phase 3 — PostgreSQL Persistence | Dockerized PostgreSQL, `Paper` model, Alembic migrations, passing DB tests |
| Phase 4A.1 — PaperDocument Model | `PaperDocument` SQLAlchemy model, Alembic migration, decoupled text artifacts |
| Phase 4A — Manual PDF Ingestion | `ManualPDFIngestionService` implemented using PyMuPDF for layout-aware text extraction and conservative normalization. Supports manual research/development papers. |
| Phase 4A.2 — Ingestion Hardening | Audited `ManualPDFIngestionService` logic. Optimized `PersistenceLayer` to return database UUIDs, avoiding subsequent queries. Nullable `publication_date` allows exact preservation of unspecified provenance without hallucination. |
| Phase 5.1 — Opportunity Domain & State Machine | Pure deterministic Python state machine linking `OpportunityRelationship` inputs to `OpportunityStatus` outcomes. SQLAlchemy schema with bounded confidence (`[0.0, 1.0]`) and rigorous evidence/history tracking. LLM is formally decoupled from status mutations. |
| Phase 5.2 — Opportunity Transition Service | `OpportunityTransitionService` orchestrates safe state transitions, atomic commits, concurrency locks via `SELECT ... FOR UPDATE`, and deduplication across `Opportunity`, `OpportunityEvidence`, and `OpportunityHistory` tables. |
| Phase 5.3 — LLM Relationship Classifier | Pure application-layer classifier generating structured validations (`SUPPORTS`, `ADDRESSES`, `CONTRADICTS`, `EXTENDS`, `UNRELATED`) behind a provider-independent `LLMProvider` abstraction. Explicitly isolated from DB state. |
| Phase 5.4 & 5.4.1 — Real LLM Provider & Evaluation | Integrated Google's `google-genai` SDK. Structured evaluations run successfully on the 25-fixture benchmark via a controlled evaluation script. API keys strictly protected. |
| Phase 5.5 — Controlled LLM → Opportunity Transition Integration | Integrated pipeline orchestration via `OpportunityClassificationIntegration`. Pure delegation: LLM classifies, code transitions. 18 integration tests pass, full suite pass, strict scope safety maintained. Real Gemini controlled test was BLOCKED by external quota. |
| Phase 6 — Controlled Opportunity Discovery | Implemented purely deterministic candidate discovery from `ResearchGapSignal` structures. Features: conservative normalization, `GapEmbedder` abstraction with `FakeGapEmbedder`, cosine similarity, configurable threshold clustering (HDBSCAN placeholder/connected components), configurable `MIN_CLUSTER_SIZE=2`. Strict preservation of trace evidence. Used synthetic fixture of 13 signals. Full tests pass (118 tests). 0 OpenAlex calls, 0 Gemini calls, corpus not frozen. |
| Phase 7 — Evidence Grounding / RAG | Implemented controlled, deterministic RAG architecture via `EvidenceRetriever` and `LocalEvidenceRetriever` for bounded local search, avoiding giant FAISS dependencies in this phase. Introduced `GroundedExplanation` and `ExplanationGenerator` constructs wrapped by `GroundedOpportunityService` to strictly validate citations and block hallucinations ("No citation, no claim" enforcement). Added 20 new tests bringing total to 138 passing tests. OpenAlex calls = 0, Gemini calls = 0, corpus not frozen. |
| Phase 8 — Temporal Opportunity Evolution | Built the deterministic temporal evaluation engine via `TemporalBatch` and `OpportunityEvolutionService`. Enforced strict temporal isolation and future-paper leakage protection through chronological date cutoff validation. Verified replay capabilities over chronological sequences. Completed with full integration into the state machine without adding automated RESOLVED behaviors or new status types (AD-22 rules preserved). 14 new tests brought the backend test suite to 152 passing tests, processing a synthetic fixture dataset of 12+ papers across up to 3 batches. OpenAlex = 0, Gemini = 0, corpus not frozen. |
| Phase 9 — Controlled End-to-End Pipeline | Orchestrated the full component stack via `IdeaGenOrchestrator`. Built a reproducible fixture dataset (9 papers spanning 4 batches). Seamlessly tied ingestion mock -> signal processing -> discovery -> candidate validation -> RAG explanation grounding -> temporal batch progression. Verified explicit future-leakage protection, chronological isolation, provider fallback bounds, duplicate/idempotency handling, and the progression `OPEN -> STRENGTHENED -> PARTIALLY_ADDRESSED`. Added 2 integration tests bringing total tests to 154 passing. Database migrations=0, OpenAlex=0, Gemini=0. |
| Phase 10 — IDEAGEN DASHBOARD / DEMO UI | Built a functional research-opportunity dashboard in React + Vite over a FastAPI backend adapter. Reused Phase 9 controlled End-to-End fixture safely without mutating PostgreSQL database or adding research logic. Visualizes temporal evolution (with future-leakage protection shown), RAG claims/citations, and status progression natively in the UI. Clean, academic layout using vanilla CSS. |
| Phase 11 — DEMO HARDENING + RESEARCH EVALUATION | Established the formal research evaluation architecture. Clearly delineated Engineering Validation (software correctness) from Controlled Prototype Evaluation (fixture testing) from Real-World Research Evaluation (future work). Configured `ideagen/evaluation` Python packages to emit JSON-serializable structured metrics for discovery precision, grounding, and temporal leakage labeled specifically as Controlled Prototype Evaluation. Defined explicit baselines (BM25, TF-IDF), ablation plans, human evaluation protocols (Krippendorff's alpha), and failure analysis categories. Hardened the API schema and UI presentation to strictly convey the dashboard as a *controlled demonstration* of the architecture, avoiding exaggerated production claims. |
| Phase 12 — IDEAGEN CORPUS SPECIFICATION FREEZE | Established and froze the exact `v1.0` research corpus specifications. |
| Phase 12.1 — OPENALEX QUERY VALIDATION | Validated candidate queries against OpenAlex syntax. Discovered v1.0 `|` without phrase matching caused massive over-recall (484k+). Superseded with `v1.1` specification (`docs/corpus_spec_v1.1.json`) using explicit Boolean syntax: `"retrieval-augmented generation" AND (hallucination OR faithfulness OR grounding OR factuality)`, achieving an optimal focused corpus size of ~11,475. |
| Phase 13 — OPENALEX CORPUS ACQUISITION + VALIDATION | Started corpus acquisition against `corpus_spec_v1.1.json` targeting the production PostgreSQL database. OpenAlex API returned HTTP 429 Too Many Requests with an 8-hour `Retry-After` block. Acquisition was gracefully aborted, producing `docs/corpus_acquisition_report_v1.json` with status **BLOCKED**. The dataset is NOT acquired, NOT validated, and NOT frozen. |
| Phase 13A — OPENALEX SNAPSHOT ACQUISITION STRATEGY | Designed the local Snapshot Acquisition Strategy. Identified a critical query-semantic incompatibility: the API searches full-text indexing, while the Snapshot only contains Title and Abstract. The strategy recommends formally adjusting the Corpus Specification to "Title and Abstract" only to preserve scientific reproducibility. A streaming `.gz` pipeline with `ingestion_checkpoint.json` resumability was designed and successfully tested against a small fixture (`backend/tests/ingestion/test_snapshot_strategy.py`). |
| Phase 13B — CORPUS SPECIFICATION v1.2 + RESOURCE GATE | Formally upgraded Corpus Specification to v1.2 (`docs/corpus_spec_v1.2.json`), explicitly limiting search matching to Title and Abstract. Evaluated OpenAlex Works manifest (2026-09-23 snapshot), identifying 659 GB total compressed size across 2040 partitions. Confirmed local disk resources (360 GB free on D: drive) cannot support full download. Designed the partition-streaming strategy (download temp `.gz`, process, delete), avoiding full storage constraints. Verified v1.2 semantic local test fixtures. |
| Phase 13C — OPENALEX SNAPSHOT STREAMING ACQUISITION | Started streaming acquisition but halted due to execution time constraints and the discovery of an unsafe `updated_date` pruning assumption. Status marked as INCOMPLETE. |
| Phase 13C.1 — ACQUISITION CORRECTNESS REPAIR | Removed unsafe `updated_date` pruning. Added `is_xpac=false` to explicitly match OpenAlex API core behavior, formalizing Corpus Specification v1.2.1. Validated database integrity (0 records inserted). Designed new per-partition checkpoint architecture (`ingestion_checkpoint_v13C.1.json`). Benchmarked transfer methods (HTTPS vs S3 anonymous). Fully tested specification changes. |
| Phase 13D — CLOUD SNAPSHOT ACQ. FEASIBILITY + COST GATE | Designed an ephemeral AWS EC2 (c7g.4xlarge) architecture to process the 659 GB snapshot using 16 parallel multiprocessing workers in the `us-east-1` region. Estimated runtime drops from ~67 hours locally to <2 hours. Total estimated cost is <$2.00. The job produces a compact `JSONL.gz` artifact (~5MB) for SCP transfer back to the local database, completely bypassing long local HTTP bottlenecks. |
| Phase 14 — REAL-DATA DEMO ONLY | Generated a prototype corpus of 20 real ArXiv papers on RAG hallucinations. Modified the `demo_service.py` to ingest these real papers into an in-memory test DB, mapping them across temporal batches. Fully demonstrated the Research Pipeline (Discovery -> Grounding -> Evolution) on real text with deterministic state transitions. Verified backend test suites and frontend build. Produced `DEMO_GUIDE.md` for local presentations. |

---

## In Progress

Nothing is currently in progress. Phase 14 is complete. Ready for Cloud EC2 deployment and actual full acquisition.

---

## Planned

| Phase | Description | Dependencies |
|-------|-------------|-------------|
| Phase 1 | OpenAlex ingestion | Phase 0 ✅ |
| Phase 2 | Paper processing | Phase 1 |
| Phase 3 | Structured information extraction | Phase 2 |
| Phase 4 | Embeddings and semantic representation | Phase 3 |
| Phase 5 | Opportunity discovery | Phase 4 |
| Phase 6 | Opportunity model and lifecycle | Phase 3 |
| Phase 7 | Temporal tracking | Phases 5, 6 |
| Phase 8 | Evidence / RAG explanation | Phase 7 |
| Phase 9 | Evaluation | Phase 8 |
| Phase 10 | API / dashboard integration | Phase 9 |

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details.

---

## Not Implemented

The following components are **designed** (documented in the project specification) but have **zero implementation** in the repository:

- Paper processing pipeline
- LLM structured extraction
- SPECTER2 embeddings
- Sentence-transformer embeddings
- FAISS vector index
- HDBSCAN clustering
- Opportunity discovery
- Opportunity model / state machine
- Opportunity lifecycle management
- Temporal tracking
- Evidence/RAG explanation
- Evaluation framework
- FastAPI application
- Database connection / PostgreSQL integration
- Database tables / models
- Frontend (React / Next.js)
- `IDEAGEN_DATABASE_URL` configuration

---

## Known Risks

| Risk | Description | Mitigation |
|------|-------------|-----------|
| Corpus size uncertainty | Expected ~2,000–4,000 works but exact count requires OpenAlex validation | Validate metadata before full retrieval |
| Open design decisions | Several design items remain open (state transition rules, confidence formula, specific models, evaluation metrics) | Resolve before implementing the corresponding phase |
| Phase 1 definition conflict | Existing `06_STATE_AND_SESSION.md` defines Phase 1 as "Database Models and Schema"; authoritative spec defines it as "OpenAlex Ingestion" | Authoritative specification takes precedence |
| Embedding model availability | SPECTER2 and sentence-transformer model availability/compatibility with Python 3.11+ | Verify during Phase 4 implementation |

---

## Open Decisions

| Decision | Related Phase | Document |
|----------|--------------|----------|
| Exact OpenAlex query expression | Phase 1 | [CORPUS_SCOPE.md](CORPUS_SCOPE.md) |
| OpenAlex API interaction approach | Phase 1 | [INGESTION_PLAN.md](INGESTION_PLAN.md) |
| Full-text vs. abstract-only processing | Phase 2 | [EXTRACTION_PLAN.md](EXTRACTION_PLAN.md) |
| Specific LLM model version | Phase 3 | [EXTRACTION_PLAN.md](EXTRACTION_PLAN.md) |
| Prompt templates / extraction schema | Phase 3 | [EXTRACTION_PLAN.md](EXTRACTION_PLAN.md) |
| Specific sentence-transformer model | Phase 4 | [EMBEDDING_STRATEGY.md](EMBEDDING_STRATEGY.md) |
| Re-embedding protocol | Phase 4 | [EMBEDDING_STRATEGY.md](EMBEDDING_STRATEGY.md) |
| Complete state transition rule table | Phase 6 | [MODEL_DESIGN.md](MODEL_DESIGN.md) |
| Confidence formula | Phase 6 | [MODEL_DESIGN.md](MODEL_DESIGN.md) |
| Near-duplicate opportunity merging | Phase 5 | [OPPORTUNITY_LIFECYCLE.md](OPPORTUNITY_LIFECYCLE.md) |
| Evaluation metrics | Phase 9 | [EVALUATION_PLAN.md](EVALUATION_PLAN.md) |
| Baseline selection | Phase 9 | [EVALUATION_PLAN.md](EVALUATION_PLAN.md) |
| Dashboard requirements | Phase 10 | [EVALUATION_PLAN.md](EVALUATION_PLAN.md) |

---

## Installed Dependencies

| Package | Version | Phase |
|---------|---------|-------|
| pydantic | 2.13.5 | Phase 0 |
| pydantic-settings | 2.15.0 | Phase 0 |
| pytest | 9.1.1 | Phase 0 (dev) |
| pytest-cov | 7.1.0 | Phase 0 (dev) |

No additional dependencies have been installed beyond Phase 0.
