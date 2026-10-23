# Current Status

> **2026-10-23 — reliability audit:** see [`RELIABILITY_AUDIT_2026-10-23.md`](RELIABILITY_AUDIT_2026-10-23.md).
> Projects are now server-side records (AD-24); research runs execute INGESTION → UNDERSTANDING →
> GAP_DISCOVERY and pause at AWAITING_REVIEW, never generating opportunities (AD-25); gap discovery and
> prior-art retrieval are project-scoped (AD-26); proxy failures are diagnosed, not bypassed (AD-27).
> Verified: 818 passed / 25 skipped (backend, temporary databases), frontend tsc/oxlint/build clean,
> headless-browser E2E 29/29, and real-provider runs (OpenAlex, qwen3:8b, nomic-embed-text) for search →
> add → run → review → explicit opportunity, failure and interruption/retry. Not verified: PostgreSQL.
> Not implemented: human gap accept/reject, server-side project rename/delete.
> The sections below predate the audit and were not re-verified unless restated in the audit report.

> **Last updated:** 2026-10-23
> Facts below were checked against the source code, test runs and experiment artifacts on this date.
> Verification limits are stated explicitly; anything not listed as executed was not executed.

---

## Current Phase

**Phase 5E bounded live discovery complete and audited; backend pipeline verified with local Ollama.**
The evidence-grounded pipeline (ingestion → relevance screening → extraction → gap discovery →
opportunity generation → temporal evolution) is implemented and verified end to end with real local models
(`nomic-embed-text` and `qwen3:8b`). Managed Ollama subprocess reliability has been hardened to eliminate
pipe buffer deadlocks (AD-28). Next step: frontend demonstration workflow and final integration on top
of the existing verified API.

---

## Current Implementation

| Area | Status | Evidence |
|------|--------|----------|
| Backend stack | FastAPI app, SQLAlchemy models, Pydantic schemas | `backend/ideagen/api/main.py`, `api/schema.py`, `db/models.py` |
| Database | PostgreSQL is the intended application store (Alembic chain, 11 revisions, head `e7b3c9a14f52`). Experiments and tests use SQLite via `create_all` | `backend/alembic/versions/`, `experiments/runner.py` |
| LLM / embeddings & lifecycle | Ollama `qwen3:8b` and `nomic-embed-text` with automated lifecycle manager (`OllamaLifecycleManager`): auto-start, external-server reuse without claimed ownership, bounded readiness polling, lease-based reference counting, safe idle shutdown, process ownership verification, conservative app shutdown, and deadlock-free log file / DEVNULL stderr redirection with safe handle cleanup (AD-28) | `ideagen/llm/ollama_lifecycle.py`, `ideagen/llm/ollama_provider.py`, `ideagen/discovery/embeddings.py` |
| PDF ingestion | PyMuPDF extraction, section detection, deterministic passage chunking, SHA-256 content-hash deduplication; no fabricated publication dates | `ideagen/ingestion/pdf.py`, `ideagen/ingestion/passage_chunker.py` |
| OpenAlex search and ingestion | Search API with strict ID/DOI query routing; ingest endpoint with SSRF-hardened open-access PDF downloader (global-unicast-only destinations, DNS pinning, redirect validation, size and signature checks) and metadata-only fallback | `ingestion/search.py`, `ingestion/openalex_ingestion.py`, `ingestion/downloader.py` |
| Relevance screening | Deterministic embedding ranking of pre-cutoff papers (no LLM); decisions recorded and reused only when the input signature (including the scored abstract) matches | `ideagen/discovery/screening.py` |
| Statement extraction | LLM extraction with passage provenance and bounded passage selection | `ideagen/understanding/` |
| Gap discovery | Agglomerative clustering (cosine), limitation eligibility filtering, fail-closed citation validation, optional temporal cutoff | `ideagen/discovery/` |
| Opportunity generation | Generation, prior-art assessment and refinement; failures return an error (no fabricated fallback). Reports return `null` for assessments that were not recorded | `ideagen/generation/`, `ideagen/reporting/report_service.py` |
| Lifecycle and evolution | Deterministic state machine (OPEN / STRENGTHENED / PARTIALLY_ADDRESSED / RESOLVED); LLM classifies, code transitions; grounding fails closed; unique (opportunity, paper) evidence; `/evolve` enforces a lower temporal bound (papers must be published on or after the opportunity's creation date) | `ideagen/opportunity/` |
| Post-ingestion evolution | Optional, **off by default**; a request cannot enable it unless the server setting `IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION` is on; at most `IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES` (default 20) eligible opportunities classified per paper | `api/main.py`, `opportunity/incremental.py` |
| Classifier prompt | Version `v2` (revised ADDRESSES/EXTENDS definitions); recorded in the runner configuration fingerprint | `ideagen/opportunity/classifier.py` |
| Controlled experiment runner | SQLite isolation, configuration fingerprint and corpus hashes, resume refusal on mismatch, Stage 3 completion marker, bounded Stage 5 classifier input, explicit Stage 5 commits | `experiments/runner.py` |
| Evaluation code | Pairwise clustering metrics, TF-IDF discovery baseline, Stage 4/4B/4D classifier evaluation harnesses (evaluation-only; not used by production paths) | `ideagen/evaluation/` |
| Frontend | React 19 + Vite + TypeScript read-only explorer (hash routing): run overview, opportunities, opportunity detail with evidence timeline, corpus inspector, statement provenance drawer. Missing values shown as "Not recorded"; verification labels follow the backend `verified` flag | `frontend/src/` |
| Demo endpoints | `/api/demo/*` use scripted classifications and synthetic embeddings; responses carry a `data_notice` stating this | `ideagen/api/demo_service.py` |

---

## Verified Results

### Automated tests (2026-10-23, local macOS, Python 3.14.8 virtualenv)

Two verified test-run configurations were audited against the backend test suite:

1. **Standard Unfiltered Run** (`.venv/bin/pytest backend/tests`):
   - **841 passed, 26 skipped, 0 failed, 0 deselected** (23.59 s).
   - **Skip breakdown**: 25 skips are PostgreSQL-dependent (requiring `IDEAGEN_DATABASE_URL`); 1 skip (`tests/test_ollama_provider.py::test_live_ollama_smoke`) requires a reachable local Ollama HTTP service (`http://localhost:11434`).
   - *Database note*: PostgreSQL integration has not been verified in this environment because PostgreSQL-dependent tests were skipped; passing offline SQLite tests does not establish PostgreSQL correctness. These counts are environment-specific and should be updated after future test runs.

2. **Deselected Live-Model Run** (used during isolated offline audit):
   ```bash
   PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/pytest -p no:cacheprovider -q backend/tests \
     --deselect tests/test_ollama_provider.py::test_live_ollama_smoke \
     --deselect tests/understanding/test_understanding.py::test_real_model_inference_or_integration
   ```
   - **689 passed, 24 skipped, 0 failed, 2 deselected**.
   - Deselecting `test_live_ollama_smoke` and `test_real_model_inference_or_integration` via command-line flags removes them from test execution, shifting their status from skipped to deselected and reducing the skip count from 26 to 24. Outbound network was blocked during the run.

- Frontend: `tsc -p tsconfig.app.json --noEmit` → 0 errors; `oxlint src` → 0 errors, 7 warnings.

### Controlled five-paper temporal experiment (2026-10-19)

Artifacts: `experiments/artifacts/five_paper_temporal_compact_20261019/`; record:
`docs/experiments/five_paper_temporal_experiment_20261019.md`.
5 papers, 502 passages, 63 statements, 1 validated gap, 1 opportunity, 3 evaluation batches,
4 evolution events, 0 status transitions, 464.58 s wall time. This run used classifier prompt v1.
It demonstrates pipeline integrity on a tiny corpus; it does not establish scientific accuracy.

### Relationship classifier evaluation (qwen3:8b, 25 hand-labelled cases)

| Run | Overall | DEV (10) | HELD_OUT (15) | Real-literature subset (11) |
|-----|---------|----------|---------------|------------------------------|
| Prompt v1 (`stage4d_live_20261020_180110`) | 0.56 | 0.50 | 0.60 | 0.18 |
| Prompt v2 (`stage4e_prompt_v2_20261021`) | 0.68 | 0.60 | 0.73 | 0.64 |

Caveats: labels are single-annotator; prompt v2 was revised after inspecting v1 errors on these same
25 cases, so its figures are not a held-out estimate; the sample is too small for statistical claims.

### Phase 5E bounded live discovery experiment (2026-10-23)

Artifacts: `phase5e_results.json`, `run_phase5e_server.py`, `run_phase5e_client.py` in session scratch; server log `task-205.log`; client log `task-209.log`; isolated DB `ideagen_demo_isolated_phase5e.db`.

- **Endpoint & Timing**: `POST /api/v1/gaps/discover?project_id=rag-benchmark` completed with HTTP 200 OK in **21.85 seconds** (against a 240.0s client timeout budget and 120.0s internal Ollama timeout budget; AD-29).
- **Embeddings & Clustering**: 127 research statements embedded using local `nomic-embed-text` via `OllamaEmbeddingProvider`. 24 semantic clusters formed, producing 14 candidate gaps.
- **Bounded Validation**: The test harness intentionally bounded candidate validation to exactly one candidate (`candidates[:1]`; AD-30) to test end-to-end connectivity, timeout propagation, and persistence without unbounded queue latency.
- **Model Evaluation**: `qwen3:8b` completed structured candidate validation in **19.968 seconds**, returning a `VALID` verdict (`status: VALIDATED`, `confidence: None`, reasoning citing two specific passages).
- **Persistence & Evidence Provenance**: Persisted exactly one validated gap (`4cc867d5e6ce449d8992a9b63539161b`) and two supporting evidence links (`eaf8a50c...`, `09ec9a2e...`) referencing verified database statements (`0c30c759...`, `7354dd77...`). Invariant audit verified zero dangling foreign keys and strict provenance adherence.
- **Database & Baseline Isolation**: Executed against an isolated SQLite clone (`ideagen_demo_isolated_phase5e.db`, SHA-256 `ed06beccf638dcf1a5890f684ab638f6fe0a6b45a25962882fa981173ce6b480`; AD-31). Upstream databases remained bit-for-bit identical to baseline checksums: Master DB (`a70ff8a5...`), Phase 4C (`87acabb0...`), Phase 5B (`7528deb0...`), and Phase 5D (`87acabb0...`).
- **Subprocess Reliability Fix**: Managed Ollama lifecycle daemon (`ideagen/llm/ollama_lifecycle.py`) was updated to route subprocess stderr to a managed log file (with `subprocess.DEVNULL` fallback) and guaranteed file-handle closure upon process exit or failure, eliminating Darwin kernel 16 KB pipe buffer deadlocks (AD-28).
- **Focused Test Suite**: 101 passed, 2 skipped (live model smoke tests skipped in offline run) across `backend/tests/test_ollama_lifecycle.py` and `backend/tests/test_ollama_provider.py`.
- **Scope & Generalization Limits**: This run was explicitly bounded (1 candidate evaluated out of 14) and isolated in SQLite; it verifies end-to-end pipeline wiring, timeout resilience, and persistence mechanics under local Ollama. It does not establish multi-candidate throughput under full load, nor does it make general claims of universal scientific novelty for discovered gaps.

---

## Not Verified / Not Executed

- **PostgreSQL**: no PostgreSQL instance was used; PostgreSQL-specific behaviour (Alembic upgrade on
  Postgres, `SELECT ... FOR UPDATE`, savepoint semantics) is untested in this environment.
- **Python 3.11–3.13**: no such interpreter was available. A static compatibility check
  (annotation names, 3.12-only f-string syntax, PEP 695 syntax) reported no issues; this is not a
  substitute for running the suite on 3.11.
- **Live services in standard test suite**: Ollama, Gemini, OpenAlex and PDF downloads are mocked in the standard unit test suite (live model tests such as `test_live_ollama_smoke` are skipped unless explicitly invoked against a live server; empirical validation is conducted in isolated bounded harnesses such as Phase 5D and Phase 5E).
- **Frontend runtime**: type-checked and linted only; no browser or end-to-end UI test.

---

## Not Implemented

- Authentication / authorization (API is open; CORS allows all origins)
- Full OpenAlex corpus acquisition (Phase 13 remains blocked/incomplete; no frozen corpus)
- Human-annotated ground truth for gap discovery; baseline comparisons on real data
- HDBSCAN clustering (agglomerative clustering is used instead)
- SPECTER2 / sentence-transformer embeddings as the default path (Ollama embeddings are used)
- Background job execution for long-running LLM operations (endpoints are synchronous)

---

## Known Limitations and Risks

| Item | Description |
|------|-------------|
| Synchronous LLM endpoints | Discovery, generation, `/evolve` and opt-in post-ingestion evolution run LLM calls inside the request |
| Temporal lower bound | Opportunities generated live (wall-clock `created_at`) cannot be evolved through `/evolve` with papers published before their creation; runner-generated opportunities are backdated to the baseline cutoff |
| Legacy experiment databases | Stage 5 databases from before prompt v2 / recorded abstract bounds refuse to resume (by design) |
| Address over-blocking | The downloader also blocks NAT64/6to4 forms of public hosts (fail-closed) |
| Undeclared dependency | `requests` is imported by the Ollama providers but not declared in `backend/pyproject.toml`; numpy/faiss/scikit-learn are in the optional `ml` extra |
| Remaining backlog | Opportunity generation failure returns 500 (502 would be more precise); evidence endpoint may report synthetic batch labels for papers outside configured Stage 5 windows; runner `--screening-only` has no effect without `--screening-topic` |
| Benchmark reports | `experiments/stage4*/stage5/stage6_*_report.md` contain known inaccuracies and are excluded from version control pending an evidence-based rewrite |

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

## Dependencies

Declared in `backend/pyproject.toml` (core, `ml` and `dev` extras) and `frontend/package.json`.
This file no longer duplicates version lists.

---

## Phase History (as recorded at the time)

The table below is preserved verbatim from earlier updates. Some entries are superseded:
Phase 6's HDBSCAN placeholder was replaced by agglomerative clustering; the Phase 10 dashboard and the
Phase 14 demo UI were replaced by the current read-only explorer; the Phase 14 demo uses scripted
relationship classifications, not live model evolution; Phase 13 corpus acquisition remains incomplete.
Test counts and environment claims in this table (including PostgreSQL test results) are historical
and were not re-verified on 2026-10-21; see "Verified Results" above for the current evidence.

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
| Phase 5D — Bounded Live-Model Validation | Validated direct `qwen3:8b` and `nomic-embed-text` provider inference against controlled benchmarks. Structured output and citation constraints verified. |
| Phase 5E — Bounded Live Discovery API Test | Successfully verified bounded live discovery API (`POST /api/v1/gaps/discover?project_id=rag-benchmark`) with local Ollama (`qwen3:8b` + `nomic-embed-text`) against an isolated SQLite database clone. Resolved Ollama daemon subprocess stderr pipe buffer deadlock via dedicated log file redirection with safe handle cleanup (AD-28). 127 statements embedded, 24 clusters formed, exactly 1 candidate validated (HTTP 200 in 21.85s). Evidence provenance and database integrity audits passed; baseline checksums unchanged. Focused unit tests: 101 passed, 2 skipped. |
