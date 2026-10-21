# IdeaGen Stage 6 — Backend Integration Audit (Revised)

- **Original audit:** 2026-10-21, before the Stage 6A–6G integration work
- **Revision:** 2026-10-21, re-verified against commit `17895a2` (`feat: add OpenAlex ingestion, incremental evolution, read-only explorer API and frontend`)
- **Scope:** FastAPI routes in `backend/ideagen/api/main.py`, the services they call, and the automated tests that exercise them.

> **Revision note.** The original Stage 6 audit was written before the OpenAlex search/ingestion routes and
> the post-ingestion evolution gate existed. Its headline finding ("no HTTP API endpoints expose OpenAlex
> search or OpenAlex paper ingestion") is **superseded**. Its test summary ("312 passed") covered only
> three test directories, and its statement that database constraints function "across SQLite and
> PostgreSQL" was **not supported by any PostgreSQL run**. This revision reports only behaviour verified
> in the current code and test runs. The original findings are summarised in §7 with their current status.

---

## 1. Summary

- The API registers **48 routes** (48 unique method-and-path pairs: 36 GET, 12 POST) served by **27 handler
  functions**. 19 handlers are registered under more than one path, accounting for 21 alias registrations
  (mostly `/api` vs `/api/v1`, plus forms such as `/health`, `/api/v1/run_metadata` and
  `/generate_opportunity`); 8 handlers have a single path. FastAPI's built-in `/openapi.json`, `/docs`,
  `/docs/oauth2-redirect` and `/redoc` are excluded. Counts were taken from the `@app.<method>("...")`
  decorators in `main.py` and confirmed against FastAPI's runtime `app.routes` (`APIRoute` objects).
  All 27 handlers are listed in §2.
- OpenAlex search (`GET /api/v1/search/openalex`) and OpenAlex ingestion
  (`POST /api/v1/papers/ingest/openalex`) are now routed and covered by SQLite-backed API tests with all
  network access mocked.
- Post-ingestion evolution is wired into OpenAlex ingestion only. It is **off by default**, can be turned
  on only by the server setting `IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION`, and classifies at most
  `IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES` (default 20) eligible opportunities per paper.
  Manual PDF upload (`POST /api/papers/upload`) never triggers evolution.
- **Not implemented:** authentication, authorization, users or workspaces. CORS allows all origins,
  methods and headers (`main.py:89-94`). The name `WorkspaceStatsDTO` refers to global corpus counts, not
  to a workspace model.
- **Not verified:** PostgreSQL behaviour (all PostgreSQL-gated tests were skipped), live LLM/OpenAlex/PDF
  behaviour (mocked in tests), and Python 3.11–3.13 at runtime.

---

## 2. Route Inventory (verified in `backend/ideagen/api/main.py` at `17895a2`)

"LLM" = calls the configured language model synchronously inside the request; "Net" = outbound network
call; "Writes" = persists to the database. Test files are those that call the route path.

| Handler (line) | Method | Paths | Side effects | Exercised by (SQLite unless noted) |
|---|---|---|---|---|
| `health_check` (102) | GET | `/api/health`, `/api/v1/health`, `/health` | `SELECT 1` | `db/test_session_and_foundation.py` — **PostgreSQL-gated, skipped** |
| `get_workspace_stats` (116) | GET | `/api/v1/stats`, `/api/stats` | read-only counts | `api/test_phase6_api.py` |
| `get_run_metadata` (183) | GET | `/api/v1/run-metadata`, `/api/v1/run_metadata`, `/api/run-metadata` | read-only (`experiment_metadata`, counts) | `api/test_read_only_contracts.py` |
| `list_papers` (273) | GET | `/api/v1/papers`, `/api/papers` | read-only | API suites |
| `get_paper_details` (316) | GET | `/api/v1/papers/{paper_id}` | read-only | API suites |
| `get_document_passages` (358) | GET | `/api/v1/documents/{id}/passages`, `/api/documents/{id}/passages` | read-only | API suites |
| `understand_paper` (379) | POST | `/api/v1/papers/{id}/understand`, `/api/papers/{id}/understand` | LLM, Writes | `api/test_stage6d_*`, `api/test_stage6f_*`; `understanding/test_understanding_api.py` is PostgreSQL-gated |
| `get_paper_knowledge` (393) | GET | `/api/v1/papers/{id}/knowledge`, `/api/papers/{id}/knowledge` | read-only | API suites |
| `get_paper_statements` (405) | GET | `/api/v1/papers/{id}/statements`, `/api/papers/{id}/statements` | read-only | `api/test_read_only_contracts.py` |
| `search_statements` (432) | GET | `/api/v1/statements` | read-only (limit 100) | `api/test_read_only_contracts.py` |
| `discover_research_gaps` (474) | POST | `/api/v1/gaps/discover`, `/api/gaps/discover` | Embeddings, LLM, Writes | `api/test_phase6_api.py`, `api/test_stage6d_*`, `api/test_stage6f_*` |
| `list_research_gaps` (489) | GET | `/api/v1/gaps`, `/api/gaps` | read-only | API suites |
| `get_research_gap` (498) | GET | `/api/v1/gaps/{gap_id}`, `/api/gaps/{gap_id}` | read-only | API suites |
| `generate_opportunity_from_gap` (530) | POST | `/api/v1/gaps/{id}/opportunity`, `/api/v1/gaps/{id}/generate_opportunity` | Embeddings, LLM, Writes; returns 400/404, and 500 on pipeline failure (no fabricated fallback) | `api/test_phase6_api.py`, `api/test_read_only_contracts.py`, `api/test_stage6d_*` |
| `list_opportunities` (611) | GET | `/api/v1/opportunities`, `/api/opportunities` | read-only | `api/test_read_only_contracts.py` |
| `get_opportunity_report` (647) | GET | `/api/v1/opportunities/{id}/report`, `/api/opportunities/{id}/report` | read-only; unrecorded assessments returned as `null` | `api/test_phase6_api.py`, `api/test_read_only_contracts.py` |
| `evolve_opportunity` (697) | POST | `/api/v1/opportunities/{id}/evolve`, `/api/opportunities/{id}/evolve` | LLM per eligible paper, Writes | `api/test_evolution_api.py` |
| `get_opportunity_history` (850) | GET | `/api/v1/opportunities/{id}/history`, `/api/opportunities/{id}/history` | read-only | `api/test_evolution_api.py`, `api/test_read_only_contracts.py`, `api/test_stage6d_*` |
| `get_opportunity_evidence` (881) | GET | `/api/v1/opportunities/{id}/evidence`, `/api/opportunities/{id}/evidence` | read-only; recomputes `verified` per row | `api/test_evolution_api.py`, `api/test_read_only_contracts.py`, `api/test_stage6d_*` |
| `get_evidence_detail` (966) | GET | `/api/v1/evidence/{statement_id}` | read-only | API suites |
| `ingest_openalex_paper` (997) | POST | `/api/v1/papers/ingest/openalex`, `/api/papers/ingest/openalex` | Net (OpenAlex, PDF), Writes; optional server-gated evolution (LLM); returns 400/404/500/502 | `api/test_openalex_ingest_api.py`, `api/test_stage6c_*`, `api/test_stage6d_*`, `api/test_stage6f_*` |
| `upload_papers` (1192) | POST | `/api/papers/upload` | Writes (PDF extraction); no evolution | `api/test_upload.py`, others |
| `run_demo_scenario` (1299) | GET | `/api/demo/scenario` | in-memory demo DB; scripted classifications | `api/test_demo.py` |
| `get_demo_opportunities` (1304) | GET | `/api/demo/opportunities` | in-memory demo DB; scripted | `api/test_demo.py` |
| `search_papers` (1318) | GET | `/api/search/papers` | Net (arXiv) | `api/test_search.py` |
| `search_openalex_papers` (1326) | GET | `/api/v1/search/openalex`, `/api/search/openalex` | Net (OpenAlex) | `api/test_openalex_search_api.py`, `api/test_stage6d_*`, `api/test_stage6f_*` |
| `analyze_search_papers` (1426) | POST | `/api/search/analyze` | runs the scripted demo pipeline on supplied papers | `api/test_search.py` |

---

## 3. Capability Status

| Capability | Status | Basis |
|---|---|---|
| Paper search | **Implemented; tested with mocked network.** OpenAlex search with strict ID/DOI routing; arXiv search retained at `/api/search/papers` | routes above; `ingestion/test_openalex_search.py`, `api/test_openalex_search_api.py` |
| OpenAlex ingestion with open-access PDF download | **Implemented; tested with mocked network.** SSRF-hardened downloader (global-unicast-only destinations, DNS pinning, redirect re-validation, size and `%PDF-` checks); metadata-only fallback when no PDF | `ingestion/openalex_ingestion.py`, `ingestion/downloader.py`; `ingestion/test_openalex_ingestion.py`, `ingestion/test_downloader_hardening.py` |
| Paper/document/passage persistence | **Implemented; tested on SQLite** | `db/persistence.py`; API and ingestion suites |
| Statement extraction | **Implemented; tested with fake LLM providers on SQLite** | `understanding/`; API suites (PostgreSQL-gated tests skipped) |
| Gap discovery | **Implemented; tested with fake providers on SQLite** | `discovery/`; `api/test_phase6_api.py`, `discovery/` tests |
| Opportunity generation and report | **Implemented; tested with fake providers on SQLite** | `generation/`, `reporting/`; API suites |
| Opportunity evolution (`/evolve`) | **Implemented; tested on SQLite.** Lower temporal bound: papers must be published on or after the opportunity's creation date; optional upper cutoff | `api/test_evolution_api.py` |
| Post-ingestion evolution | **Implemented; off by default; server-gated; capped** | `api/test_stage6c_*`, `opportunity/test_incremental_service.py` |
| Evidence and history retrieval | **Implemented; read-only; tested on SQLite** | `api/test_read_only_contracts.py` |
| Temporal cutoffs | **Implemented; tested on SQLite** (discovery, generation, evolution) | `discovery/test_temporal_filtering.py`, `ingestion/test_incremental_e2e_workflow.py` |
| Authentication / authorization / workspaces | **Not implemented** | no models, middleware or dependencies exist |
| PostgreSQL deployment | **Not verified** | PostgreSQL-gated tests skipped (no `IDEAGEN_DATABASE_URL`) |

---

## 4. End-to-End Workflows Exercised by Tests

These are exercised by `api/test_stage6d_end_to_end_workflow.py` with fake LLM/embedding providers and
mocked OpenAlex/PDF responses on in-memory SQLite. They demonstrate integration of the code paths, not
behaviour against live services or real model output.

1. Search OpenAlex → ingest a paper → understand it → discover gaps → generate an opportunity.
2. Ingest a later paper with evolution enabled server-side → evaluate existing opportunities → persist
   evidence and history → read them back through the read-only endpoints.
3. Failure paths: no accessible PDF, duplicate ingestion, invalid identifiers, upstream provider error,
   undated or temporally ineligible papers, grounding rejection, evolution failure.

Each workflow still requires several sequential client calls; there is no composite orchestration endpoint.

---

## 5. Test Evidence

Commands run from the repository root (outbound network blocked during runs):

```
PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/pytest -p no:cacheprovider -q backend/tests \
  --deselect tests/test_ollama_provider.py::test_live_ollama_smoke \
  --deselect tests/understanding/test_understanding.py::test_real_model_inference_or_integration
PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/pytest -p no:cacheprovider -q backend/tests/api
```

| Run (2026-10-21, Python 3.14.8, SQLite) | Result |
|---|---|
| Full backend suite (content of `17895a2`) | 689 passed, 0 failed, 24 skipped, 2 deselected |
| `backend/tests/api` | 104 passed, 0 failed |

All 24 skips require `IDEAGEN_DATABASE_URL` (PostgreSQL). The 2 deselected tests call live models.

---

## 6. Remaining Gaps

| Severity | Gap | Note |
|---|---|---|
| High | No authentication or authorization; CORS `*` | Write and LLM-invoking routes are open. Post-ingestion evolution is mitigated by the server gate and cap; `/understand`, `/gaps/discover`, generation and `/evolve` are not. |
| High | PostgreSQL behaviour unverified | Alembic upgrade on PostgreSQL, `SELECT ... FOR UPDATE`, savepoint semantics and `/health` are untested here. |
| Medium | Synchronous LLM work in request handlers | No background job model; long requests are expected with local inference. |
| Medium | Generation failure returns 500 | 502 would distinguish upstream model failure from server error. |
| Low | arXiv and OpenAlex search coexist | `/api/search/papers` remains arXiv; ingestion is OpenAlex-based. |
| Low | No composite workflow endpoint | Understanding, discovery and generation are separate calls. |
| Low | Demo routes remain | Responses include a `data_notice` stating that classifications are scripted. |

---

## 7. Original Findings and Current Status

| Original finding | Current status |
|---|---|
| HIGH: no OpenAlex ingestion route | **Resolved** — `POST /api/v1/papers/ingest/openalex` |
| MEDIUM: post-ingestion evolution unrouted | **Resolved with changes** — wired into OpenAlex ingestion; server-gated (default off) and capped; manual upload does not evolve |
| MEDIUM: arXiv vs OpenAlex search divergence | **Partially resolved** — OpenAlex search route added; arXiv route retained |
| LOW: no composite discovery route | **Still open** |
| LOW: no multi-tenancy / authorization | **Still open** (severity raised here because write/LLM routes are unauthenticated) |
| "312 passed, 9 skipped" as full-pipeline evidence | **Corrected** — that run covered `tests/opportunity`, `tests/evaluation` and `tests/ingestion` only; current full-suite result is in §5 |
| Constraints verified "across SQLite and PostgreSQL" | **Corrected** — only SQLite was exercised |
| "Fully verified" read-only retrieval chain | **Qualified** — verified by SQLite tests with fake providers; not against live services or PostgreSQL |
