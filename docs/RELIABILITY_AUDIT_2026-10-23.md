# IdeaGen Reliability Audit — 2026-10-23

Branch `feat/ollama-provider-integration` at HEAD `945645a`, with a large pre-existing uncommitted working tree
(user work) that was preserved. Nothing was committed, stashed, reset or reverted. Evidence artifacts:
`docs/audit/2026-10-23/` (screenshots, browser E2E script and its output).

---

## A. Executive summary

**Health before the audit.** Builds and the offline suite were green (777 passed / 25 skipped), but the core
workflow was broken. A project created in the UI existed only in the browser, so the backend could not attach
papers to it. When a research run discovered gaps, it read **every project's statements** and labelled the
results as the running project's gaps. Statement extraction inside runs crashed on every paper. Opportunities had
been generated automatically inside an ordinary run. The UI showed hardcoded numbers and health states, and 213 of
397 CSS classes used by components did not exist. `make test` ran pytest against the demo database.

**Root causes (proven).**

| # | Root cause | Symptom it explains |
|---|---|---|
| 1 | Backend started from the Antigravity sandbox inherited `HTTP(S)_PROXY=127.0.0.1:57643`, a policy proxy | BUG-001 OpenAlex "not allowed by policy"; Ollama "Port 11434 not allowed for HTTP" (stored run log) |
| 2 | Projects created only in `localStorage`; `GET /projects/{id}` fabricated projects | BUG-003 paper "added" but corpus empty; 5 cybersecurity papers orphaned in the demo DB |
| 3 | Run-time gap discovery queried all statements, then stamped gaps with the run's project | Cross-project leakage (BUG-005 / security) |
| 4 | `UNDERSTANDING` read a non-existent `ResearchKnowledgeResponseDTO.statements` | Every run paper needing extraction ended FAILED (found by the real-model run) |
| 5 | Uncommitted App.css rewrite dropped 125 classes; new components used 88 undefined classes | BUG-004 unstyled stepper / browser-default Retry / cramped modal / stray "1" |
| 6 | `fadeIn` animation kept `transform` (fill-mode `forwards`) | Modals rendered inside views were clipped to the content column |

**Fixed.** All six root causes and 30+ related defects (ledger below), with 41 new regression tests. Final state:
**818 passed / 25 skipped** backend; tsc, oxlint, vite build, `make test`, `make lint` all exit 0; headless-Chrome
E2E **29/29**. Real-provider runs (live OpenAlex, qwen3:8b, nomic-embed-text) covered: search, add, run,
review gate, explicit opportunity generation, failure, interruption and retry.

**Still open / not verified.**
- The running servers on :8000/:5173 were started by Antigravity and still run the **old backend code**. Restart is
  required (section H).
- PostgreSQL was not exercised (25 PostgreSQL-dependent tests skipped).
- Human accept/reject of individual gaps is **not implemented**. Gap status comes from the LLM validator; the
  researcher's control is the explicit "generate opportunity" action.
- Project deletion is **not implemented** server-side.
- **One demo-DB row was added during the session.** It was not added by tests or by any command I ran. The user's
  open browser tab hot-reloaded the new frontend and registered its local project through the old backend
  (section F).

---

## B. Defect ledger

Severity: C = critical, H = high, M = medium, L = low. "Test" points at the regression test file
(`AP` = `backend/tests/pipeline/test_audit_project_scoping.py`,
`ND` = `backend/tests/api/test_audit_network_diagnostics.py`).

### BUG-001 (H) OpenAlex outbound search failure
- **Symptom:** "Network proxy or policy blocked request to api.openalex.org … (HTTP 502)".
- **Root cause (proven):** backend PID 62241 environment has `HTTPS_PROXY/HTTP_PROXY=http://127.0.0.1:57643`
  (Antigravity). The same `OpenAlexSearchProvider` code from a clean shell returned 3 results (50,865 total, date
  filter respected). With those proxy variables it failed (TLS interception now; policy denial in the original
  report). Application logic was not at fault.
- **Related defects fixed:**
  - Any upstream body containing "gateway"/"proxy"/"firewall" was classified as a policy block and never retried,
    which included OpenAlex's own `502 Bad Gateway`.
  - Raw upstream bodies were echoed to clients.
- **Fix:**
  - `ingestion/search.py` classifies proxy denials by explicit policy phrases.
  - Upstream bodies are truncated in logs and never returned (OpenAlex's JSON `message` is surfaced for 400s).
  - New `core/network.py` adds a hint naming the redacted proxy and how to start without it.
  - The Makefile warns when proxy variables are set.
  - No bypass of the proxy (AD-27).
- **Tests:** `ND` (9 tests); existing `test_openalex_search_api_failures.py` still passes.
- **Remaining:** restart the backend from a regular terminal.

### BUG-002 (M) False "API Offline" / Vite proxy permissions
- **Verified:**
  - The Vite proxy targets `127.0.0.1:8000`.
  - The health probe uses `cache: 'no-store'`.
  - EPERM is an environment restriction on the sandboxed Vite process, not an application defect.
- **Fixed:**
  - The Navbar polls every 3s while unhealthy (15s when healthy) and rechecks on focus/online, so it recovers
    without a reload.
  - It distinguishes `Database Error` (503) from `API Offline`.
  - Projects re-sync when the API returns.
  - The proxy target can be overridden with `IDEAGEN_API_PROXY_TARGET`.
  - The README documents starting outside the sandbox.
- **Test:** existing `test_health_and_proxy_contracts.py` (3 passed); browser E2E A3.

### BUG-003 (C) Paper added but absent from the project corpus
- **Root cause (proven):**
  - `projectStore.createProject` wrote only to `localStorage` (`proj_<ts>_<rand>`). The cyber project never
    existed server-side; at session start the demo DB held only `rag-benchmark`.
  - The pre-session demo DB contained **5 cybersecurity OpenAlex papers (19:16 UTC) linked to no project**, the
    exact residue of the report.
  - The current backend rejects unknown projects (404). `GET /projects/{id}` created a junk project as a side
    effect (reproduced: "Proj_1761170000000_Abcde").
- **Fix (AD-24):**
  - `POST /api/v1/projects` now stores topic/description/cutoff, is idempotent, and fills only unset fields.
  - The frontend creates projects server-first, and on load registers browser-only projects under their
    existing IDs.
  - `GET` no longer creates anything; selections, association, preflight, runs and scoped discovery 404 for
    unknown projects.
  - The association endpoint is race-safe and reports `already_associated`.
  - The UI says "Added to Project" only when the server reports `project_associated`.
- **Tests:** `AP` (association, duplicate, existing-global-paper → second project, isolation, 404s, fill-unset);
  browser E2E B2–B6, C1–C2, D1–D3 (live OpenAlex).
- **Remaining:** the 5 orphaned papers are not auto-assigned (no evidence of intent). Re-adding them from Paper
  Search now associates them (the E2E did exactly this for one).

### BUG-004 (M) UI layout / visual consistency
See section G.

### BUG-005 (C) Project/research data consistency and isolation
- **Defects fixed:**
  - Run gap discovery over all projects' statements (AD-26).
  - Prior-art retrieval over all projects' statements.
  - Per-ID checks let `project_id = NULL` records pass any project scope.
  - Opportunity scoping done afterwards with raw SQL hand-formatting UUIDs (SQLite-only); `project_id` is now set
    atomically from the gap in `generation/persistence.py`.
  - `/evolve` accepted papers from other projects.
  - Evolution / Opportunity Detail / Evolve panel fetched unscoped data.
  - Workspace `paper_count` counted *selections*, not corpus papers.
  - Dashboard showed hardcoded 0 for every custom project, and fabricated text ("Stage 5 Evolution Complete · 1
    Opportunity transitioned…").
  - Project Overview showed fallbacks `|| 785`, `|| 10`, `|| 5`, "Connected & Verified" and a hardcoded model.
  - Run History showed global counts inside a project.
  - Settings hardcoded "api.openalex.org (Online)".
  - Search state saved the previous project's results under the new project's key during a switch.
  - Failed extractions counted as "analyzed".
- **Tests:** `AP` (strict scoping, evolve scope, failed-extraction semantics), browser E2E A4, D1–D3 and UI checks.

### BUG-006 (C) Pipeline stage correctness and gating
- **Defects fixed:**
  - Runs could include `OPPORTUNITY_GENERATION`, whose handler referenced an undefined `ResearchOpportunity`. The
    stored demo run shows it executed automatically. It is now rejected with 400; the handler was removed (AD-25).
  - `UNDERSTANDING` crashed on `.statements`. Found only by the real-model run; fixed via `statement_count` plus
    an `extraction_status` check.
  - Run ingestion reported network/proxy failures as "metadata not found"; the real reason is now recorded.
  - Failed `GAP_DISCOVERY` still finished as `COMPLETED`; now `FAILED` with `error_message` and the stage message
    shown in the UI.
  - `AWAITING_REVIEW` with 0 gaps; now `COMPLETED` with a "no gaps" hint.
  - A failed upstream stage plus discovered gaps is now `PARTIAL_SUCCESS` (reviewable and retryable).
  - Retry re-ran finished gap discovery (duplicate gaps) and accepted runs with nothing to retry.
  - Retry with preserved gaps now returns to `AWAITING_REVIEW`.
  - Corpus papers added via Search were never extracted by runs.
  - Failed extractions were never retried.
  - Orphaned `PENDING` runs were never recovered (30s grace).
  - Concurrent run starts raced (per-project lock; run marked active before its thread starts).
  - Gaps persisted before a mid-discovery failure were orphaned; they are now tagged with project and run at save
    time.
  - Empty project/selection runs are rejected with 400.
  - `INTERRUPTED` was not handled in the UI.
- **Tests:** `AP` (planned-stage validation, statuses, retries, recovery, corpus understanding); the updated
  `test_research_workflow.py`; real-provider Workflows E–H (section E).

### BUG-007 (H) Provider and model integration
- **Proven:** a stored run failed with `Ollama embedding API error (403): Port 11434 not allowed for HTTP`.
  Reproduced exactly with `curl -x http://127.0.0.1:57643 http://localhost:11434/api/tags`. Ollama is reachable
  directly, with `qwen3:8b` and `nomic-embed-text` installed.
- **Fixed:**
  - Ollama chat and embedding errors are actionable: endpoint, model, `ollama pull` hint on 404, proxy hint.
  - Embedding failures raise a typed `EmbeddingProviderError`.
  - Provider failures map to HTTP 502 instead of 500.
  - `POST /papers/{id}/understand` returned 200 when extraction failed (it now returns 502 with the reason).
  - New `GET /api/v1/system/status` reports configured providers, live Ollama reachability and missing models.
  - Settings, Overview and Workspaces display it instead of hardcoded names.
- **No silent fallback:** a missing provider makes runs fail visibly (verified in H4).

### BUG-008 (H) Database, migrations, test isolation
- **Proven:** the Makefile `export`s `IDEAGEN_DATABASE_URL` to the demo DB. `make test` ran pytest with it, and
  `tests/pipeline/conftest.py` and `tests/opportunity/conftest.py` run `create_all` and transactions against that
  URL.
- **Fix:**
  - `tests/conftest.py` aborts (exit 4) when the URL or `.env` points at a SQLite file under `data/`.
  - `make test` unsets the URL.
- **Tests:** `test_test_database_guard.py`, plus manual verification of absolute and relative URLs.
- **Unchanged:** Alembic head `8b2c3d4e5f6a` (already applied to the demo DB before the session); no migrations
  were added or run.

### BUG-009 (M) API contracts
- **Fixed mismatches:**
  - `RunStatus` lacked `INTERRUPTED`.
  - Log level `WARN` vs backend `WARNING`.
  - Stage failure reason sent as `message` but rendered from `error`.
  - Paper badges mapped statuses the backend never emits (`UNDERSTOOD`/`ERROR`).
  - `RunHistory` read `stage_progress.length` on an object ("N / undefined").
  - `discoverGaps` sent no `project_id`.
  - Project DTOs extended (`topic`, `temporal_cutoff`, `selection_count`, `gap_count`, …).
  - `knowledge_status` gained `EXTRACTION_FAILED`.
- See section D for the inventory.

### BUG-010 (M) Developer/runtime tooling
- **Fixed:**
  - Makefile `$(wildcard)` split the repo path on its space ("IdeaGen F"), so `make test/backend/install` used
    bare `pytest`/`uvicorn`/`pip` from PATH. `make install` could have installed into a global Python. Tools now
    resolve via a shell probe.
  - `make test` now runs the whole suite plus the frontend type check (it previously ran 3 directories while
    claiming frontend validation).
  - Proxy warning.
- **Verified:** `make stop` only kills PIDs it recorded. `make status` correctly reports :8000/:5173 as
  unmanaged.

---

## C. Feature verification matrix

Statuses: PASS (verified end to end), PARTIAL (works with stated gaps), BLOCKED (external), NOT IMPLEMENTED.
Columns: UI · API · Persistence · Isolation · Failure handling · Tests · Status.

| # | Feature | UI | API | Persist. | Isolation | Failure | Tests | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Startup & API indicator | ✔ | ✔ | n/a | n/a | ✔ 503 vs unreachable | E2E A3 | PASS |
| 2 | Dashboard metrics | ✔ | ✔ | ✔ | ✔ per-project counts | ✔ errors surfaced | E2E A4/A5 | PASS |
| 3 | Workspace/project navigation | ✔ | ✔ | ✔ | ✔ | ✔ | E2E | PASS |
| 4 | Project creation & selection | ✔ | ✔ | ✔ server | ✔ | ✔ 422/404 | AP, E2E A1/A6 | PASS |
| 5 | Project config & cutoff | ✔ create | ✔ | ✔ | ✔ | ✔ | AP | PARTIAL (no edit UI) |
| 6 | Paper search, dates, count, retry | ✔ | ✔ | none (by design) | n/a | ✔ | ND, E2E B1 (live) | PASS |
| 7 | Paper metadata normalization | ✔ | ✔ | ✔ | n/a | ✔ | existing suites | PASS |
| 8 | Add paper / duplicate / association | ✔ | ✔ | ✔ | ✔ | ✔ | AP, E2E B2–C2 | PASS |
| 9 | Corpus Inspector & counts | ✔ | ✔ | ✔ | ✔ | ✔ | E2E B4/B5 | PASS |
| 10 | Project paper selections | ✔ | ✔ | ✔ | ✔ 404 unknown | ✔ | test_research_workflow | PASS |
| 11 | Preflight | ✔ | ✔ | n/a | ✔ | ✔ empty-corpus warning | AP | PASS |
| 12 | Start research run | ✔ | ✔ | ✔ | ✔ lock | ✔ 400/409 | AP, real run | PASS |
| 13 | Stage execution & transitions | ✔ | ✔ | ✔ | ✔ | ✔ | AP, real run E | PASS |
| 14 | Progress, logs, retry, recovery | ✔ | ✔ | ✔ | ✔ | ✔ | AP, H6 real | PASS |
| 15 | Paper understanding | ✔ | ✔ | ✔ | ✔ | ✔ 502 on failure | AP, real (18 statements) | PASS |
| 16 | RAG retrieval / grounded evidence | ✔ | ✔ | ✔ | ✔ project-scoped prior art | ✔ | existing + real G | PASS |
| 17 | Gap discovery | ✔ | ✔ | ✔ | ✔ | ✔ | AP, real (2 gaps) | PASS |
| 18 | Gap evidence inspection | ✔ | ✔ | ✔ | ✔ strict | ✔ | real F | PASS |
| 18b | Human gap accept/reject | — | — | — | — | — | — | NOT IMPLEMENTED |
| 19 | Explicit opportunity generation | ✔ | ✔ | ✔ atomic scope | ✔ cross-project 404 | ✔ 502 | real G (41.6 s) | PASS |
| 20 | Opportunity report & evidence | ✔ | ✔ | ✔ | ✔ | ✔ | real G | PASS |
| 21 | Lifecycle evolution & history | ✔ | ✔ | ✔ | ✔ evolve scope | ✔ | AP; existing suites | PARTIAL (not exercised with a real model this session) |
| 22 | Settings / provider config | ✔ | ✔ | n/a | n/a | ✔ | E2E, H1 | PASS (read-only by design) |
| 23 | Health & readiness | n/a | ✔ | n/a | n/a | ✔ | existing | PASS |
| 24 | Migrations / existing DB compatibility | n/a | n/a | ✔ | n/a | n/a | backend suite | PARTIAL (SQLite only; PostgreSQL BLOCKED) |
| 25 | Accessibility / responsive / shared UI | ✔ | n/a | n/a | n/a | n/a | E2E overflow @390 on 9 views | PARTIAL (no screen-reader audit) |
| — | Project deletion | — | — | — | — | — | — | NOT IMPLEMENTED (server) |

---

## D. Architecture and API map

```
React (Vite :5173, hash router)
  projectStore (localStorage cache) ── sync ──► POST/GET /api/v1/projects   ◄─ source of truth (AD-24)
  api/client.ts ── /api/* (Vite proxy → IDEAGEN_API_PROXY_TARGET, default 127.0.0.1:8000)
FastAPI (ideagen.api.main)
  ├─ search.py ─────────► api.openalex.org  (honours HTTP(S)_PROXY / NO_PROXY; diagnosed, never bypassed)
  ├─ openalex_ingestion ► SSRF-hardened PDF download → passages
  ├─ WorkflowEngine (thread per run): INGESTION → UNDERSTANDING → GAP_DISCOVERY → AWAITING_REVIEW
  │      UNDERSTANDING / GAP_DISCOVERY / generation ─► Ollama :11434 (qwen3:8b, nomic-embed-text)
  ├─ explicit: POST /gaps/{id}/opportunity → generation + bridge (project_id from gap)
  └─ SQLAlchemy ► SQLite (demo) / PostgreSQL (target); FAISS index derived
Run statuses: PENDING → RUNNING → {AWAITING_REVIEW | COMPLETED | PARTIAL_SUCCESS | FAILED}; crash → INTERRUPTED → retry → PENDING
```

| UI feature | Client method | Route | Effects |
|---|---|---|---|
| Project list/sync/create | `listProjects`, `createProject` | `GET/POST /api/v1/projects` | `projects` row (idempotent, fills unset fields) |
| Health | `checkBackendHealth` | `GET /api/health` | none |
| Diagnostics | `getSystemStatus` | `GET /api/v1/system/status` | live Ollama `/api/tags` probe |
| Metrics | `getWorkspaceStats(p?)` | `GET /api/v1/stats[?project_id]` | read |
| Search | `searchOpenAlex` | `GET /api/v1/search/openalex` | none |
| Add paper | `ingestOpenAlex(id, doi, p)` | `POST /api/v1/papers/ingest/openalex` | `papers`, documents, passages, `project_papers` |
| Associate | `associatePaperToProject` | `POST /api/v1/projects/{p}/papers/{id}` | `project_papers` |
| Corpus | `getPapers(p)`, `getStatements` | `GET /api/v1/papers`, `/papers/{id}/statements` | read |
| Extract | `understandPaper` | `POST /api/v1/papers/{id}/understand` | `research_knowledge`, statements |
| Selections | `get/save/remove/clearProjectPaperSelections` | `/api/v1/projects/{p}/selections` | `project_paper_selections` |
| Preflight / run | `getPreflightCheck`, `createResearchRun` | `GET …/preflight`, `POST …/runs` | `research_runs`, `research_run_papers` |
| Run status/logs/retry | `getRunStatus`, `getRunLogs`, `retryRun`, `getProjectRuns` | `/api/v1/runs/{id}[/logs|/retry]` | run state |
| Gaps | `getGaps`, `getGap`, `discoverGaps(cutoff, p)` | `/api/v1/gaps[…]`, `POST /gaps/discover` | `research_gaps`, `gap_evidence` |
| Generate opportunity | `generateOpportunity(gap, cutoff, p)` | `POST /api/v1/gaps/{id}/opportunity` | `research_opportunities`, `opportunities` |
| Opportunities | `getOpportunities(p)`, `getOpportunityReport/Evidence/History(id, p)` | `/api/v1/opportunities[…]` | read |
| Evolve | `evolveOpportunity(id, body, p)` | `POST /api/v1/opportunities/{id}/evolve` | evidence, history |
| Run overview (global) | `getRunMetadata` | `GET /api/v1/run-metadata` | read (workspace-wide, labelled) |

Obsolete/legacy routes kept for compatibility (not used by the current UI): `/api/search/papers` (arXiv),
`/api/search/analyze`, `/api/demo/*`, `/api/papers/upload`.

---

## E. Test report

Backend runs below used `env -u IDEAGEN_DATABASE_URL` from `backend/`.

| Command | Exit | Result |
|---|---|---|
| Baseline `.venv/bin/python -m pytest -q -p no:cacheprovider` | 0 | 777 passed, 25 skipped |
| Final, same command | 0 | **818 passed, 25 skipped** (25 skips: PostgreSQL-dependent / live-Ollama tests) |
| `npx tsc -b --noEmit` (frontend) | 0 | — |
| `npx oxlint` | 0 | 0 errors, 24 warnings (baseline 23; same React-Compiler advisory classes) |
| `npx vite build --outDir /tmp/ideagen_audit/dist_final` | 0 | built (`frontend/dist` untouched) |
| `make test` | 0 | 818 passed, 25 skipped + tsc |
| `make lint` | 0 | — |
| `git diff --check` | 0 | — |
| Browser E2E `docs/audit/2026-10-23/e2e_browser.py` (headless Chrome, isolated backend on a DB copy, live OpenAlex) | 0 | **29 passed, 0 failed** |

New tests: `ND` (9), `AP` (30), `tests/db/test_test_database_guard.py` (2). Updated:
`test_research_workflow.py` (new contract), `test_phase6_api.py` (one assertion).

**Real-provider workflows** (isolated backend :8765 on `/tmp/ideagen_audit/e2e.db`, no proxy):
- **E (run).** Preflight plans 3 stages. `OPPORTUNITY_GENERATION` is rejected with 400. The run extracted 18
  statements with qwen3:8b and found 2 gaps. The first attempt exposed the `.statements` crash; after the fix,
  retry → `AWAITING_REVIEW`, extraction reused, no duplicate gaps.
- **F (review).** Both gaps `VALIDATED`, evidence ⊆ cyber corpus, `run_id` recorded. 0 opportunities before an
  explicit request. RAG project unchanged (2 gaps, 1 opportunity).
- **G (opportunity gate).** Cross-project generation → 404. Explicit generation → 200 in 41.6 s. Records scoped
  to the cyber project. Cross-project report/evidence → 404.
- **H (failures)**, second isolated backend :8766 with Ollama `127.0.0.1:9` and an unreachable proxy:
  - System status reports Ollama unreachable.
  - Search returns 502 naming the proxy.
  - Understand returned 200 on failure (found, fixed to 502).
  - Run → `FAILED` with the exact cause.
  - Invalid inputs → 400/422.
  - Kill -9 mid-run → `INTERRUPTED`; retry accepted, real cause recorded, no duplicates.

**Environment limits:** PostgreSQL not available. No screen-reader testing. Browser automation was via the Chrome
DevTools Protocol (no Playwright installed).

---

## F. Database safety report

- **Backup:** before any work, `sqlite3 .backup` →
  `/Users/sanket/Documents/Projects/IdeaGen/audit_backups_20261023/ideagen_demo_20261022.db.bak`
  (`integrity_check: ok`), plus row counts and SHA-256 of protected assets in the same folder.
- **Restore:** stop the backend, then
  `cp …/ideagen_demo_20261022.db.bak "data/demo/ideagen_demo_20261022.db"` (or `sqlite3 data/demo/… ".restore '…bak'"`).
- **Tests and E2E** never used the demo DB: in-memory / temporary SQLite and copies under `/tmp/ideagen_audit/`.
- **The demo DB is not byte-identical.** SHA-256 changed `7eb428f2…` → `99a8d4e4…` at 02:10 local. The only
  row-count difference is `projects 1 → 2`:
  - New row: `proj_1792695111211_gqxyw` "Adaptive AI for Zero-Day Attack Detection", `temporal_cutoff NULL`,
    created 20:40:53 UTC.
  - Cause: the user's browser tab on the Antigravity Vite (:5173) hot-reloaded the new project-sync code and
    registered its local project through the **old** backend (:8000), which ignores topic and cutoff. No test or
    command of mine wrote to the demo DB.
  - Once the restarted backend runs the new code, the browser's next sync fills the missing cutoff/topic from
    local storage without overwriting anything (`test_create_project_fills_only_unset_fields`).
  - `PRAGMA integrity_check` is ok; Alembic head is still `8b2c3d4e5f6a`.
- **Benchmark assets:** all 8 checksummed files are byte-identical (`shasum -c` OK):
  - `candidate_recall_benchmark.py`, `dense_candidate_recall_benchmark.py`, `stage4_benchmark.py`,
    `stage4b_benchmark.py`, `stage_c_embedding_cache.json`, and the 3 benchmark tests.
- **Projects:** `rag-benchmark` data is unchanged (11 papers, 2 gaps, 1 opportunity, 1 run).

---

## G. UI defect report

Screenshots: `docs/audit/2026-10-23/`.

| Defect | Cause | Fix | Status |
|---|---|---|---|
| Unstyled stepper / "stray 1" / browser-default Retry / cramped modal fields | 212 class names without CSS | New App.css sections 15b, 17–22 using current tokens; no dark styles | Fixed |
| Modal backdrop clipped to content column | `fadeIn` `forwards` retained a transform | Keyframe ends at `none`, no fill | Fixed (measured 1273×900) |
| Fabricated metrics / health / model names | Hardcoded fallbacks | Real project-scoped data or "—" | Fixed |
| "validated candidate gaps" wording | Ambiguous label | "N validated · M candidate · K rejected" | Fixed |
| Duplicate "New Research Run" | Overview header + Navbar | Overview button removed | Fixed |
| Navbar run button on a third row at 1280px | Row overflow | Laptop-width compaction | Fixed |
| Overflow at 390px (History 395px, Evolution 533px) | `nowrap` badges/buttons with long text | Shared `max-width`, wrapping badges, truncated button labels with `title` | Fixed (9 views ≤ 390) |
| "Publisher page" and "PDF" links run together | No gap | `.btn-link` inline-flex + group gap | Fixed |
| Metric sub-labels truncated | `nowrap` + ellipsis | Wrap | Fixed |
| Leftover indigo inline styles (search CTA, New Run banner) | Old theme | Removed / shared callout | Fixed |
| Clickable `div role=button` metric cards | Not keyboard-activatable | Real `<button>`s (Overview) | Fixed (Overview); RunOverview/RunHistory cards unchanged |
| Run-progress modal inline styles | Pre-existing | Left as is (functional) | Remaining (cosmetic) |
| 24 React-Compiler lint warnings | Pre-existing patterns | Not refactored (no proven defect) | Remaining |

---

## H. Remaining work and unblocking steps

1. **Restart both servers outside the sandbox** (required for every backend fix to take effect):
   `make stop` will not stop them (they are unmanaged). Stop PIDs 62241 (uvicorn) and 67217 (vite) yourself, then
   run `make dev` from a regular terminal. `make dev` prints a warning if proxy variables are still set.
2. Open the app once. The browser re-syncs projects and fills the missing cutoff of
   `proj_1792695111211_gqxyw`. Re-add the 5 orphaned cybersecurity papers from Paper Search (they will be
   associated without re-downloading).
3. PostgreSQL verification: start `docker-compose.yml`, then
   `IDEAGEN_DATABASE_URL=postgresql+psycopg2://… alembic upgrade head` on a scratch database and run pytest. The
   guard only blocks SQLite files under `data/`.
4. Not implemented, needs a product decision:
   - Human gap accept/reject endpoint and UI.
   - Server-side project rename/delete.
   - Project edit UI.

---

## I. Final diff summary (this session)

The working tree already contained extensive user changes. Only the edits below are this audit's.

**Backend**
- `ideagen/api/main.py`:
  - Project endpoints (`_project_dto`, `_require_project`; no implicit creation; idempotent create with
    fill-unset).
  - Strict scope helper; project-scoped discovery, prior art and evolve.
  - Atomic opportunity scope.
  - `system/status`.
  - Provider errors → 502; `knowledge_status`; understand failure → 502.
  - Per-project run lock; run 400/404 mapping; redacted health/DB errors.
- `ideagen/api/schema.py`: `ProjectDTO`/`ProjectCreateDTO` fields and validation.
- `ideagen/pipeline/workflow_engine.py`:
  - Stage contract and validation; removed the automatic opportunity stage.
  - Retry semantics, statuses, failure reasons.
  - Corpus understanding; `statement_count` fix.
  - Scoped discovery with save-time tagging.
  - PENDING recovery; marked-active start; `flag_modified`.
- `ideagen/discovery/pipeline.py`: `paper_ids`, `project_id` and `gap_metadata` parameters.
- `ideagen/db/persistence.py`: `save_research_gap(project_id=…)`.
- `ideagen/discovery/embeddings.py`: `EmbeddingProviderError`, actionable message.
- `ideagen/generation/persistence.py`: opportunity `project_id` from its gap.
- `ideagen/ingestion/search.py`: classification, body redaction, proxy hints.
- `ideagen/core/network.py` (new): proxy diagnostics.
- `ideagen/llm/ollama_provider.py`: actionable errors.
- `ideagen/db/models.py`: stage failure message surfaced.
- Tests:
  - New: `tests/conftest.py` guard, `tests/db/test_test_database_guard.py`,
    `tests/api/test_audit_network_diagnostics.py`, `tests/pipeline/test_audit_project_scoping.py`.
  - Updated: `tests/pipeline/test_research_workflow.py`, `tests/api/test_phase6_api.py`.

**Frontend**
- `projects/projectStore.ts`: server-backed create and sync.
- `api/client.ts`, `types.ts`: contracts.
- Views:
  - `Navbar.tsx`: health states.
  - `SettingsView.tsx`, `WorkspacesView.tsx`: live status.
  - `ProjectOverviewView.tsx`, `DashboardView.tsx`, `RunHistoryView.tsx`: real scoped data.
  - `NewRunModal.tsx`, `CreateProjectModal.tsx`: async, copy, accessibility.
  - `SearchPapersView.tsx`: association labels, state persistence, styles.
  - `RunProgressModal.tsx`: statuses, errors, polling after retry.
  - `ResearchGapsView.tsx`, `EvolutionView.tsx`, `OpportunityDetailView.tsx`, `EvolvePanel.tsx`, `App.tsx`:
    project scope.
  - `CorpusInspectorView.tsx`: failed-extraction state.
  - `ErrorBanner.tsx`: accessibility.
- Styles: `App.css` (restored and added component styles, overflow and layout fixes), `index.css` (fadeIn).
- `vite.config.ts`: overridable proxy target.

**Tooling and docs**
- `Makefile`: test isolation, venv resolution with spaces, proxy warning.
- `README.md`: running locally.
- `docs/DECISIONS.md`: AD-24 to AD-27.
- `docs/CURRENT_STATUS.md`: audit entry.
- This report and `docs/audit/2026-10-23/`.

---

## Addendum — Preflight showed an empty queue for a corpus-only project (approved fix)

- **Symptom:** "Adaptive AI for Zero-Day Attack Detection" had 3 corpus papers. Preflight showed
  Selected/In Corpus/New = 0, and "Start" was disabled.
- **Root cause:**
  - Preflight counted only queued selections (`project_paper_selections`). Papers added with "Add to Project
    Corpus" live in `project_papers`.
  - Since AD-25/26, runs also process the corpus: UNDERSTANDING extracts corpus papers, GAP_DISCOVERY reads the
    corpus, and the backend accepts corpus-only runs.
  - The review screen and its button were not updated, so they contradicted what a run would do. The frontend
    sent the correct project ID; the API returned zeros.
- **Fix:**
  - `preflight_check` returns `corpus {total, with_passages, statements_extracted, pending_extraction,
    metadata_only}`, `runnable`, and full-text warnings.
  - The modal shows a "Project Corpus" section and enables Start when `runnable`; it stays disabled when both
    the queue and the corpus are empty.
  - Labels were renamed to "Queued Papers" / "Queued, Already Stored".
- **Cutoff lost by sync:**
  - `fromRecord` let an empty server value overwrite the browser's cutoff during the first sync through the old
    backend.
  - Now an empty server field never erases a local value.
  - With approval, the project's cutoff was set to `2023-12-31` via `POST /api/v1/projects`, which only fills
    empty fields. Backup: `audit_backups_20261023/ideagen_demo_before_cutoff_20261023.db.bak`. Row counts were
    unchanged and `integrity_check` is ok.
- **Tests:**
  - 3 backend tests in `test_audit_project_scoping.py`.
  - Isolated browser check (6/6): cutoff sync and the preflight corpus section, with the Start button not
    clicked.
  - Suite: 821 passed / 25 skipped; tsc/oxlint/build exit 0.
- **No runs were started:** 0 runs exist for this project.
