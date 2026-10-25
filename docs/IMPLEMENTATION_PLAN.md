# Implementation Plan

## Recommended Implementation Order

The following phases represent **dependency order**, not a rigid time schedule.

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Foundation | ✅ **COMPLETE** |
| **Phase 1** | OpenAlex ingestion | ⚠️ **PARTIAL** (Interactive search/ingest complete; bulk snapshot acquisition incomplete) |
| **Phase 2** | Paper processing | ✅ **COMPLETE** |
| **Phase 3** | Structured information extraction | ✅ **COMPLETE** |
| **Phase 4** | Embeddings and semantic representation | ✅ **COMPLETE** |
| **Phase 5** | Opportunity discovery | ✅ **COMPLETE** |
| **Phase 6** | Opportunity model and lifecycle | ✅ **COMPLETE** |
| **Phase 7** | Evidence Grounding / RAG | ✅ **COMPLETE** |
| **Phase 8** | Temporal opportunity evolution | ✅ **COMPLETE** |
| **Phase 9** | Controlled End-to-End Pipeline & Evaluation | ⚠️ **PARTIAL** (Pipeline orchestration complete; large-scale empirical evaluation & baselines incomplete) |
| **Phase 10** | IDEAGEN DASHBOARD / DEMO UI | ⚠️ **PARTIAL** (Read-only explorer UI complete; auth, multi-user workspaces & async queues not implemented) |
| **Phase 11** | DEMO HARDENING + RESEARCH EVALUATION | ⚠️ **PARTIAL** (Evaluation methodology, metrics, TF-IDF baseline & demo data notices implemented; real-world research evaluation & human trials unevaluated) |
| **Phase 12** | IDEAGEN CORPUS SPECIFICATION FREEZE | 📜 **SUPERSEDED** (Historic v1.0 specification created in `docs/corpus_spec_v1.json`; query caused severe over-recall and was superseded by v1.1/v1.2.1) |
| **Phase 12.1** | OPENALEX QUERY VALIDATION | ✅ **COMPLETE** (Query validation scoped; v1.1 Boolean query & batch counts verified against OpenAlex API; corpus acquisition remains separate/incomplete) |
| **Phase 13** | OPENALEX CORPUS ACQUISITION + VALIDATION | 🛑 **BLOCKED** (Rate Limited) |
| **Phase 13A**| OPENALEX SNAPSHOT ACQUISITION STRATEGY | ✅ **COMPLETE** |
| **Phase 13B**| CORPUS SPECIFICATION v1.2 + RESOURCE GATE | ✅ **COMPLETE** |
| **Phase 13C**| OPENALEX SNAPSHOT STREAMING ACQUISITION | 🛑 **INCOMPLETE** |
| **Phase 13C.1**| ACQUISITION CORRECTNESS REPAIR | ✅ **COMPLETE** |
| **Phase 13D**| CLOUD SNAPSHOT ACQ. FEASIBILITY + COST GATE | ✅ **COMPLETE** |
| **Phase 14**| REAL-DATA DEMO ONLY | ✅ **COMPLETE** |

---

## Phase 0 — Foundation ✅

**Status:** COMPLETE (2026-09-23)

Delivered:

- Python virtual environment
- Installable backend package via `pyproject.toml` (hatchling build backend)
- Package structure: `ideagen/`, `ideagen/core/`
- Pydantic-settings configuration (`ideagen/core/config.py`)
- `.env.example` with all current config knobs
- `.gitignore` covering Python, Node.js, data, IDE, and OS files
- 13 configuration tests — all passing
- Backend `README.md` with setup instructions
- Data directory scaffold (`data/{raw,processed,indexes}/`)
- Documentation directory scaffold (`docs/`)

---

## Phase Dependencies

```
Phase 0 (Foundation)
    │
    ▼
Phase 1 (OpenAlex Ingestion)
    │
    ▼
Phase 2 (Paper Processing)
    │
    ▼
Phase 3 (Structured Extraction)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 4 (Embeddings)   Phase 6 (Opportunity Model)
    │                  │
    ▼                  │
Phase 5 (Discovery) ◄─┘
    │
    ▼
Phase 7 (Temporal Tracking)
    │
    ▼
Phase 8 (Evidence / RAG)
    │
    ▼
Phase 9 (Evaluation)
    │
    ▼
Phase 10 (API / Dashboard)
```

> **Note:** The exact dependency relationships may be refined as implementation progresses. Some phases may have partial overlap.

---

## Implementation Principles

- Implementation should be **incremental and testable**.
- Each phase should produce testable outputs.
- Database tables should only be created when the corresponding phase requires them.
- Avoid unnecessary enterprise complexity.
- Research methodology changes require explicit architectural review.

---

## Conflict Note

The existing [`docs/06_STATE_AND_SESSION.md`](06_STATE_AND_SESSION.md) describes Phase 1 as "Database Models and Schema." The authoritative specification defines Phase 1 as "OpenAlex ingestion." The authoritative specification takes precedence. The existing `06_STATE_AND_SESSION.md` is preserved as a historical record of the Phase 0 completion but its Phase 1 description is superseded.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Phase ordering (0–10) | **AUTHORITATIVE DESIGN** |
| Phase 0 completion | **IMPLEMENTED** |
| Core Pipeline Engineering (Phases 2–8) | **IMPLEMENTED** (see `docs/CURRENT_STATUS.md`) |
| OpenAlex Ingestion (Phase 1) | **PARTIAL** (interactive search and ingest implemented; bulk snapshot acquisition incomplete) |
| Pipeline & Evaluation (Phase 9) | **PARTIAL** (pipeline orchestration verified; empirical benchmark evaluation incomplete) |
| API & Dashboard (Phase 10) | **PARTIAL** (FastAPI backend and read-only React/Vite explorer implemented; authentication, multi-user workspace isolation, and async task queues not implemented) |
| Demo Hardening & Evaluation (Phase 11) | **PARTIAL** (evaluation framework, metrics, TF-IDF baseline, and demo notices implemented; empirical research evaluation and human trials unevaluated) |
| Corpus Specification v1.0 (Phase 12) | **SUPERSEDED** (historical v1.0 artifact in `docs/corpus_spec_v1.json`; query syntax caused over-recall and was superseded) |
| Query Validation (Phase 12.1) | **COMPLETE** (scoped query syntax validation and v1.1 definition complete; corpus acquisition remains separate and incomplete) |
| Phase dependencies | **AUTHORITATIVE DESIGN** (exact overlaps may be refined) |
| Detailed phase deliverables | **OPEN DECISION** — to be specified per phase before implementation |
