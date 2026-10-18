# Implementation Plan

## Recommended Implementation Order

The following phases represent **dependency order**, not a rigid time schedule.

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Foundation | ✅ **COMPLETE** |
| **Phase 1** | OpenAlex ingestion | ⬜ Not started |
| **Phase 2** | Paper processing | ⬜ Not started |
| **Phase 3** | Structured information extraction | ⬜ Not started |
| **Phase 4** | Embeddings and semantic representation | ⬜ Not started |
| **Phase 5** | Opportunity discovery | ✅ **COMPLETE** |
| **Phase 6** | Opportunity model and lifecycle | ✅ **COMPLETE** |
| **Phase 7** | Evidence Grounding / RAG | ✅ **COMPLETE** |
| **Phase 8** | Temporal opportunity evolution | ✅ **COMPLETE** |
| **Phase 9** | Controlled End-to-End Pipeline | ✅ **COMPLETE** |
| **Phase 10** | IDEAGEN DASHBOARD / DEMO UI | ✅ **COMPLETE** |
| **Phase 11** | DEMO HARDENING + RESEARCH EVALUATION | ✅ **COMPLETE** |
| **Phase 12** | IDEAGEN CORPUS SPECIFICATION FREEZE | ✅ **COMPLETE** |
| **Phase 12.1** | OPENALEX QUERY VALIDATION | ✅ **COMPLETE** |
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
| Phases 1–10 | **PLANNED — NOT IMPLEMENTED** |
| Phase dependencies | **AUTHORITATIVE DESIGN** (exact overlaps may be refined) |
| Detailed phase deliverables | **OPEN DECISION** — to be specified per phase before implementation |
