# AGENTS.md — IdeaGen AI Agent Operating Instructions

## What Is IdeaGen

IdeaGen is an Evidence-Grounded Research Opportunity Discovery and Evolution Tracking system using Deep Learning.

It discovers recurring, evidence-supported research opportunities from scientific literature and tracks how those opportunities evolve as new scientific evidence appears.

The key research idea is treating a research opportunity as an evolving entity rather than a static research-gap description.

---

## Source-of-Truth Documentation

All project design lives in `docs/`. Read these before making changes:

| Document | Contents |
|----------|----------|
| `docs/PROJECT_OVERVIEW.md` | Project identity, research problem, objective, research gaps |
| `docs/ARCHITECTURE.md` | System pipeline, technology stack, architectural principles |
| `docs/MODEL_DESIGN.md` | Opportunity object, lifecycle states, relationships |
| `docs/DATABASE_DESIGN.md` | PostgreSQL design, entity areas, constraints |
| `docs/CORPUS_SCOPE.md` | OpenAlex corpus, date range, temporal batches |
| `docs/INGESTION_PLAN.md` | Paper ingestion rules, duplicate handling |
| `docs/EXTRACTION_PLAN.md` | Paper processing, LLM extraction design |
| `docs/EMBEDDING_STRATEGY.md` | SPECTER2, sentence-transformers, FAISS, metadata |
| `docs/OPPORTUNITY_LIFECYCLE.md` | Discovery, lifecycle, temporal tracking |
| `docs/EVALUATION_PLAN.md` | Evaluation dimensions, temporal replay |
| `docs/IMPLEMENTATION_PLAN.md` | Phase ordering and dependencies |
| `docs/DECISIONS.md` | Established architecture decisions |
| `docs/CURRENT_STATUS.md` | Current implementation state |

---

## Architectural Invariants

These are **NON-NEGOTIABLE**. Do not violate them under any circumstances.

1. **LLM classifies, code transitions.** The LLM performs extraction, classification, and explanation. Deterministic application code applies state transitions, updates opportunity status, and maintains history. The LLM must **never** directly write `opportunity.status`.

2. **PostgreSQL is the source of truth.** All authoritative data lives in PostgreSQL. FAISS is a derived, rebuildable index.

3. **Temporal leakage must be prevented.** Future literature must never leak into historical opportunity discovery or evaluation.

4. **Evidence must remain traceable.** Every opportunity, transition, and explanation must link to source papers and extracted statements.

5. **No citation, no claim.** Explanations must not cite papers outside the permitted evidence set.

6. **Provider-specific logic must remain isolated.** Do not scatter LLM provider code throughout the application.

7. **Structured outputs must be validated.** Never trust raw LLM output without validation.

8. **Embedding versions must be tracked.** Do not mix incompatible embedding model versions in one index.

---

## Research Invariants

1. The research objective is singular: evidence-grounded, temporally aware research-opportunity intelligence.

2. Research gaps (RG1–RG4) are established through the project's literature survey. Do not alter them.

3. Opportunity lifecycle states are: `OPEN`, `STRENGTHENED`, `PARTIALLY_ADDRESSED`, `RESOLVED`. Do not add, remove, or rename states without explicit architectural review.

4. Relationship types are: `SUPPORTS`, `ADDRESSES`, `CONTRADICTS`, `EXTENDS`, `UNRELATED`. Do not modify without explicit review.

5. "NEW DIRECTION EMERGING" is behavior, not a lifecycle state. Do not create it as a state.

6. The research contribution is scoped appropriately. Do not claim "nobody has ever done temporal research-gap tracking."

---

## Implementation Workflow

1. **Check `docs/CURRENT_STATUS.md`** to understand what phase the project is at.
2. **Read the relevant `docs/` files** for the current phase before writing code.
3. **Check `docs/DECISIONS.md`** before introducing new technologies or changing approaches.
4. **Write tests** for every implementation phase.
5. **Update `docs/CURRENT_STATUS.md`** after completing work.
6. **Only create database tables** when the corresponding implementation phase requires them.
7. **Follow the phase order** in `docs/IMPLEMENTATION_PLAN.md`. Do not skip phases without justification.

---

## Testing Requirements

- Every implementation phase must include tests.
- Tests must verify the specific behavior implemented in that phase.
- Use `pytest` as the test framework.
- Tests live in `backend/tests/`.
- Run tests with `pytest` from the `backend/` directory.

---

## Documentation Update Requirements

After completing any implementation work, update:

1. `docs/CURRENT_STATUS.md` — move items between Completed / In Progress / Planned sections.
2. Any relevant design document if implementation reveals design refinements.
3. Do **not** silently remove or modify established design decisions.

---

## Prohibition Against Silent Architectural Changes

- Do **not** change architectural decisions without documenting the change in `docs/DECISIONS.md`.
- Do **not** introduce technologies not listed in `docs/ARCHITECTURE.md` without explicit review.
- Do **not** introduce LangChain or LlamaIndex as mandatory architecture.
- Do **not** hardcode literature data into source code.
- Do **not** modify the opportunity state machine without explicit architectural review.
- Do **not** let the LLM directly set opportunity status.

---

## Mandatory Session Summary Format

Every coding session must end with a session summary in this format:

```
# SESSION SUMMARY

## Completed
## Current Status
## Files Created
## Files Modified
## Architecture Changes
## Database Changes
## Code Added
## Tests Added
## Decisions Made
## Questions Pending
## Next Recommended Step
```

---

## File Organization

```
IdeaGen/
├── backend/
│   ├── ideagen/          # Installable Python package
│   │   ├── core/         # Configuration and shared utilities
│   │   └── ...           # Additional modules per phase
│   ├── tests/            # Backend test suite
│   ├── pyproject.toml    # Package definition
│   └── .env.example      # Environment variable template
├── data/
│   ├── raw/              # Raw data from sources (e.g., OpenAlex)
│   ├── processed/        # Cleaned and transformed data
│   └── indexes/          # Search indexes (FAISS)
├── docs/                 # Project documentation (source of truth)
├── frontend/             # React / Next.js frontend (not yet implemented)
├── notebooks/            # Jupyter notebooks for exploration
├── AGENTS.md             # This file
└── README.md             # Project README
```
