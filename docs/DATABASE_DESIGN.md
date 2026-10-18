# Database Design

## Core Principle

**PostgreSQL is the system of record.**

All authoritative data lives in PostgreSQL. FAISS is a derived, rebuildable vector index — not a source of truth.

---

## Planned Entity Areas

The database should eventually contain entities covering the following areas:

| Entity Area | Description | Implementation Status |
|-------------|-------------|----------------------|
| Ingestion batches | Temporal batch metadata and tracking | **NOT IMPLEMENTED** |
| Papers | Paper records from OpenAlex and manual uploads | **IMPLEMENTED** |
| Paper Documents | Extracted full text artifacts | **IMPLEMENTED** |
| Extracted information | Structured LLM extraction results | **NOT IMPLEMENTED** |
| Embeddings / embedding metadata | Vector data and model version tracking | **NOT IMPLEMENTED** |
| Opportunities | Research opportunity records | **IMPLEMENTED** |
| Opportunity evidence | Links between opportunities and supporting evidence | **IMPLEMENTED** |
| Opportunity history / transitions | State change records with timestamps and reasons | **IMPLEMENTED** |
| Paper–opportunity relationships | LLM-classified relationships between papers and opportunities | **IMPLEMENTED (Schema)** |
| Evaluation / annotation data | Evaluation results and human annotations (where required) | **NOT IMPLEMENTED** |

> **Important:** Only create database tables when the corresponding implementation phase is reached. The table list above represents the intended scope, not current implementation.

---

## Design Constraints

1. **PostgreSQL is the source of truth** — all data, including embedding vectors and metadata, is stored in PostgreSQL.
2. **FAISS is derived** — the FAISS index is rebuilt from PostgreSQL data. Loss of FAISS does not mean loss of data.
3. **Embedding metadata must be tracked** — model name, version, and generation timestamp are stored alongside vectors.
4. **Do not mix incompatible embedding model versions** in one FAISS index.
5. **OpenAlex `referenced_works` should initially be retained as raw JSON/JSONB** — a dedicated citation-edge table is deferred until citation-graph functionality is required.

---

## Database Connection

> **OPEN DECISION:** Database connection configuration (e.g., `IDEAGEN_DATABASE_URL`) has not yet been added to the application `Settings` class. This will be added when the database implementation phase begins.

The current `Settings` class in [`backend/ideagen/core/config.py`](../backend/ideagen/core/config.py) contains only general configuration (debug, log_level, environment, data paths).

---

## Schema Design Approach

The specific table schemas, column definitions, relationships, and constraints will be designed when each implementation phase requires them. The authoritative specification establishes the entity areas and principles above, not the detailed column-level schema.

Key schema design inputs:

- Opportunity object fields (see [MODEL_DESIGN.md](MODEL_DESIGN.md))
- OpenAlex paper fields (determined during ingestion phase)
- Extraction output schema (determined during extraction phase)
- Embedding metadata requirements (see [EMBEDDING_STRATEGY.md](EMBEDDING_STRATEGY.md))
- Temporal batch structure (see [CORPUS_SCOPE.md](CORPUS_SCOPE.md))

---

## Phase 5.1 — Opportunity Schema

The opportunity tables formalize the core deterministic entity for IdeaGen.

### Tables

1.  **opportunities**
    *   `id`: UUID (Primary Key)
    *   `title`: VARCHAR
    *   `problem_statement`: TEXT
    *   `description`: TEXT
    *   `status`: opportunity_status_enum (`OPEN`, `STRENGTHENED`, `PARTIALLY_ADDRESSED`, `RESOLVED`)
    *   `confidence`: FLOAT (CHECK `confidence >= 0.0 AND confidence <= 1.0`)
    *   `created_at`, `updated_at`

2.  **opportunity_evidence**
    *   `id`: UUID (Primary Key)
    *   `opportunity_id`: UUID (FK to `opportunities.id` ON DELETE CASCADE)
    *   `paper_id`: UUID (FK to `papers.id` ON DELETE RESTRICT) - Deleting an opportunity must not delete the underlying paper.
    *   `evidence_type`: opportunity_relationship_enum (`SUPPORTS`, `ADDRESSES`, `CONTRADICTS`, `EXTENDS`, `UNRELATED`)
    *   `supporting_text`: TEXT (Nullable)
    *   `created_at`

3.  **opportunity_history**
    *   `id`: UUID (Primary Key)
    *   `opportunity_id`: UUID (FK to `opportunities.id` ON DELETE CASCADE)
    *   `previous_status`: opportunity_status_enum
    *   `new_status`: opportunity_status_enum
    *   `trigger_type`: opportunity_relationship_enum
    *   `evidence_id`: UUID (Nullable, FK to `opportunity_evidence.id` ON DELETE SET NULL)
    *   `created_at`

### Integrity Highlights

*   **Evidence is Immutable to Opportunities, but Independent of Papers**: An opportunity holds links to papers as evidence. A paper itself is an independent, normalized unit of provenance. `ON DELETE RESTRICT` ensures we cannot accidentally drop a paper if it acts as evidence, and `ON DELETE CASCADE` for opportunities cleans up evidence if the parent opportunity is discarded.
*   **Bounded Confidence**: Enforced rigidly via `CHECK (confidence >= 0.0 AND confidence <= 1.0)`.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| PostgreSQL as source of truth | **AUTHORITATIVE DESIGN** |
| FAISS as derived index | **AUTHORITATIVE DESIGN** |
| Entity area scope | **AUTHORITATIVE DESIGN** |
| Embedding metadata tracking | **AUTHORITATIVE DESIGN** |
| JSON/JSONB for referenced_works | **AUTHORITATIVE DESIGN** |
| Deferred citation-edge table | **AUTHORITATIVE DESIGN** |
| Detailed column-level schemas | **PLANNED — to be designed per phase** |
| Database connection in Settings | **NOT IMPLEMENTED** |
| Any database tables | **NOT IMPLEMENTED** |
