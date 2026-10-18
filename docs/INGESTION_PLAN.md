# Ingestion Plan

## Literature Source

**OpenAlex** is the literature source for IdeaGen.

---

## Ingestion Rules

### Abstract Reconstruction

OpenAlex inverted-index abstracts must be reconstructed into normal text.

### Abstract Requirement

Papers without usable abstracts are **rejected from the extraction pipeline** because the structured extraction process depends on abstract content.

### Duplicate Handling

Potential duplicate papers should be handled using:

- **DOI matching**
- **Title similarity**

Preprint/published duplicates must be considered.

### Referenced Works

OpenAlex `referenced_works` should initially be retained as **raw JSON/JSONB**.

A dedicated citation-edge table is deferred until citation-graph functionality is actually required.

### Traceability

Raw source information should remain traceable.

### No Hardcoded Literature

Do **not** hardcode literature records into source code. All literature data must come from the ingestion pipeline.

---

## Temporal Batch Assignment

Each ingested paper must be assigned to the appropriate temporal batch based on its publication date. See [CORPUS_SCOPE.md](CORPUS_SCOPE.md) for the batch structure.

---

## Open Implementation Details

The following details are part of the implementation design, not yet specified in the authoritative design:

| Detail | Status |
|--------|--------|
| OpenAlex API interaction approach (REST, pyalex, bulk download) | **OPEN DECISION** |
| Rate limiting / pagination strategy | **OPEN DECISION** — to be determined during implementation |
| Raw data storage format (JSON files, direct-to-DB) | **OPEN DECISION** — to be determined during implementation |
| Incremental re-ingestion strategy | **OPEN DECISION** — to be determined during implementation |

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| OpenAlex as source | **AUTHORITATIVE DESIGN** |
| Abstract reconstruction | **AUTHORITATIVE DESIGN** |
| Abstract requirement for extraction | **AUTHORITATIVE DESIGN** |
| Duplicate handling (DOI + title similarity) | **AUTHORITATIVE DESIGN** |
| Referenced works as JSON/JSONB | **AUTHORITATIVE DESIGN** |
| Deferred citation-edge table | **AUTHORITATIVE DESIGN** |
| Raw source traceability | **AUTHORITATIVE DESIGN** |
| No hardcoded literature | **AUTHORITATIVE DESIGN** |
| API interaction approach | **OPEN DECISION** |
| Current implementation | **NOT IMPLEMENTED** |
