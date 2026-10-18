# Extraction Plan

## Paper Processing

Paper processing should produce normalized paper records suitable for downstream extraction and embedding.

The design should account for:

- Abstract reconstruction
- Cleaning
- Normalization
- Duplicate detection
- Metadata validation
- Temporal batch assignment
- Traceability to raw source data

The exact implementation should remain modular.

---

## Structured Information Extraction

The extraction stage uses an LLM/Transformer-based approach.

### Extraction Targets

Important information to extract includes:

| Field | Description |
|-------|-------------|
| Research problem | What problem the paper addresses |
| Method | Approach or methodology used |
| Datasets | Datasets mentioned or used |
| Limitations | Stated limitations of the work |
| Future work | Stated future work directions |
| Findings / relevant evidence | Key results and evidence |

### Output Requirements

- The extraction schema should be represented as **structured data**.
- LLM output must be **validated** before being accepted.
- Extraction results should be **cacheable**.

### LLM Provider Abstraction

The LLM provider should be abstracted so that the system is not permanently coupled to one provider.

- **Claude** is an initial provider choice.
- The architecture must permit another model/provider later.
- Do **not** place provider-specific logic throughout the application. Provider-specific code must remain isolated.

---

## Open Implementation Details

| Detail | Status |
|--------|--------|
| Specific LLM model selection | **OPEN DECISION** — Claude is the initial choice, exact model version TBD |
| Prompt templates / schemas | **OPEN DECISION** — to be designed during extraction implementation |
| Extraction output schema (exact fields and types) | **OPEN DECISION** — to be designed during extraction implementation |
| Full-text vs. abstract-only processing | **OPEN DECISION** — the abstract requirement is established; full-text processing scope is not specified |
| Validation rules for structured outputs | **OPEN DECISION** — to be designed during extraction implementation |
| Caching strategy | **OPEN DECISION** — to be determined during implementation |

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Paper processing requirements | **AUTHORITATIVE DESIGN** |
| Modularity requirement | **AUTHORITATIVE DESIGN** |
| Extraction targets (problem, method, datasets, limitations, future work, findings) | **AUTHORITATIVE DESIGN** |
| Structured data output | **AUTHORITATIVE DESIGN** |
| LLM output validation | **AUTHORITATIVE DESIGN** |
| Cacheability requirement | **AUTHORITATIVE DESIGN** |
| Provider abstraction | **AUTHORITATIVE DESIGN** |
| Claude as initial provider | **AUTHORITATIVE DESIGN** |
| Provider isolation | **AUTHORITATIVE DESIGN** |
| Prompt templates | **OPEN DECISION** |
| Exact extraction schema | **OPEN DECISION** |
| Current implementation | **NOT IMPLEMENTED** |
