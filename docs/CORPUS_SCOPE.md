# Corpus Scope

## Primary Literature Source

**OpenAlex**

---

## Research Domain

RAG / LLM reliability / hallucination / faithfulness / grounding.

---

## Date Range

**2019-01-01** through **2026-06-30**

---

## Query Strategy

The intended OpenAlex query is an intersection of:

- **RAG-related terms**

  AND

- **Reliability-related terms**

with title/abstract filtering and appropriate type/language/retraction filters.

> **OPEN DECISION:** The exact query expression has not been finalized in the authoritative design. It should be preserved in this document when available.

---

## Expected Scale

The exact final count is an empirical outcome of the frozen query filters. Previous historical tests suggested roughly 66,400 results before duplicate merging or abstract reconstruction limits were applied. 
Scientific relevance and reproducibility take precedence over forcing an arbitrary target number like 2,000–4,000.

---

## Temporal Batch Design

The corpus is divided temporally to support temporal replay and prevent future-information leakage.

The system must be able to reconstruct what was known at a particular point in time.

### Batch Structure

| Batch | Period | Purpose |
|-------|--------|---------|
| **Batch 0** | Through 2024-12-31 | Historical backfill |
| **2025 Q1** | 2025-01-01 – 2025-03-31 | Temporal tracking |
| **2025 Q2** | 2025-04-01 – 2025-06-30 | Temporal tracking |
| **2025 Q3** | 2025-07-01 – 2025-09-30 | Temporal tracking |
| **2025 Q4** | 2025-10-01 – 2025-12-31 | Temporal tracking |
| **2026 H1** | 2026-01-01 – 2026-06-30 | Temporal tracking |

### Temporal Replay Principle

- Historical papers (Batch 0) establish the initial opportunities.
- Later batches are introduced chronologically.
- A later paper must not be used when evaluating an earlier historical state.
- Future literature must never leak into historical opportunity discovery or evaluation.

---

## Leakage Prevention

The corpus is designed for temporal replay and leakage-aware evaluation.

The system must enforce that:

1. Opportunities discovered from Batch 0 use only Batch 0 papers.
2. Each subsequent batch is introduced in order.
3. No information from a future batch influences the analysis of a current or past batch.
4. Evaluation can verify temporal integrity.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| OpenAlex as primary source | **AUTHORITATIVE DESIGN** |
| Research domain | **AUTHORITATIVE DESIGN** |
| Date range (2019-01-01 – 2026-06-30) | **AUTHORITATIVE DESIGN** |
| Temporal batch structure | **AUTHORITATIVE DESIGN** |
| Leakage prevention principle | **AUTHORITATIVE DESIGN** |
| Exact query expression | **FROZEN (v1.1)** |
| Corpus filters | **FROZEN (v1.1)** |
| Actual corpus count (validated) | **NOT YET DETERMINED** |
| Corpus retrieval | **BLOCKED (Rate Limited)** |

---

## FROZEN CORPUS SPECIFICATION v1.2.1

*Date of specification change: 2026-10-05*

The Phase 13C.1 correction formally upgrades the specification to v1.2.1 by explicitly excluding the OpenAlex expansion corpus to match the behavior of the API's default search. The machine-readable version is located at `docs/corpus_spec_v1.2.1.json`.

### 1. Source and Domain
- **Primary Source**: OpenAlex Snapshot (CORE Corpus only)
- **Domain**: RAG / LLM reliability / hallucination / faithfulness / grounding / factuality

### 2. Search Query and Semantics
- **Query Concept**: `"retrieval-augmented generation" AND (hallucination OR faithfulness OR grounding OR factuality)`
- **Search Scope**: `title` and `abstract` only. **No full-text search**.
- **Matching Rules**: Case-insensitive, Unicode-normalized. No arbitrary keyword expansion, no semantic embeddings, no hyphenation/punctuation stripping beyond simple space delimitation.

### 3. OpenAlex Filters
- **Date Range**: `2019-01-01` to `2026-06-30` (Inclusive)
- **Type**: `article` OR `preprint`
- **Language**: `en` (English only)
- **Has Abstract**: `true`
- **Is Retracted**: `false`
- **Is XPAC (Expansion Corpus)**: `false` *(New in v1.2.1)*

### 4. Temporal Batches
*(Unchanged from v1.2)*
| Batch ID | Date Range |
|---|---|
| **0** | 2019-01-01 → 2024-12-31 |
| **1** | 2025-01-01 → 2025-03-31 |
| **2** | 2025-04-01 → 2025-06-30 |
| **3** | 2025-07-01 → 2025-09-30 |
| **4** | 2025-10-01 → 2025-12-31 |
| **5** | 2026-01-01 → 2026-06-30 |

---

## SUPERSEDED: FROZEN CORPUS SPECIFICATION v1.2

*Date of specification change: 2026-10-05*

The Phase 13B validation formally adjusts the specification to resolve the semantic mismatch between the OpenAlex live API and the local snapshot. Since the snapshot does not contain a full-text search index, the search scope is explicitly constrained to Title and Reconstructed Abstract. The machine-readable version is located at `docs/corpus_spec_v1.2.json`.

### 1. Source and Domain
- **Primary Source**: OpenAlex Snapshot
- **Domain**: RAG / LLM reliability / hallucination / faithfulness / grounding / factuality

### 2. Search Query and Semantics
- **Query Concept**: `"retrieval-augmented generation" AND (hallucination OR faithfulness OR grounding OR factuality)`
- **Search Scope**: `title` and `abstract` only. **No full-text search**.
- **Matching Rules**: Case-insensitive, Unicode-normalized. No arbitrary keyword expansion, no semantic embeddings, no hyphenation/punctuation stripping beyond simple space delimitation.
- **Reason Superseded**: v1.1 implicitly relied on the API's `default.search` which includes full text. Moving to the snapshot for reliability (HTTP 429 circumvention) required dropping the un-reproducible full-text matching to maintain strict determinism and reproducibility.

### 3. OpenAlex Filters
*(Unchanged from v1.1)*
- **Date Range**: `2019-01-01` to `2026-06-30` (Inclusive)
- **Type**: `article` OR `preprint`
- **Language**: `en` (English only)
- **Has Abstract**: `true`
- **Is Retracted**: `false`

### 4. Temporal Batches
*(Unchanged from v1.1)*
| Batch ID | Date Range |
|---|---|
| **0** | 2019-01-01 → 2024-12-31 |
| **1** | 2025-01-01 → 2025-03-31 |
| **2** | 2025-04-01 → 2025-06-30 |
| **3** | 2025-07-01 → 2025-09-30 |
| **4** | 2025-10-01 → 2025-12-31 |
| **5** | 2026-01-01 → 2026-06-30 |

---

## SUPERSEDED: FROZEN CORPUS SPECIFICATION v1.1

*Date of specification change: 2026-10-05*

The Phase 12.1 validation proved that the previous v1.0 query was being interpreted by OpenAlex as a broad OR, diluting precision and ballooning results to over 484,000. Version 1.1 explicitly applies Boolean AND logic with phrase matching to guarantee research relevance, returning a highly focused corpus of ~11,475 works. The machine-readable version is located at `docs/corpus_spec_v1.1.json`.

### 1. Source and Domain
- **Primary Source**: OpenAlex
- **Domain**: RAG / LLM reliability / hallucination / faithfulness / grounding / factuality

### 2. Search Query and Semantics
- **Query Expression**: `"retrieval-augmented generation" AND (hallucination OR faithfulness OR grounding OR factuality)`
- **Semantics**: Uses explicit OpenAlex Boolean syntax. Exact phrase match on "retrieval-augmented generation" intersected with one or more reliability attributes. Searches occur within title, abstract, and full-text index via the OpenAlex `/works?search=` parameter.
- **Previous v1.0 Query**: `retrieval-augmented generation hallucination|faithfulness|grounding|factuality`
- **Reason Superseded**: Validation evidence showed v1.0 returned 484,771 results, indicating OpenAlex interpreted the query space as an implicit broad OR, destroying precision. Candidate B (v1.1) yielded 11,475 highly relevant results, proving acceptable recall and optimal relevance.

### 3. OpenAlex Filters
*(Unchanged from v1.0)*
- **Date Range**: `2019-01-01` to `2026-06-30` (Inclusive)
- **Type**: `article` OR `preprint`
- **Language**: `en` (English only)
- **Has Abstract**: `true`
- **Is Retracted**: `false`

### 4. Temporal Batches
*(Unchanged from v1.0)*
| Batch ID | Date Range |
|---|---|
| **0** | 2019-01-01 → 2024-12-31 |
| **1** | 2025-01-01 → 2025-03-31 |
| **2** | 2025-04-01 → 2025-06-30 |
| **3** | 2025-07-01 → 2025-09-30 |
| **4** | 2025-10-01 → 2025-12-31 |
| **5** | 2026-01-01 → 2026-06-30 |
*(Ordering is strictly based on `publication_date`, not ingestion timestamp.)*

### 5. Abstract Policy
*(Unchanged from v1.0)*
A paper must have a reconstructable usable abstract for inclusion in the extraction corpus. Inverted-index abstracts from OpenAlex are reconstructed using their provided positional mappings. If reconstruction fails or yields an empty string, the paper is excluded from downstream extraction. Rejected papers remain observable through ingestion logs.

### 6. Identity and Duplicate Policy
*(Unchanged from v1.0)*
- **Primary Identity**: `(source_type, source_id)` is the primary composite key.
- **DOI**: Independent identifier. Missing DOIs must not cause false merging.
- **Duplicates**: Strict database constraints reject identical source artifacts. Title similarity/possible duplicates do *not* trigger automatic scientific merging.
- **Preprints**: Both 'article' and 'preprint' types are eligible. The same scientific work appearing as preprint and article may exist as distinct records unless a deterministic rule resolves them.

### 7. Leakage Policy
*(Unchanged from v1.0)*
- **Paper Leakage**: Only papers whose `publication_date` is within the permitted historical cutoff (batch boundaries) may influence discovery, creation, retrieval, classification, or transition.
- **Embedding Leakage**: *Unresolved Risk*. Models pretrained on future knowledge could leak temporal concepts independently of paper filtering.
- **LLM Leakage**: *Unresolved Risk*. Evaluating historical papers with modern LLMs risks knowledge leakage from LLM pretraining data.

### 8. Validation Checklist Before Acquisition Freeze
*(Unchanged from v1.0)*
The actual dataset will NOT be considered "frozen" until the following checks pass on the acquired data:
- [ ] all papers within date range
- [ ] temporal batches correct
- [ ] type filter correct
- [ ] language filter correct
- [ ] abstract requirement satisfied
- [ ] retracted papers excluded
- [ ] duplicate policy applied
- [ ] source provenance retained
- [ ] OpenAlex IDs retained
- [ ] DOI normalization applied
- [ ] raw metadata retained
- [ ] abstract reconstruction validated
- [ ] batch counts recorded
- [ ] metadata completeness recorded
- [ ] random/sample relevance audit completed
- [ ] no future-paper leakage
- [ ] acquisition configuration recorded

---

## SUPERSEDED: FROZEN CORPUS SPECIFICATION v1.0

The Phase 12 specification defines the reproducible corpus criteria before actual acquisition. The machine-readable version is located at `docs/corpus_spec_v1.json`.

### 1. Source and Domain
- **Primary Source**: OpenAlex
- **Domain**: RAG / LLM reliability / hallucination / faithfulness / grounding / factuality

### 2. Search Query and Semantics
- **Query Expression**: `retrieval-augmented generation hallucination|faithfulness|grounding|factuality`
- **Semantics**: Uses OpenAlex default search behavior. Phrase matching and OR semantics are explicitly provided in the query string via the pipe `|` operator. Searches occur within title, abstract, and full-text index as supported by the OpenAlex `/works?search=` parameter.

### 3. OpenAlex Filters
- **Date Range**: `2019-01-01` to `2026-06-30` (Inclusive)
- **Type**: `article` OR `preprint`
- **Language**: `en` (English only)
- **Has Abstract**: `true`
- **Is Retracted**: `false`

### 4. Temporal Batches
| Batch ID | Date Range |
|---|---|
| **0** | 2019-01-01 → 2024-12-31 |
| **1** | 2025-01-01 → 2025-03-31 |
| **2** | 2025-04-01 → 2025-06-30 |
| **3** | 2025-07-01 → 2025-09-30 |
| **4** | 2025-10-01 → 2025-12-31 |
| **5** | 2026-01-01 → 2026-06-30 |
*(Ordering is strictly based on `publication_date`, not ingestion timestamp.)*

### 5. Abstract Policy
A paper must have a reconstructable usable abstract for inclusion in the extraction corpus. Inverted-index abstracts from OpenAlex are reconstructed using their provided positional mappings. If reconstruction fails or yields an empty string, the paper is excluded from downstream extraction. Rejected papers remain observable through ingestion logs.

### 6. Identity and Duplicate Policy
- **Primary Identity**: `(source_type, source_id)` is the primary composite key.
- **DOI**: Independent identifier. Missing DOIs must not cause false merging.
- **Duplicates**: Strict database constraints reject identical source artifacts. Title similarity/possible duplicates do *not* trigger automatic scientific merging.
- **Preprints**: Both 'article' and 'preprint' types are eligible. The same scientific work appearing as preprint and article may exist as distinct records unless a deterministic rule resolves them.

### 7. Leakage Policy
- **Paper Leakage**: Only papers whose `publication_date` is within the permitted historical cutoff (batch boundaries) may influence discovery, creation, retrieval, classification, or transition.
- **Embedding Leakage**: *Unresolved Risk*. Models pretrained on future knowledge could leak temporal concepts independently of paper filtering.
- **LLM Leakage**: *Unresolved Risk*. Evaluating historical papers with modern LLMs risks knowledge leakage from LLM pretraining data.

### 8. Validation Checklist Before Acquisition Freeze
The actual dataset will NOT be considered "frozen" until the following checks pass on the acquired data:
- [ ] all papers within date range
- [ ] temporal batches correct
- [ ] type filter correct
- [ ] language filter correct
- [ ] abstract requirement satisfied
- [ ] retracted papers excluded
- [ ] duplicate policy applied
- [ ] source provenance retained
- [ ] OpenAlex IDs retained
- [ ] DOI normalization applied
- [ ] raw metadata retained
- [ ] abstract reconstruction validated
- [ ] batch counts recorded
- [ ] metadata completeness recorded
- [ ] random/sample relevance audit completed
- [ ] no future-paper leakage
- [ ] acquisition configuration recorded
