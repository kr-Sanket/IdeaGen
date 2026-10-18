# OpenAlex Snapshot Acquisition Strategy v1

## 1. Existing Snapshot Architecture
The current architecture is highly compatible with a snapshot approach:
- `OpenAlexSnapshotSource` already exists and correctly implements the `PaperSource` protocol by parsing line-by-line JSONL.
- The `OpenAlexResponseParser` handles the exact JSON schema provided by the OpenAlex snapshot (which mirrors the API).
- The `SourceType.OPENALEX_SNAPSHOT` enum is already established for tracking provenance.

## 2. Real Snapshot Compatibility Assessment
- **Expected Format**: Gzipped JSONL (`.jsonl.gz`), distributed across partitioned folders (e.g., `data/works/updated_date=.../*.gz`).
- **Record Structure**: Identical to API responses, containing `id`, `title`, `publication_date`, `type`, `language`, `is_retracted`, `abstract_inverted_index`, `authorships`, `referenced_works`.
- **Compatibility**: Fully compatible with IdeaGen's parser. No schema translation is required.

## 3. Query-Semantic Analysis & Semantic Differences
The frozen v1.1 query is:
`"retrieval-augmented generation" AND (hallucination OR faithfulness OR grounding OR factuality)`

**The Challenge**: OpenAlex's API `default.search` searches across titles, abstracts, and full-text (where available). The Snapshot provides titles and `abstract_inverted_index`, but **does NOT contain full-text fields**.

**Semantic Adjustment**: 
We cannot perfectly reproduce the API's full-text search capability locally using just the snapshot metadata.
**SNAPSHOT CANNOT EXACTLY REPRODUCE THE FROZEN QUERY SEMANTICS.**

**Scientifically Defensible Adjustment**: 
Restrict the search space explicitly to **Title and Abstract**. This is arguably a stronger research methodology because it guarantees the paper is *substantively about* RAG and reliability, rather than merely citing the term in a footnote or background section.

## 4. Local Filtering Strategy
The sequence implemented and tested in `snapshot_filter.py`:
1. **Basic metadata validity**: Verify `id` exists.
2. **Publication date filter**: Parse date; enforce `2019-01-01` to `2026-06-30`.
3. **Type filter**: Restrict to `article` or `preprint`.
4. **Language filter**: Restrict to `en`.
5. **Retraction filter**: Exclude `is_retracted == true`.
6. **Abstract reconstruction**: Reconstruct from `abstract_inverted_index`. Reject if missing/empty.
7. **Query-term matching**: Lowercase title + abstract. 
   - Require exact substring `"retrieval-augmented generation"`.
   - Require at least one of `["hallucination", "faithfulness", "grounding", "factuality"]`.
8. **Identity normalization**: Standard duplicate rejection via PostgreSQL.
9. **Temporal batch assignment**: Assign to Batch 0-5.

## 5. Storage Strategy
- **Do NOT unpack**: Unpacking the snapshot locally requires >350GB of disk space.
- **Recommendation**: Stream directly from the compressed files using Python's `gzip.open()`. This requires only the storage for the compressed snapshot (~40GB) and operates line-by-line, keeping memory overhead trivial (<50MB).

## 6. Resumability
To handle interruptions without re-processing hundreds of gigabytes, we implement a granular checkpoint mechanism:
- **State File**: `ingestion_checkpoint.json`
- **Fields**: 
  - `current_file`: Path to the currently processing `.gz` file.
  - `lines_processed`: Number of lines processed in the current file.
  - `global_stats`: Running counts of raw, rejected, and retained works.
- **Recovery**: On startup, read the checkpoint, skip already fully processed files, open the `current_file`, and iterate exactly `lines_processed` times before resuming logic. 

## 7. Resource Requirements
- **Disk**: ~40-50GB for the downloaded `.gz` snapshot files.
- **Memory**: <100MB (streaming line-by-line).
- **Processing Time**: Bound primarily by JSON parsing speed and disk read I/O. Expected processing time is 4-8 hours on a standard developer machine.

## 8. Small-Fixture Test Results
A synthetic test suite (`backend/tests/ingestion/test_snapshot_strategy.py`) was successfully run against the filter logic. It validated:
- Valid articles/preprints
- Rejection of missing abstracts, wrong languages, wrong types, and retracted works
- Strict boundary checking for the publication date
- Correct matching of RAG phrase and reliability OR logic.

## 9. API vs Snapshot Comparison
| Metric | Live API | Local Snapshot |
|---|---|---|
| **Advantages** | Zero local storage, delegates search compute to OpenAlex | 100% reproducible, zero rate limits, resilient |
| **Limitations** | **BLOCKED (HTTP 429)**, fragile over long periods | Requires ~40GB download, no full-text indexing |
| **Reproducibility** | Poor (API results fluctuate daily) | Perfect (static dataset frozen in time) |
| **Query Semantics** | Broad (Title, Abstract, Full Text) | Highly Targeted (Title, Abstract) |

## 10. Snapshot Versioning
To maintain provenance, the dataset must record the explicit snapshot version used (e.g., `openalex-snapshot-2025-01-01`). The checksum or manifest provided by OpenAlex should be stored alongside `corpus_acquisition_report_v1.json`.

## 11. Final Recommendation & Next Steps
**Recommendation**: Proceed with the OpenAlex Snapshot strategy.
**Required Actions Before Execution**:
1. Officially adjust `CORPUS_SCOPE.md` to state that the query scope is restricted to "Title and Abstract" (T&A) to reconcile the missing full-text issue.
2. Develop the resumable snapshot ingestion script.
3. Download the OpenAlex snapshot locally.
