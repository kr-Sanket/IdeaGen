# Architecture Decisions

This document records established architecture decisions for the IdeaGen project.

Decisions are recorded as established by the authoritative design specification. Dates and formal ADR numbers are not assigned unless they already exist in the repository.

---

## AD-01: OpenAlex as Primary Corpus

**Decision:** OpenAlex is the primary scientific literature source for IdeaGen.

**Context:** The project requires a comprehensive, freely accessible, structured source of scientific literature metadata with abstract content.

**Status:** Established.

---

## AD-02: PostgreSQL as Source of Truth

**Decision:** PostgreSQL is the system of record. All authoritative data lives in PostgreSQL.

**Context:** The project requires a relational database capable of storing structured paper metadata, extracted information, opportunity records, transition history, and embedding vectors with metadata.

**Status:** Established.

---

## AD-03: FAISS as Derived Vector Index

**Decision:** FAISS is the initial vector index. It is a derived, rebuildable artifact — not a source of truth.

**Context:** Fast approximate nearest-neighbor search is needed for candidate retrieval. The index must be rebuildable from PostgreSQL data without data loss.

**Status:** Established.

---

## AD-04: SPECTER2 for Whole-Paper Representations

**Decision:** SPECTER2 is used for whole-paper semantic representation.

**Context:** Scientific-paper-specific embeddings are needed for paper-level similarity and candidate retrieval. SPECTER2 is designed for scientific document representation.

**Status:** Established.

---

## AD-05: Sentence Embeddings for Extracted Statements

**Decision:** A sentence-transformer style model is used for embedding limitation and future-work statements.

**Context:** Statement-level semantic similarity is needed for clustering recurring gaps across papers. A sentence-level model provides the appropriate granularity.

**Open:** The specific sentence-transformer model has not been finalized.

**Status:** Established (model family). Open (specific model).

---

## AD-06: Clustering Method for Recurring-Gap Discovery

**Decision:** HDBSCAN was the initial design proposal for density-based clustering. In implementation, Hierarchical Agglomerative Clustering (`sklearn.cluster.AgglomerativeClustering` with average linkage, cosine distance threshold, and unit-normalized vectors) was adopted in `backend/ideagen/discovery/grouper.py`.

**Context:** The project needs semantic clustering across research statements that groups related limitations and gaps without a rigid predefined cluster count (`n_clusters=None`, using a distance threshold). Agglomerative clustering provides deterministic clustering, threshold-based cluster formation, and removes singleton noise when multiple clusters form.

**Status:** Superseded in implementation by Agglomerative Clustering (`backend/ideagen/discovery/grouper.py`). HDBSCAN preserved as initial design reference.

---

## AD-07: LLM/Provider Abstraction

**Decision:** The LLM provider is abstracted so the system is not permanently coupled to one provider. Claude is an initial provider choice.

**Context:** LLM capabilities and pricing evolve rapidly. Provider lock-in is a risk. Provider-specific logic must remain isolated and not be placed throughout the application.

**Status:** Established.

---

## AD-08: LLM Classifies, Code Transitions

**Decision:** The LLM performs extraction, classification, and explanation. Deterministic application code validates outputs, applies state-transition rules, updates opportunity status, and maintains history. The LLM must never directly write `opportunity.status`.

**Context:** Research reproducibility and system reliability require deterministic, auditable state transitions. LLM output is inherently probabilistic and must be treated as an input to a deterministic system.

**Status:** Established. **NON-NEGOTIABLE.**

---

## AD-09: Deterministic Opportunity State Machine

**Decision:** Opportunity lifecycle uses deterministic states (OPEN, STRENGTHENED, PARTIALLY_ADDRESSED, RESOLVED) with transitions applied by application code based on LLM-classified relationships.

**Context:** The research contribution centers on tracking how opportunities evolve. This requires a well-defined, reproducible state machine.

**Status:** Established.

---

## AD-10: Temporal Batch Design

**Decision:** The corpus is divided into temporal batches (Batch 0 through 2024-12-31, then quarterly 2025, then 2026 H1) to support temporal replay.

**Context:** The system must demonstrate temporal opportunity evolution without future-information leakage. Temporal batches enable chronological paper introduction and leakage-aware evaluation.

**Status:** Established.

---

## AD-11: Temporal Leakage Prevention

**Decision:** Future literature must never leak into historical opportunity discovery or evaluation. The system must support temporal replay and reconstruct what was known at any historical point.

**Context:** Temporal leakage would invalidate the research claims about opportunity evolution tracking. This is both a system requirement and an evaluation requirement.

**Status:** Established.

---

## AD-12: Thin/Custom RAG Layer

**Decision:** The initial RAG implementation should be a thin/custom retrieval layer. Do not introduce LangChain or LlamaIndex as mandatory architecture. Additional frameworks can only be introduced if there is a demonstrated engineering/research reason.

**Context:** RAG in IdeaGen is an evidence retrieval and explanation mechanism, not the core research novelty. A thin layer avoids unnecessary framework dependencies and enterprise complexity.

**Status:** Established.

---

## AD-13: No Unnecessary Enterprise Framework Complexity

**Decision:** Avoid unnecessary enterprise complexity. Implementation should be incremental and testable. Research methodology changes require explicit architectural review.

**Context:** IdeaGen is a research project. Over-engineering adds complexity without research value.

**Status:** Established.

---

## AD-14: Multi-Source Paper Acquisition

**Decision:** IdeaGen supports multiple paper acquisition sources conceptually (OPENALEX_API, OPENALEX_SNAPSHOT, MANUAL_UPLOAD). The downstream pipeline relies entirely on a generic PaperCreateDTO.

**Context:** While Phase 1 was initially designed around the live OpenAlex API, rate-limiting (HTTP 429) highlighted the risk of relying exclusively on synchronous API calls for thousands of records. Manual uploads and snapshot parsing are required for flexibility.

**Status:** Established. Phase 13's live API methodology was blocked by rate-limits. Phase 13B formally adopted the local OpenAlex Snapshot approach and upgraded the query specification to **v1.2** (`docs/corpus_spec_v1.2.json`), which strictly limits the search space to "Title and Abstract" since full-text indexing is unavailable locally. Phase 13C.1 further corrected this to **v1.2.1** by explicitly rejecting the OpenAlex Expansion Corpus (`is_xpac=false`), matching the original API core semantics. Additionally, Phase 13C.1 ruled out `updated_date` partition pruning as unsafe because an older updated_date does not guarantee an older publication_date. A full scan over the snapshot's 659 GB via partition-streaming download-and-delete strategy is required. Finally, Phase 13D approved executing this logic on a temporary AWS EC2 instance in `us-east-1` (where the public OpenAlex bucket resides) rather than locally, dramatically reducing processing time from ~67 hours to <2 hours while producing a highly compressed `JSONL.gz` subset for SCP transfer back home.

## AD-15: Dockerized PostgreSQL Persistence Foundation

**Decision:** The backend uses PostgreSQL running in a local Docker container for development. SQLAlchemy is used as the ORM, with Alembic managing database migrations.

**Context:** Ensures a robust, standardized persistence layer decoupled from the ingestion logic. Docker provides an easily reproducible environment without complex local setups.

**Status:** Established.

---

## AD-16: Separation of Paper Metadata and Extracted Text (PaperDocument)

**Decision:** Paper metadata and extracted document text are modeled separately. `PaperDocument` represents a versioned extraction artifact linked to `Paper`.

**Context:** A `Paper` represents the scientific work and its normalized bibliographic metadata. The extracted PDF full text is a potentially massive artifact that relates to extraction methodology, versioning, and provenance. Separating them prevents the `Paper` model from bloating and allows multiple extraction versions (e.g. `pdf_text`, `ocr_v2`) to exist independently.

**Status:** Established.

## AD-17: Manual PDF Extraction Library (PyMuPDF)

**Decision:** PyMuPDF (`pymupdf`) is the selected dependency for extracting plain text from manually uploaded PDF papers.

**Context:** Scientific papers frequently use complex two-column layouts, extensive references, and equations. PyMuPDF is an industry-standard, highly performant C-backed extraction library with strong heuristics for layout preservation compared to pure-Python alternatives like `pypdf`. Since this is for a limited evaluation/development corpus, a fast, layout-aware extraction library ensures the downstream processing operates on clean textual inputs. OCR is out of scope for Phase 4A.

**Status:** Established.

## AD-18: Non-hallucinated Paper Provenance (Nullable Publication Date)

**Decision:** The `publication_date` field on the `Paper` domain entity and database schema is explicitly nullable (`Optional[date]`). The ingestion layer must NOT substitute default values (like `date.today()`) if the publication date is unknown.

**Context:** The timestamp of manual ingestion or processing is already accurately captured by `PaperDocument.extracted_at` and `Paper.created_at`. Injecting the ingestion date into `publication_date` destroys research provenance, merging the concept of "when a paper was written" with "when a paper was parsed." Explicit nullability ensures missing metadata remains safely untampered without triggering downstream confusion.

**Status:** Established.

## AD-19: Deterministic Opportunity State Machine

**Decision:** The research opportunity state machine (`OPEN`, `STRENGTHENED`, `PARTIALLY_ADDRESSED`, `RESOLVED`) is driven exclusively by pure, deterministic Python functions.

**Context:** The central IdeaGen rule is: "LLM classifies, code transitions." The LLM is restricted to providing relationship inputs (e.g., `SUPPORTS`, `ADDRESSES`, `CONTRADICTS`, `EXTENDS`, `UNRELATED`). Deterministic application code evaluates these inputs to determine the next state, preventing LLM hallucination of invalid transitions or non-existent states (like `NEW_DIRECTION_EMERGING`).

**Status:** Established.

## AD-20: Multi-Paper Resolution Policy

**Decision:** A single `ADDRESSES` relationship from a single paper is insufficient to automatically transition an opportunity to `RESOLVED`. Instead, it transitions to `PARTIALLY_ADDRESSED`.

**Context:** Resolving a research gap generally requires consensus or extensive coverage, not merely one paper claiming to address it. Therefore, an explicit deterministic condition involving multiple papers will be required to reach the `RESOLVED` state. The state machine permits the state, but single-paper edges will not organically reach it.

**Status:** Established.

## AD-21: Opportunity History Event Logging Policy

**Decision:** The `OpportunityHistory` table records ONLY actual status transitions, NOT every piece of incoming evidence.

**Context:** The `OpportunityEvidence` table tracks every relationship assessment (e.g. `SUPPORTS`, `UNRELATED`) linked to an Opportunity. If a new paper `SUPPORTS` an already `STRENGTHENED` opportunity, the status remains `STRENGTHENED` (a no-op transition). To maintain a clean audit log of actual lifecycle progress, `OpportunityHistory` records are appended strictly when the state machine yields a `new_status` that differs from the `previous_status`.

**Status:** Established.

## AD-22: Idempotency and Deduplication in Transitions

**Decision:** The transition service explicitly rejects any attempt to apply relationship evidence if the specific `(opportunity_id, paper_id)` pair has already contributed evidence to that opportunity.

**Context:** A single paper should act as evidence for a specific opportunity only once. We enforce this check prior to committing a transition to prevent duplicate evidence bloat or artificially triggering cyclic state transitions (e.g., repeatedly applying the same `SUPPORTS` relationship). A unique constraint or application-level check secures this invariant.

**Status:** Established.

## AD-23: Decoupled LLM Relationship Classification

**Decision:** The relationship classifier is decoupled from database persistence. It produces a structured, Pydantic-validated `RelationshipClassification` containing the relationship, bounded confidence, rationale, and an optional evidence excerpt. It does not update the `Opportunity.status` directly.

**Context:** The LLM is restricted to the role of a pure reasoning function. This guarantees it cannot hallucinate state transitions or corrupt database integrity. A dedicated service layer independently interprets the classification result to orchestrate database operations. A protocol abstraction (`LLMProvider`) isolates the domain logic from specific vendors like OpenAI or Anthropic.

**Status:** Established.

---

## Status Classification

All decisions listed above are **AUTHORITATIVE DESIGN** — established through the project's design specification. They are not implementation artifacts.
