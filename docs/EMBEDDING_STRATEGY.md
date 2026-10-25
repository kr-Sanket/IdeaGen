# Embedding Strategy

## Two Semantic Representations

IdeaGen uses two distinct embedding approaches for different purposes.

---

## Whole-Paper Embeddings

**Model:** SPECTER2 (Design / Planned)

**Purpose:**

- Paper-level similarity
- Candidate retrieval
- Literature relationships

> **Status:** SPECTER2 remains a planned whole-paper representation option in the design specification; it is not currently implemented as an active provider in code.

---

## Extracted Statement Embeddings

**Models:** SentenceTransformer (`all-MiniLM-L6-v2`), Ollama (`nomic-embed-text`), or Gemini (`text-embedding-004`) behind `EmbeddingProvider` protocol

**Purpose:**

- Limitation statement embeddings
- Future-work statement embeddings
- Related extracted evidence embeddings
- Semantic clustering for recurring-gap discovery

> **Status:** Implemented in `backend/ideagen/discovery/embeddings.py`.

---

## Vector Index

**Initial technology:** FAISS

### Critical Principle

**PostgreSQL is the source of truth. FAISS is a derived/rebuildable artifact.**

Loss of the FAISS index does not mean loss of data. The index is rebuilt from PostgreSQL.

---

## Vector Metadata Requirements

Vector metadata must preserve:

| Metadata Field | Description |
|----------------|-------------|
| Paper ID | Link to source paper |
| Extracted field/type | Which extraction field the vector represents |
| Vector index position | Position within the FAISS index |
| Embedding model | Model name used for generation |
| Embedding model version | Specific version of the model |

---

## Version Integrity

Do **not** mix incompatible embedding model versions in one index.

When an embedding model is updated, the re-embedding strategy must account for version consistency.

> **OPEN DECISION:** The specific re-embedding protocol (full re-embed vs. versioned indices) has not been finalized in the authoritative design.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| SPECTER2 for whole-paper embeddings | **PLANNED DESIGN** (not active in code) |
| Statement embedding providers (SentenceTransformer, Ollama, Gemini) | **IMPLEMENTED** (`backend/ideagen/discovery/embeddings.py`) |
| FAISS as vector index | **IMPLEMENTED** (`backend/ideagen/retrieval/local.py`) |
| PostgreSQL as source of truth (not FAISS) | **AUTHORITATIVE DESIGN** |
| Vector metadata requirements | **AUTHORITATIVE DESIGN** |
| Version integrity (no mixing) | **AUTHORITATIVE DESIGN** |
| Re-embedding protocol | **OPEN DECISION** |
| Current implementation | **Statement embeddings and FAISS index implemented** (see `docs/CURRENT_STATUS.md`) |
