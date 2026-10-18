# Evaluation Plan

## Core Research Objective

"To design and evaluate an evidence-grounded, temporally aware research-opportunity intelligence system that discovers recurring research gaps from a bounded literature corpus and tracks their evolution as new papers emerge."

The project consolidates four research gaps:
- **RG1** — Scalable discovery of recurring, literature-supported research gaps
- **RG2** — Evidence-grounded validation and traceability
- **RG3** — Temporal tracking of opportunity evolution
- **RG4** — Leakage-aware evaluation of temporal claims

---

## Critical Distinction

It is critically important to distinguish between **Software/Engineering Validation** (tests, type checking, schema validation, deterministic replays) and **Research Evaluation** (measuring precision, recall, cluster quality, real-world temporal validity, human assessment). 

A passing unit test proving that a database row is inserted is *engineering validation*. It is *not* a research evaluation proving the system discovered a valid, novel research gap in the real world. IdeaGen currently has robust engineering validation via controlled deterministic fixtures (Phase 9/10).

**Definitions:**
- **Engineering Validation**: Software correctness, schema correctness, API correctness, deterministic replay, database constraints, service integration.
- **Controlled Prototype Evaluation**: Discovery metrics on known controlled signals, grounding citation validation on controlled claims, temporal leakage on controlled temporal fixture, offline FakeLLMProvider classifier evaluation.
- **Real Research Evaluation**: NOT-YET-EVALUATED until performed on appropriately labeled real literature / human evaluation.

---

## Evaluation Matrix

| Component | Evaluation | Evidence | Status | Scope |
|---|---|---|---|---|
| **Discovery** | controlled clustering | 13/13 fixture signals | **CONTROLLED-PROTOTYPE** | synthetic/controlled |
| **Grounding** | citation validation | 5/5 grounded, 2/2 fabricated rejected | **CONTROLLED-PROTOTYPE** | controlled |
| **Temporal** | future exclusion | 0/1 future papers admitted | **CONTROLLED-PROTOTYPE** | controlled |
| **Classifier** | relationship classification | 25/25 FakeLLMProvider | **CONTROLLED-PROTOTYPE** | offline fixture |
| **Real literature** | opportunity quality | no experiment | **NOT-YET-EVALUATED** | none |
| **Human evaluation** | researcher ratings | no participants yet | **NOT-YET-EVALUATED** | none |
| **Baseline comparison** | IdeaGen vs baselines | baselines defined only | **NOT-YET-EVALUATED** | none |

*(Note: The 13/13, 5/5, 0/1, and 25/25 metrics are Controlled Fixture Results. They do NOT represent real-literature benchmarks.)*

---

## Baselines

### A. Opportunity Discovery (RG1)
- **Baseline**: Lexical similarity / TF-IDF clustering without semantic research-gap normalization.
- **IdeaGen Approach**: Uses normalized `ResearchGapSignal` structures and semantic gap embeddings for clustering.
- **Comparison Metric**: Pairwise cluster precision/recall/F1 on labeled research signals.
- **Status**: Future Work (Requires labeled real-world dataset).

### B. Evidence Retrieval (RG2)
- **Baseline**: Standard BM25 lexical retrieval against the entire document text.
- **IdeaGen Approach**: Vector-based semantic retrieval against isolated evidence chunks with strict ID validation.
- **Comparison Metric**: Top-K retrieval precision and recall.
- **Status**: Future Work.

### C. Temporal Evolution (RG3)
- **Baseline**: Static opportunity representation (assessing the gap once at the end of the timeline without tracking intermediate states).
- **IdeaGen Approach**: Discrete, chronological, deterministic state transitions (`OPEN`, `STRENGTHENED`, `PARTIALLY_ADDRESSED`, `RESOLVED`).
- **Comparison Metric**: Ability to accurately recreate the opportunity state at arbitrary historical timestamps.
- **Status**: Controlled-Prototype (Engineering Validated).

### D. Grounding (RG2)
- **Baseline**: Explanation generation without explicit evidence-reference validation.
- **IdeaGen Approach**: Strict "No citation, no claim" enforcement via `GroundedOpportunityService`.
- **Comparison Metric**: Citation hallucination rate.
- **Status**: Controlled-Prototype (Engineering Validated).

---

## Ablation Plan

To determine the contribution of individual architectural components, the following ablations are planned:

- **Ablation A (Without temporal tracking)**: Process the entire corpus in a single batch. Measures the loss of historical state trajectories and intermediate opportunity states.
- **Ablation B (Without evidence grounding validation)**: Bypass `GroundedOpportunityService` and rely on raw LLM output. Measures the increase in hallucinated or un-traceable citations.
- **Ablation C (Without semantic gap normalization)**: Cluster raw paper abstracts instead of extracted `ResearchGapSignal` structures. Measures the degradation in cluster precision and the increase in noise.
- **Ablation D (Static vs. Evolving)**: Compare IdeaGen's state machine outputs against a simple "Open/Closed" binary classifier.

---

## Human Evaluation Protocol

For future real-world evaluation, IdeaGen opportunities will be evaluated by human researchers using the following protocol:

- **Evaluator Instructions**: Raters will be presented with an Opportunity, its Grounded Explanation, and its Timeline. They must independently score the opportunity without knowing whether it was generated by IdeaGen or a Baseline.
- **Rating Scale (1-5 Likert)**:
  - *Relevance*: Does this represent a valid scientific problem?
  - *Novelty*: Is this distinct from previously solved problems?
  - *Evidence Support*: Do the cited papers actually support the claims made?
  - *Clarity*: Is the gap statement and boundary well-defined?
  - *Usefulness*: Would this be helpful for a literature review or grant proposal?
- **Agreement Metric**: Krippendorff's alpha (chosen over Cohen's kappa to support more than two raters and handle ordinal Likert data gracefully).
- **Status**: NOT-YET-EVALUATED.

---

## Reproducibility Checklist

### FROZEN / ESTABLISHED
- **Python version/configuration**: Pinned by project (3.11+)
- **Dependency versions**: Genuinely pinned in `pyproject.toml`
- **DB schema migrations**: Established via SQLAlchemy Base
- **Lifecycle transition rules**: Established deterministic Phase 5 logic
- **Discovery implementation**: Established deterministic connected components clustering logic and min_size parameters
- **Classifier prompt**: Version `v1` established for the offline FakeLLMProvider classification evaluation
- **Definitive OpenAlex production query**: FROZEN in Phase 12 (`docs/corpus_spec_v1.json`)

### OPEN / NOT YET FROZEN
- **Final production corpus**: Not acquired (Acquisition deferred)
- **Production embedding model/version**: Not yet selected/frozen (currently mocked)
- **Large-scale LLM evaluation configuration**: Not executed/frozen (currently blocked by quota limits)

---

## Leakage Audit Framework (RG4)

| Risk | Detection Mechanism | Mitigation | Current Status |
|---|---|---|---|
| Future paper included in historical batch | Batch date bound checks (`start_date <= pub_date <= end_date`) | Strict filtering in `OpportunityEvolutionService` | **CONTROLLED-PROTOTYPE** |
| Embedder trained on future knowledge | Model timestamp verification | Use embedding models trained strictly prior to the corpus cutoff | **NOT-YET-EVALUATED** (Model not selected) |
| Duplicate ID bypasses cutoff | Unique constraints on `source_id` | Enforced at DB level, duplicates rejected/merged | **ENGINEERING-VALIDATED** |
| Missing publication date defaults to future | Null checks during batching | Papers without dates are skipped/dropped during temporal replays | **ENGINEERING-VALIDATED** |

---

## Failure Analysis Framework

| Category | Symptom | Likely Cause | Mitigation | Unresolved Risk |
|---|---|---|---|---|
| **Discovery false positive** | Unrelated papers clustered together | Similarity threshold too low, poor embeddings | Tune `similarity_threshold` | Finding optimal generic threshold |
| **Discovery false negative** | Highly related papers not clustered | Normalization too aggressive | Adjust signal extraction prompt | Missed opportunities |
| **Relationship misclassification** | Paper marked `SUPPORTS` when it `ADDRESSES` | LLM prompt ambiguity | Few-shot prompting, schema validation | LLM stochasticity |
| **Unsupported claim** | Claim generated without citation | LLM hallucination | `GroundedOpportunityService` rejects | Explanation pipeline fails entirely |
| **Citation mismatch** | Citation ID exists but points to wrong paper | Context window confusion | Chunked RAG retrieval | Fine-grained grounding limits |
| **Temporal leakage** | Future paper transitions historical state | Batching logic failure | Strict `OpportunityEvolutionService` cutoff | Pre-prints with updated dates |

