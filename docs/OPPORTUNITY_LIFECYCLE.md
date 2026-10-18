# Opportunity Lifecycle

## Overview

This document describes how research opportunities are discovered, formed, tracked, and evolved over time.

---

## Opportunity Discovery (Phase 6)

Opportunity discovery is based on recurring patterns across literature. The discovery pipeline operates strictly prior to opportunity evolution (state transitions).

### Discovery vs. Evolution

**Discovery** is the process of finding recurring gaps in the literature to propose candidate opportunities.
**Evolution** is the process of tracking how an established Opportunity changes state (e.g., OPEN to PARTIALLY_ADDRESSED) as new papers are published.

### Discovery Pipeline

The explicit discovery stage before Opportunity state transitions is structured as follows:

```
Research-gap discovery
        ↓
OpportunityCandidate
        ↓
Evidence retrieval / grounding
        ↓
Validated Opportunity
        ↓
Relationship classification
        ↓
Deterministic lifecycle transition
```

> **Note:** RAG provides evidence grounding and explanation; it is not the core temporal novelty.

### Discovery Process Details

1. Collect structured `ResearchGapSignal` statements from extracted paper information.
2. Normalize statements conservatively to remove noise while preserving domain terms.
3. Generate embeddings for these statements using a configured `GapEmbedder`.
4. Calculate semantic similarity (e.g., Cosine Similarity).
5. Cluster semantically similar statements using a deterministic threshold.
6. Identify recurring clusters.
7. Require evidence across **multiple distinct papers** (configurable `MIN_CLUSTER_SIZE`).
8. Synthesize an `OpportunityCandidate`.
9. Preserve source evidence traceably.

### Clustering Method

The initial proposed clustering method is **HDBSCAN**.

### Minimum Evidence Threshold

A candidate recurring gap should span at least **three distinct papers**.

### LLM Synthesis

The LLM may synthesize a cluster into:

- Opportunity title
- Problem statement
- Unresolved aspects
- Evidence-supported summary

### Quality Controls

- An opportunity should remain linked to the papers/statements supporting it.
- Near-duplicate opportunities should be identified and considered for merging.
- Human review may be used during evaluation/validation of discovery quality.

---

## Lifecycle States

The established lifecycle states are:

| State | Description |
|-------|-------------|
| **OPEN** | Opportunity has been discovered from evidence |
| **STRENGTHENED** | Additional evidence reinforces the opportunity |
| **PARTIALLY_ADDRESSED** | Some aspects have been addressed by new work |
| **RESOLVED** | The opportunity has been substantially addressed |

These are **deterministic application states**. See the core principle below.

---

## Paper–Opportunity Relationships

For a new paper evaluated against an existing opportunity, the LLM classifies:

| Relationship | Description |
|-------------|-------------|
| **SUPPORTS** | Paper provides additional evidence for the opportunity |
| **ADDRESSES** | Paper directly works on the opportunity |
| **CONTRADICTS** | Paper provides evidence against the opportunity |
| **EXTENDS** | Paper extends the opportunity in a new direction |
| **UNRELATED** | Paper is not meaningfully related |

Relationship classification should include an explanation and identify relevant aspects where appropriate.

### Relationship Semantics

These strict semantic definitions instruct the LLM:

*   **SUPPORTS**: The paper provides evidence that strengthens the existence or importance of the unresolved research opportunity. It does not substantially solve the opportunity.
*   **ADDRESSES**: The paper directly tackles the problem represented by the opportunity. It provides a method/result that reduces the unresolved aspect. This does NOT mean the opportunity is automatically RESOLVED.
*   **CONTRADICTS**: The paper provides evidence that conflicts with a premise, claim, or assumption underlying the opportunity.
*   **EXTENDS**: The paper builds on, broadens, or redirects the research direction in a way related to the opportunity. EXTENDS does not automatically mean the opportunity is solved.
*   **UNRELATED**: The paper is not materially relevant to the opportunity.

---

## Core Principle: LLM Classifies, Code Transitions

This is a **NON-NEGOTIABLE** architectural constraint.

- The LLM produces a relationship classification and supporting reasoning.
- **Deterministic application code** applies the state transition rules.
- The LLM must **never** directly write `opportunity.status`.
- Application code validates structured outputs.
- Application code enforces evidence constraints.
- Application code prevents invalid state transitions.

---

## EXTENDS and New Directions

`EXTENDS` is a relationship type that may result in a new or linked opportunity.

**"NEW DIRECTION EMERGING" is NOT a formal lifecycle state.** It is a behavior resulting from the relationship analysis.

---

## Temporal Tracking (Phase 8)

For each existing opportunity and each later literature batch, the system evaluates the opportunity against chronological evidence.

The temporal evolution process follows this pipeline:

```
Historical corpus
        ↓
Temporal batches
        ↓
Evidence available at cutoff
        ↓
Relationship classification
        ↓
Deterministic state transition
        ↓
Opportunity history
        ↓
Next batch
        ↓
Updated opportunity
```

### Temporal Integrity

- **Chronological Processing:** Papers are processed strictly in chronological order within their bounded batches.
- **Temporal Cutoff:** A paper with a publication date later than the current batch's cutoff date must **not** influence earlier states.
- **Future-Information Exclusion:** Future papers are strictly excluded to prevent temporal leakage.
- **Deterministic Replay:** The system can deterministically reconstruct the evolution timeline of an opportunity by replaying chronological batches.
- **AD-22 Behavior:** Processing the identical paper against the same opportunity multiple times follows idempotency logic (raises on duplicates instead of creating duplicate evidence).
- **No Automatic Resolution:** Temporal processing relies entirely on the deterministic state machine. `ADDRESSES` transitions to `PARTIALLY_ADDRESSED`, not `RESOLVED`.
- **EXTENDS behavior:** `EXTENDS` does not create a new lifecycle status; the existing status is maintained, and it operates as behavior rather than status.

---

## State Transition Rules

The complete state transition rule table is now established (Phase 5.1).

- The LLM classification is the input; the code applies the rule.
- `EXTENDS` may create a linked or new opportunity, but state-wise it preserves the current state or returns to OPEN/STRENGTHENED depending on the prior state.
- Invalid transitions are prevented and default to the current state.
- Each transition stores triggering evidence and reason in `OpportunityHistory`.

**Deterministic Rules (Phase 5.1 Baseline):**
- OPEN + SUPPORTS -> STRENGTHENED
- STRENGTHENED + SUPPORTS -> STRENGTHENED
- PARTIALLY_ADDRESSED + SUPPORTS -> PARTIALLY_ADDRESSED
- RESOLVED + SUPPORTS -> RESOLVED

- OPEN + ADDRESSES -> PARTIALLY_ADDRESSED
- STRENGTHENED + ADDRESSES -> PARTIALLY_ADDRESSED
- PARTIALLY_ADDRESSED + ADDRESSES -> PARTIALLY_ADDRESSED
- RESOLVED + ADDRESSES -> RESOLVED

- OPEN + CONTRADICTS -> OPEN
- STRENGTHENED + CONTRADICTS -> OPEN
- PARTIALLY_ADDRESSED + CONTRADICTS -> PARTIALLY_ADDRESSED
- RESOLVED + CONTRADICTS -> OPEN

- OPEN + EXTENDS -> OPEN
- STRENGTHENED + EXTENDS -> STRENGTHENED
- PARTIALLY_ADDRESSED + EXTENDS -> PARTIALLY_ADDRESSED
- RESOLVED + EXTENDS -> RESOLVED

- Any status + UNRELATED -> unchanged

### Resolution Policy
**A single `ADDRESSES` relationship does NOT automatically transition an opportunity to `RESOLVED`.** 
A single paper addressing an opportunity results in `PARTIALLY_ADDRESSED`. Reaching `RESOLVED` requires an explicit, multi-paper deterministic condition (to be defined in a future phase). For now, `RESOLVED` exists as a valid state in the schema and state machine, but no ordinary single-paper relationship should automatically trigger it.

---

## Integration Pipeline (Phase 5.5)

The full LLM-to-transition pipeline is:

```
Paper + Opportunity bounded context
        |
        v
RelationshipClassifier
        |
        v
RelationshipClassification
        |
        v
OpportunityClassificationIntegration
        |
        v
OpportunityTransitionService
        |
        v
Deterministic state machine
        |
        v
Evidence + History
        |
        v
Updated Opportunity
```

**"LLM classifies, code transitions."**

### Integration Invariants

- **Classifier failures do not mutate opportunity state.** If the LLM provider fails, returns malformed data, or input validation fails, the error propagates immediately. No evidence, history, or status change is created.
- **State transitions are deterministic.** The integration layer passes the classified relationship to `OpportunityTransitionService`, which invokes the pure `transition()` function. The integration layer does not contain transition rules.
- **AD-22 duplicate behaviour is preserved.** A duplicate `(opportunity_id, paper_id)` pair is rejected by the transition service. The integration layer does not override or silently replace.
- **Resolution is not triggered by a single ADDRESSES.** `OPEN + ADDRESSES → PARTIALLY_ADDRESSED`, never `RESOLVED`.
- **The integration layer is vendor-neutral.** It depends on `RelationshipClassifier`, not on any specific LLM provider.

---

## Confidence

Opportunity records include a confidence concept reflecting evidence quality and/or system assessment.

> **OPEN DECISION:** The exact confidence formula remains an explicit design item.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Discovery process (clustering + synthesis) | **AUTHORITATIVE DESIGN** |
| HDBSCAN as initial clustering method | **AUTHORITATIVE DESIGN** |
| Minimum 3 distinct papers threshold | **AUTHORITATIVE DESIGN** |
| Lifecycle states | **AUTHORITATIVE DESIGN** |
| Relationship types | **AUTHORITATIVE DESIGN** |
| LLM classifies, code transitions | **AUTHORITATIVE DESIGN — NON-NEGOTIABLE** |
| EXTENDS behavior | **AUTHORITATIVE DESIGN** |
| Temporal tracking protocol | **AUTHORITATIVE DESIGN** |
| Temporal integrity rules | **AUTHORITATIVE DESIGN** |
| Complete state transition rule table | **AUTHORITATIVE DESIGN (Phase 5.1)** |
| LLM Classifier Interface | **IMPLEMENTED (Phase 5.3)** |
| Integration pipeline (classify → transition) | **IMPLEMENTED (Phase 5.5)** |
| Confidence formula | **OPEN DECISION** |
| Near-duplicate merging strategy | **OPEN DECISION** |
