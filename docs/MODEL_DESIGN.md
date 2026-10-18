# Domain Model Design

## Opportunity Object

An opportunity represents an evolving research opportunity.

Conceptually it contains:

| Field | Description |
|-------|-------------|
| Identifier | Unique opportunity ID |
| Title | Synthesized opportunity title |
| Problem statement | Description of the research gap |
| Evidence | Supporting extracted statements and papers |
| Existing methods | Known approaches related to the gap |
| Unresolved aspects | What remains open |
| Confidence | Evidence quality / system assessment |
| Current status | Lifecycle state (see below) |
| History | Record of state transitions |
| Linked papers | Papers supporting or affecting this opportunity |
| Supporting extracted statements | Specific limitation/future-work statements |
| Temporal information | When the opportunity was discovered and how it evolved |

The opportunity must be traceable back to its evidence.

---

## Opportunity Lifecycle States

The established lifecycle states are:

```
OPEN → STRENGTHENED → PARTIALLY_ADDRESSED → RESOLVED
```

| State | Meaning |
|-------|---------|
| **OPEN** | Opportunity has been discovered from evidence |
| **STRENGTHENED** | Additional evidence reinforces the opportunity |
| **PARTIALLY_ADDRESSED** | Some aspects have been addressed by new work |
| **RESOLVED** | The opportunity has been substantially addressed |

These are **deterministic application states**.

- The LLM must **NOT** directly set the state.
- The LLM produces a relationship classification and supporting reasoning.
- **Application code** applies the state transition rules.

---

## Paper–Opportunity Relationships

For a new paper evaluated against an existing opportunity, the LLM may classify:

| Relationship | Meaning |
|-------------|---------|
| **SUPPORTS** | Paper provides additional evidence for the opportunity |
| **ADDRESSES** | Paper directly works on the opportunity |
| **CONTRADICTS** | Paper provides evidence against the opportunity |
| **EXTENDS** | Paper extends the opportunity in a new direction |
| **UNRELATED** | Paper is not meaningfully related |

Relationship classification should include:

- An explanation of the relationship
- Identification of relevant aspects where appropriate

These relationships are evidence used by the deterministic transition system.

---

## EXTENDS and New Directions

`EXTENDS` is a relationship type.

It may result in a new or linked opportunity.

**"NEW DIRECTION EMERGING" is NOT a formal lifecycle state.** It is a behavior resulting from the relationship analysis.

---

## Confidence

Opportunity records include a confidence concept.

Confidence should reflect evidence quality and/or system assessment.

> **OPEN DECISION:** The exact final confidence formula remains an explicit design item. No mathematically precise formula exists in the authoritative design at this time.

---

## State Transition Rules

The deterministic application code applies state transitions based on LLM-classified relationships.

The complete state transition rule table has been established in Phase 5.1.

What is established:

- Transitions are deterministic, not LLM-driven.
- Application code reads the LLM classification and applies rules.
- The LLM never writes `opportunity.status` directly.
- `EXTENDS` may create a linked or new opportunity.
- Invalid state transitions are prevented and default to the current state.
- A single `ADDRESSES` transitions to `PARTIALLY_ADDRESSED`, not `RESOLVED`.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Opportunity object concept | **AUTHORITATIVE DESIGN** |
| Lifecycle states (OPEN/STRENGTHENED/PARTIALLY_ADDRESSED/RESOLVED) | **AUTHORITATIVE DESIGN** |
| Relationship types (SUPPORTS/ADDRESSES/CONTRADICTS/EXTENDS/UNRELATED) | **AUTHORITATIVE DESIGN** |
| EXTENDS behavior | **AUTHORITATIVE DESIGN** |
| "NEW DIRECTION EMERGING" is not a state | **AUTHORITATIVE DESIGN** |
| Confidence concept | **AUTHORITATIVE DESIGN** |
| Confidence formula | **OPEN DECISION** |
| Complete state transition rule table | **AUTHORITATIVE DESIGN (Phase 5.1)** |
| LLM Classifier Interface | **IMPLEMENTED (Phase 5.3)** |
