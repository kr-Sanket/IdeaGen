# Project Overview

## Project Identity

**Project name:** IdeaGen

**Full concept:** Evidence-Grounded Research Opportunity Discovery and Evolution Tracking using Deep Learning.

IdeaGen is a research intelligence system designed to discover recurring, evidence-supported research opportunities from scientific literature and track how those opportunities evolve as new scientific evidence appears.

**Key research idea:** Treat a research opportunity as an evolving entity rather than a static research-gap description.

---

## Research Problem

Scientific literature grows continuously.

Existing systems can perform tasks such as:

- Literature search
- Paper summarization
- Information extraction
- Research-gap identification
- Research-idea generation
- Research-idea evaluation
- Literature-grounded question answering

However, IdeaGen focuses on a different problem:

> A research opportunity identified from one body of literature may change when new scientific evidence appears.

Therefore the system should not simply discover an opportunity once. It should:

1. Discover an opportunity from historical literature.
2. Preserve the evidence supporting it.
3. Introduce later literature chronologically.
4. Determine how new papers relate to the opportunity.
5. Update the opportunity's state deterministically.
6. Preserve the history of those changes.
7. Explain each transition using traceable evidence.

---

## Research Objective

The current consolidated research objective is:

> "To design and evaluate an evidence-grounded, temporally aware research-opportunity intelligence system that discovers recurring research gaps from a bounded literature corpus and tracks their evolution as new papers emerge."

This is ONE research objective. There are no competing sub-objectives.

---

## Research Gap Structure

The project consolidates four research gaps identified through the project's literature survey:

| Gap | Description |
|-----|-------------|
| **RG1** | Scalable discovery of recurring, literature-supported research gaps |
| **RG2** | Evidence-grounded validation and traceability |
| **RG3** | Temporal tracking of opportunity evolution |
| **RG4** | Leakage-aware evaluation of temporal claims |

> **Note:** These are the gaps consolidated through the project's literature survey. They are not claimed as universally recognized gaps across all research.

---

## Research Positioning

The project's novelty claim is appropriately scoped.

The contribution is described as the temporal opportunity-evolution framework identified as a gap through the project's surveyed literature.

The project does **not** claim: "Nobody has ever done temporal research-gap tracking."

### Supporting/Reference Research Resources

The project uses supporting and reference research resources for positioning and evaluation methodology, including:

- AI Idea Bench
- PRESCIENCE
- ResearchStudio-Idea
- ROAD-tv
- ScholarEval

These are supporting resources for positioning and evaluation methodology. They are **not** automatically the main IdeaGen corpus. They are **not** training datasets unless explicitly implemented and documented later.

---

## End-to-End Controlled Demonstration

IdeaGen establishes its conceptual viability through an **End-to-End Controlled Demonstration**. This automated pipeline evaluates the system's architectural correctness without the noise or overhead of live LLM endpoints or massive vector databases:
1. Validates chronological literature ingestion into PostgreSQL.
2. Forms candidate clusters deterministically from research gap signals.
3. Secures candidate selection through grounded evidence (RAG) tied exactly to the database representations.
4. Evolves opportunity state correctly across multiple subsequent synthetic batches (maintaining deterministic isolation and AD-22 rules).

This demonstration verifies the temporal architecture and structural pipeline. It is the baseline necessary before expanding to large-scale literature evaluation.

---

## Status Classification

| Aspect | Classification |
|--------|---------------|
| Project identity | **AUTHORITATIVE DESIGN** |
| Research problem | **AUTHORITATIVE DESIGN** |
| Research objective | **AUTHORITATIVE DESIGN** |
| Research gaps (RG1–RG4) | **AUTHORITATIVE DESIGN** |
| Research positioning | **AUTHORITATIVE DESIGN** |
| Current implementation | **Phase 13 corpus acquisition BLOCKED** |
