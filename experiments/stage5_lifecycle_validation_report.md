# IdeaGen Stage 5 — Lifecycle Rules and Deterministic Validation (Revised)

- **Original report:** 2026-10-21
- **Revision:** 2026-10-21, re-verified against commit `17895a2`
- **Code:** `backend/ideagen/opportunity/state_machine.py`, `service.py`, `integration.py`, `grounding.py`,
  `temporal.py`, `incremental.py`
- **Test matrix:** `backend/tests/opportunity/test_stage5_lifecycle_matrix.py`

> **Revision note.** The original report's scenario list and pass counts were broadly right, but it contained
> factual errors: its lifecycle table allowed `ADDRESSES` to move an opportunity from `OPEN` directly to
> `RESOLVED` (the code does not); it listed the Stage 4E SUPPORTS→ADDRESSES errors as
> "DEV-02, DEV-03, HELD-04, HELD-08, HELD-09" (correct: DEV-02, DEV-03, HELD-08, HELD-11, HELD-12); it counted
> "12 false EXTENDS" in Stage 4D (correct: 12 EXTENDS predictions, 10 of them false); it gave a mean latency
> of 7.5 s (recorded means: 6.59 s and 7.90 s); it attributed savepoints to `classify_and_apply()`; and it
> described the lifecycle components as "100% stable … production-ready". This revision corrects those points.
> **Nothing in this report is a real-world accuracy measurement.**

---

## 1. What Stage 5 Establishes

Stage 5 checks that, **given** a relationship label and an excerpt, the lifecycle machinery behaves as
specified: grounding fails closed, transitions follow the rules, duplicates are rejected, temporal
eligibility is enforced, and failures roll back cleanly. All scenarios use fake LLM providers with fixed
outputs on in-memory SQLite. Whether the classifier produces the *right* label is a separate question,
addressed (with significant limitations) by the Stage 4D/4E report.

---

## 2. Lifecycle Rules (verified in `state_machine.py` and `service.py`)

States: `OPEN`, `STRENGTHENED`, `PARTIALLY_ADDRESSED`, `RESOLVED`. The language model only classifies;
`transition()` is a pure function that decides the new state.

| Relationship | From | To |
|---|---|---|
| SUPPORTS | OPEN | STRENGTHENED |
| SUPPORTS | STRENGTHENED / PARTIALLY_ADDRESSED / RESOLVED | unchanged |
| ADDRESSES | OPEN / STRENGTHENED | PARTIALLY_ADDRESSED |
| ADDRESSES | PARTIALLY_ADDRESSED, ≥ 2 distinct addressing papers | RESOLVED |
| ADDRESSES | PARTIALLY_ADDRESSED, < 2 distinct addressing papers | unchanged |
| ADDRESSES | RESOLVED | unchanged |
| CONTRADICTS | STRENGTHENED / RESOLVED | OPEN |
| CONTRADICTS | OPEN / PARTIALLY_ADDRESSED | unchanged |
| EXTENDS | any | unchanged |
| UNRELATED | any | unchanged |

Supporting rules:

- **Grounding** is required for SUPPORTS, ADDRESSES, CONTRADICTS and EXTENDS (`grounding.py:22-27`):
  the excerpt must match stored passages exactly or with ≥ 0.85 fuzzy similarity, or the paper abstract when
  no passages exist; otherwise `UngroundedEvidenceError` is raised and nothing is persisted.
  UNRELATED needs no excerpt.
- **Every classified pair creates one evidence row**, including UNRELATED; a history row is written only
  when the status changes. A database unique constraint on (opportunity, paper) prevents duplicates.
- **Distinct papers** are counted by canonical identity: DOI, else normalised title (≥ 5 characters), else
  source type and ID, else database ID (`get_paper_canonical_identity`, `service.py:19-46`).
- **Reopening boundary:** after a CONTRADICTS moves RESOLVED → OPEN, only addressing evidence recorded after
  that reopening counts toward resolution again.
- **Savepoints** are opened around the evidence insert (`service.py`), each paper in a temporal batch
  (`temporal.py`) and each candidate in incremental evolution (`incremental.py`).
- **Grounding scope:** it verifies that an excerpt exists in stored text. It does not verify that the
  relationship label is semantically correct.

---

## 3. Twelve-Scenario Deterministic Matrix

Run on 2026-10-21 (`pytest backend/tests/opportunity/test_stage5_lifecycle_matrix.py`): **12 passed**.

| # | Scenario | Asserted behaviour |
|---|---|---|
| 1 | Supports an open opportunity | Grounded SUPPORTS: OPEN → STRENGTHENED, evidence and history written |
| 2 | Directly addresses | One grounded ADDRESSES: OPEN → PARTIALLY_ADDRESSED |
| 3 | Documents the problem | SUPPORTS on PARTIALLY_ADDRESSED: evidence written, status unchanged |
| 4 | Extends related work | EXTENDS: grounded evidence written, status unchanged |
| 5 | Contradicts the premise | CONTRADICTS: STRENGTHENED → OPEN with history |
| 6 | Unrelated paper | UNRELATED: evidence row written, status unchanged, no history (the test's docstring says "does not create evidence", but its assertions require the evidence row) |
| 7 | Two distinct addressing papers | A second paper with the same normalised title does not count; a distinct paper (different DOI) moves PARTIALLY_ADDRESSED → RESOLVED |
| 8 | Contradiction after resolution | RESOLVED → OPEN; a subsequent single addressing paper reaches only PARTIALLY_ADDRESSED |
| 9 | Repeated evaluation | Re-applying the same paper raises "already exists"; status preserved |
| 10 | Fabricated excerpt | `UngroundedEvidenceError`; savepoint rolled back; no evidence |
| 11 | Temporal eligibility | Undated paper → `SKIPPED_UNDATED`; opportunity created after the paper's publication date is skipped |
| 12 | Provider failure | Candidate evaluation fails and rolls back; paper and other opportunities unchanged |

---

## 4. Related Test Results (2026-10-21, Python 3.14.8, SQLite, outbound network blocked)

| Command scope | Result |
|---|---|
| `backend/tests/opportunity` | 155 passed |
| `backend/tests/opportunity`, `ingestion`, `evaluation` | 356 passed, 9 skipped |
| Full backend suite (content of `17895a2`) | 689 passed, 0 failed, 24 skipped, 2 deselected |

Skips require PostgreSQL; deselected tests call live models. The original report's "312 passed" was an earlier
count of the opportunity/evaluation/ingestion directories before later tests were added.

Additional lifecycle tests added after the original report cover the duplicate-evidence savepoint race
(`test_stage3a_correctness.py`), the post-ingestion candidate cap applied after eligibility filtering
(`test_incremental_service.py`) and the `/evolve` lower temporal bound (`api/test_evolution_api.py`).

---

## 5. How Classifier Errors Would Propagate

The state machine applies whatever label it receives. Using the Stage 4D/4E observations (25 cases,
exploratory; see that report):

- In Stage 4D (attributed to the original prompt), ADDRESSES papers were often labelled EXTENDS (6 of 10), which would leave
  opportunities unchanged instead of moving them to PARTIALLY_ADDRESSED.
- In Stage 4E (attributed to the revised prompt; prompt text unverified), SUPPORTS papers were often labelled ADDRESSES (5 of 7: DEV-02, DEV-03, HELD-08,
  HELD-11, HELD-12), which would move opportunities toward PARTIALLY_ADDRESSED and potentially RESOLVED.

Grounding does not catch either pattern when the excerpt is genuine. These are inferences from the
classifier runs combined with the verified rules, not measurements of lifecycle outcomes on real data.

The only real-data lifecycle run, `five_paper_temporal_compact_20261019` (2026-10-19, before the prompt
revision; its fingerprint predates prompt-version recording, so the prompt is inferred to be the original), recorded four
evolution events (UNRELATED, CONTRADICTS, CONTRADICTS, EXTENDS) and no status transitions. Whether those labels
are correct has not been reviewed by domain experts.

---

## 6. Limitations

1. The matrix tests fixed inputs with fake providers; it does not measure classification accuracy.
2. Only SQLite was exercised; PostgreSQL locking and savepoint behaviour are unverified.
3. Live latency figures come from the 25-case classifier runs (mean 6.59 s and 7.90 s per case on local
   hardware) and do not characterise production throughput.
4. Opportunities generated live (wall-clock creation date) cannot be evolved through `/evolve` with papers
   published before their creation; runner-generated opportunities are backdated to the baseline cutoff.

## 7. Status

The lifecycle rules, grounding gate, duplicate protection, temporal checks and rollback behaviour are
implemented as specified and covered by deterministic tests on SQLite. The relationship classifier, not the
state machine, is the main source of uncertainty in lifecycle outcomes. No claim of production readiness or
real-world accuracy is made.
