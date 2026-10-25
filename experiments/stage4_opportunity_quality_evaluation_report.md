# IdeaGen Stage 4 — Benchmark v1 and Opportunity Quality Review (Revised)

- **Original report:** 2026-10-20
- **Revision:** 2026-10-21, re-verified against commit `17895a2`
- **Code:** `backend/ideagen/evaluation/stage4_benchmark.py`, `experiments/run_stage4_benchmark.py`,
  `backend/tests/evaluation/test_stage4_opportunity_quality_eval.py`

> **Revision note.** The original report headlined "100% accuracy, macro F1 1.0000" for relationship
> classification and an "APPROVED" verdict. That figure was **not a model measurement**: the runner passes
> the gold labels as the predictions (`run_stage4_benchmark.py:55-56`), so the score is 100% by construction.
> The original also described an observed `OPEN → STRENGTHENED → PARTIALLY_ADDRESSED → RESOLVED`
> progression on 2023 literature; no pipeline run produced that sequence (see §5). This revision states what
> Benchmark v1 actually tests. Live classifier accuracy is reported separately in the Stage 4D/4E report.

---

## 1. What Benchmark v1 Is

Benchmark v1 is a **15-case deterministic regression suite**, not an empirical evaluation of model output.
It exercises three pieces of code with fixed inputs:

1. the evidence-grounding validator (`validate_evidence_grounding`, `opportunity/grounding.py`),
2. the pure lifecycle state machine (`transition`, `opportunity/state_machine.py`),
3. the metric arithmetic used for confusion matrices (`run_classification_evaluation`).

No language model, embedding model, database or PDF is involved when the benchmark runs.

---

## 2. Case Composition (verified in `get_stage4_benchmark_dataset`, `stage4_benchmark.py:56`)

| Group | Cases | Gold labels | Notes |
|---|---|---|---|
| Literature-derived | LIT-01 … LIT-05 | SUPPORTS, ADDRESSES, ADDRESSES, EXTENDS, UNRELATED | All five are scored against one seed opportunity derived from Lewis et al. (2020). |
| Synthetic | SYN-01 … SYN-10 | SUPPORTS ×2, ADDRESSES ×3, CONTRADICTS ×3, EXTENDS ×1, UNRELATED ×1 | Constructed around the grounding threshold and state-machine branches. |

Class totals: SUPPORTS 3, ADDRESSES 5, CONTRADICTS 3, EXTENDS 2, UNRELATED 2.

Provenance limitations of the "literature" cases:

- LIT-01 … LIT-04 use real paper IDs from the five-paper corpus, but their passages are **strings typed
  into the benchmark file**, not passages produced by the ingestion pipeline. Compared with the local PDFs
  they are close but mostly not verbatim; only LIT-04's excerpt appears verbatim.
- LIT-05 (He et al., 2016, ResNet) uses a **placeholder UUID** (`stage4_benchmark.py:179`) and a hard-coded
  abstract snippet; there is no ResNet PDF in the repository.
- Labels were assigned by the project author; there was no second annotator and no agreement statistic.

---

## 3. Results

Reproduced on 2026-10-21 with `backend/.venv/bin/python experiments/run_stage4_benchmark.py` (offline; writes
no files) and `pytest backend/tests/evaluation/test_stage4_opportunity_quality_eval.py` (13 passed).

### 3.1 Grounding validator — deterministic regression coverage

| Check | Result | What it shows |
|---|---|---|
| 11 cases with genuine excerpts accepted | 11/11 | The validator accepts excerpts present in the supplied passage text (exact or ≥ 0.85 fuzzy match). |
| 4 invalid cases rejected (SYN-03 fabricated, SYN-04 mismatched excerpt, SYN-05 missing excerpt, SYN-06 sub-threshold paraphrase) | 4/4 | The validator fails closed on these constructed inputs. |

Qualifications:

- All four rejected cases are synthetic; no real-literature case tests rejection.
- The passages are the strings embedded in each case, not pipeline output.
- UNRELATED skips grounding by design (`grounding.py:128-130`), so LIT-05 and SYN-09 pass without excerpts.
- SYN-04 ("cross-paper attribution") checks an excerpt against a single passage that does not contain it;
  no second paper's passages are involved.
- **Grounding verifies that an excerpt can be found in stored text. It does not verify that the
  relationship label is correct**; a correctly quoted excerpt can accompany a wrong classification.

### 3.2 Classification metrics — arithmetic check only

The runner computes a confusion matrix from `gold_preds = [c.expected_relationship for c in cases]`
(`run_stage4_benchmark.py:55`). The resulting diagonal matrix and 1.0000 precision/recall/F1 confirm only
that the metric code is arithmetically consistent. **They are not evidence of classifier accuracy.**

The original "sensitivity analysis under classification noise" (86.67%) is the unit test
`test_classification_confusion_matrix_with_simulated_misclassifications` (test file lines 190-218), which
hand-sets two wrong labels to check metric arithmetic. It is not an analysis of model behaviour.

### 3.3 State machine — deterministic regression coverage

`transition()` returns the hand-specified expected status for all 15 cases (15/15), including:
a single ADDRESSES paper reaches PARTIALLY_ADDRESSED but not RESOLVED; a second distinct addressing paper
reaches RESOLVED; CONTRADICTS reopens RESOLVED to OPEN. These inputs use the gold relationship labels.

The "end-to-end lifecycle" check (also 15/15) reads the gold `expected_grounding_valid` flag
(`stage4_benchmark.py:616-619`) instead of running grounding or the transition service, so it adds no
coverage beyond §3.1 and §3.3.

---

## 4. Opportunity Quality Observations (single-paper smoke run)

Source: `experiments/artifacts/smoke_one_paper_stage4_fix_20261019/` (opportunity
`ef562805-e2fc-4eb3-bf0c-c263c80f77b6`; corpus of one paper, `lewis2020_rag.pdf`).

- **Direction:** "Investigate the integration of task-specific knowledge modules into large pre-trained
  language models (PLMs) to enhance their performance on knowledge-intensive …" (status VALIDATED,
  confidence 0.75, 0 refinement iterations).
- **Specificity:** the stored text names TriviaQA and HotpotQA as evaluation benchmarks. The term
  "task-specific knowledge module" is not further specified (adapter, verifier, memory, etc.).
- **Traceability:** all 10 prior-art statement IDs resolve to stored statements with passage links. With
  a one-paper corpus this is necessarily the same paper, so it shows provenance linkage, not breadth of
  prior-art search.
- **Duplication:** not assessable from a one-paper run.

These are qualitative observations by the author on one generated opportunity; no rubric or second
reviewer was used.

---

## 5. Historical Pipeline Context

- **Failed run `five_paper_temporal_fixed_20261019`:** status `FAILED (Zero eligible statements found for
  temporal cutoff 2020-12-31.)`. Extraction for `lewis2020_rag.pdf`, the only pre-cutoff paper, failed with
  an Ollama timeout after 300 s; the other four papers produced 122 statements, none eligible. The pipeline
  correctly refused to discover gaps from post-cutoff statements.
- **Validated run `five_paper_temporal_compact_20261019`:** the one generated opportunity was evaluated
  against four later papers, giving UNRELATED, CONTRADICTS, CONTRADICTS and EXTENDS, with **no status
  transitions** (it remained OPEN). See `docs/experiments/five_paper_temporal_experiment_20261019.md`.
- The original report's progression to RESOLVED existed only as hand-labelled benchmark cases and manual
  state-machine sequences in tests, not as an observed pipeline outcome. That claim is withdrawn.

---

## 6. Limitations

1. Classification accuracy is not measured by this benchmark (see Stage 4D/4E for live runs).
2. 15 cases, one seed opportunity, one annotator; synthetic cases were designed around the implementation.
3. Benchmark passages are hand-entered text, not ingestion output; one "literature" case has no source file.
4. Grounding checks textual presence, not semantic correctness of the relationship.

## 7. Status

Benchmark v1 is retained as a **deterministic regression suite** for the grounding validator, state
machine and metric arithmetic (13 tests passing). It does not support claims about classifier accuracy,
opportunity quality at scale, or scientific validity. The original "APPROVED" verdict is withdrawn.
