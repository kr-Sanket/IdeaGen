# IdeaGen Stage 4B — Benchmark Credibility Audit and Benchmark v2 (Revised)

- **Original report:** 2026-10-20
- **Revision:** 2026-10-21, re-verified against commit `17895a2`
- **Code:** `backend/ideagen/evaluation/stage4b_benchmark.py`, `experiments/run_stage4b_benchmark.py`,
  `backend/tests/evaluation/test_stage4b_benchmark_robustness.py`

> **Revision note.** The original audit correctly found that Benchmark v1 compared gold labels with
> themselves. It then reported a **held-out macro F1 of 1.0000 for Benchmark v2 computed the same way**
> (`run_stage4b_benchmark.py:55-56`), described the held-out split as "strictly isolated", described the
> pilot cases as coming from "vetted pilot fixtures", and stated that the benchmark carries "full
> provenance hashes". None of these hold (see §3–§4). This revision keeps the audit findings that are
> correct and corrects the rest. Live classifier results on this benchmark are in the Stage 4D/4E report.

---

## 1. Audit Findings on Benchmark v1 (retained, verified)

1. **Gold-vs-gold classification.** Benchmark v1's classification metrics use the expected labels as the
   predictions; the 100% score is an arithmetic identity, not a model measurement.
2. **Labels derived from the implementation.** Several synthetic cases were built around the 0.85
   grounding threshold and individual state-machine branches.
3. **No development/evaluation separation.** All 15 cases were used for both design and reporting.
4. **Single annotator.** No inter-annotator agreement was measured.
5. **Synthetic and literature results were aggregated** into one headline figure.
6. **Weak provenance for one case.** LIT-05 uses a placeholder paper ID and hard-coded text.

---

## 2. Benchmark v2 Composition (verified in `get_stage4b_benchmark_dataset`, `stage4b_benchmark.py:82`)

25 cases, `benchmark_version = "v2.0-robustness"`, with case IDs DEV-01…DEV-10 and HELD-01…HELD-15.

| Category | DEV | HELD_OUT | Total | Source of text |
|---|---|---|---|---|
| REAL_LITERATURE | 3 | 8 | 11 | Passages typed into the benchmark file from five-paper-corpus PDFs (close to, but mostly not verbatim with, the PDFs) plus one ResNet control (HELD-15) with no local PDF |
| HUMAN_CURATED_PILOT | 6 | 7 | 13 | **Fictional author-written vignettes** (no DOIs, authors or real paper IDs). DEV-03…DEV-07 come from `backend/tests/opportunity/fixtures/classification_fixtures.json`; the others were written for this benchmark |
| SYNTHETIC_STRESS_TEST | 1 | 0 | 1 | DEV-10, constructed cross-paper excerpt |

Gold-label distribution: ADDRESSES 10, SUPPORTS 7, CONTRADICTS 4, EXTENDS 2, UNRELATED 2.
DEV has no EXTENDS or UNRELATED cases. Six cases carry a borderline flag with an alternative label
(DEV-06, HELD-01, HELD-02, HELD-06, HELD-08, HELD-12).

"Pilot" therefore means hand-written scenario text, not real publications.

---

## 3. Validity Problems Found in Benchmark v2

| Issue | Evidence | Consequence |
|---|---|---|
| Held-out metrics were gold-vs-gold | `run_stage4b_benchmark.py:55` | The reported 1.0000 held-out macro F1 is not a measurement and is withdrawn. |
| DEV macro of 0.6000 was an artefact | Macro averages divide by all five classes and score empty classes 0 (`stage4b_benchmark.py:988-989`) | DEV's 0.6000 reflects missing EXTENDS/UNRELATED cases, not errors. |
| **HELD-14 duplicates DEV-08** | Identical paper ID (`pilot-fl-003`), title, opportunity, passages, excerpt and gold label | The classifier receives the same input in both splits, so the held-out split is not independent. |
| Held-out cases were inspected during development | Stated in the original Stage 4D report; the Stage 4E run (attributed to a revised prompt; prompt text unverified) used the same 25 cases | Results on HELD_OUT are not held-out estimates. |
| "Provenance hashes" | Claimed in the module docstring (`stage4b_benchmark.py:5`); no hash field exists | Claim withdrawn. |
| DEV-10 label is debatable | Gold SUPPORTS, but its excerpt does not match its passage's topic | Both live runs predicted UNRELATED; the label itself may be wrong. |
| Classifier sees fabricated excerpts | The Stage 4D evaluator passes `claimed_excerpt` as the prompt excerpt (`stage4d_evaluator.py:107`) | For DEV-10, HELD-04 and HELD-11, the model is shown the invalid excerpt. |

---

## 4. Results That Are Valid

Reproduced on 2026-10-21 with `backend/.venv/bin/python experiments/run_stage4b_benchmark.py` (offline;
writes no files) and `pytest backend/tests/evaluation/test_stage4b_benchmark_robustness.py` (10 passed).

### 4.1 Grounding validator — deterministic regression coverage, stratified

| Category | Valid excerpts accepted | Invalid excerpts rejected |
|---|---|---|
| REAL_LITERATURE | 9/9 | 2/2 (HELD-04 fabricated claim; HELD-11 loose paraphrase) |
| HUMAN_CURATED_PILOT | 13/13 | — (no invalid pilot cases) |
| SYNTHETIC_STRESS_TEST | — | 1/1 (DEV-10 cross-paper excerpt) |

DEV-09 confirms that an excerpt typed without the `ﬁ` ligature is accepted against a passage containing it.
These checks run the real `validate_evidence_grounding` on the case-embedded passages using the gold
relationship. They show that the validator behaves as designed on these inputs. They do **not** show that
relationship labels are correct, and they say nothing about pipeline-produced passages.

### 4.2 Classifier evaluation

Benchmark v2 itself contains no model predictions. The first live predictions on these 25 cases are the
Stage 4D and Stage 4E runs (14/25 and 17/25 correct respectively; see that report, including why these are
exploratory rather than held-out results).

### 4.3 State machine

The test file checks state-machine outcomes using gold labels and the gold `expected_grounding_valid` flag
(`test_stage4b_benchmark_robustness.py:215-230`). This duplicates Benchmark v1's state-machine coverage;
it is not a separate evaluation over 25 cases.

---

## 5. Borderline Cases (retained)

| Case | Gold | Alternative | Reason for ambiguity |
|---|---|---|---|
| DEV-06 | SUPPORTS | UNRELATED | Survey restating a known limitation |
| HELD-01 (Gao 2023, ALCE) | EXTENDS | ADDRESSES | Introduces a benchmark and evaluates baselines |
| HELD-02 (Asai 2023, Self-RAG) | ADDRESSES | EXTENDS | New paradigm vs direct mitigation |
| HELD-06 | CONTRADICTS | SUPPORTS | Impossibility result could support difficulty or refute feasibility |
| HELD-08 | SUPPORTS | UNRELATED | Survey without new data |
| HELD-12 (Asai 2023) | SUPPORTS | ADDRESSES | Excerpt describes prior-art defects while the paper addresses them |

These reflect single-annotator judgement and have not been adjudicated by independent reviewers.

---

## 6. Tests and Reproduction

```
backend/.venv/bin/python experiments/run_stage4b_benchmark.py
PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/pytest -p no:cacheprovider -q backend/tests/evaluation
```

On 2026-10-21 `backend/tests/evaluation` gave **58 passed** (Stage 4 v1: 13, Stage 4B: 10, Stage 4D: 5,
clustering metrics: 11, baselines: 17, schemas: 2). The original report's "53 passed" was an earlier count.

## 7. Status

Benchmark v2 is useful as **stratified regression coverage for the grounding validator** and as a
**fixed input set** for live classifier runs. It is not an independent held-out benchmark: its held-out
split duplicates a DEV case, was inspected during development, and was reused for the Stage 4E run, which is attributed to a revised prompt.
Most of its non-literature cases are fictional vignettes. The original "APPROVED" verdict is withdrawn.

A credible held-out evaluation would need newly collected real-literature cases, labelled by at least two
annotators with measured agreement, and kept unseen until a prompt version is frozen.
