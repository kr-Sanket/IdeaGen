# IdeaGen Stage 4D/4E — Live Relationship Classifier Evaluation (Revised)

- **Original report:** 2026-10-20
- **Revision:** 2026-10-21, re-verified against commit `17895a2` and the run artifacts below
- **Code:** `backend/ideagen/evaluation/stage4d_evaluator.py`, `experiments/run_stage4d_classifier_evaluation.py`,
  `backend/tests/evaluation/test_stage4d_classifier_eval.py`
- **Classifier:** `RelationshipClassifier.classify()` (`backend/ideagen/opportunity/classifier.py`)
- **Cases:** Stage 4B Benchmark v2 (25 cases: DEV-01…DEV-10, HELD-01…HELD-15)

> **Revision note.** The original report stated that empirical accuracy was "WITHHELD / NOT MEASURED",
> named the model `deepseek-r1:8b`, used four classes (ADDRESSED/EXTENDED/CONTRADICTED/UNRELATED), listed
> case IDs such as `stage4b-heldout-001`, gave a composition of 11/4/10, and documented a `--provider gemini`
> flag. **None of this matches the code or the artifacts.** The evaluator uses the five relationship classes
> and DEV/HELD case IDs; two live runs with `qwen3:8b` exist and are reported here; the runner has no
> `--provider` flag. The sections on input isolation and abstention handling were broadly correct and are kept.

---

## 1. Summary

| Run | Artifact | Correct | Accuracy | 95% Wilson score interval\* |
|---|---|---|---|---|
| Stage 4D (attributed: original prompt) | `experiments/artifacts/stage4d_live_20261020_180110/` | 14/25 | **56%** | 0.37 – 0.73 |
| Stage 4E (attributed: revised prompt) | `experiments/artifacts/stage4e_prompt_v2_20261021/` | 17/25 | **68%** | 0.48 – 0.83 |

\* Intervals are not part of the run artifacts; they were computed for this revision from the correct/total
counts (two-sided 95% Wilson score interval, z = 1.96) and cross-checked with
`scipy.stats.binomtest(k, n).proportion_ci(method="wilson")` (SciPy 1.18.1). They describe binomial sampling
uncertainty only, not label or prompt uncertainty.

Prompt attribution in this report is **inferred, not verified** (§2): both artifact headers record
`Prompt Version: v1`, and the prompt text was not saved.

**These are exploratory results, not independent held-out validation.** The Stage 4E run used the same
25 cases whose Stage 4D errors are reported to have motivated the prompt revision, and the HELD_OUT split
contains a duplicate of a DEV case (§5). The intervals overlap heavily; with 25 cases the improvement is not statistically
established. Neither run demonstrates generalisation to new literature.

The revision traded one error mode for another: EXTENDS over-prediction disappeared, ADDRESSES recall rose to
10/10, but **SUPPORTS recall fell from 3/7 to 1/7** as supporting papers were relabelled ADDRESSES.

---

## 2. Method (verified in code)

- **Input isolation.** `build_isolated_classifier_input` (`stage4d_evaluator.py:93-108`) passes the
  opportunity title, problem and direction, the paper title, the case passages joined as the abstract, and the
  case's `claimed_excerpt` as the prompt excerpt. The test suite asserts that the gold label and grounding flag
  are absent from the classifier input (`test_stage4d_classifier_eval.py:88-90`); the input type has no
  lifecycle-status field.
- **Consequence:** for the three invalid-grounding cases (DEV-10, HELD-04, HELD-11) the model is shown the
  fabricated or mismatched excerpt. Classification scoring ignores grounding: HELD-04's fabricated excerpt
  predicted as ADDRESSES counts as correct.
- **Abstention handling.** Provider failures and malformed outputs are recorded as non-SUCCESS statuses and
  excluded from scoring rather than replaced with gold labels. Both runs had 25/25 successful predictions.
- **Provider.** `OllamaLLMProvider`, model `qwen3:8b` (both runs; recorded in each `stage4d_live_metrics.json`).

### Prompt provenance

The artifacts do not store the prompt text. Both artifact reports record `Prompt Version: v1` (line 7),
including the run in the `stage4e_prompt_v2_20261021` directory. The prompt text in the working tree had been
revised without changing the version constant; the constant was changed to `v2` only in commit `17895a2`.
The 4E run is therefore attributed to the revised prompt on the basis of the directory name, its timing
(42 minutes after 4D) and its changed behaviour. **The exact prompt text used in either run cannot be
reproduced from the artifacts** (UNVERIFIED).

---

## 3. Results

### 3.1 By split and category (from `metrics_by_split` / `metrics_by_category` in each JSON)

| Subset | Stage 4D | Stage 4E |
|---|---|---|
| DEV (10) | 5/10 | 6/10 |
| HELD_OUT (15) | 9/15 | 11/15 |
| REAL_LITERATURE (11) | 2/11 (Wilson\* 0.05 – 0.48) | 7/11 (0.35 – 0.85) |
| HUMAN_CURATED_PILOT (13, fictional vignettes) | 12/13 (0.67 – 0.99) | 10/13 (0.50 – 0.92) |
| SYNTHETIC_STRESS_TEST (1) | 0/1 | 0/1 |

The real-literature subset improved; the pilot subset regressed (DEV-03 and HELD-08).

### 3.2 Confusion matrices (rows = gold, columns = predicted; computed from per-case records)

Stage 4D:

| Gold \ Pred | SUPPORTS | ADDRESSES | CONTRADICTS | EXTENDS | UNRELATED | Recall |
|---|---|---|---|---|---|---|
| SUPPORTS (7) | 3 | 0 | 0 | 3 | 1 | 3/7 |
| ADDRESSES (10) | 0 | 4 | 0 | 6 | 0 | 4/10 |
| CONTRADICTS (4) | 0 | 0 | 3 | 1 | 0 | 3/4 |
| EXTENDS (2) | 0 | 0 | 0 | 2 | 0 | 2/2 |
| UNRELATED (2) | 0 | 0 | 0 | 0 | 2 | 2/2 |

Stage 4E:

| Gold \ Pred | SUPPORTS | ADDRESSES | CONTRADICTS | EXTENDS | UNRELATED | Recall |
|---|---|---|---|---|---|---|
| SUPPORTS (7) | 1 | 5 | 0 | 0 | 1 | 1/7 |
| ADDRESSES (10) | 0 | 10 | 0 | 0 | 0 | 10/10 |
| CONTRADICTS (4) | 0 | 1 | 3 | 0 | 0 | 3/4 |
| EXTENDS (2) | 0 | 1 | 0 | 1 | 0 | 1/2 |
| UNRELATED (2) | 0 | 0 | 0 | 0 | 2 | 2/2 |

Per-class precision / recall / F1:

| Class | 4D P / R / F1 | 4E P / R / F1 |
|---|---|---|
| SUPPORTS | 1.00 / 0.43 / 0.60 | 1.00 / 0.14 / 0.25 |
| ADDRESSES | 1.00 / 0.40 / 0.57 | 0.59 / 1.00 / 0.74 |
| CONTRADICTS | 1.00 / 0.75 / 0.86 | 1.00 / 0.75 / 0.86 |
| EXTENDS | 0.17 / 1.00 / 0.29 | 1.00 / 0.50 / 0.67 |
| UNRELATED | 0.67 / 1.00 / 0.80 | 0.67 / 1.00 / 0.80 |

### 3.3 Errors

**Stage 4D (11 errors):** DEV-01, DEV-09, HELD-02, HELD-03, HELD-04, HELD-13 (ADDRESSES→EXTENDS);
DEV-02, HELD-11, HELD-12 (SUPPORTS→EXTENDS); DEV-05 (CONTRADICTS→EXTENDS); DEV-10 (SUPPORTS→UNRELATED).
EXTENDS was predicted 12 times, 10 of them incorrectly.

**Stage 4E (8 errors):** DEV-02, DEV-03, HELD-08, HELD-11, HELD-12 (SUPPORTS→ADDRESSES); DEV-05
(CONTRADICTS→ADDRESSES); HELD-01 (EXTENDS→ADDRESSES); DEV-10 (SUPPORTS→UNRELATED).
ADDRESSES was predicted 17 times, 7 of them incorrectly.

Persistent errors in both runs: DEV-02, DEV-05, DEV-10, HELD-11, HELD-12. DEV-10's gold label is itself
debatable (its excerpt does not match its passage), so both runs' UNRELATED prediction may be reasonable.

### 3.4 Run statistics

| | Stage 4D | Stage 4E |
|---|---|---|
| Timestamp (UTC) | 2026-10-20 17:58:25 | 2026-10-20 18:40:10 |
| Duration | 164.8 s | 197.4 s |
| Latency mean / median (s) | 6.59 / 6.58 | 7.90 / 7.46 |
| Reported confidence values | 0.85 or 0.95 | 0.95 for every case |

Constant reported confidence in 4E means classifier confidence carries no information about correctness here.

---

## 4. Lifecycle Implications

The lifecycle state machine applies the predicted label deterministically. With the observed error patterns:

- **Original prompt:** ADDRESSES papers labelled EXTENDS would leave opportunities unchanged instead of
  advancing them to PARTIALLY_ADDRESSED.
- **Revised prompt:** SUPPORTS papers labelled ADDRESSES would advance opportunities toward
  PARTIALLY_ADDRESSED and, with two distinct papers, RESOLVED. This is the more consequential risk for a system
  that tracks whether a research gap remains open.

Evidence grounding does not mitigate either error mode when the quoted excerpt is genuine: grounding confirms
that the excerpt exists in stored text, not that the relationship label is correct.

---

## 5. Validity Limitations

1. **Same cases used for tuning and evaluation.** The Stage 4E run used the same 25 cases that revealed the
   Stage 4D errors (verified from the case IDs). Per the project record the prompt was revised against those
   errors; the prompt text itself is unverified (§2). HELD_OUT figures are not held-out estimates.
2. **HELD-14 duplicates DEV-08** as classifier input.
3. **Sample size.** 25 cases; per-class supports of 2–10; confidence intervals are wide and overlapping.
4. **Case realism.** 13 of 25 cases are fictional vignettes; real-literature passages were typed into the
   benchmark rather than produced by the ingestion pipeline; one real-literature case (HELD-15) has no
   local source file.
5. **Labels.** Assigned by the project author; no inter-annotator agreement; six cases are flagged borderline.
6. **Prompt provenance** cannot be reproduced from the artifacts (§2).
7. **Single model and setting.** One local model, one run per prompt, no repeated sampling.

---

## 6. Evaluation-Harness Tests

`backend/tests/evaluation/test_stage4d_classifier_eval.py` — 5 tests, all passing on 2026-10-21 with a mock
provider: predictions are never derived from gold labels; off-diagonal confusion entries are recorded
correctly; provider and malformed-output failures become abstentions; split/category partitions are disjoint
(10/15; 11/13/1); a no-provider run reports no accuracy.

Note: the runner's "SAFEGUARD VERIFIED" and four "[PASS]" lines
(`run_stage4d_classifier_evaluation.py:296-302`) are unconditional print statements, not checks. The test
suite, not those messages, is the evidence for the safeguards.

---

## 7. Reproduction

```
# Live run (requires a reachable Ollama server with qwen3:8b; writes a new directory under experiments/artifacts/)
backend/.venv/bin/python experiments/run_stage4d_classifier_evaluation.py --model qwen3:8b
# Offline check: no provider calls, no files written
backend/.venv/bin/python experiments/run_stage4d_classifier_evaluation.py --dry-run-check
```

Options are `--dry-run-check`, `--timeout`, `--model` and `--output-dir`. The provider is taken from
settings (`IDEAGEN_LLM_PROVIDER`), not a command-line flag. A rerun with the current code would use the
committed `v2` prompt text and is not guaranteed to reproduce either run above.

---

## 8. Next Steps for a Credible Evaluation

Freeze a prompt version and record its text in the run artifact; collect new real-literature cases from
pipeline output; label them with at least two annotators and report agreement; keep them unseen until
evaluation; report per-class results with intervals; and repeat runs to measure variability.
