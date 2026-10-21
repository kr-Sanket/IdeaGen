"""Topic-based relevance screening for research opportunity discovery (Phase 3).

Executes deterministic, embedding-based relevance ranking of discovery-eligible
papers before Stage 2 LLM extraction.

Design & Safety:
- Operates after Stage 1 ingestion and before Stage 2 LLM extraction.
- Does NOT delete papers or passages from the database.
- Bounded to pre-cutoff discovery candidates; post-cutoff papers remain
  unfiltered for Stage 5 temporal evolution.
- Combines paper-level (title + abstract) relevance with top body-passage relevance.
- Excludes reference-list sections from scoring.
- Completely offline and deterministic (no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import datetime
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np

from ideagen.discovery.embeddings import EmbeddingProvider
from ideagen.discovery.temporal import extract_paper_publication_date, is_paper_eligible
from ideagen.opportunity.integration import resolve_paper_abstract

# Abstract bound for the paper header embedded during scoring; the input signature hashes
# the same bounded text so a changed abstract invalidates reused decisions.
SCREENING_ABSTRACT_MAX_CHARS = 2000

logger = logging.getLogger(__name__)

DEFAULT_SCREENING_THRESHOLD: float = 0.50
DEFAULT_SCREENING_MIN_PAPERS: int = 0
DEFAULT_SCREENING_ALPHA: float = 0.50
REFERENCE_SECTION_KEYWORDS: Tuple[str, ...] = (
    "reference",
    "references",
    "bibliography",
    "works cited",
    "citations",
)


def is_reference_passage(passage: Any) -> bool:
    """Returns True if the passage belongs to a reference or bibliography section."""
    sec = (getattr(passage, "section", None) or "").strip().lower()
    if not sec:
        return False
    return any(kw in sec for kw in REFERENCE_SECTION_KEYWORDS)


def is_abstract_passage(passage: Any) -> bool:
    """Returns True if the passage belongs to an abstract section."""
    sec = (getattr(passage, "section", None) or "").strip().lower()
    return bool(sec and "abstract" in sec)


def compute_relevance_score(
    paper_sim: float,
    body_sims: Sequence[float],
    alpha: float = DEFAULT_SCREENING_ALPHA,
) -> float:
    """Combines paper-level and body-passage relevance scores deterministically.

    Formula:
        Score = alpha * paper_sim + (1 - alpha) * avg(top_3_body_sims)

    If no body passages exist, falls back to paper_sim.
    Averages top-3 body passages to prevent a single noisy passage from
    dominating the paper's ranking.

    Args:
        paper_sim: Cosine similarity of topic with paper header (title + abstract).
        body_sims: Sequence of cosine similarities of non-reference body passages.
        alpha: Weight for paper-level relevance (default 0.5).

    Returns:
        Combined score rounded to 6 decimal places.
    """
    if not body_sims:
        return round(float(paper_sim), 6)

    sorted_body = sorted([float(s) for s in body_sims], reverse=True)
    top_m = sorted_body[: min(3, len(sorted_body))]
    body_avg = sum(top_m) / len(top_m)

    combined = alpha * float(paper_sim) + (1.0 - alpha) * body_avg
    return round(float(combined), 6)


@dataclass(frozen=True)
class PassageSnippet:
    """Summary of a top relevant passage for auditability."""

    passage_id: str
    passage_index: int
    section: Optional[str]
    page_start: int
    page_end: int
    similarity: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "passage_index": self.passage_index,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "similarity": round(self.similarity, 4),
            "snippet": self.snippet,
        }


@dataclass
class PaperScreeningResult:
    """Individual screening outcome and provenance for a paper."""

    paper_id: uuid.UUID
    filename: str
    title: Optional[str]
    publication_date: Optional[str]
    score: float
    rank: int
    decision: str  # SELECTED_ABOVE_THRESHOLD, SELECTED_BELOW_THRESHOLD_FLOOR, EXCLUDED_BELOW_THRESHOLD, EXCLUDED_CAPACITY_LIMIT
    reason: str
    paper_level_similarity: float
    body_similarity: float
    top_passages: List[PassageSnippet] = field(default_factory=list)
    floor_included: bool = False
    is_pre_cutoff: bool = True
    temporal_workflow_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": str(self.paper_id),
            "filename": self.filename,
            "title": self.title,
            "publication_date": self.publication_date,
            "score": round(self.score, 4),
            "rank": self.rank,
            "decision": self.decision,
            "reason": self.reason,
            "paper_level_similarity": round(self.paper_level_similarity, 4),
            "body_similarity": round(self.body_similarity, 4),
            "floor_included": self.floor_included,
            "is_pre_cutoff": self.is_pre_cutoff,
            "temporal_workflow_note": self.temporal_workflow_note,
            "top_passages": [p.to_dict() for p in self.top_passages],
        }


@dataclass
class ScreeningReport:
    """Full auditable screening report for an experiment run."""

    topic: str
    threshold: float
    min_papers: int
    max_papers: Optional[int]
    total_eligible_papers: int
    selected_count: int
    excluded_count: int
    floor_warning: Optional[str] = None
    results: List[PaperScreeningResult] = field(default_factory=list)
    metric_interpretation_note: str = (
        "Embedding cosine similarity is an automated relevance heuristic for candidate prioritization, "
        "not definitive proof of scientific relevance or methodological quality."
    )
    input_signature: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "enabled": True,
            "topic": self.topic,
            "threshold": self.threshold,
            "min_papers": self.min_papers,
            "max_papers": self.max_papers,
            "total_eligible_papers": self.total_eligible_papers,
            "selected_count": self.selected_count,
            "excluded_count": self.excluded_count,
            "floor_warning": self.floor_warning,
            "metric_interpretation_note": self.metric_interpretation_note,
            "ranked_papers": [r.to_dict() for r in self.results],
        }
        if self.input_signature is not None:
            d["input_signature"] = self.input_signature
        return d


def compute_screening_input_signature(
    candidates: Sequence[Tuple[Any, Sequence[Any], str]],  # (paper, passages, filename)
    topic: str,
    threshold: float,
    min_papers: int,
    max_papers: Optional[int],
    embedding_provider_name: str,
    embedding_model: str,
    embedding_dimension: int,
    temporal_cutoff: Optional[Union[datetime.date, datetime.datetime, str]] = None,
) -> Dict[str, Any]:
    """Generates a deterministic signature for the exact relevance screening input set.

    Binds the candidate papers, passage contents, topic, threshold, min/max limits,
    embedding provider/model/dimension, and cutoff into a canonical SHA-256 hash.
    """
    clean_topic = str(topic).strip() if topic else ""
    norm_cutoff_str = None
    if temporal_cutoff is not None:
        if isinstance(temporal_cutoff, (datetime.date, datetime.datetime)):
            norm_cutoff_str = temporal_cutoff.isoformat().split("T")[0]
        else:
            norm_cutoff_str = str(temporal_cutoff).split("T")[0]

    candidate_records: List[Dict[str, Any]] = []
    for paper, passages, filename in candidates:
        paper_id_str = str(getattr(paper, "id", ""))
        title = (getattr(paper, "title", None) or "").strip()
        pub_date = getattr(paper, "publication_date", None)
        pub_date_str = pub_date.isoformat() if pub_date else None

        # Build non-reference body passage text digests
        valid_body_passages = [
            p for p in passages
            if not is_reference_passage(p) and not is_abstract_passage(p) and getattr(p, "text", None)
        ]
        passages_hasher = hashlib.sha256()
        for p in sorted(valid_body_passages, key=lambda x: int(getattr(x, "passage_index", 0))):
            p_idx = getattr(p, "passage_index", 0)
            p_sec = getattr(p, "section", "") or ""
            p_text = (getattr(p, "text", "") or "").strip()
            passages_hasher.update(f"{p_idx}:{p_sec}:{p_text}\n".encode("utf-8"))

        # Header abstract exactly as embedded by screen_papers (abstract passages are excluded above).
        header_abstract = resolve_paper_abstract(paper, passages, max_abstract_chars=SCREENING_ABSTRACT_MAX_CHARS) or ""
        abstract_digest = hashlib.sha256(header_abstract.strip().encode("utf-8")).hexdigest()

        paper_hasher = hashlib.sha256()
        paper_hasher.update(
            f"{paper_id_str}:{filename}:{title}:{pub_date_str}:{abstract_digest}:{passages_hasher.hexdigest()}".encode("utf-8")
        )

        candidate_records.append({
            "paper_id": paper_id_str,
            "filename": str(filename),
            "candidate_hash": paper_hasher.hexdigest(),
        })

    candidate_records.sort(key=lambda c: (c["filename"].lower(), c["paper_id"]))
    candidates_hasher = hashlib.sha256()
    for c in candidate_records:
        candidates_hasher.update(f"{c['filename']}:{c['paper_id']}:{c['candidate_hash']}\n".encode("utf-8"))

    canonical = {
        "topic": clean_topic,
        "threshold": float(threshold),
        "min_papers": int(min_papers),
        "max_papers": int(max_papers) if max_papers is not None else None,
        "embedding_provider": str(embedding_provider_name),
        "embedding_model": str(embedding_model),
        "embedding_dimension": int(embedding_dimension),
        "temporal_cutoff": norm_cutoff_str,
        "candidate_count": len(candidate_records),
        "candidates_hash": candidates_hasher.hexdigest(),
    }
    sig_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    canonical["signature_hash"] = sig_hash
    return canonical


def validate_screening_report_for_reuse(
    stored_report: Dict[str, Any],
    current_signature: Dict[str, Any],
    candidate_paper_ids: Sequence[Union[str, uuid.UUID]],
) -> Tuple[bool, Optional[str]]:
    """Checks whether a persisted screening report is valid and safe for reuse.

    Returns:
        (True, None) if safe to reuse.
        (False, reason_str) if invalid or stale.
    """
    if not isinstance(stored_report, dict):
        return False, "Stored screening report is not a dictionary."

    # Validate essential fields
    required_keys = ("topic", "threshold", "min_papers", "max_papers", "ranked_papers")
    for k in required_keys:
        if k not in stored_report:
            return False, f"Stored screening report missing required field: '{k}'."

    ranked = stored_report.get("ranked_papers")
    if not isinstance(ranked, list):
        return False, "Stored screening report 'ranked_papers' is not a list."

    # If the stored report has an input_signature, compare signature hashes
    stored_sig = stored_report.get("input_signature")
    if isinstance(stored_sig, dict) and "signature_hash" in stored_sig:
        if stored_sig.get("signature_hash") != current_signature.get("signature_hash"):
            return False, (
                f"Screening input signature mismatch: stored={stored_sig.get('signature_hash')}, "
                f"current={current_signature.get('signature_hash')}."
            )
        return True, None

    # Fallback validation for reports without input_signature (backward compatibility):
    # Verify core settings match
    for k in ("topic", "threshold", "min_papers", "max_papers"):
        if stored_report.get(k) != current_signature.get(k):
            return False, f"Screening setting mismatch for '{k}': stored={stored_report.get(k)}, current={current_signature.get(k)}."

    # Verify candidate count and paper membership match
    stored_paper_ids = {str(r.get("paper_id")) for r in ranked if isinstance(r, dict) and r.get("paper_id")}
    cur_paper_ids = {str(pid) for pid in candidate_paper_ids}
    if stored_paper_ids != cur_paper_ids:
        return False, (
            f"Candidate paper membership mismatch: stored has {len(stored_paper_ids)} papers, "
            f"current candidates have {len(cur_paper_ids)} papers."
        )

    return True, None


class RelevanceScreeningService:
    """Deterministic embedding-based paper relevance screening."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        threshold: float = DEFAULT_SCREENING_THRESHOLD,
        min_papers: int = DEFAULT_SCREENING_MIN_PAPERS,
        max_papers: Optional[int] = None,
        alpha: float = DEFAULT_SCREENING_ALPHA,
    ) -> None:
        if max_papers is not None and max_papers < min_papers:
            raise ValueError(
                f"max_papers ({max_papers}) cannot be less than min_papers ({min_papers})"
            )
        self.embedding_provider = embedding_provider
        self.threshold = float(threshold)
        self.min_papers = max(0, int(min_papers))
        self.max_papers = int(max_papers) if max_papers is not None else None
        self.alpha = float(alpha)

    def screen_papers(
        self,
        topic: str,
        candidates: Sequence[Tuple[Any, Sequence[Any], str]],  # (paper, passages, filename)
        input_signature: Optional[Dict[str, Any]] = None,
    ) -> ScreeningReport:
        """Ranks eligible papers against topic and assigns selection decisions.

        Args:
            topic: Topic or research question string.
            candidates: Sequence of (Paper, list[PaperPassage], filename) tuples.

        Returns:
            A complete ``ScreeningReport`` with per-paper decisions and audit trail.

        Raises:
            ValueError: If topic is empty.
            Exception: If embedding provider fails (fails closed).
        """
        clean_topic = topic.strip() if topic else ""
        if not clean_topic:
            raise ValueError("Screening topic cannot be empty.")

        if not candidates:
            return ScreeningReport(
                topic=clean_topic,
                threshold=self.threshold,
                min_papers=self.min_papers,
                max_papers=self.max_papers,
                total_eligible_papers=0,
                selected_count=0,
                excluded_count=0,
                results=[],
                input_signature=input_signature,
            )

        # 1. Embed topic vector (fail-closed if provider fails)
        topic_embs = self.embedding_provider.embed_texts([clean_topic])
        if len(topic_embs) == 0:
            raise RuntimeError("Embedding provider returned empty vectors for topic.")
        topic_vec = topic_embs[0]
        t_norm = np.linalg.norm(topic_vec)
        if t_norm > 0:
            topic_vec = topic_vec / t_norm

        evaluated_papers: List[Dict[str, Any]] = []

        # 2. Score each paper deterministically
        for paper, passages, filename in candidates:
            # Paper-level header text: title + bounded abstract
            resolved_abs = resolve_paper_abstract(paper, passages, max_abstract_chars=SCREENING_ABSTRACT_MAX_CHARS)
            title = (getattr(paper, "title", None) or "").strip()
            if title and resolved_abs:
                header_text = f"{title}\n\n{resolved_abs}"
            elif title:
                header_text = title
            else:
                header_text = resolved_abs or "Scientific Paper"

            # Filter body passages: exclude reference list and abstract section
            body_passages = [
                p for p in passages
                if not is_reference_passage(p) and not is_abstract_passage(p)
            ]

            # Batch embed header + body passages for efficiency
            all_texts = [header_text] + [p.text.strip() for p in body_passages if getattr(p, "text", None)]
            embeddings = self.embedding_provider.embed_texts(all_texts)
            if len(embeddings) != len(all_texts):
                raise RuntimeError(
                    f"Embedding count mismatch for paper '{filename}': expected {len(all_texts)}, got {len(embeddings)}"
                )

            # Header similarity
            header_vec = embeddings[0]
            h_norm = np.linalg.norm(header_vec)
            if h_norm > 0:
                header_vec = header_vec / h_norm
            paper_sim = float(np.dot(topic_vec, header_vec))

            # Passage similarities
            passage_sims: List[Tuple[float, Any]] = []
            valid_body_passages = [p for p in body_passages if getattr(p, "text", None)]
            for i, p in enumerate(valid_body_passages, start=1):
                p_vec = embeddings[i]
                p_norm = np.linalg.norm(p_vec)
                if p_norm > 0:
                    p_vec = p_vec / p_norm
                sim = float(np.dot(topic_vec, p_vec))
                passage_sims.append((sim, p))

            # Sort passages by similarity descending
            passage_sims.sort(key=lambda item: item[0], reverse=True)
            body_sim_values = [item[0] for item in passage_sims]

            score = compute_relevance_score(
                paper_sim=paper_sim,
                body_sims=body_sim_values,
                alpha=self.alpha,
            )

            # Compute top-3 body passage average
            if body_sim_values:
                top_3_body = body_sim_values[: min(3, len(body_sim_values))]
                body_avg = sum(top_3_body) / len(top_3_body)
            else:
                body_avg = paper_sim

            # Build top snippets for auditability (up to 3)
            snippets: List[PassageSnippet] = []
            for sim, p in passage_sims[:3]:
                raw_text = (getattr(p, "text", "") or "").strip()
                clean_snippet = raw_text[:150] + ("..." if len(raw_text) > 150 else "")
                snippets.append(
                    PassageSnippet(
                        passage_id=str(getattr(p, "id", "")),
                        passage_index=int(getattr(p, "passage_index", 0)),
                        section=getattr(p, "section", None),
                        page_start=int(getattr(p, "page_start", 1)),
                        page_end=int(getattr(p, "page_end", 1)),
                        similarity=sim,
                        snippet=clean_snippet,
                    )
                )

            pub_date_val = extract_paper_publication_date(paper)
            evaluated_papers.append({
                "paper": paper,
                "filename": filename,
                "title": getattr(paper, "title", None),
                "publication_date": pub_date_val.isoformat() if pub_date_val else None,
                "score": score,
                "paper_level_similarity": paper_sim,
                "body_similarity": body_avg,
                "top_passages": snippets,
            })

        # 3. Deterministic sorting: primary score descending, tie-breaker filename ascending
        evaluated_papers.sort(
            key=lambda item: (-round(item["score"], 6), item["filename"].lower(), str(item["paper"].id))
        )

        # 4. Partition decisions according to thresholds, capacity, and floor
        qualifying = [item for item in evaluated_papers if item["score"] >= self.threshold]
        below_threshold = [item for item in evaluated_papers if item["score"] < self.threshold]

        results: List[PaperScreeningResult] = []
        floor_warning: Optional[str] = None

        if len(qualifying) >= self.min_papers:
            # Threshold qualified enough papers to satisfy min_papers
            if self.max_papers is not None:
                selected_items = qualifying[: self.max_papers]
                capacity_excluded = qualifying[self.max_papers :]
            else:
                selected_items = qualifying
                capacity_excluded = []

            for rank, item in enumerate(selected_items, start=1):
                results.append(
                    PaperScreeningResult(
                        paper_id=item["paper"].id,
                        filename=item["filename"],
                        title=item["title"],
                        publication_date=item["publication_date"],
                        score=item["score"],
                        rank=rank,
                        decision="SELECTED_ABOVE_THRESHOLD",
                        reason=(
                            f"Score {item['score']:.4f} >= threshold {self.threshold:.4f}"
                            + (f" and within max_papers {self.max_papers}." if self.max_papers is not None else ".")
                        ),
                        paper_level_similarity=item["paper_level_similarity"],
                        body_similarity=item["body_similarity"],
                        top_passages=item["top_passages"],
                        floor_included=False,
                    )
                )

            current_rank = len(selected_items) + 1
            for item in capacity_excluded:
                results.append(
                    PaperScreeningResult(
                        paper_id=item["paper"].id,
                        filename=item["filename"],
                        title=item["title"],
                        publication_date=item["publication_date"],
                        score=item["score"],
                        rank=current_rank,
                        decision="EXCLUDED_CAPACITY_LIMIT",
                        reason=(
                            f"Score {item['score']:.4f} >= threshold {self.threshold:.4f} "
                            f"but excluded by max_papers limit {self.max_papers}."
                        ),
                        paper_level_similarity=item["paper_level_similarity"],
                        body_similarity=item["body_similarity"],
                        top_passages=item["top_passages"],
                        floor_included=False,
                        temporal_workflow_note="Pre-cutoff paper excluded from discovery; not evaluated in Stage 5 temporal evolution.",
                    )
                )
                current_rank += 1

            for item in below_threshold:
                results.append(
                    PaperScreeningResult(
                        paper_id=item["paper"].id,
                        filename=item["filename"],
                        title=item["title"],
                        publication_date=item["publication_date"],
                        score=item["score"],
                        rank=current_rank,
                        decision="EXCLUDED_BELOW_THRESHOLD",
                        reason=f"Score {item['score']:.4f} < threshold {self.threshold:.4f}.",
                        paper_level_similarity=item["paper_level_similarity"],
                        body_similarity=item["body_similarity"],
                        top_passages=item["top_passages"],
                        floor_included=False,
                        temporal_workflow_note="Pre-cutoff paper excluded from discovery; not evaluated in Stage 5 temporal evolution.",
                    )
                )
                current_rank += 1

        else:
            # Qualifying papers are fewer than min_papers
            if self.min_papers == 0:
                # 0 papers qualified and min_papers == 0
                for rank, item in enumerate(evaluated_papers, start=1):
                    results.append(
                        PaperScreeningResult(
                            paper_id=item["paper"].id,
                            filename=item["filename"],
                            title=item["title"],
                            publication_date=item["publication_date"],
                            score=item["score"],
                            rank=rank,
                            decision="EXCLUDED_BELOW_THRESHOLD",
                            reason=f"Score {item['score']:.4f} < threshold {self.threshold:.4f}.",
                            paper_level_similarity=item["paper_level_similarity"],
                            body_similarity=item["body_similarity"],
                            top_passages=item["top_passages"],
                            floor_included=False,
                            temporal_workflow_note="Pre-cutoff paper excluded from discovery; not evaluated in Stage 5 temporal evolution.",
                        )
                    )
            else:
                # Floor pulls in lower-scoring papers up to min_papers
                floor_target = min(self.min_papers, len(evaluated_papers))
                if self.max_papers is not None:
                    floor_target = min(floor_target, self.max_papers)

                selected_items = evaluated_papers[:floor_target]
                excluded_items = evaluated_papers[floor_target:]

                floor_count = sum(1 for it in selected_items if it["score"] < self.threshold)
                if floor_count > 0:
                    floor_warning = (
                        f"Minimum selection floor ({self.min_papers}) caused {floor_count} paper(s) "
                        f"below similarity threshold ({self.threshold:.4f}) to be included for discovery."
                    )
                    logger.warning(floor_warning)

                for rank, item in enumerate(selected_items, start=1):
                    if item["score"] >= self.threshold:
                        decision = "SELECTED_ABOVE_THRESHOLD"
                        reason = f"Score {item['score']:.4f} >= threshold {self.threshold:.4f}."
                        floor_inc = False
                    else:
                        decision = "SELECTED_BELOW_THRESHOLD_FLOOR"
                        reason = (
                            f"Score {item['score']:.4f} < threshold {self.threshold:.4f} "
                            f"but included to satisfy minimum selection floor ({self.min_papers})."
                        )
                        floor_inc = True

                    results.append(
                        PaperScreeningResult(
                            paper_id=item["paper"].id,
                            filename=item["filename"],
                            title=item["title"],
                            publication_date=item["publication_date"],
                            score=item["score"],
                            rank=rank,
                            decision=decision,
                            reason=reason,
                            paper_level_similarity=item["paper_level_similarity"],
                            body_similarity=item["body_similarity"],
                            top_passages=item["top_passages"],
                            floor_included=floor_inc,
                        )
                    )

                current_rank = len(selected_items) + 1
                for item in excluded_items:
                    results.append(
                        PaperScreeningResult(
                            paper_id=item["paper"].id,
                            filename=item["filename"],
                            title=item["title"],
                            publication_date=item["publication_date"],
                            score=item["score"],
                            rank=current_rank,
                            decision="EXCLUDED_BELOW_THRESHOLD",
                            reason=f"Score {item['score']:.4f} < threshold {self.threshold:.4f}.",
                            paper_level_similarity=item["paper_level_similarity"],
                            body_similarity=item["body_similarity"],
                            top_passages=item["top_passages"],
                            floor_included=False,
                            temporal_workflow_note="Pre-cutoff paper excluded from discovery; not evaluated in Stage 5 temporal evolution.",
                        )
                    )
                    current_rank += 1

        selected_count = sum(1 for r in results if r.decision.startswith("SELECTED"))
        excluded_count = len(results) - selected_count

        return ScreeningReport(
            topic=clean_topic,
            threshold=self.threshold,
            min_papers=self.min_papers,
            max_papers=self.max_papers,
            total_eligible_papers=len(candidates),
            selected_count=selected_count,
            excluded_count=excluded_count,
            floor_warning=floor_warning,
            results=results,
            input_signature=input_signature,
        )
