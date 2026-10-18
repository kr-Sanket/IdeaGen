"""Deterministic passage selection for AI Paper Understanding (Phase 3).

Selects relevant passages based on section boundaries and position without
introducing embeddings or vector search, keeping provenance intact.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import uuid

from ideagen.db.models import PaperPassage


class SelectedPassage:
    """Represents a passage selected for model inference with preserved provenance."""
    def __init__(
        self,
        passage_id: uuid.UUID,
        passage_index: int,
        page_start: int,
        page_end: int,
        section: Optional[str],
        text: str,
    ):
        self.passage_id = passage_id
        self.passage_index = passage_index
        self.page_start = page_start
        self.page_end = page_end
        self.section = section
        self.text = text


class PassageSelector:
    """Selects and orders relevant passages for scientific paper extraction."""

    # Canonical priority sections for scientific claims
    PRIORITY_SECTIONS = {
        "abstract": 1,
        "introduction": 2,
        "conclusion": 3,
        "limitations": 4,
        "future work": 5,
        "results": 6,
        "discussion": 7,
        "methodology": 8,
        "methods": 9,
        "experiments": 10,
        "background": 11,
        "related work": 12,
    }

    def __init__(self, max_passages: int = 12, max_total_words: int = 4000):
        self.max_passages = max_passages
        self.max_total_words = max_total_words

    def select_passages(self, passages: Sequence[PaperPassage]) -> List[SelectedPassage]:
        """Selects a representative, bounded subset of passages for AI understanding.

        Preserves passage indexes and metadata for provenance tracing.
        """
        if not passages:
            return []

        # Convert to local representation
        candidates = [
            SelectedPassage(
                passage_id=p.id,
                passage_index=p.passage_index,
                page_start=p.page_start,
                page_end=p.page_end,
                section=p.section,
                text=p.text.strip(),
            )
            for p in passages
            if p.text and p.text.strip()
        ]

        if len(candidates) <= self.max_passages:
            return candidates

        # Group by section relevance
        selected: List[SelectedPassage] = []
        selected_indices = set()

        def add_passage(p: SelectedPassage):
            if p.passage_index not in selected_indices:
                selected.append(p)
                selected_indices.add(p.passage_index)

        # 1. Abstract (crucial for problem, contributions, findings)
        for p in candidates:
            if p.section and "abstract" in p.section.lower():
                add_passage(p)

        # 2. Limitations and Future Work (crucial for gaps and constraints)
        for p in candidates:
            if p.section and any(k in p.section.lower() for k in ("limitation", "future work", "threats")):
                add_passage(p)

        # 3. Introduction & Conclusion
        for p in candidates:
            if p.section and any(k in p.section.lower() for k in ("introduction", "conclusion")):
                if len(selected) < self.max_passages:
                    add_passage(p)

        # 4. Methods, Results, Discussion
        for p in candidates:
            if p.section and any(k in p.section.lower() for k in ("method", "result", "discussion", "experiment")):
                if len(selected) < self.max_passages:
                    add_passage(p)

        # 5. If under limit, add first and last passages (structural fallback)
        if len(selected) < self.max_passages:
            # First few passages
            for p in candidates[:3]:
                if len(selected) < self.max_passages:
                    add_passage(p)
            # Last few passages
            for p in candidates[-3:]:
                if len(selected) < self.max_passages:
                    add_passage(p)

        # 6. Sort selected passages by original passage_index to maintain document flow
        selected.sort(key=lambda p: p.passage_index)

        # 7. Check word count budget
        budgeted: List[SelectedPassage] = []
        current_words = 0
        for p in selected:
            word_count = len(p.text.split())
            if current_words + word_count <= self.max_total_words or not budgeted:
                budgeted.append(p)
                current_words += word_count
            else:
                break

        return budgeted

    @staticmethod
    def format_passages_for_prompt(passages: List[SelectedPassage]) -> str:
        """Formats passages into structured context blocks for model ingestion."""
        blocks = []
        for p in passages:
            section_label = p.section if p.section else "Unknown"
            header = f"[PASSAGE {p.passage_index} | Pages {p.page_start}-{p.page_end} | Section: {section_label}]"
            blocks.append(f"{header}\n{p.text}")
        return "\n\n".join(blocks)
