"""Scientific Paper Understanding Extractor (Phase 3).

Executes structured extraction using a pretrained language model provider,
enforcing strict Pydantic validation, error handling, and passage-level provenance.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import uuid

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from .models import (
    StructuredPaperExtractionOutput,
    ResearchKnowledgeCreateDTO,
    ResearchStatementCreateDTO,
    ExtractionStatus,
)
from .passage_selector import SelectedPassage, PassageSelector

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert scientific paper analyzer.
Analyze the provided excerpts of a scientific paper and extract structured research knowledge.

CRITICAL RULES:
1. Ground all extractions in the provided text. Do NOT fabricate, invent, or extrapolate information.
2. If a field (e.g., datasets, limitations) is not mentioned in the passages, return an empty list or null.
3. Every statement in the statements list must identify the exact `passage_index` of the passage providing evidence.
4. Categorize statements accurately into:
   - PROBLEM: Research gap, challenge, or problem addressed.
   - QUESTION: Research questions or hypotheses formulated.
   - METHOD: Algorithms, architectures, frameworks, or procedures proposed.
   - DATASET: Benchmarks, corpora, or data utilized or evaluated.
   - FINDING: Empirical results, performance metrics, or experimental discoveries.
   - LIMITATION: Constraints, threats to validity, or computational limits identified.
   - FUTURE_WORK: Proposed directions, unresolved questions, or next steps.
   - ASSUMPTION: Explicit assumptions made by the methodology.
   - CONTRIBUTION: Core novel contributions claimed by the authors.
"""


class ScientificPaperExtractor:
    """Extracts structured research knowledge and statements from paper passages."""

    def __init__(
        self,
        provider: LLMProvider,
        model_name: str,
        model_version: Optional[str] = None,
        schema_version: str = "v1.0",
    ):
        self.provider = provider
        self.model_name = model_name
        self.model_version = model_version
        self.schema_version = schema_version

    def extract(
        self,
        paper_id: uuid.UUID,
        selected_passages: List[SelectedPassage],
        paper_title: Optional[str] = None,
    ) -> ResearchKnowledgeCreateDTO:
        """Extracts structured research knowledge from selected passages.

        Handles model failures cleanly without fabricating fallback values.
        """
        if not selected_passages:
            return ResearchKnowledgeCreateDTO(
                paper_id=paper_id,
                model_name=self.model_name,
                model_version=self.model_version,
                extraction_schema_version=self.schema_version,
                extraction_status=ExtractionStatus.FAILED,
                error_message="No passages available for extraction.",
            )

        passage_context = PassageSelector.format_passages_for_prompt(selected_passages)
        title_context = f"Paper Title: {paper_title}\n\n" if paper_title else ""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{title_context}"
            f"--- PASSAGES ---\n"
            f"{passage_context}\n\n"
            f"Extract the structured research knowledge following the requested schema."
        )

        try:
            raw_output: StructuredPaperExtractionOutput = self.provider.generate_structured(
                prompt=prompt,
                response_schema=StructuredPaperExtractionOutput,
            )
        except (ProviderFailureError, MalformedResponseError, Exception) as e:
            logger.error(f"Inference error extracting knowledge for paper {paper_id}: {e}")
            return ResearchKnowledgeCreateDTO(
                paper_id=paper_id,
                model_name=self.model_name,
                model_version=self.model_version,
                extraction_schema_version=self.schema_version,
                extraction_status=ExtractionStatus.FAILED,
                error_message=str(e),
            )

        # Build passage index lookup table
        passage_map: Dict[int, SelectedPassage] = {
            p.passage_index: p for p in selected_passages
        }

        # Resolve provenance for each research statement
        resolved_statements: List[ResearchStatementCreateDTO] = []
        for raw_stmt in raw_output.statements:
            source_p: Optional[SelectedPassage] = None
            if raw_stmt.passage_index is not None and raw_stmt.passage_index in passage_map:
                source_p = passage_map[raw_stmt.passage_index]
            else:
                # Text match fallback if index wasn't populated
                for p in selected_passages:
                    if raw_stmt.statement_text in p.text:
                        source_p = p
                        break

            statement_dto = ResearchStatementCreateDTO(
                statement_text=raw_stmt.statement_text,
                statement_type=raw_stmt.statement_type,
                source_passage_id=source_p.passage_id if source_p else None,
                source_page_start=source_p.page_start if source_p else None,
                source_page_end=source_p.page_end if source_p else None,
                section=source_p.section if source_p else None,
                confidence=raw_stmt.confidence,
                metadata={
                    "model_reported_passage_index": raw_stmt.passage_index,
                },
            )
            resolved_statements.append(statement_dto)

        return ResearchKnowledgeCreateDTO(
            paper_id=paper_id,
            research_problem=raw_output.research_problem,
            research_questions=raw_output.research_questions,
            methodology=raw_output.methodology,
            datasets=raw_output.datasets,
            findings=raw_output.findings,
            limitations=raw_output.limitations,
            future_work=raw_output.future_work,
            research_statements=resolved_statements,
            model_name=self.model_name,
            model_version=self.model_version,
            extraction_schema_version=self.schema_version,
            extraction_status=ExtractionStatus.SUCCESS,
            metadata={
                "selected_passage_count": len(selected_passages),
                "total_statement_count": len(resolved_statements),
            },
        )
