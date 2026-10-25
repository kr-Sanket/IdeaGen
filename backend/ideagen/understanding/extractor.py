"""Scientific Paper Understanding Extractor (Phase 3).

Executes structured extraction using a pretrained language model provider,
enforcing strict Pydantic validation, error handling, and passage-level provenance.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Type
import uuid

from ideagen.core.config import get_settings
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, ProviderConnectionError, MalformedResponseError
from .models import (
    StructuredPaperExtractionOutput,
    ResearchKnowledgeCreateDTO,
    ResearchStatementCreateDTO,
    ExtractionStatus,
)
from .passage_selector import SelectedPassage, PassageSelector

logger = logging.getLogger(__name__)

INLINE_PASSAGE_REF_PATTERN = re.compile(
    r"\s*[\(\[]\s*passage(?:_index)?:\s*(\d+)\s*[\)\]]\s*(\.?)\s*$",
    re.IGNORECASE,
)


SYSTEM_PROMPT = """You are an expert scientific paper analyzer.
Analyze the provided excerpts of a scientific paper and extract structured research knowledge.

CRITICAL RULES:
1. Ground all extractions strictly in the provided text. Do NOT fabricate, invent, or extrapolate information.
2. Keep narrative overview fields (research_problem, methodology) concise (1-2 sentences each).
3. Keep list overview fields (research_questions, datasets, findings, limitations, future_work) to 1-3 concise items. If not mentioned in passages, return an empty list or null.
4. Extract approximately 3-8 high-priority research statements when supported by the text. If the passages genuinely support fewer statements, return only the well-supported statements (do not force a minimum or invent statements).
5. Every statement in the statements list must identify the exact `passage_index` of the input passage providing direct evidence.
6. Categorize statements accurately into:
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

RECOVERY_INSTRUCTION_TRUNCATED = """--- RECOVERY INSTRUCTION ---
The previous extraction output was truncated or malformed.
Re-extract strictly the most critical research knowledge in a concise format:
- Keep all overview fields to at most 1 brief sentence each.
- Extract between 3 and 5 high-priority research statements with exact passage_index values.
- Return strictly valid JSON adhering to the schema.
"""

RECOVERY_INSTRUCTION_EMPTY = """--- RECOVERY INSTRUCTION ---
The previous extraction returned 0 research statements.
Analyze the excerpts carefully to identify and extract between 3 and 5 grounded research statements
(claims, methods, findings, problems, or limitations) with their exact passage_index values.
"""


class ScientificPaperExtractor:
    """Extracts structured research knowledge and statements from paper passages."""

    def __init__(
        self,
        provider: LLMProvider,
        model_name: str,
        model_version: Optional[str] = None,
        schema_version: str = "v1.0",
        first_attempt_num_predict: Optional[int] = None,
        recovery_num_predict: Optional[int] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.model_version = model_version
        self.schema_version = schema_version

        cfg = get_settings()
        self.first_attempt_num_predict = (
            first_attempt_num_predict
            if first_attempt_num_predict is not None
            else cfg.ollama_num_predict
        )
        self.recovery_num_predict = (
            recovery_num_predict
            if recovery_num_predict is not None
            else cfg.ollama_recovery_num_predict
        )

    def _call_provider(
        self,
        prompt: str,
        response_schema: Type[StructuredPaperExtractionOutput],
        num_predict: Optional[int] = None,
    ) -> StructuredPaperExtractionOutput:
        try:
            return self.provider.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
                num_predict=num_predict,
            )
        except TypeError:
            return self.provider.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
            )

    def extract(
        self,
        paper_id: uuid.UUID,
        selected_passages: List[SelectedPassage],
        paper_title: Optional[str] = None,
    ) -> ResearchKnowledgeCreateDTO:
        """Extracts structured research knowledge from selected passages.

        Handles model failures cleanly with bounded recovery without fabricating fallback values.
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

        raw_output: Optional[StructuredPaperExtractionOutput] = None
        last_error: Optional[Exception] = None
        can_retry: bool = True

        # Attempt 1: Standard extraction with initial token budget (default 1024)
        try:
            raw_output = self._call_provider(
                prompt=prompt,
                response_schema=StructuredPaperExtractionOutput,
                num_predict=self.first_attempt_num_predict,
            )
        except MalformedResponseError as e:
            logger.warning(
                f"Initial extraction attempt for paper {paper_id} produced malformed/truncated output ({e}). "
                "Attempting bounded recovery."
            )
            last_error = e
        except ProviderConnectionError as e:
            logger.error(
                f"Provider connection error during extraction for paper {paper_id}: {e}. Skipping recovery."
            )
            last_error = e
            can_retry = False
        except ProviderFailureError as e:
            msg = str(e).lower()
            if any(term in msg for term in ("connect", "timed out", "timeout", "refused", "unreachable")):
                logger.error(
                    f"Provider connection/timeout failure for paper {paper_id}: {e}. Skipping recovery."
                )
                last_error = e
                can_retry = False
            else:
                logger.warning(
                    f"Provider failure for paper {paper_id} ({e}). Attempting bounded recovery."
                )
                last_error = e
        except Exception as e:
            logger.error(
                f"Unexpected non-retryable error during extraction for paper {paper_id}: {e}. Skipping recovery."
            )
            last_error = e
            can_retry = False

        # Attempt 2: Bounded recovery if attempt 1 failed retryably or if full-text paper produced 0 statements
        needs_empty_recovery = (
            can_retry
            and raw_output is not None
            and len(raw_output.statements) == 0
            and len(selected_passages) > 0
        )
        if can_retry and (raw_output is None or needs_empty_recovery):
            recovery_instruction = RECOVERY_INSTRUCTION_EMPTY if needs_empty_recovery else RECOVERY_INSTRUCTION_TRUNCATED
            recovery_prompt = f"{prompt}\n\n{recovery_instruction}"
            try:
                recovered_output: StructuredPaperExtractionOutput = self._call_provider(
                    prompt=recovery_prompt,
                    response_schema=StructuredPaperExtractionOutput,
                    num_predict=self.recovery_num_predict,
                )
                if raw_output is None or len(recovered_output.statements) > 0:
                    raw_output = recovered_output
                    last_error = None
            except Exception as rec_err:
                logger.warning(f"Recovery extraction attempt for paper {paper_id} failed: {rec_err}")
                if raw_output is None:
                    last_error = rec_err

        if raw_output is None:
            logger.error(f"Inference error extracting knowledge for paper {paper_id}: {last_error}")
            return ResearchKnowledgeCreateDTO(
                paper_id=paper_id,
                model_name=self.model_name,
                model_version=self.model_version,
                extraction_schema_version=self.schema_version,
                extraction_status=ExtractionStatus.FAILED,
                error_message=str(last_error),
            )

        # Build passage index lookup table
        passage_map: Dict[int, SelectedPassage] = {
            p.passage_index: p for p in selected_passages
        }

        # Resolve provenance for each research statement
        resolved_statements: List[ResearchStatementCreateDTO] = []
        for raw_stmt in raw_output.statements:
            statement_text = raw_stmt.statement_text.strip()
            effective_passage_index = raw_stmt.passage_index
            recovered_from_inline = False

            # Defensive fallback: detect trailing inline reference (e.g. "(passage_index: 14)")
            match = INLINE_PASSAGE_REF_PATTERN.search(statement_text)
            if match:
                inline_idx = int(match.group(1))
                trailing_period = match.group(2)
                prefix = statement_text[:match.start()].rstrip()
                if trailing_period and not prefix.endswith("."):
                    statement_text = prefix + trailing_period
                else:
                    statement_text = prefix

                if effective_passage_index is None:
                    effective_passage_index = inline_idx
                    recovered_from_inline = True

            source_p: Optional[SelectedPassage] = None

            # Resolve index against the actual selected-passage map
            if effective_passage_index is not None:
                if effective_passage_index in passage_map:
                    source_p = passage_map[effective_passage_index]
                else:
                    logger.warning(
                        f"Statement passage_index {effective_passage_index} not found in selected passages "
                        f"(valid indexes: {sorted(passage_map.keys())}) for paper {paper_id}. "
                        "Leaving provenance unlinked to prevent false attribution."
                    )
            else:
                # Text match fallback if index wasn't populated or detected inline
                for p in selected_passages:
                    if statement_text and statement_text in p.text:
                        source_p = p
                        break

            if source_p is None:
                logger.warning(
                    f"Provenance could not be resolved for statement: '{statement_text[:80]}...' "
                    f"(paper_id={paper_id}). Preserving statement with unlinked provenance."
                )

            metadata_dict = {
                "model_reported_passage_index": raw_stmt.passage_index,
            }
            if recovered_from_inline:
                metadata_dict["recovered_inline_passage_index"] = effective_passage_index

            statement_dto = ResearchStatementCreateDTO(
                statement_text=statement_text,
                statement_type=raw_stmt.statement_type,
                source_passage_id=source_p.passage_id if source_p else None,
                source_page_start=source_p.page_start if source_p else None,
                source_page_end=source_p.page_end if source_p else None,
                section=source_p.section if source_p else None,
                confidence=raw_stmt.confidence,
                metadata=metadata_dict,
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
