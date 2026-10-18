"""AI Paper Understanding Pipeline (Phase 3).

Coordinates the end-to-end understanding pipeline:
Paper passages → Passage selection → Pretrained model inference →
Pydantic validation → Provenance mapping → PostgreSQL persistence.
"""

from __future__ import annotations

import logging
from typing import Optional
import uuid

from ideagen.core.config import get_settings
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.db.models import Paper
from ideagen.llm.gemini_provider import GeminiLLMProvider
from ideagen.llm.provider import LLMProvider
from .models import (
    ResearchKnowledgeCreateDTO,
    ResearchKnowledgeResponseDTO,
)
from .passage_selector import PassageSelector
from .extractor import ScientificPaperExtractor

logger = logging.getLogger(__name__)


class PaperUnderstandingPipeline:
    """Orchestrates structured scientific understanding for ingested papers."""

    def __init__(
        self,
        persistence: SQLAlchemyPersistenceLayer,
        provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ):
        self.persistence = persistence
        settings = get_settings()

        if provider is not None:
            self.provider = provider
            self.model_name = model_name or "custom-provider"
        else:
            # Default to configured Gemini provider using real pretrained model
            api_key = settings.gemini_api_key or ""
            self.model_name = model_name or settings.gemini_model_name
            self.provider = GeminiLLMProvider(api_key=api_key, model_name=self.model_name)

        self.selector = PassageSelector()
        self.extractor = ScientificPaperExtractor(
            provider=self.provider,
            model_name=self.model_name,
            model_version="1.0",
            schema_version="v1.0",
        )

    def understand_paper(self, paper_id: uuid.UUID) -> ResearchKnowledgeResponseDTO:
        """Executes the complete paper understanding pipeline for a paper."""
        # 1. Fetch paper and passages
        paper = self.persistence.session.query(Paper).filter_by(id=paper_id).first()
        if not paper:
            raise ValueError(f"Paper with id {paper_id} not found.")

        passages = self.persistence.get_passages_by_paper(paper_id)
        if not passages:
            raise ValueError(f"No passages found for paper {paper_id}. Ingest passages first.")

        # 2. Select relevant passages preserving provenance
        selected_passages = self.selector.select_passages(passages)

        # 3. Extract structured research knowledge and statements
        knowledge_dto: ResearchKnowledgeCreateDTO = self.extractor.extract(
            paper_id=paper_id,
            selected_passages=selected_passages,
            paper_title=paper.title,
        )

        # 4. Persist result and statements to PostgreSQL
        knowledge_id = self.persistence.save_research_knowledge(knowledge_dto)

        # 5. Format response DTO
        return ResearchKnowledgeResponseDTO(
            id=knowledge_id,
            paper_id=paper_id,
            research_problem=knowledge_dto.research_problem,
            research_questions=knowledge_dto.research_questions,
            methodology=knowledge_dto.methodology,
            datasets=knowledge_dto.datasets,
            findings=knowledge_dto.findings,
            limitations=knowledge_dto.limitations,
            future_work=knowledge_dto.future_work,
            statement_count=len(knowledge_dto.research_statements),
            model_name=knowledge_dto.model_name,
            extraction_schema_version=knowledge_dto.extraction_schema_version,
            extracted_at=knowledge_dto.extracted_at,
            extraction_status=knowledge_dto.extraction_status.value if hasattr(knowledge_dto.extraction_status, 'value') else str(knowledge_dto.extraction_status),
            error_message=knowledge_dto.error_message,
        )

    def get_paper_knowledge(self, paper_id: uuid.UUID) -> Optional[ResearchKnowledgeResponseDTO]:
        """Retrieves previously extracted research knowledge for a paper."""
        knowledge = self.persistence.get_research_knowledge_by_paper(paper_id)
        if not knowledge:
            return None

        statements = self.persistence.get_statements_by_knowledge(knowledge.id)
        return ResearchKnowledgeResponseDTO(
            id=knowledge.id,
            paper_id=knowledge.paper_id,
            research_problem=knowledge.research_problem,
            research_questions=knowledge.research_questions or [],
            methodology=knowledge.methodology,
            datasets=knowledge.datasets or [],
            findings=knowledge.findings or [],
            limitations=knowledge.limitations or [],
            future_work=knowledge.future_work or [],
            statement_count=len(statements),
            model_name=knowledge.model_name,
            extraction_schema_version=knowledge.extraction_schema_version,
            extracted_at=knowledge.extracted_at,
            extraction_status=knowledge.extraction_status,
            error_message=knowledge.error_message,
        )
