import uuid
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from ideagen.ingestion.models import PaperCreateDTO, SourceType
from ideagen.ingestion.pipeline import PersistenceLayer
from .models import Paper

class SQLAlchemyPersistenceLayer:
    def __init__(self, session: Session):
        self.session = session
        
    def paper_exists_by_source_id(self, source_type: str, source_id: str) -> bool:
        stmt = select(Paper.id).where(
            Paper.source_type == source_type,
            Paper.source_id == source_id
        )
        return self.session.scalar(stmt) is not None
        
    def paper_exists_by_doi(self, doi: str) -> bool:
        stmt = select(Paper.id).where(Paper.doi == doi)
        return self.session.scalar(stmt) is not None
        
    def save_paper(self, paper_dto: PaperCreateDTO) -> uuid.UUID:
        paper = Paper(
            source_type=paper_dto.source_type.value,
            source_id=paper_dto.source_id,
            doi=paper_dto.doi,
            title=paper_dto.title,
            publication_date=paper_dto.publication_date,
            batch_id=paper_dto.batch_id,
            abstract=paper_dto.abstract,
            authors=paper_dto.authors,
            referenced_works=paper_dto.referenced_works,
            raw_metadata=paper_dto.raw_metadata
        )
        try:
            self.session.add(paper)
            self.session.commit()
            return paper.id
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"IntegrityError saving paper: {e}")
        except Exception as e:
            self.session.rollback()
            raise e

    def document_exists_by_hash(self, content_hash: str) -> bool:
        if not content_hash:
            return False
        from .models import PaperDocument
        stmt = select(PaperDocument.id).where(PaperDocument.content_hash == content_hash)
        return self.session.scalar(stmt) is not None

    def get_document_by_hash(self, content_hash: str) -> Optional['PaperDocument']:
        if not content_hash:
            return None
        from .models import PaperDocument
        stmt = select(PaperDocument).where(PaperDocument.content_hash == content_hash)
        return self.session.scalar(stmt)

    def save_paper_document(self, doc_dto: 'ideagen.ingestion.models.PaperDocumentCreateDTO') -> uuid.UUID:
        from .models import PaperDocument
        doc = PaperDocument(
            paper_id=doc_dto.paper_id,
            full_text=doc_dto.full_text,
            extraction_method=doc_dto.extraction_method,
            extraction_version=doc_dto.extraction_version,
            extracted_at=doc_dto.extracted_at,
            content_hash=getattr(doc_dto, "content_hash", None),
            original_filename=getattr(doc_dto, "original_filename", None),
            acquisition_timestamp=getattr(doc_dto, "acquisition_timestamp", None),
            provenance_metadata=getattr(doc_dto, "provenance_metadata", None),
        )
        try:
            self.session.add(doc)
            self.session.commit()
            return doc.id
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"IntegrityError saving paper document: {e}")
        except Exception as e:
            self.session.rollback()
            raise e

    def save_passages(self, passage_dtos: list['ideagen.ingestion.models.PassageCreateDTO']) -> None:
        if not passage_dtos:
            return
        from .models import PaperPassage
        passages = [
            PaperPassage(
                paper_id=dto.paper_id,
                document_id=dto.document_id,
                passage_index=dto.passage_index,
                page_start=dto.page_start,
                page_end=dto.page_end,
                section=dto.section,
                text=dto.text,
            )
            for dto in passage_dtos
        ]
        try:
            self.session.add_all(passages)
            self.session.commit()
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"IntegrityError saving paper passages: {e}")
        except Exception as e:
            self.session.rollback()
            raise e

    def get_passages_by_document(self, document_id: uuid.UUID) -> list['PaperPassage']:
        from .models import PaperPassage
        stmt = select(PaperPassage).where(PaperPassage.document_id == document_id).order_by(PaperPassage.passage_index)
        return list(self.session.scalars(stmt).all())

    def get_passages_by_paper(self, paper_id: uuid.UUID) -> list['PaperPassage']:
        from .models import PaperPassage
        stmt = select(PaperPassage).where(PaperPassage.paper_id == paper_id).order_by(PaperPassage.passage_index)
        return list(self.session.scalars(stmt).all())

    def save_research_knowledge(self, dto: 'ideagen.understanding.models.ResearchKnowledgeCreateDTO') -> uuid.UUID:
        from .models import ResearchKnowledge, ResearchStatement
        # Check if record already exists for this paper
        existing = self.session.query(ResearchKnowledge).filter_by(paper_id=dto.paper_id).first()
        if existing:
            # Delete existing knowledge and cascade statements
            self.session.delete(existing)
            self.session.flush()

        knowledge = ResearchKnowledge(
            paper_id=dto.paper_id,
            research_problem=dto.research_problem,
            research_questions=dto.research_questions,
            methodology=dto.methodology,
            datasets=dto.datasets,
            findings=dto.findings,
            limitations=dto.limitations,
            future_work=dto.future_work,
            model_name=dto.model_name,
            model_version=dto.model_version,
            extraction_schema_version=dto.extraction_schema_version,
            extracted_at=dto.extracted_at,
            extraction_status=dto.extraction_status.value if hasattr(dto.extraction_status, 'value') else dto.extraction_status,
            error_message=dto.error_message,
            metadata_json=dto.metadata,
        )

        statements = [
            ResearchStatement(
                knowledge=knowledge,
                paper_id=dto.paper_id,
                passage_id=stmt_dto.source_passage_id,
                statement_text=stmt_dto.statement_text,
                statement_type=stmt_dto.statement_type.value if hasattr(stmt_dto.statement_type, 'value') else stmt_dto.statement_type,
                page_start=stmt_dto.source_page_start,
                page_end=stmt_dto.source_page_end,
                section=stmt_dto.section,
                confidence=stmt_dto.confidence,
                metadata_json=stmt_dto.metadata,
            )
            for stmt_dto in dto.research_statements
        ]

        try:
            self.session.add(knowledge)
            self.session.add_all(statements)
            self.session.commit()
            return knowledge.id
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"IntegrityError saving research knowledge: {e}")
        except Exception as e:
            self.session.rollback()
            raise e

    def get_research_knowledge_by_paper(self, paper_id: uuid.UUID) -> Optional['ResearchKnowledge']:
        from .models import ResearchKnowledge
        stmt = select(ResearchKnowledge).where(ResearchKnowledge.paper_id == paper_id)
        return self.session.scalars(stmt).first()

    def get_statements_by_knowledge(self, knowledge_id: uuid.UUID) -> list['ResearchStatement']:
        from .models import ResearchStatement
        stmt = select(ResearchStatement).where(ResearchStatement.knowledge_id == knowledge_id).order_by(ResearchStatement.created_at)
        return list(self.session.scalars(stmt).all())

    def get_statements_by_paper(self, paper_id: uuid.UUID) -> list['ResearchStatement']:
        from .models import ResearchStatement
        stmt = select(ResearchStatement).where(ResearchStatement.paper_id == paper_id).order_by(ResearchStatement.created_at)
        return list(self.session.scalars(stmt).all())

    def get_all_research_statements(self) -> list['ResearchStatement']:
        from .models import ResearchStatement
        stmt = select(ResearchStatement).order_by(ResearchStatement.created_at)
        return list(self.session.scalars(stmt).all())

    def save_research_gap(self, dto: 'ideagen.discovery.models.ResearchGapCreateDTO') -> uuid.UUID:
        from .models import ResearchGap, GapEvidence
        gap = ResearchGap(
            gap_statement=dto.gap_statement,
            research_problem=dto.research_problem,
            status=dto.status.value if hasattr(dto.status, "value") else dto.status,
            evidence_count=dto.evidence_count,
            limitations=dto.limitations,
            future_work=dto.future_work,
            validation_reasoning=dto.validation_reasoning,
            validation_confidence=dto.validation_confidence,
            metadata_json=dto.metadata,
        )

        evidence_records = [
            GapEvidence(
                gap=gap,
                statement_id=item.statement_id,
                paper_id=item.paper_id,
                evidence_role=item.evidence_role.value if hasattr(item.evidence_role, "value") else item.evidence_role,
            )
            for item in dto.evidence_items
        ]

        try:
            self.session.add(gap)
            self.session.add_all(evidence_records)
            self.session.commit()
            return gap.id
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"IntegrityError saving research gap: {e}")
        except Exception as e:
            self.session.rollback()
            raise e

    def get_research_gap(self, gap_id: uuid.UUID) -> Optional['ResearchGap']:
        from .models import ResearchGap
        stmt = select(ResearchGap).where(ResearchGap.id == gap_id)
        return self.session.scalars(stmt).first()

    def get_all_research_gaps(self, status: Optional[str] = None) -> list['ResearchGap']:
        from .models import ResearchGap
        stmt = select(ResearchGap)
        if status:
            stmt = stmt.where(ResearchGap.status == status)
        stmt = stmt.order_by(ResearchGap.created_at.desc())
        return list(self.session.scalars(stmt).all())

    def get_gap_evidence(self, gap_id: uuid.UUID) -> list['GapEvidence']:
        from .models import GapEvidence
        stmt = select(GapEvidence).where(GapEvidence.gap_id == gap_id).order_by(GapEvidence.created_at)
        return list(self.session.scalars(stmt).all())



