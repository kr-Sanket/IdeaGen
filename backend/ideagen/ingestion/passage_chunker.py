"""Deterministic passage/chunk creation from structured scientific text.

Produces reproducible, page-referenced, and section-annotated passages.
"""

import re
from typing import List, Optional, Tuple, NamedTuple
import uuid

from .models import PassageCreateDTO


class BlockUnit(NamedTuple):
    text: str
    page_number: int
    section: Optional[str]


class PassageChunker:
    """Creates reproducible passages from structured document text blocks."""

    def __init__(
        self,
        target_chunk_size: int = 1000,
        max_chunk_size: int = 1600,
        min_chunk_size: int = 150,
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_oversized_block(
        self,
        text: str,
        page_number: int,
        section: Optional[str],
    ) -> List[BlockUnit]:
        """Splits an oversized paragraph into sentence-bounded sub-blocks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sub_blocks: List[BlockUnit] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            if current_length + len(sentence_clean) > self.max_chunk_size and current_chunk:
                sub_blocks.append(
                    BlockUnit(
                        text=" ".join(current_chunk),
                        page_number=page_number,
                        section=section,
                    )
                )
                current_chunk = [sentence_clean]
                current_length = len(sentence_clean)
            else:
                current_chunk.append(sentence_clean)
                current_length += len(sentence_clean) + 1

        if current_chunk:
            sub_blocks.append(
                BlockUnit(
                    text=" ".join(current_chunk),
                    page_number=page_number,
                    section=section,
                )
            )

        return sub_blocks

    def create_passages(
        self,
        blocks: List[BlockUnit],
        paper_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> List[PassageCreateDTO]:
        """Converts an ordered list of text blocks into deterministic passages."""
        # 1. Expand any oversized blocks into sub-blocks
        normalized_blocks: List[BlockUnit] = []
        for b in blocks:
            b_text = b.text.strip()
            if not b_text:
                continue
            if len(b_text) > self.max_chunk_size:
                normalized_blocks.extend(
                    self._split_oversized_block(b_text, b.page_number, b.section)
                )
            else:
                normalized_blocks.append(b)

        if not normalized_blocks:
            return []

        passages: List[PassageCreateDTO] = []
        current_texts: List[str] = []
        current_page_start = normalized_blocks[0].page_number
        current_page_end = normalized_blocks[0].page_number
        current_section = normalized_blocks[0].section
        current_length = 0
        passage_index = 0

        def emit_current():
            nonlocal passage_index, current_texts, current_page_start, current_page_end, current_section, current_length
            if current_texts:
                combined_text = "\n\n".join(current_texts)
                passages.append(
                    PassageCreateDTO(
                        paper_id=paper_id,
                        document_id=document_id,
                        passage_index=passage_index,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        section=current_section,
                        text=combined_text,
                    )
                )
                passage_index += 1
                current_texts = []
                current_length = 0

        for block in normalized_blocks:
            # If section changes and we already have content, emit the previous chunk
            if current_texts and block.section != current_section:
                emit_current()
                current_page_start = block.page_number
                current_page_end = block.page_number
                current_section = block.section

            # If adding this block exceeds target chunk size and current chunk is large enough, emit
            if (
                current_length + len(block.text) > self.target_chunk_size
                and current_length >= self.min_chunk_size
            ):
                emit_current()
                current_page_start = block.page_number
                current_page_end = block.page_number
                current_section = block.section

            # Append block
            current_texts.append(block.text)
            current_length += len(block.text)
            current_page_end = max(current_page_end, block.page_number)
            if not current_section and block.section:
                current_section = block.section

        emit_current()
        return passages
