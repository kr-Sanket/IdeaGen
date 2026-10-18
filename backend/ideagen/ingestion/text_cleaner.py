"""Text cleaning utilities for scientific literature extraction.

Cleans extracted PDF text carefully without removing scientific terminology,
citations, numbers, equations, or punctuation.
"""

import re


class TextCleaner:
    """Deterministic cleaner for extracted PDF text."""

    # Matches words hyphenated across a line-break (e.g., "method-\nology" or "method- \nology" -> "methodology")
    DEHYPHENATION_PATTERN = re.compile(r"([a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,})")

    # Matches standalone page number lines or arXiv stamps
    STANDALONE_ARTIFACT_PATTERN = re.compile(
        r"^(?:(?:page\s+\d+(?:\s+of\s+\d+)?)|(?:\d+)|(?:arxiv:\d+\.\d+(?:v\d+)?(?:\[\S+\])?))\s*$",
        re.IGNORECASE,
    )

    # Control characters excluding tab and newline
    CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Applies conservative cleaning to a text segment.

        Preserves citations, scientific terms, equations, and paragraph boundaries.
        """
        if not text:
            return ""

        # 1. Normalize line endings to LF
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip control characters
        cleaned = cls.CONTROL_CHARS_PATTERN.sub("", cleaned)

        # 3. De-hyphenate words broken at line breaks
        cleaned = cls.DEHYPHENATION_PATTERN.sub(r"\1\2", cleaned)

        # 4. Filter obvious page headers/footers and artifacts line-by-line
        lines = []
        for line in cleaned.split("\n"):
            line_str = line.strip()
            if cls.STANDALONE_ARTIFACT_PATTERN.match(line_str):
                continue
            lines.append(line)

        reconstructed = "\n".join(lines)

        # 5. Handle line wraps inside paragraphs:
        # Separate text into paragraphs (split by 2 or more newlines)
        paragraphs = re.split(r"\n{2,}", reconstructed)
        cleaned_paragraphs = []

        for p in paragraphs:
            # Within a paragraph, join broken line wrapping with spaces
            p_lines = [l.strip() for l in p.split("\n") if l.strip()]
            if not p_lines:
                continue
            para_text = " ".join(p_lines)
            # Normalize repeated horizontal spaces
            para_text = re.sub(r"[ \t]+", " ", para_text).strip()
            if para_text:
                cleaned_paragraphs.append(para_text)

        return "\n\n".join(cleaned_paragraphs)

    @classmethod
    def clean_block(cls, block_text: str) -> str:
        """Cleans a single block/paragraph extracted from a PDF page."""
        if not block_text:
            return ""
        # Remove control chars
        cleaned = cls.CONTROL_CHARS_PATTERN.sub("", block_text)
        # De-hyphenate
        cleaned = cls.DEHYPHENATION_PATTERN.sub(r"\1\2", cleaned)
        # Collapse internal newlines and whitespace
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        if not lines:
            return ""
        single_str = " ".join(lines)
        single_str = re.sub(r"[ \t]+", " ", single_str).strip()
        return single_str
