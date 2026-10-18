"""Deterministic section detection for scientific literature.

Detects standard scientific paper sections based on rule-based heading patterns
without artificial or fabricated AI labels.
"""

import re
from typing import Optional, List, Tuple


class SectionDetector:
    """Rule-based detector for scientific paper section headers."""

    # Heading prefix pattern (e.g., '1.', '1.2', 'IV.', 'A.', 'Section 3:')
    HEADING_PREFIX_PATTERN = re.compile(
        r"^(?:(?:section\s+)?(?:[0-9]+(?:\.[0-9]+)*|[ivxlcdm]+(?:\.[0-9]+)*|[a-z]\.))\s*[-.:]?\s*",
        re.IGNORECASE,
    )

    # Ordered list of canonical sections and regex matchers
    SECTION_RULES: List[Tuple[str, re.Pattern]] = [
        ("Abstract", re.compile(r"^abstract$", re.IGNORECASE)),
        ("Introduction", re.compile(r"^introduction$", re.IGNORECASE)),
        ("Related Work", re.compile(r"^(?:related\s+work|prior\s+work)$", re.IGNORECASE)),
        ("Background", re.compile(r"^(?:background|background\s+and\s+related\s+work)$", re.IGNORECASE)),
        ("Methods", re.compile(r"^(?:methods?|methodology|proposed\s+methods?|proposed\s+approach|system\s+overview)$", re.IGNORECASE)),
        ("Experiments", re.compile(r"^(?:experiments?|experimental\s+setup|experimental\s+evaluation)$", re.IGNORECASE)),
        ("Results", re.compile(r"^(?:results?|empirical\s+results?|experimental\s+results?|findings)$", re.IGNORECASE)),
        ("Discussion", re.compile(r"^discussion$", re.IGNORECASE)),
        ("Limitations", re.compile(r"^(?:limitations?|limitations?\s+and\s+future\s+work|threats\s+to\s+validity)$", re.IGNORECASE)),
        ("Conclusion", re.compile(r"^(?:conclusions?|concluding\s+remarks|summary\s+and\s+conclusions?)$", re.IGNORECASE)),
        ("Future Work", re.compile(r"^(?:future\s+work|future\s+directions?)$", re.IGNORECASE)),
        ("References", re.compile(r"^(?:references?|bibliography)$", re.IGNORECASE)),
    ]

    @classmethod
    def detect_heading(cls, text: str) -> Optional[str]:
        """Determines if a given block or line text is a section heading.

        Returns the canonical section name if confidently detected, else None.
        """
        if not text:
            return None

        clean_line = text.strip()

        # Headings are generally short titles, not long body sentences
        if len(clean_line) > 80:
            return None

        # Remove trailing colon or period
        normalized = clean_line.rstrip(":.").strip()

        # Remove leading section numbering (e.g. "1. Introduction" -> "Introduction")
        without_prefix = cls.HEADING_PREFIX_PATTERN.sub("", normalized).strip()

        # Check if the title text itself contains multiple sentences
        if re.search(r"[.!?]\s+[A-Za-z]", without_prefix):
            return None

        for canonical_name, pattern in cls.SECTION_RULES:
            if pattern.match(without_prefix):
                return canonical_name

        return None
