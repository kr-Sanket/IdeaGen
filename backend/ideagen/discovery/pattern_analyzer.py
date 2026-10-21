"""Research-pattern analysis and candidate gap detection (Phase 4).

Analyzes structured patterns within semantic clusters (recurring limitations,
unfulfilled future work, contradictions, unresolved problems) to synthesize
candidate research gaps grounded in empirical evidence.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set
import uuid

from .models import GapCandidate
from .grouper import SemanticCluster

logger = logging.getLogger(__name__)

# Patterns indicating positive sufficiency, feasibility, or capability assertions
INELIGIBLE_LIMITATION_PATTERNS = [
    # Sufficiency claims (when not negated)
    re.compile(r"(?<!not\s)(?<!hardly\s)(?<!scarcely\s)(?<!in)\bsufficient\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are|was|were|proves?|remains?)\s+sufficient\b", re.IGNORECASE),
    # Feasibility, adaptability, and extensibility claims
    re.compile(r"\b(?:can|could|is able to|are able to)\s+be\s+(?:easily\s+|readily\s+)?(?:adapted|extended|applied)\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+capable\s+of\b", re.IGNORECASE),
    re.compile(r"\b(?:provides?|demonstrates?)\s+a\s+(?:general-purpose|flexible|viable|practical)\b", re.IGNORECASE),
    # High-performance or state-of-the-art empirical claims
    re.compile(r"\b(?:achieves?|sets?)\s+(?:new\s+)?state-of-the-art\b", re.IGNORECASE),
    re.compile(r"\boutperforms?\b", re.IGNORECASE),
    re.compile(r"\b(?:superior|optimal|strong)\s+performance\b", re.IGNORECASE),
]

# Patterns indicating genuine constraints, bottlenecks, or deficits
DEFICIT_OR_CONSTRAINT_PATTERNS = [
    re.compile(r"\b(?:not|cannot|can\s+not|unable|fails?|failure|lacks?|lacking|deficit|deficiency)\b", re.IGNORECASE),
    re.compile(r"\b(?:limit|limits|limited|limitation|limitations|constraint|constraints|bottleneck|bottlenecks)\b", re.IGNORECASE),
    re.compile(r"\b(?:overhead|costly|expensive|slow|latency|degrade|degrades|degradation|suffer|suffers|impede|impedes)\b", re.IGNORECASE),
    re.compile(r"\b(?:struggle|struggles|vulnerable|vulnerability|trade-?off|trade-?offs|restrict|restricted|restriction)\b", re.IGNORECASE),
    re.compile(r"\b(?:unresolved|unclear|insufficient|inadequate|suboptimal|intractable|flaw|flaws|error|errors)\b", re.IGNORECASE),
    re.compile(r"\b(?:only\s+evaluated|restricted\s+to|does\s+not\s+scale|hallucinat\w*|noise|distractor\w*)\b", re.IGNORECASE),
]


def is_eligible_limitation(text: str) -> bool:
    """Verifies that a statement genuinely describes a deficit, constraint, or limitation.

    Filters out positive capability, sufficiency, feasibility, or adaptation claims
    that may have been misclassified as limitations.
    """
    if not text or not text.strip():
        return False

    has_positive = any(pat.search(text) for pat in INELIGIBLE_LIMITATION_PATTERNS)
    has_deficit = any(pat.search(text) for pat in DEFICIT_OR_CONSTRAINT_PATTERNS)

    # If the statement asserts positive sufficiency, feasibility, or capability without a deficit,
    # it is not an eligible limitation.
    if has_positive and not has_deficit:
        return False

    # If it expresses an explicit deficit or constraint, it is an eligible limitation.
    if has_deficit:
        return True

    # If it has neither positive claims nor explicit deficit indicators,
    # it is not a defensible limitation (conservative gating).
    return False


class ResearchPatternAnalyzer:
    """Analyzes research patterns across clustered statements to detect candidate gaps."""

    def __init__(self, min_supporting_evidence: int = 1):
        self.min_supporting_evidence = min_supporting_evidence

    def analyze_clusters(self, clusters: List[SemanticCluster]) -> List[GapCandidate]:
        """Analyzes semantic clusters and proposes candidate research gaps."""
        candidates: List[GapCandidate] = []

        for cluster in clusters:
            gap = self._analyze_cluster(cluster)
            if gap is not None:
                candidates.append(gap)

        return candidates

    def _analyze_cluster(self, cluster: SemanticCluster) -> GapCandidate | None:
        """Analyzes a single cluster for gap patterns."""
        statements = cluster.statements
        if not statements:
            return None

        # Categorize statements by type
        problems: List[Any] = []
        methods: List[Any] = []
        findings: List[Any] = []
        limitations: List[Any] = []
        future_works: List[Any] = []
        assumptions: List[Any] = []

        for s in statements:
            st_type = str(s.statement_type if hasattr(s, "statement_type") else s["statement_type"]).upper()
            text = s.statement_text if hasattr(s, "statement_text") else s["statement_text"]

            if "PROBLEM" in st_type or "QUESTION" in st_type:
                problems.append(s)
            elif "METHOD" in st_type or "CONTRIBUTION" in st_type:
                methods.append(s)
            elif "FINDING" in st_type:
                findings.append(s)
            elif "LIMITATION" in st_type:
                # Eligibility check: ensure statement genuinely describes a deficit, constraint, or unmet need,
                # rather than a positive capability, feasibility, or sufficiency claim.
                if is_eligible_limitation(text):
                    limitations.append(s)
            elif "ASSUMPTION" in st_type:
                assumptions.append(s)
            elif "FUTURE_WORK" in st_type:
                future_works.append(s)

        # Signal 1: Presence of explicit limitations or future work
        has_limitations = len(limitations) > 0
        has_future_work = len(future_works) > 0
        has_problems = len(problems) > 0

        # We need at least some evidence of a gap, limitation, or future direction
        if not (has_limitations or has_future_work or has_problems):
            return None

        # Collect supporting statements and papers
        supporting_stmts = limitations + future_works + problems
        if not supporting_stmts:
            supporting_stmts = statements

        if len(supporting_stmts) < self.min_supporting_evidence:
            return None

        supporting_stmt_ids: List[uuid.UUID] = []
        supporting_paper_ids: Set[uuid.UUID] = set()

        for s in supporting_stmts:
            s_id = s.id if hasattr(s, "id") else s["id"]
            if isinstance(s_id, str):
                s_id = uuid.UUID(s_id)
            supporting_stmt_ids.append(s_id)

            p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
            if isinstance(p_id, str):
                p_id = uuid.UUID(p_id)
            supporting_paper_ids.add(p_id)

        # Synthesize problem description from problem statements or cluster context
        if problems:
            primary_problem = (problems[0].statement_text if hasattr(problems[0], "statement_text") else problems[0]["statement_text"])
        else:
            primary_problem = (statements[0].statement_text if hasattr(statements[0], "statement_text") else statements[0]["statement_text"])

        # Determine gap statement based on dominant pattern
        lim_texts = [
            (l.statement_text if hasattr(l, "statement_text") else l["statement_text"])
            for l in limitations
        ]
        fw_texts = [
            (f.statement_text if hasattr(f, "statement_text") else f["statement_text"])
            for f in future_works
        ]

        if has_limitations and has_future_work:
            pattern_type = "LIMITATION_AND_FUTURE_WORK"
            gap_stmt = (
                f"Unresolved methodological limitation and open research direction: {lim_texts[0]}. "
                f"Proposed future direction remains unaddressed: {fw_texts[0]}."
            )
            distinct_papers = len(supporting_paper_ids)
            if distinct_papers > 1:
                why_insufficient = (
                    f"Existing approaches demonstrate documented constraints ({lim_texts[0]}), "
                    f"while subsequent work explicitly designates this problem for future investigation without resolution."
                )
            else:
                why_insufficient = (
                    f"Current work demonstrates documented constraints ({lim_texts[0]}), "
                    f"while explicitly designating this problem for future investigation without resolution."
                )
        elif has_limitations:
            distinct_papers = len(supporting_paper_ids)
            if distinct_papers > 1:
                pattern_type = "RECURRING_LIMITATIONS"
                gap_stmt = f"Persistent scientific limitation across current methodologies: {lim_texts[0]}."
                why_insufficient = f"While current methods establish baseline findings, they explicitly encounter limitations across multiple papers: {'; '.join(lim_texts[:2])}."
            else:
                pattern_type = "DOCUMENTED_LIMITATION"
                gap_stmt = f"Documented scientific limitation in methodology: {lim_texts[0]}."
                why_insufficient = f"Current work identifies an explicit limitation: {lim_texts[0]}."
        elif has_future_work:
            pattern_type = "OPEN_FUTURE_WORK"
            gap_stmt = f"Unexplored research direction identified across scientific literature: {fw_texts[0]}."
            why_insufficient = f"Prior work highlights this challenge as an open milestone, but no verified method has resolved it within the analyzed evidence."
        else:
            pattern_type = "UNRESOLVED_PROBLEM"
            gap_stmt = f"Identified research problem lacking comprehensive empirical resolution: {primary_problem}."
            why_insufficient = f"The underlying research question is recognized, yet experimental verification remains partial."

        return GapCandidate(
            gap_statement=gap_stmt,
            research_problem=primary_problem,
            why_existing_work_insufficient=why_insufficient,
            supporting_statement_ids=supporting_stmt_ids,
            supporting_paper_ids=list(supporting_paper_ids),
            relevant_limitations=lim_texts,
            relevant_future_work=fw_texts,
            signals={
                "pattern_type": pattern_type,
                "cluster_size": len(statements),
                "contributing_paper_count": len(supporting_paper_ids),
                "limitation_count": len(limitations),
                "future_work_count": len(future_works),
                "assumption_count": len(assumptions),
            },
        )
