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

# Patterns describing why superseded, external, or traditional baseline paradigms were replaced
BACKGROUND_MOTIVATION_PATTERNS = [
    re.compile(r"\b(?:traditional|conventional|centralized|classical|standard)\s+(?:learning|methods?|approaches?|systems?|models?)\s+(?:requires?|relies?|suffers?|poses?|raises?)\b", re.IGNORECASE),
    re.compile(r"\b(?:privacy\s+risks\s+and\s+storage\s+demands\s+of\s+centralized|storing\s+all\s+data\s+centrally)\b", re.IGNORECASE),
    re.compile(r"\b(?:motivation\s+for|in\s+order\s+to\s+avoid|to\s+alleviate\s+the\s+need\s+for)\s+(?:centralized|traditional)\b", re.IGNORECASE),
]

# Patterns describing generic importance, necessity, or purpose rationale that explain why a field exists
GENERIC_PURPOSE_MOTIVATION_PATTERNS = [
    re.compile(r"\b(?:is|are|was|were)\s+(?:required|essential|crucial|needed|vital|necessary|imperative)\s+to\s+(?:safeguard|ensure|achieve|enable|optimize|maintain|balance)\b", re.IGNORECASE),
    re.compile(r"\b(?:crucial|essential|required|vital|necessary)\s+for\s+(?:the\s+)?(?:planning|stability|reliability|integration|management|operation)\s+of\b", re.IGNORECASE),
    re.compile(r"\b(?:optimally|efficiently)\s+balance\s+(?:energy\s+)?supply\s+and\s+demand\b", re.IGNORECASE),
    re.compile(r"\b(?:growing|increasing)\s+integration\s+of\s+variable\s+(?:renewable|solar|wind)\s+(?:energy\s+sources|generation)\s+poses\s+new\s+challenges\b", re.IGNORECASE),
]

# Patterns describing narrow single-study experimental setups or author disclaimers
STUDY_SPECIFIC_DISCLAIMER_PATTERNS = [
    re.compile(r"\b(?:simulated\s+environment|simulation\s+(?:setting|framework|data)|limited\s+to\s+a\s+simulated|real-world\s+hospital\s+data,\s+which\s+may\s+not\s+fully\s+represent|single\s+hospital\s+data|specific\s+hospital\s+data|cohort\s+size\s+is\s+limited)\b", re.IGNORECASE),
]


def is_background_motivation_claim(text: str) -> bool:
    """Detects whether a claim represents introductory background motivation or domain rationale.

    Protects against false positives: if the text ALSO expresses an explicit empirical deficit,
    constraint, or technical failure (via DEFICIT_OR_CONSTRAINT_PATTERNS), it is NOT rejected as
    pure background motivation.
    """
    if not text or not text.strip():
        return False

    has_bg_pattern = any(pat.search(text) for pat in BACKGROUND_MOTIVATION_PATTERNS) or any(
        pat.search(text) for pat in GENERIC_PURPOSE_MOTIVATION_PATTERNS
    )
    if not has_bg_pattern:
        return False

    # False-positive guard: If it describes a genuine deficit/bottleneck/constraint,
    # it is a technical claim, not pure background motivation.
    has_deficit = any(pat.search(text) for pat in DEFICIT_OR_CONSTRAINT_PATTERNS)
    if has_deficit:
        return False

    return True


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
        supporting_passage_ids: Set[uuid.UUID] = set()

        for s in supporting_stmts:
            s_id = s.id if hasattr(s, "id") else s["id"]
            if isinstance(s_id, str):
                s_id = uuid.UUID(s_id)
            supporting_stmt_ids.append(s_id)

            p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
            if isinstance(p_id, str):
                p_id = uuid.UUID(p_id)
            supporting_paper_ids.add(p_id)

            raw_pid = getattr(s, "passage_id", None) if hasattr(s, "passage_id") else (
                s.get("passage_id") if isinstance(s, dict) else None
            )
            if raw_pid:
                try:
                    if isinstance(raw_pid, str):
                        supporting_passage_ids.add(uuid.UUID(raw_pid.strip()))
                    elif isinstance(raw_pid, uuid.UUID):
                        supporting_passage_ids.add(raw_pid)
                except (ValueError, AttributeError):
                    pass

        distinct_passages = len(supporting_passage_ids)
        is_same_passage = (distinct_passages <= 1 and len(supporting_stmts) > 1 and len(supporting_passage_ids) > 0)

        # Synthesize problem description from problem statements or cluster context
        if problems:
            primary_problem = (problems[0].statement_text if hasattr(problems[0], "statement_text") else problems[0]["statement_text"])
        else:
            primary_problem = (statements[0].statement_text if hasattr(statements[0], "statement_text") else statements[0]["statement_text"])

        # Determine gap statement based on dominant pattern
        lim_texts = [
            (l.statement_text if hasattr(l, "statement_text") else l["statement_text"]).strip()
            for l in limitations
        ]
        fw_texts = [
            (f.statement_text if hasattr(f, "statement_text") else f["statement_text"]).strip()
            for f in future_works
        ]

        lim_0_clean = lim_texts[0].rstrip(".") if lim_texts else ""
        fw_0_clean = fw_texts[0].rstrip(".") if fw_texts else ""
        primary_clean = primary_problem.strip().rstrip(".")

        if has_limitations and has_future_work:
            pattern_type = "LIMITATION_AND_FUTURE_WORK"
            distinct_papers = len(supporting_paper_ids)
            if distinct_papers > 1:
                gap_stmt = (
                    f"Unresolved methodological limitation and open research direction: {lim_0_clean}. "
                    f"Proposed future direction remains unaddressed: {fw_0_clean}."
                )
                why_insufficient = (
                    f"Existing approaches demonstrate documented constraints ({lim_0_clean}), "
                    f"while subsequent work explicitly designates this problem for future investigation without resolution."
                )
            else:
                if is_same_passage:
                    gap_stmt = (
                        f"Documented limitation and proposed direction from single source passage: {lim_0_clean}. "
                        f"Proposed direction: {fw_0_clean}."
                    )
                    why_insufficient = (
                        f"Current work demonstrates documented constraints ({lim_0_clean}), "
                        f"while noting this problem for exploration ({fw_0_clean}) within the same source passage."
                    )
                else:
                    gap_stmt = (
                        f"Documented scientific limitation and open direction within single study: {lim_0_clean}. "
                        f"Proposed direction: {fw_0_clean}."
                    )
                    why_insufficient = (
                        f"Current work demonstrates documented constraints ({lim_0_clean}), "
                        f"while explicitly designating this problem for future investigation without resolution."
                    )
        elif has_limitations:
            distinct_papers = len(supporting_paper_ids)
            if distinct_papers > 1:
                pattern_type = "RECURRING_LIMITATIONS"
                gap_stmt = f"Persistent scientific limitation across current methodologies: {lim_0_clean}."
                why_insufficient = f"While current methods establish baseline findings, they explicitly encounter limitations across multiple papers: {'; '.join(t.rstrip('.') for t in lim_texts[:2])}."
            else:
                pattern_type = "DOCUMENTED_LIMITATION"
                gap_stmt = f"Documented scientific limitation in methodology: {lim_0_clean}."
                why_insufficient = f"Current work identifies an explicit limitation: {lim_0_clean}."
        elif has_future_work:
            pattern_type = "OPEN_FUTURE_WORK"
            gap_stmt = f"Unexplored research direction identified across scientific literature: {fw_0_clean}."
            why_insufficient = f"Prior work highlights this challenge as an open milestone, but no verified method has resolved it within the analyzed evidence."
        else:
            has_deficit = any(pat.search(primary_problem) for pat in DEFICIT_OR_CONSTRAINT_PATTERNS)
            if has_deficit:
                pattern_type = "UNRESOLVED_PROBLEM"
                gap_stmt = f"Identified research problem with documented constraints: {primary_clean}."
                why_insufficient = f"The underlying research question involves documented technical bottlenecks ({primary_clean}), requiring empirical resolution."
            else:
                pattern_type = "UNRESOLVED_PROBLEM"
                gap_stmt = f"Identified research question from literature: {primary_clean}."
                why_insufficient = f"The literature describes this research objective ({primary_clean}), but empirical validation remains to be established."

        # Detect potential background motivation and study-specific constraint signals
        is_bg_motivation = is_background_motivation_claim(primary_problem)
        is_study_specific = (
            any(pat.search(lim_0_clean) for pat in STUDY_SPECIFIC_DISCLAIMER_PATTERNS)
            if (len(supporting_paper_ids) <= 1 and lim_0_clean) else False
        )

        return GapCandidate(
            gap_statement=gap_stmt,
            research_problem=primary_problem,
            why_existing_work_insufficient=why_insufficient,
            supporting_statement_ids=supporting_stmt_ids,
            supporting_paper_ids=list(supporting_paper_ids),
            supporting_passage_ids=list(supporting_passage_ids),
            relevant_limitations=lim_texts,
            relevant_future_work=fw_texts,
            signals={
                "pattern_type": pattern_type,
                "cluster_size": len(statements),
                "contributing_paper_count": len(supporting_paper_ids),
                "distinct_passage_count": distinct_passages,
                "same_passage_evidence": is_same_passage,
                "limitation_count": len(limitations),
                "future_work_count": len(future_works),
                "assumption_count": len(assumptions),
                "potential_background_motivation": is_bg_motivation,
                "potential_study_specific_limitation": is_study_specific,
            },
        )
