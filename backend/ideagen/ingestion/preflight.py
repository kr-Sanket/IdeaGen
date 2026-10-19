"""Lightweight offline corpus metadata and publication-date preflight validation."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CorpusPreflightReport:
    """Structured report of offline corpus publication-date and identity preflight."""

    total_papers: int
    dated_papers_count: int
    undated_papers_count: int
    malformed_dates_count: int
    dated_papers: List[str]
    undated_papers: List[str]
    malformed_dates: List[Dict[str, Any]]
    duplicate_identities: Dict[str, List[str]]
    date_range: Dict[str, Optional[str]]
    distribution_by_year: Dict[str, int]
    temporal_evaluation_impact: str
    warnings: List[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_papers": self.total_papers,
            "dated_papers_count": self.dated_papers_count,
            "undated_papers_count": self.undated_papers_count,
            "malformed_dates_count": self.malformed_dates_count,
            "dated_papers": self.dated_papers,
            "undated_papers": self.undated_papers,
            "malformed_dates": self.malformed_dates,
            "duplicate_identities": self.duplicate_identities,
            "date_range": self.date_range,
            "distribution_by_year": self.distribution_by_year,
            "temporal_evaluation_impact": self.temporal_evaluation_impact,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [
            "=== Corpus Date & Metadata Preflight Report ===",
            f"Total Papers: {self.total_papers}",
            f"Dated Papers: {self.dated_papers_count}",
            f"Undated Papers: {self.undated_papers_count}",
            f"Malformed Dates: {self.malformed_dates_count}",
            f"Date Range: {self.date_range.get('earliest', 'N/A')} to {self.date_range.get('latest', 'N/A')}",
        ]
        if self.distribution_by_year:
            lines.append("Yearly Distribution:")
            for yr, count in sorted(self.distribution_by_year.items()):
                lines.append(f"  {yr}: {count}")
        if self.duplicate_identities:
            lines.append("Duplicate Identities Detected:")
            for ident, fnames in self.duplicate_identities.items():
                lines.append(f"  {ident}: {', '.join(fnames)}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append(f"Temporal Evaluation Impact:\n  {self.temporal_evaluation_impact}")
        return "\n".join(lines)


def run_corpus_preflight(input_dir: Path | str) -> CorpusPreflightReport:
    """Runs a lightweight offline preflight check over input PDF directory and manifest."""
    dir_path = Path(input_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {dir_path}")

    pdf_files = sorted(list(dir_path.glob("*.pdf")))
    manifest_data: Dict[str, Dict[str, Any]] = {}
    for m_name in ["manifest.json", "metadata.json"]:
        m_path = dir_path / m_name
        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict) and "filename" in item:
                            manifest_data[item["filename"]] = item
                elif isinstance(raw, dict):
                    if "papers" in raw and isinstance(raw["papers"], list):
                        for item in raw["papers"]:
                            if isinstance(item, dict) and "filename" in item:
                                manifest_data[item["filename"]] = item
                    else:
                        for k, v in raw.items():
                            if isinstance(v, dict):
                                manifest_data[k] = v
            except Exception as e:
                logger.warning(f"Could not parse manifest {m_path}: {e}")
            break

    dated_papers: List[str] = []
    pure_undated_papers: List[str] = []
    malformed_dates: List[Dict[str, Any]] = []
    parsed_dates: List[datetime.date] = []
    distribution_by_year: Dict[str, int] = {}
    warnings: List[str] = []

    doi_map: Dict[str, List[str]] = {}
    title_map: Dict[str, List[str]] = {}

    for pdf in pdf_files:
        meta = manifest_data.get(pdf.name, {})
        raw_date = meta.get("publication_date")

        # Track DOI duplicate
        doi = meta.get("doi")
        if doi and isinstance(doi, str) and doi.strip():
            doi_norm = doi.strip().lower()
            doi_map.setdefault(doi_norm, []).append(pdf.name)

        # Track title duplicate
        title = meta.get("title")
        if title and isinstance(title, str) and title.strip():
            title_norm = title.strip().lower()
            title_map.setdefault(title_norm, []).append(pdf.name)

        if raw_date is None or (isinstance(raw_date, str) and not raw_date.strip()):
            pure_undated_papers.append(pdf.name)
            continue

        # Attempt to parse date
        try:
            date_str = str(raw_date).split("T")[0]
            d = datetime.date.fromisoformat(date_str)
            dated_papers.append(pdf.name)
            parsed_dates.append(d)
            yr = str(d.year)
            distribution_by_year[yr] = distribution_by_year.get(yr, 0) + 1
        except Exception as parse_err:
            malformed_dates.append({
                "filename": pdf.name,
                "raw_date": raw_date,
                "error": str(parse_err),
            })

    duplicate_identities: Dict[str, List[str]] = {}
    for doi_key, files in doi_map.items():
        if len(files) > 1:
            duplicate_identities[f"doi:{doi_key}"] = files
            warnings.append(f"Duplicate DOI '{doi_key}' found in files: {files}")

    for title_key, files in title_map.items():
        if len(files) > 1:
            duplicate_identities[f"title:{title_key}"] = files
            warnings.append(f"Duplicate title '{title_key}' found in files: {files}")

    if pure_undated_papers:
        warnings.append(
            f"{len(pure_undated_papers)} paper(s) lack publication dates and will be excluded from temporal filtering."
        )

    if malformed_dates:
        warnings.append(
            f"{len(malformed_dates)} paper(s) have malformed publication dates that failed ISO parsing."
        )

    earliest_str = min(parsed_dates).isoformat() if parsed_dates else None
    latest_str = max(parsed_dates).isoformat() if parsed_dates else None
    date_range = {"earliest": earliest_str, "latest": latest_str}

    # Construct impact statement
    impact_parts = []
    total_missing = len(pure_undated_papers) + len(malformed_dates)
    if total_missing > 0:
        impact_parts.append(
            f"{total_missing} out of {len(pdf_files)} papers lack verified ISO publication dates "
            f"({len(pure_undated_papers)} undated, {len(malformed_dates)} malformed). "
            f"Under strict temporal isolation, these papers will be excluded from historical cutoff filtering "
            f"and all Stage 5 chronological evolution batches, reducing the effective evaluation corpus."
        )
    else:
        impact_parts.append(
            f"All {len(pdf_files)} papers possess valid publication dates. "
            f"Temporal isolation will cover 100% of the corpus without date-based exclusions."
        )

    if duplicate_identities:
        impact_parts.append(
            f"Found {len(duplicate_identities)} duplicate identity groups. "
            f"Identical papers may have redundant evidence deduplicated during lifecycle transitions."
        )

    temporal_evaluation_impact = " ".join(impact_parts)

    return CorpusPreflightReport(
        total_papers=len(pdf_files),
        dated_papers_count=len(dated_papers),
        undated_papers_count=len(pure_undated_papers),
        malformed_dates_count=len(malformed_dates),
        dated_papers=dated_papers,
        undated_papers=pure_undated_papers,
        malformed_dates=malformed_dates,
        duplicate_identities=duplicate_identities,
        date_range=date_range,
        distribution_by_year=distribution_by_year,
        temporal_evaluation_impact=temporal_evaluation_impact,
        warnings=warnings,
    )
