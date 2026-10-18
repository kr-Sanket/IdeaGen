"""Validation harness for testing OpenAlex query scope candidates."""

import logging
from typing import Dict, Any, List
from .client import OpenAlexClient
from .models import IngestionConfig
from .parser import BATCHES

logger = logging.getLogger(__name__)

PREDEFINED_CANDIDATES = {
    "C": {"title_and_abstract.search": '"retrieval-augmented generation" factuality'},
    "D": {"title_and_abstract.search": '"retrieval-augmented generation" grounding'},
    "E": {
        "title_and_abstract.search": '"retrieval-augmented generation"',
        "default.search": "hallucination|faithfulness|factuality"
    },
    "F": {
        "title_and_abstract.search": '"retrieval-augmented generation"|"retrieval augmented generation"|RAG',
        "default.search": "hallucination|faithfulness|factuality"
    },
    "G": {
        "title_and_abstract.search": '"large language model"|"large language models"|LLM|LLMs',
        "default.search": "hallucination|faithfulness|factuality"
    }
}

BASELINE_RESULTS = {
    "A": {
        "query_id": "A",
        "description": '"retrieval-augmented generation" hallucination',
        "filtered_count": 1341, # From Phase-2 initial run
        "note": "Previously validated baseline. Defensible but potentially too narrow."
    },
    "B": {
        "query_id": "B",
        "description": '"retrieval-augmented generation" hallucination|faithfulness|grounding|factuality',
        "filtered_count": 66400, # From Phase-1 run
        "note": "Previously validated baseline. Rejected due to extensive domain leakage on 'grounding'."
    }
}

class ValidationHarness:
    def __init__(self, polite_email: str):
        self.polite_email = polite_email
        self.base_filters = {
            "from_publication_date": "2019-01-01",
            "to_publication_date": "2026-06-30",
        }
        self.fully_filtered = {
            **self.base_filters,
            "type": "article|preprint",
            "language": "en",
            "has_abstract": "true",
            "is_retracted": "false"
        }
        
    def _construct_url(self, config: IngestionConfig, filters: dict, per_page: int) -> str:
        f_str = ",".join(f"{k}:{v}" for k, v in filters.items())
        return f"{config.base_url}?filter={f_str}&per_page={per_page}&mailto={self.polite_email}"
        
    def evaluate_candidate(self, query_id: str, query_filters: dict) -> Dict[str, Any]:
        """Evaluates a single candidate query against the API without freezing it."""
        report = {
            "query_id": query_id,
            "exact_filters": query_filters,
            "raw_count": -1,
            "filtered_count": -1,
            "temporal_counts": {},
            "samples": [],
            "metadata_quality": {},
            "errors": []
        }
        
        # Raw client config
        raw_config = IngestionConfig(
            query_filters={**self.base_filters, **query_filters},
            polite_email=self.polite_email,
            per_page=1
        )
        raw_client = OpenAlexClient(raw_config)
        
        # Filtered client config
        filtered_config = IngestionConfig(
            query_filters={**self.fully_filtered, **query_filters},
            polite_email=self.polite_email,
            per_page=10
        )
        filtered_client = OpenAlexClient(filtered_config)
        
        try:
            # 1. Raw count
            url_raw = self._construct_url(raw_config, raw_config.query_filters, 1)
            raw_data = raw_client._make_request(url_raw)
            report["raw_count"] = raw_data.get("meta", {}).get("count", -1)
            
            # 2. Filtered count and samples
            url_full = self._construct_url(filtered_config, filtered_config.query_filters, 10)
            full_data = filtered_client._make_request(url_full)
            report["filtered_count"] = full_data.get("meta", {}).get("count", -1)
            
            # 3. Temporal counts
            for batch in BATCHES:
                batch_filters = {
                    **self.fully_filtered, 
                    **query_filters, 
                    "from_publication_date": batch.start_date.isoformat(),
                    "to_publication_date": batch.end_date.isoformat()
                }
                url_batch = self._construct_url(filtered_config, batch_filters, 1)
                batch_data = filtered_client._make_request(url_batch)
                report["temporal_counts"][batch.id] = batch_data.get("meta", {}).get("count", -1)
            
            # 4. Samples and Metadata Quality
            samples = full_data.get("results", [])
            
            has_doi = 0
            is_preprint = 0
            has_abstract = 0
            
            for s in samples:
                report["samples"].append({
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "doi": s.get("doi"),
                    "type": s.get("type"),
                })
                if s.get("doi"): has_doi += 1
                if s.get("type") == "preprint": is_preprint += 1
                if s.get("abstract_inverted_index"): has_abstract += 1
                
            report["metadata_quality"] = {
                "total_samples": len(samples),
                "has_doi": has_doi,
                "is_preprint": is_preprint,
                "has_abstract": has_abstract
            }
                
        except Exception as e:
            logger.error(f"Error evaluating candidate {query_id}: {e}")
            report["errors"].append(str(e))
            
        return report
