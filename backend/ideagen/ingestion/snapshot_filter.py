import json
from datetime import date
from typing import Iterator, Dict, Any

def passes_snapshot_filters(work: Dict[str, Any]) -> bool:
    """
    Applies the frozen corpus rules to a raw OpenAlex Snapshot work record.
    """
    import unicodedata
    # 1. Basic metadata validity
    if not work.get("id"):
        return False
        
    # 2. Publication date filter (2019-01-01 to 2026-06-30)
    pub_date_str = work.get("publication_date")
    if not pub_date_str:
        return False
    try:
        pub_date = date.fromisoformat(pub_date_str)
    except ValueError:
        return False
    if not (date(2019, 1, 1) <= pub_date <= date(2026, 6, 30)):
        return False
        
    # 3. Type filter
    work_type = work.get("type", "")
    if work_type not in ("article", "preprint"):
        return False
        
    # 4. Language filter
    if work.get("language") != "en":
        return False
        
    # 5. Retraction filter
    if work.get("is_retracted") is True:
        return False
        
    # 6. Core vs Expansion Corpus filter (v1.2.1)
    if work.get("is_xpac") is True:
        return False
        
    # 6. Abstract availability/reconstruction
    abstract_index = work.get("abstract_inverted_index")
    if not abstract_index:
        return False
        
    # Reconstruct abstract
    word_positions = []
    for word, positions in abstract_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    word_positions.append((pos, word))
    word_positions.sort()
    abstract_text = " ".join(word for _, word in word_positions)
    if not abstract_text.strip():
        return False
        
    # 7. Query-term matching (Title + Abstract approximation)
    title_text = work.get("title") or ""
    
    def normalize(text: str) -> str:
        if not text: return ""
        # Unicode normalization NFKC and lowercasing
        return unicodedata.normalize("NFKC", text).lower()
        
    searchable_text = normalize(title_text) + " " + normalize(abstract_text)
    
    # A. Phrase match (conservative exact substring)
    if "retrieval-augmented generation" not in searchable_text:
        return False
        
    # B. OR match
    reliability_terms = ["hallucination", "faithfulness", "grounding", "factuality"]
    if not any(term in searchable_text for term in reliability_terms):
        return False
        
    return True
