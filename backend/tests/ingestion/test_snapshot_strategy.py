import pytest
from ideagen.ingestion.snapshot_filter import passes_snapshot_filters

def create_work(
    id="W123",
    publication_date="2025-01-01",
    type="article",
    language="en",
    is_retracted=False,
    is_xpac=False,
    title="A Study on Retrieval-Augmented Generation",
    abstract_words=["This", "fixes", "hallucination"]
):
    # build abstract_inverted_index
    index = {}
    for pos, word in enumerate(abstract_words):
        if word not in index:
            index[word] = []
        index[word].append(pos)
        
    return {
        "id": id,
        "publication_date": publication_date,
        "type": type,
        "language": language,
        "is_retracted": is_retracted,
        "is_xpac": is_xpac,
        "title": title,
        "abstract_inverted_index": index if abstract_words else None
    }

def test_valid_article():
    work = create_work(type="article")
    assert passes_snapshot_filters(work) is True

def test_valid_preprint():
    work = create_work(type="preprint")
    assert passes_snapshot_filters(work) is True

def test_missing_abstract():
    work = create_work(abstract_words=[])
    assert passes_snapshot_filters(work) is False

def test_wrong_language():
    work = create_work(language="fr")
    assert passes_snapshot_filters(work) is False

def test_wrong_type():
    work = create_work(type="book_chapter")
    assert passes_snapshot_filters(work) is False

def test_retracted():
    work = create_work(is_retracted=True)
    assert passes_snapshot_filters(work) is False

def test_is_xpac():
    work_core = create_work(is_xpac=False)
    assert passes_snapshot_filters(work_core) is True
    
    work_xpac = create_work(is_xpac=True)
    assert passes_snapshot_filters(work_xpac) is False

def test_date_outside_range():
    work = create_work(publication_date="2018-12-31")
    assert passes_snapshot_filters(work) is False
    work2 = create_work(publication_date="2026-07-01")
    assert passes_snapshot_filters(work2) is False

def test_missing_phrase_match():
    work = create_work(title="Just a study on generation", abstract_words=["fixes", "hallucination"])
    assert passes_snapshot_filters(work) is False

def test_missing_or_match():
    # Has RAG but missing hallucination/faithfulness/grounding/factuality
    work = create_work(title="Retrieval-augmented generation", abstract_words=["improves", "speed"])
    assert passes_snapshot_filters(work) is False

def test_match_in_abstract():
    # Phrase in abstract, OR term in title
    work = create_work(
        title="Reducing hallucination",
        abstract_words=["using", "retrieval-augmented", "generation"]
    )
    assert passes_snapshot_filters(work) is True

def test_case_variation():
    work = create_work(
        title="RETRIEVAL-AUGMENTED GENERATION",
        abstract_words=["HALLUCINATION"]
    )
    assert passes_snapshot_filters(work) is True

def test_unicode_normalization():
    # Title with non-standard Unicode (e.g. full-width or accented matching base if normalized)
    work = create_work(
        title="retrieval-augmented generation",
        abstract_words=["h\u0061llucination"] # normal 'a'
    )
    assert passes_snapshot_filters(work) is True
    
    # \uFB01 is 'fi' ligature, NFKC normalizes it to 'fi'
    work_ligature = create_work(
        title="retrieval-augmented generation",
        abstract_words=["fai\uFB01thfulness"] # fai + fi + thfulness (wait, fi is \uFB01, so fai+fi is faifi, actually faith is f a i t h. Let's do f a i t h f u l n e s s. 'fi' ligature isn't here. Let's just do 'factuality' with a ligature? No, 'ff' in faithfulness?)
    )
    # let's just use \uFF46 fullwidth f for factuality
    work_fullwidth = create_work(
        title="retrieval-augmented generation",
        abstract_words=["\uFF46actuality"] # fullwidth f
    )
    assert passes_snapshot_filters(work_fullwidth) is True

def test_punctuation_behavior():
    # If phrase is hyphenated, it must exactly match hyphenated in our conservative approach.
    work_no_hyphen = create_work(
        title="retrieval augmented generation",
        abstract_words=["hallucination"]
    )
    assert passes_snapshot_filters(work_no_hyphen) is False
