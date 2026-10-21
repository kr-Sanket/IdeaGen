from typing import List, Protocol, Optional, Dict, Any
from pydantic import BaseModel, Field
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import uuid
import datetime
import logging
import random
import re
import time

from ideagen.core.config import get_settings
from ideagen.ingestion.parser import PaperNormalizer, OpenAlexResponseParser

logger = logging.getLogger(__name__)

# A query is routed to an identifier filter only if it is entirely an identifier.
_OPENALEX_ID_QUERY_RE = re.compile(
    r"^(?:(?:https?://)?(?:api\.)?openalex\.org/(?:works/)?)?[Ww]\d+$"
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

# Upper bound for any single retry wait (including server-supplied Retry-After).
MAX_RETRY_DELAY_SECONDS = 10.0


class PaperSearchResult(BaseModel):
    id: str
    source: str
    title: str
    authors: List[str]
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    abstract: str
    url: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    is_oa: Optional[bool] = None
    pdf_url: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class PaperSearchProvider(Protocol):
    def search(
        self,
        query: str,
        limit: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PaperSearchResult]:
        ...


class OpenAlexSearchError(Exception):
    """Exception raised when an OpenAlex search request fails."""
    pass


class OpenAlexSearchProvider:
    """Production literature search provider backed by the OpenAlex /works API."""

    def __init__(
        self,
        email: Optional[str] = None,
        base_url: str = "https://api.openalex.org/works",
        timeout: float = 15.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
    ):
        if email is not None:
            self.email = email.strip() or None
        else:
            settings = get_settings()
            self.email = settings.openalex_email.strip() if settings.openalex_email else None
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_on_error = raise_on_error
        self.last_total_count: Optional[int] = None

    def _build_params(
        self,
        query: str,
        limit: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, str]:
        params: Dict[str, str] = {
            "per_page": str(max(1, min(limit, 100))),
        }
        if self.email:
            params["mailto"] = self.email

        filters: List[str] = []
        q = query.strip()

        # 1. Direct ID / DOI detection: only when the whole query is an identifier,
        # so free text such as "What makes RAG faithful" or "llama 10.5" stays a search.
        norm_doi = PaperNormalizer.normalize_doi(q)
        if _OPENALEX_ID_QUERY_RE.match(q):
            filters.append(f"openalex:{PaperNormalizer.normalize_openalex_id(q)}")
        elif norm_doi and _DOI_RE.match(norm_doi):
            filters.append(f"doi:{norm_doi}")
        else:
            params["search"] = q

        # 2. Date filters
        if start_date and start_date.strip():
            filters.append(f"from_publication_date:{start_date.strip()}")
        if end_date and end_date.strip():
            filters.append(f"to_publication_date:{end_date.strip()}")

        if filters:
            params["filter"] = ",".join(filters)

        return params

    def _execute_request(self, url: str) -> Optional[Dict[str, Any]]:
        user_agent = f"IdeaGen (mailto:{self.email})" if self.email else "IdeaGen"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )
        last_error = None
        base_delay = 1.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8")
                        return json.loads(raw)
                    if self.raise_on_error:
                        raise OpenAlexSearchError(f"Unexpected status: {resp.status}")
                    return None
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code in (429, 500, 502, 503, 504):
                    if attempt >= self.max_retries - 1:
                        break  # no further attempt; don't sleep for nothing
                    retry_after = e.headers.get("Retry-After") if e.code == 429 and e.headers else None
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    # A server-supplied Retry-After must not block a request thread indefinitely.
                    delay = min(delay, MAX_RETRY_DELAY_SECONDS)
                    logger.warning(f"OpenAlex {e.code} response. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"OpenAlex search HTTP error {e.code}: {e.reason}")
                    if self.raise_on_error:
                        raise OpenAlexSearchError(f"HTTP Error {e.code}: {e.reason}") from e
                    return None
            except (urllib.error.URLError, TimeoutError) as e:
                last_error = e
                logger.warning(f"OpenAlex search network error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                if self.raise_on_error:
                    raise OpenAlexSearchError(f"Network error: {e}") from e
                return None
            except Exception as e:
                last_error = e
                logger.warning(f"OpenAlex search unexpected error: {e}")
                if self.raise_on_error:
                    raise OpenAlexSearchError(f"Unexpected error: {e}") from e
                return None

        if self.raise_on_error and last_error is not None:
            raise OpenAlexSearchError(f"Max retries exceeded: {last_error}")
        return None

    def _parse_work_to_result(self, work: Dict[str, Any]) -> PaperSearchResult:
        raw_id = work.get("id") or ""
        norm_id = PaperNormalizer.canonical_openalex_source_id(str(raw_id)) or str(raw_id)

        title = work.get("title") or "Untitled Work"

        # Authors
        authors: List[str] = []
        for a in work.get("authorships", []):
            if isinstance(a, dict):
                author_obj = a.get("author") or {}
                name = author_obj.get("display_name") or a.get("raw_author_name")
                if name:
                    authors.append(name.strip())

        # Dates
        pub_date = work.get("publication_date")
        pub_year = work.get("publication_year")
        if pub_year is None and pub_date:
            try:
                pub_year = int(pub_date[:4])
            except (ValueError, TypeError):
                pub_year = None

        # Abstract
        abstract = OpenAlexResponseParser.reconstruct_abstract(work.get("abstract_inverted_index"))

        # DOI
        doi = PaperNormalizer.normalize_doi(work.get("doi"))

        # Citation count
        citation_count = work.get("cited_by_count", 0)

        # Open Access status
        oa_obj = work.get("open_access") or {}
        is_oa = bool(oa_obj.get("is_oa", False))

        # Locations & PDF URL
        best_oa = work.get("best_oa_location") or {}
        primary_loc = work.get("primary_location") or {}
        pdf_url = best_oa.get("pdf_url") or primary_loc.get("pdf_url")

        # Landing URL
        landing_url = (
            best_oa.get("landing_page_url")
            or primary_loc.get("landing_page_url")
            or (f"https://doi.org/{doi}" if doi else None)
            or (f"https://openalex.org/{norm_id}" if norm_id else raw_id)
        )

        return PaperSearchResult(
            id=norm_id,
            source="OPENALEX",
            title=title,
            authors=authors,
            publication_year=pub_year,
            publication_date=pub_date,
            abstract=abstract,
            url=landing_url,
            doi=doi,
            citation_count=citation_count,
            is_oa=is_oa,
            pdf_url=pdf_url,
            raw_metadata=work,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PaperSearchResult]:
        """Executes search against OpenAlex /works endpoint with optional date bounds."""
        if not query or not query.strip():
            return []

        params = self._build_params(
            query=query,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        data = self._execute_request(url)
        if not data or not isinstance(data, dict):
            self.last_total_count = None
            return []

        meta = data.get("meta")
        if isinstance(meta, dict) and "count" in meta and isinstance(meta["count"], int):
            self.last_total_count = meta["count"]
        else:
            self.last_total_count = None

        results: List[PaperSearchResult] = []
        for item in data.get("results", []):
            if isinstance(item, dict):
                results.append(self._parse_work_to_result(item))
        return results


class ArxivSearchProvider:
    def search(
        self,
        query: str,
        limit: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PaperSearchResult]:
        limit = min(limit, 20)

        # arXiv query syntax
        arxiv_query = urllib.parse.quote(query)
        url = f'http://export.arxiv.org/api/query?search_query=all:{arxiv_query}&start=0&max_results={limit}'

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
        except Exception as e:
            # Gracefully handle network/API errors
            logger.warning(f"ArXiv search failed: {e}")
            return []

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return []

        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        results = []

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text
            year = int(published[:4])
            arxiv_id_full = entry.find('atom:id', ns).text
            arxiv_id = arxiv_id_full.split('/')[-1]

            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]

            # Extract DOI if present
            doi = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'doi':
                    doi = link.attrib.get('href')

            if not doi:
                doi = f"https://doi.org/10.48550/arXiv.{arxiv_id}"

            res = PaperSearchResult(
                id=f"W{uuid.uuid4().hex[:10]}",
                source="ARXIV",
                title=title,
                authors=authors,
                publication_year=year,
                publication_date=published[:10],
                abstract=summary,
                url=arxiv_id_full,
                doi=doi,
                raw_metadata={
                    "id": f"W{uuid.uuid4().hex[:10]}",
                    "title": title,
                    "publication_date": published[:10],
                    "publication_year": year,
                    "doi": doi
                }
            )
            results.append(res)

        return results
