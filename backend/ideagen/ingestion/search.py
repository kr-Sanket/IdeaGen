from typing import List, Protocol, Optional, Dict, Any
from pydantic import BaseModel
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import uuid
import datetime

class PaperSearchResult(BaseModel):
    id: str
    source: str
    title: str
    authors: List[str]
    publication_year: int
    publication_date: str
    abstract: str
    url: Optional[str]
    doi: Optional[str]
    raw_metadata: Dict[str, Any]

class PaperSearchProvider(Protocol):
    def search(self, query: str, limit: int = 10) -> List[PaperSearchResult]:
        ...

class ArxivSearchProvider:
    def search(self, query: str, limit: int = 10) -> List[PaperSearchResult]:
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
            print(f"ArXiv search failed: {e}")
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
                id=f"W{uuid.uuid4().hex[:10]}", # Mock OpenAlex ID style for compatibility
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
