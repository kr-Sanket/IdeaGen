import time
import json
import urllib.request
import urllib.parse
import urllib.error
import logging
import random
from typing import Dict, Any, Iterator
from .models import IngestionConfig

logger = logging.getLogger(__name__)

class OpenAlexError(Exception):
    pass

class OpenAlexRateLimitError(OpenAlexError):
    pass

class OpenAlexClient:
    def __init__(self, config: IngestionConfig):
        self.config = config
        
    def _make_request(self, url: str) -> Dict[str, Any]:
        user_agent = f"IdeaGen (mailto:{self.config.polite_email})" if self.config.polite_email else "IdeaGen"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": user_agent}
        )
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        time.sleep(0.15)
                        return data
                    else:
                        raise OpenAlexError(f"Unexpected status: {resp.status}")
            except urllib.error.HTTPError as e:
                retry_after = e.headers.get("Retry-After")
                
                if e.code == 429:
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"HTTP 429 Too Many Requests. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                elif e.code in (500, 502, 503, 504):
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"HTTP {e.code} Server Error. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise OpenAlexError(f"HTTP Error {e.code}: {e.reason}")
            except urllib.error.URLError as e:
                if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"Timeout. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                raise OpenAlexError(f"URL Error: {e.reason}")
                
        raise OpenAlexRateLimitError("Max retries exceeded for OpenAlex request.")

    def fetch_works(self, cursor: str = "*") -> Iterator[Dict[str, Any]]:
        """Fetch works using cursor pagination."""
        params = {
            "filter": ",".join(f"{k}:{v}" for k, v in self.config.query_filters.items()),
            "per_page": str(self.config.per_page),
        }
        if self.config.polite_email:
            params["mailto"] = self.config.polite_email
        
        current_cursor = cursor
        
        while current_cursor:
            params["cursor"] = current_cursor
            url = f"{self.config.base_url}?{urllib.parse.urlencode(params)}"
            
            logger.info(f"Fetching OpenAlex works with cursor {current_cursor}")
            data = self._make_request(url)
            
            results = data.get("results", [])
            if not results:
                break
                
            yield from results
            
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            
            if not next_cursor or next_cursor == current_cursor:
                break
                
            current_cursor = next_cursor
