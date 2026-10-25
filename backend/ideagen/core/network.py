"""Outbound-network diagnostics shared by HTTP clients (OpenAlex, Ollama).

The backend honours the standard HTTP(S)_PROXY / NO_PROXY environment variables. When it is
started from a sandboxed tool, those variables can point at a policy proxy that rejects requests
(for example "not allowed by policy" for api.openalex.org, or "Port 11434 not allowed" for a local
Ollama). These helpers only describe that routing so errors are actionable; they never bypass it.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Optional


def _redact(proxy_url: str) -> str:
    """Drops any credentials from a proxy URL."""
    parsed = urllib.parse.urlsplit(proxy_url)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    scheme = f"{parsed.scheme}://" if parsed.scheme else ""
    return f"{scheme}{host}{port}" if host else "<unparseable proxy URL>"


def proxy_for_url(url: str) -> Optional[str]:
    """Returns the (redacted) environment proxy that requests to ``url`` are routed through, if any."""
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        return None
    proxies = urllib.request.getproxies_environment()
    proxy = proxies.get(parsed.scheme)
    if not proxy:
        return None
    host = parsed.hostname if parsed.port is None else f"{parsed.hostname}:{parsed.port}"
    if urllib.request.proxy_bypass_environment(host, proxies):
        return None
    return _redact(proxy)


def proxy_hint(url: str) -> str:
    """A sentence explaining proxy routing for ``url``, or an empty string when no proxy applies."""
    proxy = proxy_for_url(url)
    if proxy is None:
        return ""
    return (
        f" The backend process routes this request through the proxy {proxy} configured in its "
        "environment (HTTP_PROXY/HTTPS_PROXY), so this response may come from that proxy rather than "
        "the service. If that proxy is a sandbox or policy proxy, start the backend from a regular "
        "terminal without those variables (e.g. `make backend`), or add the host to NO_PROXY."
    )


def looks_like_proxy_denial(body: str) -> bool:
    """True when a response body reads like a proxy/egress-policy rejection rather than an API error."""
    lowered = (body or "").lower()
    return (
        "not allowed by policy" in lowered
        or "blocked by policy" in lowered
        or ("not allowed for http" in lowered and "port" in lowered)
        or "proxy authentication required" in lowered
    )
