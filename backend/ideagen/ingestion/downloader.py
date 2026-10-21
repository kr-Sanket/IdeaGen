"""Open-Access PDF Downloader for scientific literature.

Features:
- Extracts open-access PDF URLs from OpenAlex work objects (best_oa_location first, then primary_location).
- Validates URL scheme (HTTP/HTTPS only) and prevents SSRF / unsafe file URL access.
- Validates every redirect destination BEFORE connecting to it (blocking loopback, private,
  link-local, multicast, reserved, and cloud-metadata addresses).
- Supports standard and alternative numeric/hex/octal/IPv6-mapped IP representations.
- Prevents forwarding sensitive headers (e.g. mailto User-Agent, auth) across different origins.
- Streams responses with timeout and 50 MB size limit (both Content-Length pre-check and stream byte counter).
- Validates PDF header signature (%PDF-) and rejects HTML landing pages or binary junk.
- Handles HTTP 403/404, network errors, timeouts, and oversized responses gracefully.
- Guarantees cleanup of temporary download files upon context manager exit.
- Never prints or logs sensitive configuration.
"""

from __future__ import annotations

import os
import socket
import ipaddress
import tempfile
import urllib.request
import urllib.parse
import urllib.error
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, Union

from ideagen.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_PDF_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
DEFAULT_DOWNLOAD_TIMEOUT: float = 30.0  # seconds
PDF_SIGNATURE: bytes = b"%PDF-"

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "metadata",
}

BLOCKED_DOMAIN_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
)


class PDFDownloadError(Exception):
    """Base exception for PDF download errors."""
    pass


class SSRFProtectionError(PDFDownloadError):
    """Raised when URL attempts connection to private, loopback, or metadata addresses."""
    pass


class PDFUnavailableError(PDFDownloadError):
    """Raised when PDF is not accessible (403, 404, 410, etc.)."""
    pass


class OversizedPDFError(PDFDownloadError):
    """Raised when PDF exceeds the maximum allowed size."""
    pass


class InvalidPDFSignatureError(PDFDownloadError):
    """Raised when downloaded file is not a valid PDF (e.g. HTML landing page)."""
    pass


def _parse_octet(p: str) -> int:
    """Parses a dotted IPv4 octet accepting decimal, hex (0x..), or octal (0..)."""
    if p.startswith("0x") or p.startswith("0X"):
        return int(p, 16)
    if p.startswith("0") and len(p) > 1 and all(c in "01234567" for c in p):
        return int(p, 8)
    return int(p, 10)


def parse_ip_literal(host: str) -> Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    """Detects and parses standard and non-standard IPv4 and IPv6 address literals.

    Handles:
    - Standard IPv4 / IPv6 literals (e.g. 127.0.0.1, [::1])
    - Decimal integers (e.g. 2130706433)
    - Hexadecimal literals (e.g. 0x7f000001)
    - Dotted octal / hex quads (e.g. 0177.0.0.1, 0x7f.0.0.1)
    - IPv6-mapped IPv4 literals (e.g. ::ffff:127.0.0.1)
    """
    cleaned = host.strip("[]").strip()
    if not cleaned:
        return None

    # 1. Direct standard IP parsing
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        pass

    # 2. Dotted notation with alternative octal/hex bases
    parts = cleaned.split(".")
    if len(parts) == 4:
        try:
            octets = [_parse_octet(p) for p in parts]
            if all(0 <= o <= 255 for o in octets):
                return ipaddress.IPv4Address(".".join(str(o) for o in octets))
        except (ValueError, OverflowError):
            pass

    # 3. Single integer / hex integer notation (0 <= N <= 2^32 - 1)
    try:
        val = int(cleaned, 0) if (cleaned.startswith("0x") or cleaned.startswith("0X")) else (
            int(cleaned, 8) if cleaned.startswith("0") and len(cleaned) > 1 and all(c in "01234567" for c in cleaned) else int(cleaned)
        )
        if 0 <= val <= 0xFFFFFFFF:
            return ipaddress.IPv4Address(val)
    except (ValueError, OverflowError):
        pass

    return None


_NAT64_NETWORKS = (
    ipaddress.IPv6Network("64:ff9b::/96"),    # RFC 6052 well-known prefix
    ipaddress.IPv6Network("64:ff9b:1::/48"),  # RFC 8215 local-use prefix
)
_IPV4_COMPATIBLE_NETWORK = ipaddress.IPv6Network("::/96")  # deprecated ::a.b.c.d


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """Returns the IPv4 address carried by an IPv6 transition address, if any."""
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip.sixtofour:
        return ip.sixtofour
    if ip.teredo:
        return ip.teredo[1]
    if any(ip in net for net in _NAT64_NETWORKS) or (ip in _IPV4_COMPATIBLE_NETWORK and int(ip) > 1):
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def is_blocked_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    """Returns True unless the address is globally routable unicast.

    Covers loopback, private, link-local, multicast, reserved, unspecified, shared address
    space (100.64.0.0/10), documentation/benchmarking ranges, and IPv6 transition addresses
    (IPv4-mapped, 6to4, Teredo, NAT64, IPv4-compatible) that embed a blocked IPv4 address.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None and is_blocked_ip(embedded):
            return True

    return (
        not ip.is_global
        or ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_host(hostname: Optional[str], resolve_dns: bool = False) -> None:
    """Validates that hostname does not target private, loopback, or metadata addresses."""
    if not hostname or not isinstance(hostname, str) or not hostname.strip():
        raise PDFDownloadError("Invalid URL: missing hostname")

    raw_host = hostname.strip()
    norm_host = raw_host.rstrip(".").lower()
    if not norm_host:
        raise PDFDownloadError("Invalid URL: missing hostname")

    # 1. Block prohibited hostnames and suffixes (normalized trailing dots)
    if norm_host in BLOCKED_HOSTNAMES:
        raise SSRFProtectionError(f"Blocked connection to internal/metadata hostname: '{raw_host}'")

    for suffix in BLOCKED_DOMAIN_SUFFIXES:
        if norm_host.endswith(suffix):
            raise SSRFProtectionError(f"Blocked connection to internal domain suffix '{suffix}': '{raw_host}'")

    # 2. Check for IP literals (IPv4, IPv6, integer, octal, hex)
    ip_literal = parse_ip_literal(norm_host)
    if ip_literal is not None:
        if is_blocked_ip(ip_literal):
            raise SSRFProtectionError(f"Blocked connection to private or loopback IP address: {ip_literal}")
        return

    # 3. If explicit DNS pre-validation is requested (fail closed)
    if resolve_dns:
        try:
            addr_info = socket.getaddrinfo(raw_host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except (socket.gaierror, OSError) as e:
            raise SSRFProtectionError(f"DNS resolution failed for host '{raw_host}': {e}") from e

        if not addr_info:
            raise SSRFProtectionError(f"DNS resolution returned no addresses for host '{raw_host}'")

        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                if is_blocked_ip(ip_obj):
                    raise SSRFProtectionError(
                        f"Blocked connection: domain '{raw_host}' resolves to restricted IP: {ip_obj}"
                    )
            except ValueError:
                raise SSRFProtectionError(f"Unparseable resolved IP '{ip_str}' for host '{raw_host}'")


def validate_url(url: str) -> None:
    """Validates URL scheme, syntax, and destination safety."""
    if not url or not isinstance(url, str) or not url.strip():
        raise PDFDownloadError("Empty or invalid URL provided")

    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise PDFDownloadError(
            f"Disallowed URL scheme: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    if not parsed.hostname:
        raise PDFDownloadError(f"Invalid URL missing domain host: {url}")

    validate_host(parsed.hostname)


def ssrf_safe_create_connection(
    address: tuple[str, int],
    timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: Optional[tuple[str, int]] = None,
) -> socket.socket:
    """Connection factory that pins DNS resolution and strictly validates destination IPs.

    Eliminates TOCTOU / DNS rebinding attacks:
    - Resolves host once and fails closed if DNS lookup fails.
    - Validates ALL returned IP addresses against loopback, private, link-local,
      multicast, reserved, and cloud-metadata addresses.
    - Connects directly to the validated IP literal socket address without a second lookup.
    """
    host, port = address
    validate_host(host, resolve_dns=False)

    raw_host = host.strip()
    norm_host = raw_host.rstrip(".")

    ip_literal = parse_ip_literal(norm_host)
    if ip_literal is not None:
        if is_blocked_ip(ip_literal):
            raise SSRFProtectionError(f"Blocked connection to private or loopback IP address: {ip_literal}")
        res_list = socket.getaddrinfo(str(ip_literal), port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    else:
        try:
            res_list = socket.getaddrinfo(raw_host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except (socket.gaierror, OSError) as e:
            raise SSRFProtectionError(f"DNS resolution failed for host '{raw_host}': {e}") from e

    if not res_list:
        raise SSRFProtectionError(f"DNS resolution returned no addresses for host '{host}'")

    # Validate ALL returned IP addresses before attempting ANY connection
    for af, socktype, proto, canonname, sa in res_list:
        ip_str = sa[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if is_blocked_ip(ip_obj):
                raise SSRFProtectionError(
                    f"Blocked connection: domain '{host}' resolved to restricted IP: {ip_obj}"
                )
        except ValueError:
            raise SSRFProtectionError(f"Unparseable resolved IP '{ip_str}' for host '{host}'")

    # Connect to the first validated address that succeeds
    exceptions = []
    for af, socktype, proto, canonname, sa in res_list:
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            exceptions.clear()
            return sock
        except OSError as exc:
            exceptions.append(exc)
            if sock is not None:
                sock.close()

    if exceptions:
        raise exceptions[0]
    raise socket.error(f"Could not connect to any validated address for '{host}'")


import http.client


class SSRFSafeHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that enforces SSRF safety and IP pinning on socket creation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = ssrf_safe_create_connection


class SSRFSafeHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that enforces SSRF safety, IP pinning, and standard TLS verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = ssrf_safe_create_connection


class SSRFSafeHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler utilizing SSRFSafeHTTPConnection."""

    def http_open(self, req):
        return self.do_open(SSRFSafeHTTPConnection, req)


class SSRFSafeHTTPSHandler(urllib.request.HTTPSHandler):
    """urllib handler utilizing SSRFSafeHTTPSConnection with standard SSL context."""

    def https_open(self, req):
        return self.do_open(SSRFSafeHTTPSConnection, req, context=self._context)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP redirect handler that validates destinations BEFORE connecting.

    Enforces SSRF validation on every redirect jump and strips sensitive headers
    when crossing origins.
    """

    def __init__(self, validator_func=validate_url, max_redirects: int = 5):
        super().__init__()
        self.validator_func = validator_func
        self.max_redirects = max_redirects
        self.redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirect_count += 1
        if self.redirect_count > self.max_redirects:
            raise PDFDownloadError(f"Exceeded maximum redirect limit ({self.max_redirects})")

        # 1. Resolve relative URLs against previous request full_url
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)

        # 2. Validate destination BEFORE connecting
        self.validator_func(resolved_url)

        # 3. Build new request using base handler
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None

        # 4. Strip sensitive headers when crossing origins
        orig_split = urllib.parse.urlsplit(req.full_url)
        new_split = urllib.parse.urlsplit(resolved_url)
        if (orig_split.scheme, orig_split.netloc) != (new_split.scheme, new_split.netloc):
            # Replace mailto User-Agent with generic User-Agent
            new_req.headers["User-agent"] = "IdeaGen"
            # Strip authorization or cookie headers across origins
            for h in list(new_req.headers.keys()):
                if h.lower() in ("authorization", "cookie", "proxy-authorization"):
                    del new_req.headers[h]

        return new_req


class OpenAccessPDFDownloader:
    """Production downloader for open-access scientific literature PDFs."""

    def __init__(
        self,
        email: Optional[str] = None,
        max_size_bytes: int = DEFAULT_MAX_PDF_SIZE_BYTES,
        timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ):
        if email is not None:
            self.email = email.strip() or None
        else:
            settings = get_settings()
            self.email = settings.openalex_email.strip() if settings.openalex_email else None
        self.max_size_bytes = max_size_bytes
        self.timeout = timeout
        self.opener = opener

    @staticmethod
    def extract_pdf_url(work: Dict[str, Any]) -> Optional[str]:
        """Resolves best open-access PDF URL from an OpenAlex work metadata object.

        Checks best_oa_location.pdf_url first, then primary_location.pdf_url.
        """
        if not work or not isinstance(work, dict):
            return None

        # 1. Check best_oa_location
        best_oa = work.get("best_oa_location")
        if isinstance(best_oa, dict):
            pdf_url = best_oa.get("pdf_url")
            if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
                return pdf_url.strip()

        # 2. Fall back to primary_location
        primary_loc = work.get("primary_location")
        if isinstance(primary_loc, dict):
            pdf_url = primary_loc.get("pdf_url")
            if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
                return pdf_url.strip()

        # 3. Fall back to locations array if present
        locations = work.get("locations")
        if isinstance(locations, list):
            for loc in locations:
                if isinstance(loc, dict):
                    pdf_url = loc.get("pdf_url")
                    if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
                        return pdf_url.strip()

        return None

    validate_url = staticmethod(validate_url)

    @staticmethod
    def verify_pdf_signature(header_bytes: bytes) -> None:
        """Verifies that initial bytes contain %PDF- signature and rejects HTML/text."""
        if not header_bytes:
            raise InvalidPDFSignatureError("Downloaded file is empty (0 bytes)")

        # Reject HTML landing pages
        lower_header = header_bytes[:1024].lower()
        if b"<html" in lower_header or b"<!doctype html" in lower_header:
            raise InvalidPDFSignatureError("Response is an HTML landing page, not a binary PDF")

        # Must have %PDF- in the first 1024 bytes
        if PDF_SIGNATURE not in header_bytes[:1024]:
            raise InvalidPDFSignatureError("File lacks valid '%PDF-' header signature")

    def _open_request(self, req: urllib.request.Request):
        """Opens request using SafeRedirectHandler and SSRFSafe connection handlers."""
        if self.opener is not None:
            return self.opener.open(req, timeout=self.timeout)

        proxy_handler = urllib.request.ProxyHandler({})
        redirect_handler = SafeRedirectHandler(self.validate_url, max_redirects=5)
        http_handler = SSRFSafeHTTPHandler()
        https_handler = SSRFSafeHTTPSHandler()
        opener = urllib.request.build_opener(proxy_handler, redirect_handler, http_handler, https_handler)
        return opener.open(req, timeout=self.timeout)

    @contextmanager
    def download_pdf(self, url: str) -> Iterator[Path]:
        """Streams and downloads a PDF to a temporary file with size/signature validation.

        Guarantees cleanup of temporary file upon exit of context manager.
        """
        self.validate_url(url)

        user_agent = f"IdeaGen (mailto:{self.email})" if self.email else "IdeaGen"
        req = urllib.request.Request(
            url.strip(),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
            },
        )

        fd, temp_file_path_str = tempfile.mkstemp(suffix=".pdf", prefix="ideagen_oa_")
        os.close(fd)
        temp_path = Path(temp_file_path_str)

        try:
            try:
                resp = self._open_request(req)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404, 410):
                    raise PDFUnavailableError(f"PDF unavailable (HTTP {e.code}: {e.reason})") from e
                raise PDFDownloadError(f"HTTP Error {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                raise PDFDownloadError(f"Network error downloading PDF: {e.reason}") from e
            except TimeoutError as e:
                raise PDFDownloadError(f"Timeout downloading PDF after {self.timeout}s") from e

            with resp:
                # Verify final redirected URL has not switched to an unsafe destination
                final_url = resp.geturl() if hasattr(resp, "geturl") else url
                if final_url and final_url != url:
                    self.validate_url(final_url)

                # 1. Early Content-Length check if provided by remote server
                content_length = resp.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    if int(content_length) > self.max_size_bytes:
                        raise OversizedPDFError(
                            f"PDF Content-Length ({int(content_length)} bytes) exceeds limit ({self.max_size_bytes} bytes)"
                        )

                # 2. Stream download in chunks
                downloaded_bytes = 0
                first_chunk = True

                with open(temp_path, "wb") as f_out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break

                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > self.max_size_bytes:
                            raise OversizedPDFError(
                                f"Downloaded bytes exceeded limit ({self.max_size_bytes} bytes)"
                            )

                        if first_chunk:
                            # Verify PDF signature on initial bytes
                            self.verify_pdf_signature(chunk)
                            first_chunk = False

                        f_out.write(chunk)

                if downloaded_bytes == 0:
                    raise InvalidPDFSignatureError("Downloaded file is empty (0 bytes)")

            yield temp_path

        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError as err:
                    logger.warning(f"Could not delete temporary file {temp_path}: {err}")
