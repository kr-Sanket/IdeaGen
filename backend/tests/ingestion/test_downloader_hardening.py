"""F8 regression tests: SSRF address coverage, no production test bypass, test seam
fails closed. All network access is mocked; nothing leaves the process."""
import ipaddress
import socket
import urllib.request

import pytest

from ideagen.ingestion.downloader import (
    OpenAccessPDFDownloader,
    PDFDownloadError,
    SSRFProtectionError,
    is_blocked_ip,
    ssrf_safe_create_connection,
    validate_host,
)
from ideagen.ingestion import openalex_ingestion
from tests.conftest import PatchedUrlopenOpener


@pytest.mark.parametrize(
    "address",
    [
        "100.64.0.1",                              # shared address space (RFC 6598)
        "100.100.100.200",                         # cloud metadata inside 100.64.0.0/10
        "198.18.0.1",                              # benchmarking
        "192.0.2.10",                              # documentation
        "64:ff9b::a9fe:a9fe",                      # NAT64 -> 169.254.169.254
        "64:ff9b::7f00:1",                         # NAT64 -> 127.0.0.1
        "64:ff9b:1::a00:1",                        # local-use NAT64 -> 10.0.0.1
        "2002:c0a8:101::",                         # 6to4 -> 192.168.1.1
        "2002:a9fe:a9fe::",                        # 6to4 -> 169.254.169.254
        "2001:0:4136:e378:8000:63bf:3f57:fdf2",    # Teredo
        "::7f00:1",                                # IPv4-compatible -> 127.0.0.1
        "::ffff:10.0.0.1",                         # IPv4-mapped private
    ],
)
def test_transition_and_shared_ranges_are_blocked(address):
    assert is_blocked_ip(ipaddress.ip_address(address)) is True


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111", "::ffff:8.8.8.8"])
def test_global_unicast_addresses_are_allowed(address):
    assert is_blocked_ip(ipaddress.ip_address(address)) is False


@pytest.mark.parametrize("host", ["100.100.100.200", "[64:ff9b::a9fe:a9fe]", "2002:a9fe:a9fe::"])
def test_validate_host_rejects_new_ranges_as_literals(host):
    with pytest.raises(SSRFProtectionError):
        validate_host(host.strip("[]"))


@pytest.mark.parametrize(
    "family,sockaddr",
    [
        (socket.AF_INET, ("100.100.100.200", 80)),
        (socket.AF_INET6, ("64:ff9b::a9fe:a9fe", 80, 0, 0)),
    ],
)
def test_connection_time_dns_to_new_ranges_is_blocked(monkeypatch, family, sockaddr):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [(family, socket.SOCK_STREAM, 6, "", sockaddr)],
    )
    with pytest.raises(SSRFProtectionError, match="restricted IP"):
        ssrf_safe_create_connection(("rebind.attacker.org", 80))


def test_patched_urlopen_does_not_bypass_ssrf_safe_opener(monkeypatch):
    """Production downloads must ignore a monkeypatched urllib.request.urlopen."""
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))],
    )
    downloader = OpenAccessPDFDownloader()
    with pytest.raises(SSRFProtectionError, match="restricted IP: 169.254.169.254"):
        downloader._open_request(urllib.request.Request("http://rebind.attacker.org/paper.pdf"))
    assert calls == []


def test_test_seam_fails_closed_without_patched_urlopen():
    assert getattr(urllib.request.urlopen, "__name__", "") == "urlopen"
    with pytest.raises(PDFDownloadError, match="Network access is disabled in tests"):
        PatchedUrlopenOpener().open(urllib.request.Request("https://example.org/a.pdf"), timeout=1)


def test_ingestion_service_default_downloader_uses_test_seam():
    service = openalex_ingestion.OpenAlexPaperIngestionService(persistence=None)
    assert isinstance(service.downloader.opener, PatchedUrlopenOpener)
