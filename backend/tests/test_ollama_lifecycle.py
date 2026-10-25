"""Comprehensive Unit and Integration Tests for Ollama Lifecycle Manager (Phase 2 & 3).

Tests verify:
- Startup and readiness polling.
- Startup failure and timeout.
- External-server reuse without claiming ownership.
- Missing-model behavior without downloads.
- Model tag variant matching.
- Safe lease acquisition and release with context manager exception safety.
- Concurrent leases across threads.
- Idle shutdown scheduling, cancellation, and execution.
- Race conditions between new work and idle shutdown.
- Proof that external processes are never terminated.
- Proof that only the exact manager-owned child may be terminated.
- Disabled auto-management behavior.
- Backward compatibility of the system-status endpoint.
- FastAPI conservative shutdown handling.
- Integration with OllamaLLMProvider and OllamaEmbeddingProvider.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from ideagen.api.main import app, get_db, on_shutdown
from ideagen.core.config import Settings
from ideagen.discovery.embeddings import OllamaEmbeddingProvider
from ideagen.llm.exceptions import ProviderFailureError
from ideagen.llm.ollama_lifecycle import (
    OllamaLifecycleManager,
    get_ollama_lifecycle_manager,
    set_ollama_lifecycle_manager,
)
from ideagen.llm.ollama_provider import OllamaLLMProvider


class SampleOutput(BaseModel):
    statement: str
    valid: bool


@pytest.fixture(autouse=True)
def clean_lifecycle_manager():
    """Ensure global manager is reset before and after each test."""
    set_ollama_lifecycle_manager(None)
    yield
    set_ollama_lifecycle_manager(None)


# ── Manager Startup & Readiness Tests ─────────────────────────────────────────

def test_manager_defaults():
    mgr = OllamaLifecycleManager()
    assert mgr.base_url == "http://localhost:11434"
    assert mgr.auto_manage is True
    assert mgr.idle_timeout == 60.0
    assert mgr.startup_timeout == 30.0
    assert mgr.polling_interval == 0.5
    assert mgr.binary_path == "ollama"
    assert mgr.is_owned is False
    assert mgr.active_leases == 0
    assert mgr.process_running is False


@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_manager_startup_and_readiness_success(mock_get, mock_popen, mock_which):
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    # First call during check: False, second call during polling: True
    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Offline"),
        MagicMock(status_code=200, json=lambda: {"models": [{"name": "qwen3:8b"}]}),
    ]

    mgr = OllamaLifecycleManager(startup_timeout=2.0, polling_interval=0.05)
    mgr.ensure_running()

    assert mgr.is_owned is True
    assert mgr.process_running is True
    assert mock_popen.call_count == 1
    call_args, call_kwargs = mock_popen.call_args
    assert call_args[0] == ["/usr/local/bin/ollama", "serve"]
    assert call_kwargs["stdout"] == subprocess.DEVNULL
    assert call_kwargs["stderr"] != subprocess.PIPE
    assert call_kwargs["text"] is True
    mgr.shutdown(force=True)


@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_startup_immediate_process_termination(mock_get, mock_popen, mock_which, tmp_path):
    mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = 1
    mock_popen.return_value = mock_proc

    test_log = tmp_path / "test_ollama_startup_fail.log"
    test_log.write_text("bind: address already in use\n", encoding="utf-8")

    mgr = OllamaLifecycleManager(startup_timeout=2.0, polling_interval=0.05, log_file_path=str(test_log))
    with pytest.raises(ProviderFailureError, match="terminated immediately with exit code 1: bind: address already in use"):
        mgr.ensure_running()

    assert mgr.is_owned is False
    assert mgr.process_running is False


@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_startup_timeout_bounded_polling(mock_get, mock_popen, mock_which):
    mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mgr = OllamaLifecycleManager(startup_timeout=0.1, polling_interval=0.02)
    with pytest.raises(ProviderFailureError, match="Timed out waiting 0.1s"):
        mgr.ensure_running()

    assert mgr.is_owned is False
    mock_proc.terminate.assert_called_once()


# ── External Server Reuse & Ownership Guard Tests ────────────────────────────

@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_external_server_reuse(mock_get, mock_popen):
    """If Ollama is already reachable, manager must reuse it and never claim ownership."""
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager()
    mgr.ensure_running()

    assert mgr.is_owned is False
    assert mgr.process_running is False
    mock_popen.assert_not_called()


@patch("requests.get")
def test_disabled_auto_manage_mode(mock_get):
    """When auto_manage is False and Ollama is unreachable, fail without spawning."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Offline")

    mgr = OllamaLifecycleManager(auto_manage=False)
    with pytest.raises(ProviderFailureError, match="auto_manage is disabled"):
        mgr.ensure_running()


# ── Model Verification Tests ─────────────────────────────────────────────────

@patch("requests.get")
def test_missing_model_behavior_without_downloads(mock_get):
    """When required models are missing, raise ProviderFailureError and do not download."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3:latest"}, {"name": "nomic-embed-text:latest"}]},
    )

    mgr = OllamaLifecycleManager()
    with pytest.raises(ProviderFailureError) as exc_info:
        mgr.verify_models(["qwen3:8b", "nomic-embed-text"])

    err_msg = str(exc_info.value)
    assert "Required Ollama model(s) missing: 'qwen3:8b'" in err_msg
    assert "Please run `ollama pull <model>`" in err_msg


@patch("requests.get")
def test_model_tag_matching_variants(mock_get):
    """Symmetric resolution between explicit tag and default ':latest'."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "models": [
                {"name": "qwen3:8b"},
                {"name": "nomic-embed-text:latest"},
            ]
        },
    )

    mgr = OllamaLifecycleManager()
    # 1. Exact match
    installed = mgr.verify_models(["qwen3:8b"])
    assert "qwen3:8b" in installed

    # 2. Required has no tag, installed has :latest
    installed = mgr.verify_models(["nomic-embed-text"])
    assert "nomic-embed-text:latest" in installed


# ── Lease Tracking & Concurrency Tests ───────────────────────────────────────

@patch("requests.get")
def test_lease_reference_counting(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager()
    assert mgr.active_leases == 0

    mgr.acquire()
    assert mgr.active_leases == 1

    mgr.acquire()
    assert mgr.active_leases == 2

    mgr.release()
    assert mgr.active_leases == 1

    mgr.release()
    assert mgr.active_leases == 0


@patch("requests.get")
def test_lease_context_manager_exception_safety(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager()
    try:
        with mgr.lease():
            assert mgr.active_leases == 1
            raise RuntimeError("Failure inside inference block")
    except RuntimeError:
        pass

    assert mgr.active_leases == 0


@patch("requests.get")
def test_concurrent_lease_requests(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager()
    num_threads = 20
    iterations = 10

    def worker():
        for _ in range(iterations):
            with mgr.lease():
                time.sleep(0.001)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mgr.active_leases == 0


# ── Idle Shutdown & Race Condition Tests ─────────────────────────────────────

@patch("requests.get")
def test_idle_shutdown_scheduled_and_executed(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager(idle_timeout=0.05)
    mock_proc = MagicMock()
    mock_proc.pid = 4321
    mock_proc.poll.return_value = None

    # Simulate owned process
    mgr._process = mock_proc
    mgr._is_owned = True

    mgr.acquire()
    assert mgr.active_leases == 1

    mgr.release()
    assert mgr.active_leases == 0
    assert mgr.get_status()["idle_timer_pending"] is True

    # Wait for idle timeout to fire
    time.sleep(0.12)

    assert mgr.is_owned is False
    assert mgr._process is None
    mock_proc.terminate.assert_called_once()


@patch("requests.get")
def test_idle_shutdown_cancelled_by_new_work(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager(idle_timeout=0.1)
    mock_proc = MagicMock()
    mock_proc.pid = 5555
    mock_proc.poll.return_value = None

    mgr._process = mock_proc
    mgr._is_owned = True

    mgr.acquire()
    mgr.release()
    assert mgr.get_status()["idle_timer_pending"] is True

    # New work arrives before idle timeout
    time.sleep(0.02)
    mgr.acquire()
    assert mgr.get_status()["idle_timer_pending"] is False

    # Sleep past the original deadline
    time.sleep(0.12)
    mock_proc.terminate.assert_not_called()
    assert mgr.is_owned is True
    assert mgr.active_leases == 1

    mgr.release()


@patch("requests.get")
def test_race_condition_between_new_work_and_idle_timeout(mock_get):
    """Even if _on_idle_timeout begins, if a new lease was acquired, it must not terminate."""
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

    mgr = OllamaLifecycleManager(idle_timeout=0.01)
    mock_proc = MagicMock()
    mock_proc.pid = 6666
    mock_proc.poll.return_value = None

    mgr._process = mock_proc
    mgr._is_owned = True

    with mgr._lock:
        # Simulate timer thread entering right when active_leases is 1
        mgr._active_leases = 1
        mgr._on_idle_timeout()

    # Process was NOT stopped because active_leases > 0
    mock_proc.terminate.assert_not_called()
    assert mgr.is_owned is True


# ── Process Safety & Non-Termination of External Servers ─────────────────────

def test_external_process_never_terminated():
    """An unowned external server must never be terminated by stop, timeout, or shutdown."""
    mgr = OllamaLifecycleManager()
    mgr._is_owned = False
    mgr._process = None

    # None of these should raise or touch any subprocess
    mgr._stop_owned_process()
    mgr._on_idle_timeout()
    mgr.shutdown(force=True)

    assert mgr.is_owned is False


def test_only_exact_manager_owned_child_terminated():
    """Only the exact process handle held in _process is terminated."""
    mgr = OllamaLifecycleManager()
    owned_proc = MagicMock()
    owned_proc.pid = 1111
    owned_proc.poll.return_value = None
    foreign_proc = MagicMock()
    foreign_proc.pid = 2222
    foreign_proc.poll.return_value = None

    mgr._process = owned_proc
    mgr._is_owned = True

    mgr._stop_owned_process()

    owned_proc.terminate.assert_called_once()
    foreign_proc.terminate.assert_not_called()


# ── Application Shutdown Conservatism ────────────────────────────────────────

def test_shutdown_conservative_with_active_leases():
    mgr = OllamaLifecycleManager()
    mock_proc = MagicMock()
    mock_proc.pid = 3333
    mock_proc.poll.return_value = None
    mgr._process = mock_proc
    mgr._is_owned = True
    mgr._active_leases = 2

    # Non-forced shutdown must skip stopping process when leases remain
    mgr.shutdown(force=False)
    mock_proc.terminate.assert_not_called()
    assert mgr.is_owned is True

    # Forced shutdown terminates
    mgr.shutdown(force=True)
    mock_proc.terminate.assert_called_once()
    assert mgr.is_owned is False


def test_fastapi_shutdown_event_conservative():
    mock_mgr = MagicMock(spec=OllamaLifecycleManager)
    set_ollama_lifecycle_manager(mock_mgr)

    on_shutdown()
    mock_mgr.shutdown.assert_called_once_with(force=False)


# ── System Status Endpoint Compatibility ─────────────────────────────────────

def test_system_status_endpoint_backward_compatibility():
    def override_get_db():
        mock_db = MagicMock()
        mock_db.execute.return_value = None
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_db.get_bind.return_value = mock_bind
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)

        with patch("ideagen.api.main._check_ollama") as mock_check:
            mock_check.return_value = {
                "reachable": True,
                "installed_models": ["qwen3:8b", "nomic-embed-text"],
                "missing_models": [],
                "error": None,
            }

            resp = client.get("/api/v1/system/status")
            assert resp.status_code == 200
            data = resp.json()

            # Backward compatible root structure
            assert "database" in data
            assert "llm" in data
            assert "embedding" in data
            assert "ollama" in data
            assert "openalex" in data

            ollama = data["ollama"]
            # Required historical fields
            assert ollama["reachable"] is True
            assert "qwen3:8b" in ollama["installed_models"]
            assert ollama["missing_models"] == []
            assert ollama["error"] is None

            # Lifecycle fields present
            assert "auto_manage" in ollama
            assert "is_owned" in ollama
            assert "active_leases" in ollama
            assert "idle_timeout" in ollama
            assert "process_running" in ollama
    finally:
        app.dependency_overrides.pop(get_db, None)


# ── Provider Integration Tests ───────────────────────────────────────────────

@patch("requests.post")
def test_ollama_llm_provider_with_lifecycle_manager(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"message": {"content": json.dumps({"statement": "test", "valid": True})}},
    )

    mock_mgr = MagicMock(spec=OllamaLifecycleManager)
    # mock lease context manager
    mock_mgr.lease.return_value.__enter__ = MagicMock(return_value=mock_mgr)
    mock_mgr.lease.return_value.__exit__ = MagicMock(return_value=None)

    provider = OllamaLLMProvider(model_name="qwen3:8b", lifecycle_manager=mock_mgr)
    res = provider.generate_structured("test prompt", SampleOutput)

    assert isinstance(res, SampleOutput)
    assert res.statement == "test"
    mock_mgr.lease.assert_called_once_with(required_models=["qwen3:8b"])


@patch("requests.post")
def test_ollama_embedding_provider_with_lifecycle_manager(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"embeddings": [[0.5] * 768]},
    )

    mock_mgr = MagicMock(spec=OllamaLifecycleManager)
    mock_mgr.lease.return_value.__enter__ = MagicMock(return_value=mock_mgr)
    mock_mgr.lease.return_value.__exit__ = MagicMock(return_value=None)

    provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text",
        dimension=768,
        lifecycle_manager=mock_mgr,
    )
    embeddings = provider.embed_texts(["sample statement"])

    assert embeddings.shape == (1, 768)
    mock_mgr.lease.assert_called_once_with(required_models=["nomic-embed-text"])


# ── Regression Tests: Pipe Deadlock Prevention & Handle Safety ────────────────

@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_subprocess_does_not_use_pipe_stderr_avoids_deadlock(mock_get, mock_popen, mock_which, tmp_path):
    """Regression test: verifies subprocess does NOT use unread subprocess.PIPE for stderr.
    
    Using subprocess.PIPE without active reading causes 16KB pipe buffer saturation
    and hangs the Ollama daemon permanently on macOS/Linux.
    """
    mock_proc = MagicMock()
    mock_proc.pid = 8888
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Offline"),
        MagicMock(status_code=200, json=lambda: {"models": [{"name": "qwen3:8b"}]}),
    ]

    log_path = tmp_path / "custom_ollama.log"
    mgr = OllamaLifecycleManager(
        startup_timeout=2.0,
        polling_interval=0.05,
        log_file_path=str(log_path),
    )
    mgr.ensure_running()

    assert mock_popen.call_count == 1
    _, kwargs = mock_popen.call_args
    # Explicitly ensure subprocess.PIPE is NEVER passed to stderr
    assert kwargs["stderr"] != subprocess.PIPE
    # Ensure it is a valid file object (or devnull)
    assert hasattr(kwargs["stderr"], "write") or kwargs["stderr"] == subprocess.DEVNULL
    mgr.shutdown(force=True)


@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_subprocess_log_file_handle_closed_on_shutdown(mock_get, mock_popen, mock_which, tmp_path):
    """Regression test: verifies daemon log file handle is closed on process shutdown."""
    mock_proc = MagicMock()
    mock_proc.pid = 8888
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Offline"),
        MagicMock(status_code=200, json=lambda: {"models": [{"name": "qwen3:8b"}]}),
    ]

    log_path = tmp_path / "lifecycle_cleanup.log"
    mgr = OllamaLifecycleManager(
        startup_timeout=2.0,
        polling_interval=0.05,
        log_file_path=str(log_path),
    )
    mgr.ensure_running()

    handle = mgr._log_file_handle
    assert handle is not None
    assert not handle.closed

    mgr.shutdown(force=True)
    assert mgr._log_file_handle is None
    assert handle.closed


@patch("ideagen.llm.ollama_lifecycle.shutil.which", return_value="/usr/local/bin/ollama")
@patch("ideagen.llm.ollama_lifecycle.subprocess.Popen")
@patch("requests.get")
def test_subprocess_devnull_fallback_when_configured(mock_get, mock_popen, mock_which):
    """Regression test: verifies subprocess.DEVNULL is used when DEVNULL is configured."""
    mock_proc = MagicMock()
    mock_proc.pid = 8888
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Offline"),
        MagicMock(status_code=200, json=lambda: {"models": [{"name": "qwen3:8b"}]}),
    ]

    mgr = OllamaLifecycleManager(
        startup_timeout=2.0,
        polling_interval=0.05,
        log_file_path="DEVNULL",
    )
    mgr.ensure_running()

    assert mock_popen.call_count == 1
    _, kwargs = mock_popen.call_args
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert mgr._log_file_handle is None
    mgr.shutdown(force=True)
