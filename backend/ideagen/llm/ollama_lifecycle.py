"""Ollama process lifecycle and lease manager.

Provides automatic startup, model readiness verification, lease-based reference counting,
idle shutdown, and process ownership tracking for local Ollama instances.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from ideagen.core.config import get_settings
from ideagen.core.network import proxy_hint
from ideagen.llm.exceptions import ProviderFailureError

logger = logging.getLogger(__name__)


class OllamaLifecycleManager:
    """Manages the lifecycle of an Ollama server instance.

    Features:
    - Verifies server connectivity and installed models without auto-downloading.
    - If auto_manage is True and Ollama is not running, starts an owned subprocess.
    - Tracks active leases (reference counting) across threads so Ollama is kept alive while in use.
    - Shuts down the owned process only after an idle timeout once active leases reach zero.
    - Cancels pending shutdown when new work arrives.
    - Never terminates an external, unmanaged Ollama process that was already running.
    - Only terminates the exact process handle spawned by this manager.
    - Supports conservative shutdown when active leases remain.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        auto_manage: bool = True,
        idle_timeout: float = 60.0,
        startup_timeout: float = 30.0,
        polling_interval: float = 0.5,
        binary_path: str = "ollama",
        log_file_path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auto_manage = auto_manage
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.polling_interval = max(0.05, polling_interval)
        self.binary_path = binary_path
        self.log_file_path = log_file_path

        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._is_owned: bool = False
        self._active_leases: int = 0
        self._idle_timer: Optional[threading.Timer] = None
        self._log_file_handle: Optional[Any] = None
        self._effective_log_path: Optional[str] = None

    @property
    def is_owned(self) -> bool:
        """Whether the currently running Ollama server was started by this manager."""
        with self._lock:
            return self._is_owned

    @property
    def active_leases(self) -> int:
        """The number of active leases currently holding Ollama open."""
        with self._lock:
            return self._active_leases

    @property
    def process_running(self) -> bool:
        """Whether a subprocess handle owned by this manager is currently alive."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def is_server_reachable(self, timeout: float = 2.0) -> bool:
        """Checks if the Ollama server is responding to HTTP requests."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_installed_models(self, timeout: float = 3.0) -> List[str]:
        """Returns the list of installed model names from the Ollama server."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=timeout)
            if resp.status_code != 200:
                body = " ".join((resp.text or "").split())[:200]
                raise ProviderFailureError(
                    f"Ollama at {self.base_url} returned HTTP {resp.status_code}: {body}.{proxy_hint(tags_url)}"
                )
            data = resp.json()
            return sorted({m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)})
        except requests.exceptions.RequestException as e:
            raise ProviderFailureError(
                f"Cannot reach Ollama at {self.base_url}: {e.__class__.__name__}.{proxy_hint(tags_url)}"
            ) from e

    def verify_models(self, required_models: Sequence[str]) -> List[str]:
        """Verifies that all required models are installed without pulling.
        
        Raises ProviderFailureError if any are missing.
        """
        if not required_models:
            return []

        installed = self.get_installed_models()

        def _present(m: str) -> bool:
            if m in installed:
                return True
            if ":" not in m and f"{m}:latest" in installed:
                return True
            if m.endswith(":latest") and m[:-7] in installed:
                return True
            return False

        missing = [m for m in required_models if not _present(m)]
        if missing:
            models_str = ", ".join(f"'{m}'" for m in missing)
            raise ProviderFailureError(
                f"Required Ollama model(s) missing: {models_str}. "
                f"Installed models: {installed}. "
                f"Please run `ollama pull <model>` to install the missing model(s)."
            )
        return installed

    def _start_server_process(self) -> None:
        """Starts the Ollama daemon subprocess and waits until it is ready."""
        bin_path = shutil.which(self.binary_path) or self.binary_path
        if not os.path.exists(bin_path) and not shutil.which(self.binary_path):
            raise ProviderFailureError(
                f"Cannot auto-start Ollama: binary '{self.binary_path}' not found in PATH. "
                "Please install Ollama from https://ollama.ai or ensure it is running."
            )

        logger.info(f"Starting managed Ollama daemon via '{bin_path} serve'...")

        effective_log_path = self.log_file_path or os.environ.get(
            "IDEAGEN_OLLAMA_LOG_PATH",
            os.path.join(tempfile.gettempdir(), f"ideagen_ollama_{os.getpid()}.log"),
        )
        self._effective_log_path = effective_log_path

        try:
            if effective_log_path == os.devnull or (isinstance(effective_log_path, str) and effective_log_path.upper() == "DEVNULL"):
                stderr_dest = subprocess.DEVNULL
            else:
                self._log_file_handle = open(effective_log_path, "a", encoding="utf-8")
                stderr_dest = self._log_file_handle

            self._process = subprocess.Popen(
                [bin_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=stderr_dest,
                text=True,
            )
            self._is_owned = True
        except Exception as e:
            self._close_log_file()
            self._is_owned = False
            self._process = None
            raise ProviderFailureError(f"Failed to spawn Ollama process: {e}") from e

        # Wait for server to become responsive
        deadline = time.time() + self.startup_timeout
        started = False
        while time.time() < deadline:
            ret = self._process.poll()
            if ret is not None:
                stderr_text = ""
                if self._log_file_handle:
                    try:
                        self._log_file_handle.flush()
                    except Exception:
                        pass
                if self._effective_log_path and os.path.exists(self._effective_log_path):
                    try:
                        with open(self._effective_log_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                            stderr_text = "".join(lines[-20:])[:500]
                    except Exception:
                        pass
                self._close_log_file()
                self._is_owned = False
                self._process = None
                detail = f": {stderr_text.strip()}" if stderr_text.strip() else ""
                raise ProviderFailureError(
                    f"Ollama process terminated immediately with exit code {ret}{detail}"
                )

            if self.is_server_reachable(timeout=min(1.0, self.polling_interval)):
                started = True
                break
            time.sleep(self.polling_interval)

        if not started:
            self._stop_owned_process()
            raise ProviderFailureError(
                f"Timed out waiting {self.startup_timeout}s for Ollama server to become ready at {self.base_url}."
            )

        logger.info("Managed Ollama daemon started and verified successfully.")

    def ensure_running(self, required_models: Optional[Sequence[str]] = None) -> None:
        """Ensures that the Ollama server is running and optional required models exist."""
        with self._lock:
            # If an idle shutdown was queued, cancel it immediately
            self._cancel_idle_timer()

            if self.is_server_reachable(timeout=1.5):
                # Server is already alive. If we did not start a live owned process, mark unowned
                if self._process is None or self._process.poll() is not None:
                    self._is_owned = False
                    self._process = None
            else:
                # Server is not reachable
                if not self.auto_manage:
                    raise ProviderFailureError(
                        f"Ollama server is not reachable at {self.base_url} and auto_manage is disabled. "
                        "Please start Ollama manually."
                    )
                self._start_server_process()

            if required_models:
                self.verify_models(required_models)

    def acquire(self, required_models: Optional[Sequence[str]] = None) -> None:
        """Acquires a lease, starting the server if necessary and incrementing reference count."""
        with self._lock:
            self._cancel_idle_timer()
            self.ensure_running(required_models=required_models)
            self._active_leases += 1
            logger.debug(f"Acquired Ollama lease (active: {self._active_leases})")

    def release(self) -> None:
        """Releases a lease, decrementing reference count and starting idle timer if zero."""
        with self._lock:
            if self._active_leases > 0:
                self._active_leases -= 1
            logger.debug(f"Released Ollama lease (active: {self._active_leases})")

            # Only schedule shutdown if leases hit 0, we OWN the process, and process is alive
            if self._active_leases == 0 and self._is_owned and self._process is not None:
                if self._process.poll() is None:
                    if self.idle_timeout <= 0:
                        logger.info("Active leases reached 0; idle timeout is <= 0. Stopping Ollama immediately.")
                        self._stop_owned_process()
                    else:
                        logger.debug(f"Scheduling Ollama idle shutdown in {self.idle_timeout}s")
                        self._cancel_idle_timer()
                        self._idle_timer = threading.Timer(self.idle_timeout, self._on_idle_timeout)
                        self._idle_timer.daemon = True
                        self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        """Cancels any pending idle shutdown timer."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self) -> None:
        """Callback executed when the idle timer expires."""
        with self._lock:
            self._idle_timer = None
            # Recheck: leases must still be 0, must be owned, and process must still be running
            if self._active_leases == 0 and self._is_owned and self._process is not None:
                if self._process.poll() is None:
                    logger.info(f"Ollama idle timeout ({self.idle_timeout}s) expired with 0 leases. Shutting down daemon.")
                    self._stop_owned_process()

    def _close_log_file(self) -> None:
        """Closes the daemon log file handle if open."""
        with self._lock:
            if self._log_file_handle is not None:
                try:
                    self._log_file_handle.close()
                except Exception:
                    pass
                self._log_file_handle = None

    def _stop_owned_process(self) -> None:
        """Gracefully terminates and reaps only the exact manager-owned child process."""
        with self._lock:
            proc = self._process
            is_owned = self._is_owned
            self._process = None
            self._is_owned = False
            self._cancel_idle_timer()

            if proc is None or not is_owned:
                self._close_log_file()
                return

            try:
                if proc.poll() is None:
                    pid_str = getattr(proc, "pid", "unknown")
                    logger.info(f"Terminating owned Ollama process (pid={pid_str})...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Ollama process (pid={pid_str}) did not terminate within 5s; killing.")
                        proc.kill()
                        proc.wait(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping owned Ollama process: {e}")
            finally:
                self._close_log_file()

    def shutdown(self, force: bool = False) -> None:
        """Shuts down the owned Ollama daemon on application exit or manual request.
        
        If force is False and active leases remain, shutdown is skipped to avoid interrupting work.
        """
        with self._lock:
            self._cancel_idle_timer()
            if not force and self._active_leases > 0:
                logger.warning(
                    f"Conservative shutdown skipped: {self._active_leases} active lease(s) remain."
                )
                return

            if force:
                self._active_leases = 0

            # Recheck ownership immediately before stopping
            if self._is_owned and self._process is not None:
                self._stop_owned_process()

    @contextlib.contextmanager
    def lease(self, required_models: Optional[Sequence[str]] = None):
        """Context manager providing an active lease on the Ollama daemon."""
        self.acquire(required_models=required_models)
        try:
            yield self
        finally:
            self.release()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state of the Ollama lifecycle manager."""
        with self._lock:
            reachable = self.is_server_reachable(timeout=1.0)
            return {
                "base_url": self.base_url,
                "auto_manage": self.auto_manage,
                "idle_timeout": self.idle_timeout,
                "startup_timeout": self.startup_timeout,
                "polling_interval": self.polling_interval,
                "reachable": reachable,
                "is_owned": self._is_owned,
                "active_leases": self._active_leases,
                "process_running": self._process is not None and self._process.poll() is None,
                "idle_timer_pending": self._idle_timer is not None,
            }


# ── Global Singleton & Lifecycle Registration ─────────────────────────

_manager_instance: Optional[OllamaLifecycleManager] = None
_manager_lock = threading.Lock()


def get_ollama_lifecycle_manager() -> OllamaLifecycleManager:
    """Returns the global OllamaLifecycleManager instance configured from application settings."""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                settings = get_settings()
                _manager_instance = OllamaLifecycleManager(
                    base_url=settings.ollama_base_url,
                    auto_manage=settings.ollama_auto_manage,
                    idle_timeout=settings.ollama_idle_timeout,
                    startup_timeout=settings.ollama_startup_timeout,
                    polling_interval=settings.ollama_polling_interval,
                    binary_path=settings.ollama_binary_path,
                )
    return _manager_instance


def set_ollama_lifecycle_manager(manager: Optional[OllamaLifecycleManager]) -> None:
    """Sets or resets the global OllamaLifecycleManager instance (useful for testing)."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is not None and _manager_instance is not manager:
            try:
                _manager_instance.shutdown(force=True)
            except Exception:
                pass
        _manager_instance = manager


@atexit.register
def _cleanup_ollama_on_exit() -> None:
    """Ensures owned Ollama processes are cleaned up when the Python process exits."""
    global _manager_instance
    if _manager_instance is not None:
        try:
            _manager_instance.shutdown(force=True)
        except Exception:
            pass
