"""LLM Integration Module."""

from __future__ import annotations

from ideagen.core.config import get_settings
from .provider import LLMProvider
from .exceptions import LLMError, ProviderFailureError, MalformedResponseError, MissingInputError
from .ollama_provider import OllamaLLMProvider
from .gemini_provider import GeminiLLMProvider
from .ollama_lifecycle import (
    OllamaLifecycleManager,
    get_ollama_lifecycle_manager,
    set_ollama_lifecycle_manager,
)


def get_default_llm_provider() -> LLMProvider:
    """Return the configured default LLM provider."""
    settings = get_settings()
    provider_name = (settings.llm_provider or "ollama").lower()
    if provider_name == "ollama":
        lifecycle_mgr = get_ollama_lifecycle_manager() if settings.ollama_auto_manage else None
        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model,
            timeout=settings.ollama_timeout,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
            lifecycle_manager=lifecycle_mgr,
        )
    elif provider_name == "gemini":
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key or "",
            model_name=settings.gemini_model_name,
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name}")


__all__ = [
    "LLMProvider",
    "LLMError",
    "ProviderFailureError",
    "MalformedResponseError",
    "MissingInputError",
    "OllamaLLMProvider",
    "GeminiLLMProvider",
    "OllamaLifecycleManager",
    "get_ollama_lifecycle_manager",
    "set_ollama_lifecycle_manager",
    "get_default_llm_provider",
]
