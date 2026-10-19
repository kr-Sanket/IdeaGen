from __future__ import annotations

import logging
import re
from typing import Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaLLMProvider(LLMProvider):
    """Local LLM Provider backed by Ollama with structured JSON schema output."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        timeout: float = 300.0,
        num_ctx: int = 8192,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.num_ctx = num_ctx

    def generate_structured(self, prompt: str, response_schema: Type[T]) -> T:
        """Generates a structured response validated against the provided Pydantic schema.

        Args:
            prompt: The instruction and context prompt.
            response_schema: A Pydantic model class defining the expected output.

        Returns:
            An instance of response_schema.

        Raises:
            ProviderFailureError: If Ollama is unreachable, returns HTTP error, or times out.
            MalformedResponseError: If Ollama output cannot be parsed or fails schema validation.
        """
        chat_url = f"{self.base_url}/api/chat"

        try:
            schema_dict = response_schema.model_json_schema()
        except Exception:
            schema_dict = "json"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema_dict,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            response = requests.post(
                chat_url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise ProviderFailureError(f"Ollama request timed out after {self.timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            raise ProviderFailureError(f"Ollama connection error: {e}") from e

        if response.status_code != 200:
            raise ProviderFailureError(
                f"Ollama API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise ProviderFailureError(f"Failed to parse Ollama HTTP response as JSON: {e}") from e

        raw_content = data.get("message", {}).get("content", "").strip()
        if not raw_content:
            raise ProviderFailureError("Ollama returned an empty response.")

        # Clean potential markdown fences (e.g., ```json ... ```)
        cleaned_content = raw_content
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```(?:json)?\s*", "", cleaned_content, flags=re.IGNORECASE)
            cleaned_content = re.sub(r"\s*```$", "", cleaned_content).strip()

        try:
            return response_schema.model_validate_json(cleaned_content)
        except ValidationError as e:
            raise MalformedResponseError(
                f"Response failed schema validation: {e}\nRaw output: {raw_content}"
            ) from e
        except Exception as e:
            raise MalformedResponseError(
                f"Failed to parse JSON response: {e}\nRaw output: {raw_content}"
            ) from e
