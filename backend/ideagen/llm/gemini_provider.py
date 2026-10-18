from typing import Type, TypeVar
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError

T = TypeVar('T', bound=BaseModel)

class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_structured(self, prompt: str, response_schema: Type[T]) -> T:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0
                )
            )
        except Exception as e:
            raise ProviderFailureError(f"Gemini API failure: {str(e)}") from e
            
        if not response.text:
            raise ProviderFailureError("Gemini returned an empty or blocked response.")

        try:
            data = json.loads(response.text)
            return response_schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"Failed to parse JSON response: {str(e)}\nRaw output: {response.text}") from e
        except ValidationError as e:
            raise MalformedResponseError(f"Response failed schema validation: {str(e)}\nRaw output: {response.text}") from e
