from typing import Type, TypeVar, Any
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError

T = TypeVar('T')

class FakeLLMProvider(LLMProvider):
    def __init__(self, response: Any = None, should_fail: bool = False, should_malform: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.should_malform = should_malform
        self.last_prompt = None
        self.last_schema = None

    def generate_structured(self, prompt: str, response_schema: Type[T]) -> T:
        self.last_prompt = prompt
        self.last_schema = response_schema

        if self.should_fail:
            raise ProviderFailureError("Fake provider simulated failure.")
            
        if self.should_malform:
            raise MalformedResponseError("Fake provider simulated malformed response.")
            
        if self.response is not None:
            # If the response is a dict, parse it using the schema
            if isinstance(self.response, dict):
                try:
                    return response_schema(**self.response)
                except Exception as e:
                    raise MalformedResponseError(f"Failed to parse response: {e}")
            return self.response
            
        raise ValueError("FakeLLMProvider requires a response to be set unless should_fail or should_malform is True.")
