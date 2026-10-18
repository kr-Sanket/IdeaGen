from typing import Protocol, TypeVar, Type

T = TypeVar('T')

class LLMProvider(Protocol):
    def generate_structured(self, prompt: str, response_schema: Type[T]) -> T:
        """
        Generates a structured response based on the provided prompt and schema.
        
        Args:
            prompt: The instruction and context.
            response_schema: A Pydantic model class defining the expected output.
            
        Returns:
            An instance of the response_schema.
            
        Raises:
            ProviderFailureError: If the provider API fails.
            MalformedResponseError: If the provider returns data not matching the schema.
        """
        ...
