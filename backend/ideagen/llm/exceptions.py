class LLMError(Exception):
    """Base class for LLM related errors."""
    pass

class ProviderFailureError(LLMError):
    """Raised when the underlying LLM provider fails."""
    pass

class ProviderConnectionError(ProviderFailureError):
    """Raised when the LLM provider cannot be reached (e.g. connection refused, network failure, or timeout)."""
    pass

class MalformedResponseError(LLMError):
    """Raised when the LLM returns an unparseable or schema-violating response."""
    pass

class MissingInputError(LLMError):
    """Raised when required input for the LLM is missing."""
    pass
