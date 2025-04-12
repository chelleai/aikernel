"""Error types for the aikernel package.

This module defines the error types that can be raised by the aikernel package,
providing a structured way to handle errors that may occur when interacting with LLMs.
"""

from enum import StrEnum


class AIErrorType(StrEnum):
    """Enumeration of possible AI error types."""
    MODEL_UNAVAILABLE = "model_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    NO_RESPONSE = "no_response"
    TOOL_CALL_ERROR = "tool_call_error"
    INVALID_CONVERSATION_DUMP = "invalid_conversation_dump"


class AIError(Exception):
    """Base exception class for all AI-related errors.
    
    This exception serves as the parent class for all errors that may occur
    when interacting with AI models.
    """
    def __init__(self, *, error_type: AIErrorType) -> None:
        self.error_type = error_type
        super().__init__(f"Error calling AI model: {error_type}")


class ModelUnavailableError(AIError):
    """Error raised when the requested AI model is unavailable."""
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.MODEL_UNAVAILABLE)


class RateLimitExceededError(AIError):
    """Error raised when API rate limits have been exceeded."""
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.RATE_LIMIT_EXCEEDED)


class NoResponseError(AIError):
    """Error raised when the AI model does not provide a response."""
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.NO_RESPONSE)


class ToolCallError(AIError):
    """Error raised when there is an issue with a tool call."""
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.TOOL_CALL_ERROR)


class InvalidConversationDumpError(AIError):
    """Error raised when a conversation dump cannot be parsed."""
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.INVALID_CONVERSATION_DUMP)
