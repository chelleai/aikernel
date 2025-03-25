from enum import StrEnum


class AIErrorType(StrEnum):
    MODEL_UNAVAILABLE = "model_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    NO_RESPONSE = "no_response"
    TOOL_CALL_ERROR = "tool_call_error"
    INVALID_CONVERSATION_DUMP = "invalid_conversation_dump"


class AIError(Exception):
    def __init__(self, *, error_type: AIErrorType) -> None:
        self.error_type = error_type
        super().__init__(f"Error calling AI model: {error_type}")


class ModelUnavailableError(AIError):
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.MODEL_UNAVAILABLE)


class RateLimitExceededError(AIError):
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.RATE_LIMIT_EXCEEDED)


class NoResponseError(AIError):
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.NO_RESPONSE)


class ToolCallError(AIError):
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.TOOL_CALL_ERROR)


class InvalidConversationDumpError(AIError):
    def __init__(self) -> None:
        super().__init__(error_type=AIErrorType.INVALID_CONVERSATION_DUMP)
