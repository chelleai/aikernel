from enum import StrEnum

from pydantic_core import ErrorDetails


class AIErrorType(StrEnum):
    MODEL_UNAVAILABLE = "model_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    NO_RESPONSE = "no_response"
    TOOL_CALL_ERROR = "tool_call_error"
    INVALID_MODEL_NAME = "invalid_model_name"
    BAD_REQUEST_ERROR = "bad_request_error"
    SCHEMA_NOT_FOLLOWED = "schema_not_followed"


class AIError(Exception):
    def __init__(self, *, error_type: AIErrorType, message: str) -> None:
        self.error_type = error_type
        self.message = message
        super().__init__(f"Error calling AI model ({error_type}): {message}")


class ModelUnavailableError(AIError):
    def __init__(self, *, model_name: str) -> None:
        super().__init__(error_type=AIErrorType.MODEL_UNAVAILABLE, message=f"Model {model_name} is unavailable")


class RateLimitExceededError(AIError):
    def __init__(self, *, model_name: str) -> None:
        super().__init__(
            error_type=AIErrorType.RATE_LIMIT_EXCEEDED, message=f"Rate limit exceeded for model {model_name}"
        )


class NoResponseError(AIError):
    def __init__(self, *, model_name: str) -> None:
        super().__init__(error_type=AIErrorType.NO_RESPONSE, message=f"No response from model {model_name}")


class ToolCallError(AIError):
    def __init__(self, *, model_name: str) -> None:
        super().__init__(error_type=AIErrorType.TOOL_CALL_ERROR, message=f"Tool call error for model {model_name}")


class InvalidModelNameError(AIError):
    def __init__(self, *, model_name: str) -> None:
        super().__init__(error_type=AIErrorType.INVALID_MODEL_NAME, message=f"Invalid model name: {model_name}")


class LLMRequestError(AIError):
    def __init__(self, *, message: str) -> None:
        super().__init__(error_type=AIErrorType.BAD_REQUEST_ERROR, message=message)


class SchemaNotFollowedError(AIError):
    def __init__(self, *, raw_response_text: str, errors: list[ErrorDetails]) -> None:
        super().__init__(
            error_type=AIErrorType.SCHEMA_NOT_FOLLOWED,
            message=f"Provided schema not followed by LLM. Raw response text: {raw_response_text}. Errors: {errors}",
        )
