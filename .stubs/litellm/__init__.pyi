from typing import Any, Literal, NotRequired, TypedDict

_LiteLLMGeminiModel = Literal[
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.0-flash-lite",
    "vertex_ai/gemini-2.0-pro-exp-02-05",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "gemini/gemini-2.0-pro-exp-02-05",
]
_LiteLLMEmbeddingModel = Literal[
    "gemini/text-embedding-004",
]
_MessageRole = Literal["system", "user", "assistant", "tool"]


# request
class _LiteLLMTextMessageContent(TypedDict):
    type: Literal["text"]
    text: str


class _LiteLLMMediaMessageContent(TypedDict):
    type: Literal["image_url"]
    image_url: str


class _LiteLLMToolMessageContent(TypedDict):
    type: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


class _LiteLLMCacheControl(TypedDict):
    type: Literal["ephemeral"]


class _LiteLLMFunctionCall(TypedDict):
    name: str
    arguments: str


class _LiteLLMToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: _LiteLLMFunctionCall


class _LiteLLMMessage(TypedDict):
    role: _MessageRole
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    content: list[_LiteLLMTextMessageContent | _LiteLLMMediaMessageContent] | str | None
    tool_calls: NotRequired[list[_LiteLLMToolCall]]
    cache_control: NotRequired[_LiteLLMCacheControl]


class _LiteLLMFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _LiteLLMTool(TypedDict):
    type: Literal["function"]
    function: _LiteLLMFunction


# response
class _LiteLLMModelResponseChoiceToolCallFunction:
    name: str
    arguments: str


class _LiteLLMModelResponseChoiceToolCall:
    id: str
    function: _LiteLLMModelResponseChoiceToolCallFunction
    type: Literal["function"]


class _LiteLLMModelResponseChoiceMessage:
    role: Literal["assistant"]
    content: str
    tool_calls: list[_LiteLLMModelResponseChoiceToolCall] | None

class _LiteLLMModelResponseChoice:
    finish_reason: Literal["stop"]
    index: int
    message: _LiteLLMModelResponseChoiceMessage


class _LiteLLMUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ModelResponse:
    id: str
    created: int
    model: _LiteLLMGeminiModel
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[_LiteLLMModelResponseChoice]
    usage: _LiteLLMUsage


class _LiteLLMEmbeddingData(TypedDict):
    index: int
    object: Literal["embedding"]
    embedding: list[float]


class EmbeddingResponse:
    data: list[_LiteLLMEmbeddingData]
    model: _LiteLLMEmbeddingModel
    usage: _LiteLLMUsage


def completion(
    *,
    model: _LiteLLMGeminiModel,
    messages: list[_LiteLLMMessage],
    response_format: Any = None,
    tools: list[_LiteLLMTool] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
) -> ModelResponse: ...


async def acompletion(
    *,
    model: _LiteLLMGeminiModel,
    messages: list[_LiteLLMMessage],
    response_format: Any = None,
    tools: list[_LiteLLMTool] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
) -> ModelResponse: ...

def embedding(model: _LiteLLMEmbeddingModel, input: list[str]) -> EmbeddingResponse: ...

async def aembedding(model: _LiteLLMEmbeddingModel, input: list[str]) -> EmbeddingResponse: ...