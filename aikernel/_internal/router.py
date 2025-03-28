from typing import Any, Literal, NotRequired, TypedDict

from litellm import Router
from pydantic import BaseModel

from aikernel._internal.types.provider import LiteLLMMessage, LiteLLMTool

LLMModelAlias = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
]
LLMModelName = Literal[
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.0-flash-lite",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
]

MODEL_ALIAS_MAPPING: dict[LLMModelAlias, LLMModelName] = {
    "gemini-2.0-flash": "vertex_ai/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "vertex_ai/gemini-2.0-flash-lite",
    "claude-3.5-sonnet": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3.7-sonnet": "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
}


class ModelResponseChoiceToolCallFunction(BaseModel):
    name: str
    arguments: str


class ModelResponseChoiceToolCall(BaseModel):
    id: str
    function: ModelResponseChoiceToolCallFunction
    type: Literal["function"]


class ModelResponseChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    tool_calls: list[ModelResponseChoiceToolCall] | None


class ModelResponseChoice(BaseModel):
    finish_reason: Literal["stop"]
    index: int
    message: ModelResponseChoiceMessage


class ModelResponseUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ModelResponse(BaseModel):
    id: str
    created: int
    model: str
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[ModelResponseChoice]
    usage: ModelResponseUsage


class RouterModelLitellmParams(TypedDict):
    model: str
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    rpm: NotRequired[int]


class RouterModel[ModelT: LLMModelAlias](TypedDict):
    model_name: ModelT
    litellm_params: RouterModelLitellmParams


class LLMRouter[ModelT: LLMModelAlias](Router):
    def __init__(self, *, model_list: list[RouterModel[ModelT]], fallbacks: list[dict[ModelT, list[ModelT]]]) -> None:
        super().__init__(model_list=model_list, fallbacks=fallbacks)  # type: ignore

    def complete(
        self,
        *,
        model: ModelT,
        messages: list[LiteLLMMessage],
        response_format: Any | None = None,
        tools: list[LiteLLMTool],
        tool_choice: Literal["auto", "required"] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
    ) -> ModelResponse:
        raw_response = super().completion(
            model=MODEL_ALIAS_MAPPING[model],
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return ModelResponse.model_validate(raw_response)

    async def acomplete(
        self,
        *,
        model: ModelT,
        messages: list[LiteLLMMessage],
        response_format: Any | None = None,
        tools: list[LiteLLMTool],
        tool_choice: Literal["auto", "required"] | None = None,
        temperature: float = 1.0,
    ) -> ModelResponse:
        raw_response = await super().acompletion(
            model=MODEL_ALIAS_MAPPING[model],
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        return ModelResponse.model_validate(raw_response)


def get_router[ModelT: LLMModelAlias](*, model_priority_list: list[ModelT]) -> LLMRouter[ModelT]:
    model_list: list[RouterModel[ModelT]] = [
        {"model_name": model, "litellm_params": {"model": MODEL_ALIAS_MAPPING[model]}} for model in model_priority_list
    ]
    fallbacks = [
        {model: [other_model for other_model in model_priority_list if other_model != model]}
        for model in model_priority_list
    ]
    return LLMRouter(model_list=model_list, fallbacks=fallbacks)
