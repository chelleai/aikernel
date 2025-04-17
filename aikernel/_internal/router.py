import functools
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, NoReturn, NotRequired, TypedDict, cast

from litellm import Router
from litellm.exceptions import BadRequestError, RateLimitError, ServiceUnavailableError
from pydantic import BaseModel

from aikernel._internal.types.provider import LiteLLMMessage, LiteLLMTool
from aikernel.errors import (
    InvalidModelNameError,
    LLMRequestError,
    ModelUnavailableError,
    RateLimitExceededError,
)


class LLMModelName(StrEnum):
    GEMINI_20_FLASH = "gemini/gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini/gemini-2.0-flash-lite"
    CLAUDE_35_SONNET = "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_37_SONNET = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"


def disable_method[**P, R](func: Callable[P, R]) -> Callable[P, NoReturn]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
        raise NotImplementedError(f"{func.__name__} is not implemented")

    return wrapper


class ModelResponseChoiceToolCallFunction(BaseModel):
    name: str
    arguments: str


class ModelResponseChoiceToolCall(BaseModel):
    id: str
    function: ModelResponseChoiceToolCallFunction
    type: Literal["function"]


class ModelResponseChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ModelResponseChoiceToolCall] | None


class ModelResponseChoice(BaseModel):
    finish_reason: Literal["stop", "tool_calls"]
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


class RouterModel[ModelT: LLMModelName](TypedDict):
    model_name: ModelT
    litellm_params: RouterModelLitellmParams


class LLMRouter[ModelT: LLMModelName](Router):
    def __init__(self, *, model_list: list[RouterModel[ModelT]], fallbacks: list[dict[ModelT, list[ModelT]]]) -> None:
        super().__init__(model_list=model_list, fallbacks=fallbacks)  # type: ignore

    @property
    def primary_model(self) -> ModelT:
        model_names = self.model_names

        if len(model_names) == 0:
            raise ValueError("No models available")

        return cast(ModelT, model_names[0])

    def complete(
        self,
        *,
        messages: list[LiteLLMMessage],
        response_format: Any | None = None,
        tools: list[LiteLLMTool] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        num_retries: int = 0,
    ) -> ModelResponse:
        try:
            raw_response = super().completion(
                model=self.primary_model,
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                num_retries=num_retries,
            )
        except RateLimitError:
            raise RateLimitExceededError(model_name=self.primary_model)
        except ServiceUnavailableError:
            raise ModelUnavailableError(model_name=self.primary_model)
        except BadRequestError as error:
            raise LLMRequestError(message=error.message)

        return ModelResponse.model_validate(raw_response, from_attributes=True)

    async def acomplete(
        self,
        *,
        messages: list[LiteLLMMessage],
        response_format: Any | None = None,
        tools: list[LiteLLMTool] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        temperature: float = 1.0,
        num_retries: int = 0,
    ) -> ModelResponse:
        try:
            raw_response = await super().acompletion(
                model=self.primary_model,
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                num_retries=num_retries,
            )
        except RateLimitError:
            raise RateLimitExceededError(model_name=self.primary_model)
        except ServiceUnavailableError:
            raise ModelUnavailableError(model_name=self.primary_model)

        return ModelResponse.model_validate(raw_response, from_attributes=True)

    def translate_model_name(self, *, model_name: str) -> LLMModelName:
        table = {model_name.value.split("/")[-1]: model_name for model_name in LLMModelName}

        if model_name not in table:
            raise InvalidModelNameError(model_name=model_name)

        return table[model_name]

    @disable_method
    def completion(self, *args: Any, **kwargs: Any) -> NoReturn: ...

    @disable_method
    def acompletion(self, *args: Any, **kwargs: Any) -> NoReturn: ...


class RouterRegistry:
    def __init__(self) -> None:
        self._routers: dict[tuple[LLMModelName, ...], LLMRouter[LLMModelName]] = {}

    def get_router[ModelT: LLMModelName](self, *, models: tuple[ModelT, ...]) -> LLMRouter[ModelT]:
        if models in self._routers:
            return self._routers[models]  # type: ignore

        model_list: list[RouterModel[ModelT]] = [
            {"model_name": model, "litellm_params": {"model": model.value}} for model in models
        ]
        fallbacks = [{model: [other_model for other_model in models if other_model != model]} for model in models]

        router = LLMRouter(model_list=model_list, fallbacks=fallbacks)
        self._routers[models] = router

        return router


router_registry = RouterRegistry()


def get_router[ModelT: LLMModelName](*, models: tuple[ModelT, ...]) -> LLMRouter[ModelT]:
    return router_registry.get_router(models=models)
