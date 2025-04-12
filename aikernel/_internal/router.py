"""Router implementation for managing LLM model selection and fallbacks.

This module provides a router implementation that manages LLM model selection,
handles fallbacks between models, and provides a consistent interface for
making requests to LLMs through the litellm library.
"""

import functools
from collections.abc import Callable
from typing import Any, Literal, NoReturn, NotRequired, TypedDict, cast

from litellm import Router
from pydantic import BaseModel

from aikernel._internal.types.provider import LiteLLMMessage, LiteLLMTool

LLMModelAlias = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
]
"""Simplified aliases for LLM models supported by the system."""

LLMModelName = Literal[
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.0-flash-lite",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
]
"""Full model names as used by the underlying providers."""

MODEL_ALIAS_MAPPING: dict[LLMModelAlias, LLMModelName] = {
    "gemini-2.0-flash": "vertex_ai/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "vertex_ai/gemini-2.0-flash-lite",
    "claude-3.5-sonnet": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3.7-sonnet": "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
}
"""Mapping from simplified model aliases to full provider-specific model names."""


def disable_method[**P, R](func: Callable[P, R]) -> Callable[P, NoReturn]:
    """Decorator to disable a method by raising NotImplementedError when called."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
        raise NotImplementedError(f"{func.__name__} is not implemented")

    return wrapper


class ModelResponseChoiceToolCallFunction(BaseModel):
    """Function call details in a model response."""
    name: str
    arguments: str


class ModelResponseChoiceToolCall(BaseModel):
    """Tool call in a model response."""
    id: str
    function: ModelResponseChoiceToolCallFunction
    type: Literal["function"]


class ModelResponseChoiceMessage(BaseModel):
    """Message in a model response choice."""
    role: Literal["assistant"]
    content: str
    tool_calls: list[ModelResponseChoiceToolCall] | None


class ModelResponseChoice(BaseModel):
    """A choice in a model response."""
    finish_reason: Literal["stop"]
    index: int
    message: ModelResponseChoiceMessage


class ModelResponseUsage(BaseModel):
    """Token usage metrics in a model response."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ModelResponse(BaseModel):
    """Complete response from an LLM model."""
    id: str
    created: int
    model: str
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[ModelResponseChoice]
    usage: ModelResponseUsage


class RouterModelLitellmParams(TypedDict):
    """Parameters for configuring a model in the litellm router."""
    model: str
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    rpm: NotRequired[int]


class RouterModel[ModelT: LLMModelAlias](TypedDict):
    """Configuration for a model in the router."""
    model_name: ModelT
    litellm_params: RouterModelLitellmParams


class LLMRouter[ModelT: LLMModelAlias](Router):
    """Router for managing LLM model selection and fallbacks.
    
    This class extends the litellm Router to provide a more type-safe and
    streamlined interface for working with LLM models.
    """
    def __init__(self, *, model_list: list[RouterModel[ModelT]], fallbacks: list[dict[ModelT, list[ModelT]]]) -> None:
        """Initialize the router with a list of models and fallback configurations.
        
        Args:
            model_list: List of models to use in the router
            fallbacks: List of fallback configurations
        """
        super().__init__(model_list=model_list, fallbacks=fallbacks)  # type: ignore

    @property
    def primary_model(self) -> ModelT:
        """Get the primary model for this router.
        
        Returns:
            The primary model alias
            
        Raises:
            ValueError: If no models are available
        """
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
    ) -> ModelResponse:
        """Make a synchronous completion request to the primary model.
        
        Args:
            messages: List of messages in the conversation
            response_format: Optional response format specification
            tools: Optional list of tools the model can use
            tool_choice: Optional tool choice setting
            max_tokens: Optional maximum number of tokens to generate
            temperature: Sampling temperature (default: 1.0)
            
        Returns:
            The model's response
        """
        raw_response = super().completion(
            model=MODEL_ALIAS_MAPPING[self.primary_model],
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return ModelResponse.model_validate(raw_response, from_attributes=True)

    async def acomplete(
        self,
        *,
        messages: list[LiteLLMMessage],
        response_format: Any | None = None,
        tools: list[LiteLLMTool] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        temperature: float = 1.0,
    ) -> ModelResponse:
        """Make an asynchronous completion request to the primary model.
        
        Args:
            messages: List of messages in the conversation
            response_format: Optional response format specification
            tools: Optional list of tools the model can use
            tool_choice: Optional tool choice setting
            temperature: Sampling temperature (default: 1.0)
            
        Returns:
            The model's response
        """
        raw_response = await super().acompletion(
            model=MODEL_ALIAS_MAPPING[self.primary_model],
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        return ModelResponse.model_validate(raw_response, from_attributes=True)

    @disable_method
    def completion(self, *args: Any, **kwargs: Any) -> NoReturn: 
        """Disabled method to prevent direct use of litellm's completion."""
        ...

    @disable_method
    def acompletion(self, *args: Any, **kwargs: Any) -> NoReturn: 
        """Disabled method to prevent direct use of litellm's acompletion."""
        ...


@functools.cache
def get_router[ModelT: LLMModelAlias](*, models: tuple[ModelT, ...]) -> LLMRouter[ModelT]:
    """Get a configured router instance for the given models.
    
    This function creates a router with the specified models and configures
    fallbacks between them. Results are cached based on the input models.
    
    Args:
        models: Tuple of model aliases to use
        
    Returns:
        A configured router instance
    """
    model_list: list[RouterModel[ModelT]] = [
        {"model_name": model, "litellm_params": {"model": MODEL_ALIAS_MAPPING[model]}} for model in models
    ]
    fallbacks = [{model: [other_model for other_model in models if other_model != model]} for model in models]
    return LLMRouter(model_list=model_list, fallbacks=fallbacks)
