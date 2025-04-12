"""Functions for getting structured responses from LLMs.

This module provides functions for getting structured responses from LLMs,
where the model is asked to generate output conforming to a Pydantic model.
It provides both synchronous and asynchronous versions of the function.
"""

from typing import Any

from litellm.exceptions import RateLimitError, ServiceUnavailableError
from pydantic import BaseModel

from aikernel._internal.router import LLMRouter
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMSystemMessage,
    LLMTool,
    LLMToolMessage,
    LLMUserMessage,
)
from aikernel._internal.types.response import LLMResponseUsage, LLMStructuredResponse
from aikernel.errors import ModelUnavailableError, NoResponseError, RateLimitExceededError

AnyLLMTool = LLMTool[Any]


def llm_structured_sync[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
    response_model: type[T],
) -> LLMStructuredResponse[T]:
    """Get a structured response from an LLM synchronously.
    
    This function sends a conversation to an LLM and asks it to generate
    a response that conforms to the provided Pydantic model.
    
    Args:
        messages: The conversation messages to send to the LLM
        router: The LLM router to use for making the request
        response_model: The Pydantic model that the response should conform to
        
    Returns:
        A structured response containing the raw text and parsed model
        
    Raises:
        ModelUnavailableError: If the model is unavailable
        RateLimitExceededError: If rate limits have been exceeded
        NoResponseError: If the model didn't provide a response
    """
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    try:
        response = router.complete(messages=rendered_messages, response_format=response_model)
    except ServiceUnavailableError:
        raise ModelUnavailableError()
    except RateLimitError:
        raise RateLimitExceededError()

    if len(response.choices) == 0:
        raise NoResponseError()

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMStructuredResponse(text=text, structure=response_model, usage=usage)


async def llm_structured[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
    response_model: type[T],
) -> LLMStructuredResponse[T]:
    """Get a structured response from an LLM asynchronously.
    
    This function sends a conversation to an LLM and asks it to generate
    a response that conforms to the provided Pydantic model.
    
    Args:
        messages: The conversation messages to send to the LLM
        router: The LLM router to use for making the request
        response_model: The Pydantic model that the response should conform to
        
    Returns:
        A structured response containing the raw text and parsed model
        
    Raises:
        ModelUnavailableError: If the model is unavailable
        RateLimitExceededError: If rate limits have been exceeded
        NoResponseError: If the model didn't provide a response
    """
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    try:
        response = await router.acomplete(messages=rendered_messages, response_format=response_model)
    except ServiceUnavailableError:
        raise ModelUnavailableError()
    except RateLimitError:
        raise RateLimitExceededError()

    if len(response.choices) == 0:
        raise NoResponseError()

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMStructuredResponse(text=text, structure=response_model, usage=usage)
