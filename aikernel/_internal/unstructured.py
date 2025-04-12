"""Functions for getting unstructured (text) responses from LLMs.

This module provides functions for getting unstructured (plain text) responses
from LLMs. It provides both synchronous and asynchronous versions of the function.
"""

from typing import Any

from aikernel._internal.router import LLMRouter
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
)
from aikernel._internal.types.response import LLMResponseUsage, LLMUnstructuredResponse
from aikernel.errors import NoResponseError


def llm_unstructured_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
) -> LLMUnstructuredResponse:
    """Get an unstructured (text) response from an LLM synchronously.
    
    This function sends a conversation to an LLM and gets a plain text response.
    
    Args:
        messages: The conversation messages to send to the LLM
        router: The LLM router to use for making the request
        
    Returns:
        An unstructured response containing the text and usage information
        
    Raises:
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

    response = router.complete(messages=rendered_messages)

    if len(response.choices) == 0:
        raise NoResponseError()

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMUnstructuredResponse(text=text, usage=usage)


async def llm_unstructured(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
) -> LLMUnstructuredResponse:
    """Get an unstructured (text) response from an LLM asynchronously.
    
    This function sends a conversation to an LLM and gets a plain text response.
    
    Args:
        messages: The conversation messages to send to the LLM
        router: The LLM router to use for making the request
        
    Returns:
        An unstructured response containing the text and usage information
        
    Raises:
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

    response = await router.acomplete(messages=rendered_messages)

    if len(response.choices) == 0:
        raise NoResponseError()

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMUnstructuredResponse(text=text, usage=usage)
