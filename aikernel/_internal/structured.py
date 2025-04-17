from typing import Any

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
from aikernel.errors import NoResponseError

AnyLLMTool = LLMTool[Any]


def llm_structured_sync[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
    response_model: type[T],
) -> LLMStructuredResponse[T]:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    response = router.complete(messages=rendered_messages, response_format=response_model, num_retries=2)

    if len(response.choices) == 0:
        raise NoResponseError(model_name=router.primary_model)

    text = response.choices[0].message.content or ""
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    response = LLMStructuredResponse(
        text=text, structure=response_model, model=router.translate_model_name(model_name=response.model), usage=usage
    )
    return response


async def llm_structured[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[Any],
    response_model: type[T],
) -> LLMStructuredResponse[T]:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    response = await router.acomplete(messages=rendered_messages, response_format=response_model, num_retries=2)

    if len(response.choices) == 0:
        raise NoResponseError(model_name=router.primary_model)

    text = response.choices[0].message.content or ""
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    response = LLMStructuredResponse(
        text=text, structure=response_model, model=router.translate_model_name(model_name=response.model), usage=usage
    )
    return response
