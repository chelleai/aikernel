from litellm import acompletion, completion

from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMModel,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
)
from aikernel._internal.types.response import LLMResponseUsage, LLMUnstructuredResponse


def llm_unstructured_sync(
    *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage], model: LLMModel
) -> LLMUnstructuredResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    response = completion(messages=rendered_messages, model=model.value)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMUnstructuredResponse(text=text, usage=usage)


async def llm_unstructured(
    *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage], model: LLMModel
) -> LLMUnstructuredResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    response = await acompletion(messages=rendered_messages, model=model.value)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    text = response.choices[0].message.content
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return LLMUnstructuredResponse(text=text, usage=usage)
