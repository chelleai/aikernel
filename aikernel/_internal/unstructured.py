from litellm import acompletion, completion

from aikernel._internal.core import render_message
from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import LLMAssistantMessage, LLMModel, LLMSystemMessage, LLMUserMessage
from aikernel._internal.types.response import LLMUsage, UnstructuredLLMResponse


def llm_unstructured_sync(
    *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage], model: LLMModel
) -> UnstructuredLLMResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    response = completion(messages=rendered_messages, model=model.value)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    return UnstructuredLLMResponse(
        text=response.choices[0].message.content,
        usage=LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens),
    )


async def llm_unstructured(
    *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage], model: LLMModel
) -> UnstructuredLLMResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    response = await acompletion(messages=rendered_messages, model=model.value)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    return UnstructuredLLMResponse(
        text=response.choices[0].message.content,
        usage=LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens),
    )
