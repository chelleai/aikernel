from typing import Any

from litellm import acompletion, completion
from pydantic import BaseModel

from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import LLMAssistantMessage, LLMModel, LLMSystemMessage, LLMTool, LLMUserMessage
from aikernel._internal.types.response import LLMUsage, StructuredLLMResponse

AnyLLMTool = LLMTool[Any]


def llm_structured_sync[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    response_model: type[T],
) -> StructuredLLMResponse[T]:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(message.render())

    response = completion(messages=rendered_messages, model=model.value, response_format=response_model)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    text = response.choices[0].message.content
    usage = LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return StructuredLLMResponse(text=text, structure=response_model, usage=usage)


async def llm_structured[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    response_model: type[T],
) -> StructuredLLMResponse[T]:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(message.render())

    response = await acompletion(messages=rendered_messages, model=model.value, response_format=response_model)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    text = response.choices[0].message.content
    usage = LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return StructuredLLMResponse(text=text, structure=response_model, usage=usage)
