from litellm import acompletion, completion
from pydantic import BaseModel

from aikernel._internal.core import render_message
from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import LLMAssistantMessage, LLMModel, LLMSystemMessage, LLMUserMessage


def llm_structured_sync[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    response_model: type[T],
) -> T:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    response = completion(messages=rendered_messages, model=model.value, response_format=response_model)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    parsed = response_model.model_validate_json(response.choices[0].message.content)

    return parsed


async def llm_structured[T: BaseModel](
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    response_model: type[T],
) -> T:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    response = await acompletion(messages=rendered_messages, model=model.value, response_format=response_model)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    parsed = response_model.model_validate_json(response.choices[0].message.content)

    return parsed
