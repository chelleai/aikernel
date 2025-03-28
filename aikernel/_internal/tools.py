import json
from typing import Any, Literal, overload

from litellm import Router
from litellm.exceptions import RateLimitError, ServiceUnavailableError

from aikernel._internal.router import LLMModelAlias
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMSystemMessage,
    LLMTool,
    LLMToolMessage,
    LLMUserMessage,
)
from aikernel._internal.types.response import (
    LLMAutoToolResponse,
    LLMRequiredToolResponse,
    LLMResponseToolCall,
    LLMResponseUsage,
)
from aikernel.errors import ModelUnavailableError, NoResponseError, RateLimitExceededError, ToolCallError

AnyLLMTool = LLMTool[Any]


@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
    router: Router,
) -> LLMAutoToolResponse: ...
@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
    router: Router,
) -> LLMRequiredToolResponse: ...


def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto", "required"],
    router: Router,
) -> LLMAutoToolResponse | LLMRequiredToolResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    rendered_tools = [tool.render() for tool in tools]

    response = router.completion(messages=rendered_messages, model=model, tools=rendered_tools, tool_choice=tool_choice)

    if len(response.choices) == 0:
        raise NoResponseError()

    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise ToolCallError()
        else:
            return LLMAutoToolResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError) as error:
        raise ToolCallError() from error

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as error:
        raise ToolCallError() from error

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)
    return LLMAutoToolResponse(tool_call=tool_call, usage=usage)


@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
    router: Router,
) -> LLMAutoToolResponse: ...
@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
    router: Router,
) -> LLMRequiredToolResponse: ...


async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModelAlias,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto", "required"] = "auto",
    router: Router,
) -> LLMAutoToolResponse | LLMRequiredToolResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        if isinstance(message, LLMToolMessage):
            invocation_message, response_message = message.render_call_and_response()
            rendered_messages.append(invocation_message)
            rendered_messages.append(response_message)
        else:
            rendered_messages.append(message.render())

    rendered_tools = [tool.render() for tool in tools]

    try:
        response = await router.acompletion(
            messages=rendered_messages, model=model, tools=rendered_tools, tool_choice=tool_choice
        )
    except ServiceUnavailableError:
        raise ModelUnavailableError()
    except RateLimitError:
        raise RateLimitExceededError()

    if len(response.choices) == 0:
        raise NoResponseError()

    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise ToolCallError()
        else:
            return LLMAutoToolResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError) as error:
        raise ToolCallError() from error

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as error:
        raise ToolCallError() from error

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)
    return LLMRequiredToolResponse(tool_call=tool_call, usage=usage)
