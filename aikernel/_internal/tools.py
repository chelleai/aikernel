import json
from typing import Any, Literal, overload

from litellm import acompletion, completion

from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMModel,
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

AnyLLMTool = LLMTool[Any]


@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
) -> LLMAutoToolResponse: ...
@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
) -> LLMRequiredToolResponse: ...


def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto", "required"],
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

    response = completion(messages=rendered_messages, model=model.value, tools=rendered_tools, tool_choice=tool_choice)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise AIError("No tool call found in response")
        else:
            return LLMAutoToolResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError):
        raise AIError(f"No tool call found in response: {tool_calls}")

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        raise AIError(f"Invalid tool call arguments: {tool_calls[0].function.arguments}")

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)
    return LLMAutoToolResponse(tool_call=tool_call, usage=usage)


@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
) -> LLMAutoToolResponse: ...
@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
) -> LLMRequiredToolResponse: ...


async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    model: LLMModel,
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto", "required"] = "auto",
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

    response = await acompletion(
        messages=rendered_messages, model=model.value, tools=rendered_tools, tool_choice=tool_choice
    )

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise AIError("No tool call found in response")
        else:
            return LLMAutoToolResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError):
        raise AIError(f"No tool call found in response: {tool_calls}")

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        raise AIError(f"Invalid tool call arguments: {tool_calls[0].function.arguments}")

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)
    return LLMRequiredToolResponse(tool_call=tool_call, usage=usage)
