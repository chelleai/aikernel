import json
from typing import Literal, overload

from litellm import acompletion, completion

from aikernel._internal.core import render_message
from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMMessage
from aikernel._internal.types.request import LLMAssistantMessage, LLMModel, LLMSystemMessage, LLMTool, LLMUserMessage
from aikernel._internal.types.response import (
    LLMToolCall,
    LLMUsage,
    StrictToolLLMResponse,
    ToolLLMResponse,
    UnstructuredLLMResponse,
)


@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["auto"],
) -> ToolLLMResponse: ...
@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["required"],
) -> StrictToolLLMResponse: ...
@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["none"],
) -> UnstructuredLLMResponse: ...


def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["auto", "none", "required"],
) -> ToolLLMResponse | StrictToolLLMResponse | UnstructuredLLMResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    rendered_tools = [tool.render() for tool in tools]

    response = completion(messages=rendered_messages, model=model.value, tools=rendered_tools, tool_choice=tool_choice)

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    usage = LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise AIError("No tool call found in response")
        elif tool_choice == "auto":
            return ToolLLMResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)
        else:
            return UnstructuredLLMResponse(text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError):
        raise AIError(f"No tool call found in response: {tool_calls}")

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        raise AIError(f"Invalid tool call arguments: {tool_calls[0].function.arguments}")

    tool_call = LLMToolCall(tool_name=chosen_tool.name, arguments=arguments)
    return StrictToolLLMResponse(tool_call=tool_call, usage=usage)


@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["auto"],
) -> ToolLLMResponse: ...
@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["required"],
) -> StrictToolLLMResponse: ...
@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["none"],
) -> UnstructuredLLMResponse: ...


async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
    model: LLMModel,
    tools: list[LLMTool],
    tool_choice: Literal["auto", "none", "required"] = "auto",
) -> ToolLLMResponse | StrictToolLLMResponse | UnstructuredLLMResponse:
    rendered_messages: list[LiteLLMMessage] = []
    for message in messages:
        rendered_messages.append(render_message(message))

    rendered_tools = [tool.render() for tool in tools]

    response = await acompletion(
        messages=rendered_messages, model=model.value, tools=rendered_tools, tool_choice=tool_choice
    )

    if len(response.choices) == 0:
        raise AIError("No response from LLM")

    usage = LLMUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise AIError("No tool call found in response")
        elif tool_choice == "auto":
            return ToolLLMResponse(tool_call=None, text=response.choices[0].message.content, usage=usage)
        else:
            return UnstructuredLLMResponse(text=response.choices[0].message.content, usage=usage)

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError):
        raise AIError(f"No tool call found in response: {tool_calls}")

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        raise AIError(f"Invalid tool call arguments: {tool_calls[0].function.arguments}")

    tool_call = LLMToolCall(tool_name=chosen_tool.name, arguments=arguments)
    return StrictToolLLMResponse(tool_call=tool_call, usage=usage)
