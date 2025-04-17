import json
from typing import Any, Literal, overload

from aikernel._internal.router import LLMModelName, LLMRouter
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
from aikernel.errors import (
    NoResponseError,
    ToolCallError,
)

AnyLLMTool = LLMTool[Any]


@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
    router: LLMRouter[LLMModelName],
) -> LLMAutoToolResponse: ...
@overload
def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
    router: LLMRouter[LLMModelName],
) -> LLMRequiredToolResponse: ...


def llm_tool_call_sync(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[LLMModelName],
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

    response = router.complete(messages=rendered_messages, tools=rendered_tools, tool_choice=tool_choice)
    used_model = router.translate_model_name(model_name=response.model)

    if len(response.choices) == 0:
        raise NoResponseError(model_name=router.primary_model)

    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise ToolCallError(model_name=router.primary_model)
        else:
            return LLMAutoToolResponse(
                tool_call=None, text=response.choices[0].message.content, model=used_model, usage=usage
            )

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError) as error:
        raise ToolCallError(model_name=router.primary_model) from error

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as error:
        raise ToolCallError(model_name=router.primary_model) from error

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)
    response = LLMAutoToolResponse(tool_call=tool_call, model=used_model, usage=usage)

    return response


@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[LLMModelName],
    tools: list[AnyLLMTool],
    tool_choice: Literal["auto"],
) -> LLMAutoToolResponse: ...
@overload
async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[LLMModelName],
    tools: list[AnyLLMTool],
    tool_choice: Literal["required"],
) -> LLMRequiredToolResponse: ...


async def llm_tool_call(
    *,
    messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage],
    router: LLMRouter[LLMModelName],
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

    response = await router.acomplete(messages=rendered_messages, tools=rendered_tools, tool_choice=tool_choice)

    if len(response.choices) == 0:
        raise NoResponseError(model_name=router.primary_model)

    used_model = router.translate_model_name(model_name=response.model)
    usage = LLMResponseUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    tool_calls = response.choices[0].message.tool_calls or []
    if len(tool_calls) == 0:
        if tool_choice == "required":
            raise ToolCallError(model_name=used_model)
        else:
            return LLMAutoToolResponse(
                tool_call=None, text=response.choices[0].message.content, model=used_model, usage=usage
            )

    try:
        chosen_tool = next(tool for tool in tools if tool.name == tool_calls[0].function.name)
    except (StopIteration, IndexError) as error:
        raise ToolCallError(model_name=used_model) from error

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as error:
        raise ToolCallError(model_name=used_model) from error

    tool_call = LLMResponseToolCall(id=tool_calls[0].id, tool_name=chosen_tool.name, arguments=arguments)

    if tool_choice == "required":
        response = LLMRequiredToolResponse(tool_call=tool_call, model=used_model, usage=usage)
    else:
        response = LLMAutoToolResponse(tool_call=tool_call, model=used_model, usage=usage)

    return response
