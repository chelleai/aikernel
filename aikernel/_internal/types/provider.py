"""Provider-specific type definitions for litellm library interoperability.

This module defines TypedDict classes that mirror the structure of objects
required by the litellm library. These types are used for serializing aikernel's
message and tool formats to the formats expected by litellm.
"""

from typing import Any, Literal, NotRequired, TypedDict


class LiteLLMTextMessagePart(TypedDict):
    """Text part of a litellm message."""
    type: Literal["text"]
    text: str


class LiteLLMMediaMessagePart(TypedDict):
    """Media part of a litellm message (e.g. an image)."""
    type: Literal["image_url"]
    image_url: str


class LiteLLMCacheControl(TypedDict):
    """Cache control settings for litellm messages."""
    type: Literal["ephemeral"]


class LiteLLMFunctionCall(TypedDict):
    """Function call made by an LLM."""
    name: str
    arguments: str


class LiteLLMToolCall(TypedDict):
    """Tool call made by an LLM."""
    id: str
    type: Literal["function"]
    function: LiteLLMFunctionCall


class LiteLLMMessage(TypedDict):
    """Message format expected by the litellm library."""
    role: Literal["system", "user", "assistant", "tool"]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    content: list[LiteLLMTextMessagePart | LiteLLMMediaMessagePart] | str | None
    tool_calls: NotRequired[list[LiteLLMToolCall]]
    cache_control: NotRequired[LiteLLMCacheControl]


class LiteLLMToolFunction(TypedDict):
    """Function definition for a tool in litellm format."""
    name: str
    description: str
    parameters: dict[str, Any]


class LiteLLMTool(TypedDict):
    """Tool definition in litellm format."""
    type: Literal["function"]
    function: LiteLLMToolFunction
