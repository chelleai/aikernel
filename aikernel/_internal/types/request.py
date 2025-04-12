"""Request type definitions for interacting with LLMs.

This module defines the types used for constructing requests to LLMs,
including different message types (system, user, assistant, tool) and
their content formats.
"""

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, NoReturn, Self

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from aikernel._internal.types.provider import (
    LiteLLMMediaMessagePart,
    LiteLLMMessage,
    LiteLLMTextMessagePart,
    LiteLLMTool,
)


class LLMMessageRole(StrEnum):
    """Enum for different message roles in an LLM conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessageContentType(StrEnum):
    """Enum for different content types that can be included in messages."""
    TEXT = "text"
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    WAV = "audio/wav"
    MP3 = "audio/mp3"
    PDF = "application/pdf"


class LLMMessagePart(BaseModel):
    """A part of a message, which can be text or media."""
    content: str
    content_type: LLMMessageContentType = LLMMessageContentType.TEXT


class _LLMMessage(BaseModel):
    """Base class for all message types.
    
    This is an abstract base class that provides common functionality
    for all message types. It should not be instantiated directly.
    """
    parts: list[LLMMessagePart]
    cache: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def render_parts(self) -> list[LiteLLMMediaMessagePart | LiteLLMTextMessagePart]:
        """Convert message parts to litellm format."""
        parts: list[LiteLLMMediaMessagePart | LiteLLMTextMessagePart] = []
        for part in self.parts:
            if part.content_type == LLMMessageContentType.TEXT:
                parts.append({"type": "text", "text": part.content})
            else:
                parts.append({"type": "image_url", "image_url": f"data:{part.content_type};base64,{part.content}"})

        return parts

    def render(self) -> LiteLLMMessage:
        """Convert message to litellm format.
        
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class LLMSystemMessage(_LLMMessage):
    """System message providing instructions to the LLM."""
    @computed_field
    @property
    def role(self) -> Literal[LLMMessageRole.SYSTEM]:
        """The role of this message."""
        return LLMMessageRole.SYSTEM

    def render(self) -> LiteLLMMessage:
        """Convert system message to litellm format."""
        message: LiteLLMMessage = {"role": "system", "content": self.render_parts()}
        if self.cache:
            message["cache_control"] = {"type": "ephemeral"}

        return message


class LLMUserMessage(_LLMMessage):
    """User message in a conversation with an LLM."""
    @computed_field
    @property
    def role(self) -> Literal[LLMMessageRole.USER]:
        """The role of this message."""
        return LLMMessageRole.USER

    def render(self) -> LiteLLMMessage:
        """Convert user message to litellm format."""
        message: LiteLLMMessage = {"role": "user", "content": self.render_parts()}
        if self.cache:
            message["cache_control"] = {"type": "ephemeral"}

        return message


class LLMAssistantMessage(_LLMMessage):
    """Assistant (AI) message in a conversation."""
    @computed_field
    @property
    def role(self) -> Literal[LLMMessageRole.ASSISTANT]:
        """The role of this message."""
        return LLMMessageRole.ASSISTANT

    @model_validator(mode="after")
    def no_media_parts(self) -> Self:
        """Validate that assistant messages don't contain media parts."""
        if any(part.content_type != LLMMessageContentType.TEXT for part in self.parts):
            raise ValueError("Assistant messages can not have media parts")

        return self

    def render(self) -> LiteLLMMessage:
        """Convert assistant message to litellm format."""
        message: LiteLLMMessage = {"role": "assistant", "content": self.render_parts()}
        if self.cache:
            message["cache_control"] = {"type": "ephemeral"}

        return message


class LLMToolMessageFunctionCall(BaseModel):
    """Function call details for tool messages."""
    name: str
    arguments: dict[str, Any]


class LLMToolMessage(_LLMMessage):
    """Tool message representing a tool call and its response."""
    tool_call_id: str
    name: str
    response: dict[str, Any]
    function_call: LLMToolMessageFunctionCall

    parts: list[LLMMessagePart] = []  # disabling from the base class

    @model_validator(mode="after")
    def no_parts(self) -> Self:
        """Validate that tool messages don't have parts."""
        if len(self.parts) > 0:
            raise ValueError("Tool messages can not have parts")

        return self

    @field_validator("cache", mode="after")
    def cannot_cache_tool_message(cls, value: bool) -> bool:
        """Validate that tool messages aren't cached."""
        if value:
            raise ValueError("Tool messages can not be cached")

        return value

    @computed_field
    @property
    def role(self) -> Literal[LLMMessageRole.TOOL]:
        """The role of this message."""
        return LLMMessageRole.TOOL

    def render(self) -> NoReturn:
        """Not implemented for tool messages."""
        raise TypeError("Tool messages can not be rendered directly, please use render_call_and_response instead")

    def render_call_and_response(self) -> tuple[LiteLLMMessage, LiteLLMMessage]:
        """Convert tool message to a pair of litellm messages for the call and response."""
        invocation_message: LiteLLMMessage = {
            "role": "assistant",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": None,
            "tool_calls": [
                {
                    "id": self.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": json.dumps(self.function_call.arguments, default=str),
                    },
                }
            ],
        }
        response_message: LiteLLMMessage = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": json.dumps(self.response, default=str),
        }

        return invocation_message, response_message


class LLMTool[ParametersT: BaseModel](BaseModel):
    """Definition of a tool that can be called by an LLM."""
    name: str
    description: str
    parameters: type[ParametersT]

    @field_validator("name", mode="after")
    @classmethod
    def validate_function_name(cls, value: str) -> str:
        """Validate that function names are alphanumeric with underscores."""
        if not value.replace("_", "").isalnum():
            raise ValueError("Function name must be alphanumeric plus underscores")

        return value

    def render(self) -> LiteLLMTool:
        """Convert tool definition to litellm format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }
