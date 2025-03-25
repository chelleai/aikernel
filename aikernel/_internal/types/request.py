from datetime import UTC, datetime
from enum import StrEnum
from typing import NoReturn, Self

from pydantic import BaseModel, Field, Json, computed_field, field_validator, model_validator

from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import (
    LiteLLMMediaMessagePart,
    LiteLLMMessage,
    LiteLLMTextMessagePart,
    LiteLLMTool,
)


class LLMModel(StrEnum):
    VERTEX_GEMINI_2_0_FLASH = "vertex_ai/gemini-2.0-flash"
    VERTEX_GEMINI_2_0_FLASH_LITE = "vertex_ai/gemini-2.0-flash-lite"
    VERTEX_GEMINI_2_0_PRO_EXP_02_05 = "vertex_ai/gemini-2.0-pro-exp-02-05"
    GEMINI_2_0_FLASH = "gemini/gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini/gemini-2.0-flash-lite"
    GEMINI_2_0_PRO_EXP_02_05 = "gemini/gemini-2.0-pro-exp-02-05"


class LLMMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessageContentType(StrEnum):
    TEXT = "text"
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    WAV = "audio/wav"
    MP3 = "audio/mp3"
    PDF = "application/pdf"


class LLMMessagePart(BaseModel):
    content: str
    content_type: LLMMessageContentType = LLMMessageContentType.TEXT


class _LLMMessage(BaseModel):
    parts: list[LLMMessagePart]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def render_parts(self) -> list[LiteLLMMediaMessagePart | LiteLLMTextMessagePart]:
        parts: list[LiteLLMMediaMessagePart | LiteLLMTextMessagePart] = []
        for part in self.parts:
            if part.content_type == LLMMessageContentType.TEXT:
                parts.append({"type": "text", "text": part.content})
            else:
                parts.append({"type": "image_url", "image_url": f"data:{part.content_type};base64,{part.content}"})

        return parts

    def render(self) -> LiteLLMMessage:
        raise NotImplementedError("Subclasses must implement this method")


class LLMSystemMessage(_LLMMessage):
    @computed_field
    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.SYSTEM

    def render(self) -> LiteLLMMessage:
        return {"role": "system", "content": self.render_parts()}


class LLMUserMessage(_LLMMessage):
    @computed_field
    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.USER

    def render(self) -> LiteLLMMessage:
        return {"role": "user", "content": self.render_parts()}


class LLMAssistantMessage(_LLMMessage):
    @computed_field
    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.ASSISTANT

    @model_validator(mode="after")
    def no_media_parts(self) -> Self:
        if any(part.content_type != LLMMessageContentType.TEXT for part in self.parts):
            raise AIError("Assistant messages can not have media parts")

        return self

    def render(self) -> LiteLLMMessage:
        return {"role": "assistant", "content": self.render_parts()}


class LLMToolMessageFunctionCall(BaseModel):
    name: str
    arguments: Json[str]


class LLMToolMessage(_LLMMessage):
    tool_call_id: str
    name: str
    response: Json[str]
    function_call: LLMToolMessageFunctionCall

    parts: list[LLMMessagePart] = []  # disabling from the base class

    @model_validator(mode="after")
    def no_parts(self) -> Self:
        if len(self.parts) > 0:
            raise AIError("Tool messages can not have parts")

        return self

    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.TOOL

    def render(self) -> NoReturn:
        raise AIError("Tool messages can not be rendered directly, please use render_call_and_response instead")

    def render_call_and_response(self) -> tuple[LiteLLMMessage, LiteLLMMessage]:
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
                        "arguments": self.function_call.arguments,
                    },
                }
            ],
        }
        response_message: LiteLLMMessage = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.response,
        }

        return invocation_message, response_message


class LLMTool[ParametersT: BaseModel](BaseModel):
    name: str
    description: str
    parameters: type[ParametersT]

    @field_validator("name", mode="after")
    @classmethod
    def validate_function_name(cls, value: str) -> str:
        if not value.replace("_", "").isalnum():
            raise ValueError("Function name must be alphanumeric plus underscores")

        return value

    def render(self) -> LiteLLMTool:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }
