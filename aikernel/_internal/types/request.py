from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator

from aikernel._internal.errors import AIError
from aikernel._internal.types.provider import LiteLLMTool


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
    created_at: datetime = datetime.now(UTC)


class LLMSystemMessage(_LLMMessage):
    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.SYSTEM


class LLMUserMessage(_LLMMessage):
    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.USER


class LLMAssistantMessage(_LLMMessage):
    @classmethod
    def from_text(cls, text: str, /) -> Self:
        return cls(
            parts=[LLMMessagePart(content_type=LLMMessageContentType.TEXT, content=text)], created_at=datetime.now(UTC)
        )

    @property
    def role(self) -> LLMMessageRole:
        return LLMMessageRole.ASSISTANT

    @model_validator(mode="after")
    def no_media_parts(self) -> Self:
        if any(part.content_type != LLMMessageContentType.TEXT for part in self.parts):
            raise AIError("Assistant messages can not have media parts")

        return self


class LLMToolParameter(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "boolean", "array", "object"] = "string"
    optional: bool = False


class LLMTool(BaseModel):
    name: str
    description: str
    parameters: type[BaseModel]

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
