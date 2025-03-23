import json
from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, computed_field, model_validator

from aikernel._internal.errors import AIError
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessageContentType,
    LLMMessagePart,
    LLMToolMessage,
    LLMToolMessageFunctionCall,
)


class LLMResponseToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: dict[str, Any]


class LLMResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class LLMUnstructuredResponse(BaseModel):
    text: str
    usage: LLMResponseUsage

    def as_message(self, *, created_at: datetime = datetime.now(UTC)) -> LLMAssistantMessage:
        return LLMAssistantMessage(
            parts=[LLMMessagePart(content_type=LLMMessageContentType.TEXT, content=self.text)],
            created_at=created_at,
        )


class LLMStructuredResponse[T: BaseModel](BaseModel):
    text: str
    structure: type[T]
    usage: LLMResponseUsage

    @computed_field
    @property
    def structured_response(self) -> T:
        return self.structure.model_validate_json(self.text)

    def as_message(self, *, created_at: datetime = datetime.now(UTC)) -> LLMAssistantMessage:
        return LLMAssistantMessage(
            parts=[LLMMessagePart(content_type=LLMMessageContentType.TEXT, content=self.text)],
            created_at=created_at,
        )


class LLMAutoToolResponse(BaseModel):
    tool_call: LLMResponseToolCall | None = None
    text: str | None = None
    usage: LLMResponseUsage

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        if self.tool_call is None and self.text is None:
            raise AIError("At least one of tool_call or text must be provided")

        return self

    def as_message(
        self, *, created_at: datetime | None = None, return_value: str | None = None
    ) -> LLMAssistantMessage | LLMToolMessage:
        if self.tool_call is not None:
            if return_value is None:
                raise AIError("Return value is required for tool messages")

            return LLMToolMessage(
                tool_call_id=self.tool_call.id,
                name=self.tool_call.tool_name,
                response=return_value,
                function_call=LLMToolMessageFunctionCall(
                    name=self.tool_call.tool_name,
                    arguments=json.dumps(self.tool_call.arguments),
                ),
                created_at=created_at or datetime.now(UTC),
            )
        else:
            if self.text is None:
                raise AIError("Text is required for assistant messages")

            return LLMAssistantMessage(
                parts=[LLMMessagePart(content_type=LLMMessageContentType.TEXT, content=self.text)],
                created_at=created_at or datetime.now(UTC),
            )


class LLMRequiredToolResponse(BaseModel):
    tool_call: LLMResponseToolCall
    usage: LLMResponseUsage
