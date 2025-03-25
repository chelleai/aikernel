from typing import Any, Self

from pydantic import BaseModel, computed_field, model_validator


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


class LLMStructuredResponse[T: BaseModel](BaseModel):
    text: str
    structure: type[T]
    usage: LLMResponseUsage

    @computed_field
    @property
    def structured_response(self) -> T:
        return self.structure.model_validate_json(self.text)


class LLMAutoToolResponse(BaseModel):
    tool_call: LLMResponseToolCall | None = None
    text: str | None = None
    usage: LLMResponseUsage

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        if self.tool_call is None and self.text is None:
            raise ValueError("At least one of tool_call or text must be provided")

        return self


class LLMRequiredToolResponse(BaseModel):
    tool_call: LLMResponseToolCall
    usage: LLMResponseUsage
