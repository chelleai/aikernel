from typing import Any, Self

from pydantic import BaseModel, computed_field, model_validator

from aikernel._internal.errors import AIError


class LLMToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class UnstructuredLLMResponse(BaseModel):
    text: str


class StructuredLLMResponse[T: BaseModel](BaseModel):
    text: str
    structure: type[T]

    @computed_field
    @property
    def structured_response(self) -> T:
        return self.structure.model_validate_json(self.text)


class ToolLLMResponse(BaseModel):
    tool_call: LLMToolCall | None = None
    text: str | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        if self.tool_call is None and self.text is None:
            raise AIError("At least one of tool_call or text must be provided")

        return self


class StrictToolLLMResponse(BaseModel):
    tool_call: LLMToolCall
