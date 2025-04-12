"""Response type definitions for LLM responses.

This module defines the types used for handling responses from LLMs,
including structured responses, unstructured text responses, and tool call responses.
"""

from typing import Any, Self

from pydantic import BaseModel, computed_field, model_validator


class LLMResponseToolCall(BaseModel):
    """A tool call response from an LLM."""
    id: str
    tool_name: str
    arguments: dict[str, Any]


class LLMResponseUsage(BaseModel):
    """Token usage metrics for an LLM request/response."""
    input_tokens: int
    output_tokens: int


class LLMUnstructuredResponse(BaseModel):
    """An unstructured (text) response from an LLM."""
    text: str
    usage: LLMResponseUsage


class LLMStructuredResponse[T: BaseModel](BaseModel):
    """A structured response from an LLM that can be parsed into a Pydantic model."""
    text: str
    structure: type[T]
    usage: LLMResponseUsage

    @computed_field
    @property
    def structured_response(self) -> T:
        """Parse the text response into a structured model."""
        return self.structure.model_validate_json(self.text)


class LLMAutoToolResponse(BaseModel):
    """Response when tool_choice is set to 'auto'.
    
    With tool_choice='auto', the model can either call a tool or respond with text.
    """
    tool_call: LLMResponseToolCall | None = None
    text: str | None = None
    usage: LLMResponseUsage

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        """Validate that either tool_call or text is provided."""
        if self.tool_call is None and self.text is None:
            raise ValueError("At least one of tool_call or text must be provided")

        return self


class LLMRequiredToolResponse(BaseModel):
    """Response when tool_choice is set to 'required'.
    
    With tool_choice='required', the model must call a tool.
    """
    tool_call: LLMResponseToolCall
    usage: LLMResponseUsage
