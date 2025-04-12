"""Utilities for fewshot prompting with LLMs.

This module provides utilities for creating fewshot prompts for LLMs,
which include a system message and a set of examples of inputs and outputs.
"""

from pydantic import BaseModel

from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessagePart,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
)

Message = LLMSystemMessage | LLMUserMessage | LLMAssistantMessage | LLMToolMessage


class FewshotExample[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """An example of an input and output for fewshot prompting.
    
    This class represents a single example in a fewshot prompt,
    consisting of an input and the corresponding expected output.
    
    Type Parameters:
        InputT: The type of the input model
        OutputT: The type of the output model
    
    Attributes:
        input: The input for this example
        output: The expected output for this example
    """
    input: InputT
    output: OutputT


class FewshotPrompt[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """A fewshot prompt for an LLM.
    
    This class represents a fewshot prompt, which consists of a system message
    and a list of examples. When rendered, it produces a list of messages that
    can be sent to an LLM, with the examples formatted as alternating user and
    assistant messages.
    
    Type Parameters:
        InputT: The type of the input model for examples
        OutputT: The type of the output model for examples
    
    Attributes:
        system: The system message for the prompt
        examples: The list of examples to include in the prompt
    """
    system: LLMSystemMessage
    examples: list[FewshotExample[InputT, OutputT]]

    def render(self) -> list[Message]:
        """Render the fewshot prompt as a list of messages.
        
        This method renders the prompt as a list of messages that can be sent to
        an LLM. The system message is first, followed by the examples formatted
        as alternating user and assistant messages.
        
        Returns:
            A list of messages representing the prompt
        """
        messages: list[Message] = [self.system]
        for example in self.examples:
            messages.append(LLMUserMessage(parts=[LLMMessagePart(content=example.input.model_dump_json())]))
            messages.append(LLMAssistantMessage(parts=[LLMMessagePart(content=example.output.model_dump_json())]))

        return messages
