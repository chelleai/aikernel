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
    input: InputT
    output: OutputT


class FewshotPrompt[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    system: LLMSystemMessage
    examples: list[FewshotExample[InputT, OutputT]]

    def render(self) -> list[Message]:
        messages: list[Message] = [self.system]
        for example in self.examples:
            messages.append(
                LLMUserMessage(
                    parts=[LLMMessagePart(content=example.input.model_dump_json())],
                    cache=True,
                )
            )
            messages.append(
                LLMAssistantMessage(
                    parts=[LLMMessagePart(content=example.output.model_dump_json())],
                    cache=True,
                )
            )

        return messages
