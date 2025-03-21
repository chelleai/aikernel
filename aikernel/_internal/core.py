from aikernel._internal.types.provider import LiteLLMMediaMessagePart, LiteLLMMessage, LiteLLMTextMessagePart
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessageContentType,
    LLMSystemMessage,
    LLMUserMessage,
)


def render_message(message: LLMUserMessage | LLMAssistantMessage | LLMSystemMessage, /) -> LiteLLMMessage:
    role = (
        "system"
        if isinstance(message, LLMSystemMessage)
        else "user"
        if isinstance(message, LLMUserMessage)
        else "assistant"
    )

    content: list[LiteLLMTextMessagePart | LiteLLMMediaMessagePart] = []
    for part in message.parts:
        if part.content_type == LLMMessageContentType.TEXT:
            content.append({"type": "text", "text": part.content})
        else:
            image_url = f"data:{part.content_type};base64,{part.content}"
            content.append({"type": "image_url", "image_url": image_url})

    return {"role": role, "content": content}
