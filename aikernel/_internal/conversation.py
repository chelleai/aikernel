import json
from typing import NewType

from pydantic import ValidationError

from aikernel._internal.errors import AIError
from aikernel._internal.types.request import LLMAssistantMessage, LLMSystemMessage, LLMUserMessage

ConversationDump = NewType("ConversationDump", str)


class Conversation:
    def __init__(self) -> None:
        self._user_messages: list[LLMUserMessage] = []
        self._assistant_messages: list[LLMAssistantMessage] = []
        self._system_message: LLMSystemMessage | None = None

    def add_user_message(self, *, message: LLMUserMessage) -> None:
        self._user_messages.append(message)

    def add_assistant_message(self, *, message: LLMAssistantMessage) -> None:
        self._assistant_messages.append(message)

    def set_system_message(self, *, message: LLMSystemMessage) -> None:
        self._system_message = message

    def render(self) -> list[LLMSystemMessage | LLMUserMessage | LLMAssistantMessage]:
        messages = [self._system_message] if self._system_message is not None else []
        messages += sorted(self._user_messages + self._assistant_messages, key=lambda message: message.created_at)

        return messages

    def dump(self) -> str:
        conversation_dump = {
            "system": self._system_message.model_dump_json() if self._system_message is not None else None,
            "user": [message.model_dump_json() for message in self._user_messages],
            "assistant": [message.model_dump_json() for message in self._assistant_messages],
        }

        return json.dumps(conversation_dump)

    @classmethod
    def load(cls, *, dump: str) -> "Conversation":
        try:
            conversation_dump = json.loads(dump)
        except json.JSONDecodeError as error:
            raise AIError("Invalid conversation dump") from error

        conversation = cls()

        try:
            if conversation_dump["system"] is not None:
                conversation.set_system_message(
                    message=LLMSystemMessage.model_validate_json(conversation_dump["system"])
                )

            for user_message in conversation_dump["user"]:
                conversation.add_user_message(message=LLMUserMessage.model_validate_json(user_message))

            for assistant_message in conversation_dump["assistant"]:
                conversation.add_assistant_message(message=LLMAssistantMessage.model_validate_json(assistant_message))
        except ValidationError as error:
            raise AIError("Invalid conversation dump") from error

        return conversation
