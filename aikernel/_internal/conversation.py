import json
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from aikernel._internal.errors import AIError
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessagePart,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
)


class Conversation:
    def __init__(self) -> None:
        self._user_messages: list[LLMUserMessage] = []
        self._assistant_messages: list[LLMAssistantMessage] = []
        self._tool_messages: list[LLMToolMessage] = []
        self._system_message: LLMSystemMessage | None = None

    def add_user_message(self, *, message: LLMUserMessage) -> None:
        self._user_messages.append(message)

    def add_assistant_message(self, *, message: LLMAssistantMessage) -> None:
        self._assistant_messages.append(message)

    def add_tool_message(self, *, tool_message: LLMToolMessage) -> None:
        self._tool_messages.append(tool_message)

    def set_system_message(self, *, message: LLMSystemMessage) -> None:
        self._system_message = message

    def render(self) -> list[LLMSystemMessage | LLMUserMessage | LLMAssistantMessage | LLMToolMessage]:
        messages = [self._system_message] if self._system_message is not None else []
        messages += sorted(
            self._user_messages + self._assistant_messages + self._tool_messages, key=lambda message: message.created_at
        )

        return messages

    @contextmanager
    def with_temporary_system_message(self, *, message_part: LLMMessagePart) -> Iterator[None]:
        if self._system_message is None:
            raise AIError("No system message to modify")

        self._system_message.parts.append(message_part)
        yield
        self._system_message.parts.pop()

    @contextmanager
    def session(self) -> Iterator[None]:
        num_user_messages = len(self._user_messages)
        num_assistant_messages = len(self._assistant_messages)
        num_tool_messages = len(self._tool_messages)

        try:
            yield
        except Exception:
            self._user_messages = self._user_messages[:num_user_messages]
            self._assistant_messages = self._assistant_messages[:num_assistant_messages]
            self._tool_messages = self._tool_messages[:num_tool_messages]
            raise

    def dump(self) -> str:
        conversation_dump = {
            "system": self._system_message.model_dump_json() if self._system_message is not None else None,
            "user": [message.model_dump_json() for message in self._user_messages],
            "assistant": [message.model_dump_json() for message in self._assistant_messages],
            "tool": [message.model_dump_json() for message in self._tool_messages],
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

            for tool_message in conversation_dump["tool"]:
                conversation.add_tool_message(tool_message=LLMToolMessage.model_validate_json(tool_message))
        except ValidationError as error:
            raise AIError("Invalid conversation dump") from error

        return conversation
