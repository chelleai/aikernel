"""Conversation management for LLM interactions.

This module provides a Conversation class that manages the state of a 
conversation with an LLM, including system, user, assistant, and tool messages.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessagePart,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
)
from aikernel.errors import InvalidConversationDumpError


class Conversation:
    """Manages a conversation between a user and an LLM.
    
    This class stores and manages the messages in a conversation, including
    system, user, assistant, and tool messages. It provides methods for adding
    messages, rendering the conversation for sending to an LLM, and serializing
    and deserializing conversations.
    """
    def __init__(self) -> None:
        """Initialize an empty conversation."""
        self._user_messages: list[LLMUserMessage] = []
        self._assistant_messages: list[LLMAssistantMessage] = []
        self._tool_messages: list[LLMToolMessage] = []
        self._system_message: LLMSystemMessage | None = None

    @property
    def user_messages(self) -> list[LLMUserMessage]:
        """Get all user messages in the conversation."""
        return self._user_messages

    @property
    def assistant_messages(self) -> list[LLMAssistantMessage]:
        """Get all assistant messages in the conversation."""
        return self._assistant_messages

    @property
    def tool_messages(self) -> list[LLMToolMessage]:
        """Get all tool messages in the conversation."""
        return self._tool_messages

    @property
    def system_message(self) -> LLMSystemMessage | None:
        """Get the system message for the conversation, if any."""
        return self._system_message

    def add_user_message(self, *, message: LLMUserMessage) -> None:
        """Add a user message to the conversation.
        
        Args:
            message: The user message to add
        """
        self._user_messages.append(message)

    def add_assistant_message(self, *, message: LLMAssistantMessage) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            message: The assistant message to add
        """
        self._assistant_messages.append(message)

    def add_tool_message(self, *, tool_message: LLMToolMessage) -> None:
        """Add a tool message to the conversation.
        
        Args:
            tool_message: The tool message to add
        """
        self._tool_messages.append(tool_message)

    def set_system_message(self, *, message: LLMSystemMessage) -> None:
        """Set the system message for the conversation.
        
        Args:
            message: The system message to set
        """
        self._system_message = message

    def render(self) -> list[LLMSystemMessage | LLMUserMessage | LLMAssistantMessage | LLMToolMessage]:
        """Render the conversation as a list of messages.
        
        This method returns all messages in the conversation, sorted by creation time.
        The system message, if present, is always first.
        
        Returns:
            A list of all messages in the conversation
        """
        messages = [self._system_message] if self._system_message is not None else []
        messages += sorted(
            self._user_messages + self._assistant_messages + self._tool_messages, key=lambda message: message.created_at
        )

        return messages

    @contextmanager
    def with_temporary_system_message(self, *, message_part: LLMMessagePart) -> Iterator[None]:
        """Temporarily add a message part to the system message.
        
        This context manager adds a message part to the system message for the duration
        of the context, and then removes it when the context exits.
        
        Args:
            message_part: The message part to temporarily add
            
        Yields:
            None
            
        Raises:
            ValueError: If there is no system message
        """
        if self._system_message is None:
            raise ValueError("No system message to modify")

        self._system_message.parts.append(message_part)
        yield
        self._system_message.parts.pop()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Create a session that can be rolled back on exception.
        
        This context manager keeps track of the number of messages at the start
        of the session, and if an exception occurs, it rolls back to that state.
        
        Yields:
            None
        """
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
        """Serialize the conversation to a JSON string.
        
        Returns:
            A JSON string representation of the conversation
        """
        conversation_dump = {
            "system": self._system_message.model_dump() if self._system_message is not None else None,
            "user": [message.model_dump() for message in self._user_messages],
            "assistant": [message.model_dump() for message in self._assistant_messages],
            "tool": [message.model_dump() for message in self._tool_messages],
        }

        return json.dumps(conversation_dump, default=str)

    @classmethod
    def load(cls, *, dump: str) -> "Conversation":
        """Deserialize a conversation from a JSON string.
        
        Args:
            dump: A JSON string representation of a conversation
            
        Returns:
            The deserialized conversation
            
        Raises:
            InvalidConversationDumpError: If the JSON is invalid or doesn't
                                          represent a valid conversation
        """
        try:
            conversation_dump = json.loads(dump)
        except json.JSONDecodeError as error:
            raise InvalidConversationDumpError() from error

        conversation = cls()

        try:
            if conversation_dump["system"] is not None:
                conversation.set_system_message(message=LLMSystemMessage.model_validate(conversation_dump["system"]))

            for user_message in conversation_dump["user"]:
                conversation.add_user_message(message=LLMUserMessage.model_validate(user_message))

            for assistant_message in conversation_dump["assistant"]:
                conversation.add_assistant_message(message=LLMAssistantMessage.model_validate(assistant_message))

            for tool_message in conversation_dump["tool"]:
                conversation.add_tool_message(tool_message=LLMToolMessage.model_validate(tool_message))
        except ValidationError as error:
            raise InvalidConversationDumpError() from error

        return conversation
