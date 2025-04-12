"""AI Kernel - A collection of types and functions wrapping LiteLLM for better typing and use cases.

This package provides a set of utilities and functions to interact with large language models
through the LiteLLM library with improved type safety and better interfaces for different use cases.
It provides structured interfaces for sending messages, receiving responses, and handling tool calls.
"""

from aikernel._internal.conversation import Conversation
from aikernel._internal.router import LLMModelAlias, LLMRouter, get_router
from aikernel._internal.structured import llm_structured, llm_structured_sync
from aikernel._internal.tools import llm_tool_call, llm_tool_call_sync
from aikernel._internal.types.provider import (
    LiteLLMCacheControl,
    LiteLLMMediaMessagePart,
    LiteLLMMessage,
    LiteLLMTextMessagePart,
    LiteLLMTool,
    LiteLLMToolFunction,
)
from aikernel._internal.types.request import (
    LLMAssistantMessage,
    LLMMessageContentType,
    LLMMessagePart,
    LLMMessageRole,
    LLMSystemMessage,
    LLMTool,
    LLMToolMessage,
    LLMToolMessageFunctionCall,
    LLMUserMessage,
)
from aikernel._internal.types.response import (
    LLMAutoToolResponse,
    LLMRequiredToolResponse,
    LLMResponseToolCall,
    LLMResponseUsage,
    LLMStructuredResponse,
    LLMUnstructuredResponse,
)
from aikernel._internal.unstructured import llm_unstructured, llm_unstructured_sync

__all__ = [
    "llm_structured_sync",
    "llm_structured",
    "llm_tool_call_sync",
    "llm_tool_call",
    "llm_unstructured_sync",
    "llm_unstructured",
    "get_router",
    "Conversation",
    "LiteLLMCacheControl",
    "LiteLLMMediaMessagePart",
    "LiteLLMMessage",
    "LiteLLMTextMessagePart",
    "LiteLLMTool",
    "LiteLLMToolFunction",
    "LLMMessageContentType",
    "LLMMessagePart",
    "LLMMessageRole",
    "LLMModelAlias",
    "LLMUserMessage",
    "LLMAssistantMessage",
    "LLMSystemMessage",
    "LLMRouter",
    "LLMToolMessage",
    "LLMToolMessageFunctionCall",
    "LLMTool",
    "LLMStructuredResponse",
    "LLMUnstructuredResponse",
    "LLMAutoToolResponse",
    "LLMRequiredToolResponse",
    "LLMResponseToolCall",
    "LLMResponseUsage",
]
