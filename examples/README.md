# AI Kernel Examples

This directory contains example usage patterns for the AI Kernel library, demonstrating its main capabilities.

## Running the Examples

These examples are designed to be run directly:

```bash
# Install the package first if you haven't already
pip install aikernel

# Run an example
python examples/structured_response.py
```

## Example Files

### Basics
- **[structured_response.py](structured_response.py)** - Demonstrates getting responses in a structured format using Pydantic models
- **[unstructured_response.py](unstructured_response.py)** - Shows how to get plain text responses from LLMs

### Advanced Features
- **[tool_calls.py](tool_calls.py)** - Shows how to define tools that LLMs can call and process the results
- **[conversation_management.py](conversation_management.py)** - Demonstrates managing conversation state, serialization, and deserialization
- **[fewshot_learning.py](fewshot_learning.py)** - Shows how to use few-shot examples to guide model responses
- **[model_routing_fallback.py](model_routing_fallback.py)** - Demonstrates working with multiple models and automatic fallback between them
- **[async_vs_sync.py](async_vs_sync.py)** - Compares synchronous and asynchronous APIs for improved performance

## Key Concepts

### Response Types
The library provides multiple ways to interact with LLMs:
- **Structured responses** - Get responses as typed Pydantic models
- **Unstructured responses** - Get plain text responses
- **Tool calls** - Let the model call functions with typed parameters

### Conversation Management
The `Conversation` class helps you manage the state of a conversation with an LLM, including:
- Adding system, user, assistant, and tool messages
- Rendering the conversation for sending to an LLM
- Serializing/deserializing conversations for persistence

### Model Routing
The router allows you to:
- Work with multiple models through a unified API
- Configure automatic fallbacks between models
- Choose between synchronous and asynchronous APIs