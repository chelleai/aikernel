# AI Kernel

This is a collection of types and functions wrapping LiteLLM to give it stricter typing and better delineate use cases.

## Features

- **Type-safe LLM interactions**: Strongly-typed interfaces to LLM services through litellm with proper Pydantic models
- **Multiple response formats**:
  - Structured responses (JSON that conforms to Pydantic models)
  - Unstructured responses (plain text)
  - Tool calls (model can invoke defined tools with typed arguments)
- **Conversation management**:
  - Track message history with proper typing
  - Support for system/user/assistant/tool messages
  - Easy serialization/deserialization of conversations
- **Multi-model routing**:
  - Use different models (Claude, Gemini) through a unified interface
  - Built-in fallback between models
  - Both synchronous and asynchronous APIs
- **Few-shot learning patterns**:
  - Create examples with input/output pairs
  - Format them properly for the model

## Getting Started

Install the package:

```bash
pip install aikernel
```

### Basic Usage

```python
from pydantic import BaseModel
from aikernel import (
    llm_structured_sync, 
    llm_unstructured_sync, 
    LLMUserMessage, 
    LLMMessagePart, 
    get_router
)

# Create a router with the models you want to use
router = get_router(models=("claude-3.7-sonnet",))

# For unstructured (text) responses
messages = [
    LLMUserMessage(parts=[LLMMessagePart(content="What is the capital of France?")])
]
response = llm_unstructured_sync(messages=messages, router=router)
print(response.text)  # Paris

# For structured responses
class Capital(BaseModel):
    city: str
    country: str

messages = [
    LLMUserMessage(parts=[LLMMessagePart(content="What is the capital of France?")])
]
response = llm_structured_sync(
    messages=messages, 
    router=router, 
    response_model=Capital
)
print(response.structured_response.city)  # Paris
print(response.structured_response.country)  # France
```

### Tool Calls

```python
from pydantic import BaseModel
from aikernel import (
    llm_tool_call_sync, 
    LLMUserMessage, 
    LLMMessagePart, 
    LLMTool,
    get_router
)

class WeatherParams(BaseModel):
    location: str
    unit: str = "celsius"

weather_tool = LLMTool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters=WeatherParams
)

messages = [
    LLMUserMessage(parts=[LLMMessagePart(content="What's the weather in Paris?")])
]

router = get_router(models=("claude-3.7-sonnet",))
response = llm_tool_call_sync(
    messages=messages,
    model="claude-3.7-sonnet",
    tools=[weather_tool],
    tool_choice="auto",
    router=router
)

if response.tool_call:
    location = response.tool_call.arguments["location"]
    print(f"Getting weather for {location}")
else:
    print(response.text)  # Model chose to respond with text instead
```

## License

[MIT License](LICENSE)
