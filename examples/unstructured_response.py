"""Example of getting unstructured (text) responses from an LLM.

This example demonstrates how to get a plain text response from an LLM.
"""

from aikernel import (
    LLMMessagePart,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_unstructured_sync,
)


def main():
    # Create a router with the model(s) we want to use
    router = get_router(models=("claude-3.7-sonnet",))

    # Create a system message that instructs the model
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a helpful assistant who provides concise answers."
            )
        ]
    )

    # Create a user message asking a question
    user_message = LLMUserMessage(
        parts=[
            LLMMessagePart(
                content="What are three key benefits of adopting a microservices architecture?"
            )
        ]
    )

    # Send the messages and get an unstructured (text) response
    response = llm_unstructured_sync(
        messages=[system_message, user_message],
        router=router,
    )

    # Print the text response
    print("Response:")
    print(response.text)
    
    # You can also access token usage statistics
    print(f"\nToken usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()