"""Example of using multiple models with routing and fallback.

This example demonstrates how to set up a router with multiple models
and how the router handles fallbacks between models.
"""

import asyncio
from aikernel import (
    LLMModelAlias,
    LLMMessagePart,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_unstructured_sync,
    llm_unstructured,
)


def perform_sync_request(router, messages, model_name):
    """Make a synchronous request to a specific model."""
    print(f"\nAttempting synchronous request with {model_name}...")
    try:
        # Note: In a real scenario, if the primary model fails, the router
        # will automatically fall back to the secondary model
        response = llm_unstructured_sync(messages=messages, router=router)
        print(f"Response from {router.primary_model}:")
        print(response.text[:300] + "..." if len(response.text) > 300 else response.text)
        print(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
    except Exception as e:
        print(f"Error: {e}")


async def perform_async_request(router, messages):
    """Make an asynchronous request."""
    print("\nPerforming asynchronous request...")
    try:
        response = await llm_unstructured(messages=messages, router=router)
        print(f"Response from {router.primary_model}:")
        print(response.text[:300] + "..." if len(response.text) > 300 else response.text)
        print(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


async def main():
    # Create messages for our request
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a helpful assistant that provides concise, accurate information."
            )
        ]
    )
    user_message = LLMUserMessage(
        parts=[
            LLMMessagePart(
                content="Explain the concept of quantum entanglement in simple terms."
            )
        ]
    )
    messages = [system_message, user_message]
    
    # Create a router with multiple models to demonstrate fallback
    # The models are tried in the order provided
    router1 = get_router(models=("gemini-2.0-flash", "gemini-2.0-flash-lite"))
    print(f"Router 1 primary model: {router1.primary_model}")
    
    # Make a synchronous request with the first router
    perform_sync_request(router1, messages, "claude-3.7-sonnet")
    
    # Create a router with models in a different order
    router2 = get_router(models=("gemini-2.0-flash-lite", "gemini-2.0-flash"))
    print(f"\nRouter 2 primary model: {router2.primary_model}")
    
    # Make a synchronous request with the second router
    perform_sync_request(router2, messages, "gemini-2.0-flash")
    
    # Make an asynchronous request
    await perform_async_request(router1, messages)
    
    # Example of parallel async requests with different routers
    print("\nPerforming parallel async requests...")
    responses = await asyncio.gather(
        perform_async_request(router1, messages),
        perform_async_request(router2, messages)
    )
    
    # Both requests should have succeeded, potentially using different models
    if all(responses):
        print("\nBoth parallel requests completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
