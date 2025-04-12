"""Example of managing conversations with the Conversation class.

This example demonstrates how to manage an ongoing conversation,
including adding messages, serializing, and deserializing.
"""

from aikernel import (
    Conversation,
    LLMAssistantMessage,
    LLMMessagePart,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_unstructured_sync,
)


def main():
    # Create a router with the model(s) we want to use
    router = get_router(models=("claude-3.7-sonnet",))

    # Create a new conversation
    conversation = Conversation()

    # Set the system message
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a helpful math tutor who explains concepts clearly and concisely."
            )
        ]
    )
    conversation.set_system_message(message=system_message)

    # Add the first user message
    user_message1 = LLMUserMessage(
        parts=[LLMMessagePart(content="Can you explain what a derivative is in calculus?")]
    )
    conversation.add_user_message(message=user_message1)

    # Get response from the LLM
    print("User: Can you explain what a derivative is in calculus?")
    response1 = llm_unstructured_sync(
        messages=conversation.render(),
        router=router,
    )
    
    # Add the assistant's response to the conversation
    assistant_message1 = LLMAssistantMessage(
        parts=[LLMMessagePart(content=response1.text)]
    )
    conversation.add_assistant_message(message=assistant_message1)
    
    print(f"Assistant: {response1.text}\n")

    # Continue the conversation with a follow-up question
    user_message2 = LLMUserMessage(
        parts=[LLMMessagePart(content="Can you give me a simple example of finding a derivative?")]
    )
    conversation.add_user_message(message=user_message2)

    print("User: Can you give me a simple example of finding a derivative?")
    response2 = llm_unstructured_sync(
        messages=conversation.render(),
        router=router,
    )
    
    # Add the assistant's response to the conversation
    assistant_message2 = LLMAssistantMessage(
        parts=[LLMMessagePart(content=response2.text)]
    )
    conversation.add_assistant_message(message=assistant_message2)
    
    print(f"Assistant: {response2.text}\n")

    # Serialize the conversation to save it
    serialized = conversation.dump()
    print("Conversation serialized to JSON:")
    print(serialized[:300] + "..." if len(serialized) > 300 else serialized)
    
    print("\nDeserializing conversation...")
    # In a real application, you would typically save the serialized conversation
    # and load it later when needed
    loaded_conversation = Conversation.load(dump=serialized)
    
    # Add another message to the loaded conversation
    user_message3 = LLMUserMessage(
        parts=[LLMMessagePart(content="What's the derivative of f(x) = x²?")]
    )
    loaded_conversation.add_user_message(message=user_message3)
    
    print("User: What's the derivative of f(x) = x²?")
    response3 = llm_unstructured_sync(
        messages=loaded_conversation.render(),
        router=router,
    )
    
    print(f"Assistant: {response3.text}")


if __name__ == "__main__":
    main()