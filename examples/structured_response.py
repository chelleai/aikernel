"""Example of getting structured responses from an LLM.

This example demonstrates how to get a response from an LLM
that conforms to a specific Pydantic model structure.
"""

from pydantic import BaseModel, Field

from aikernel import (
    LLMMessagePart,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_structured_sync,
)


# Define the structure we want the LLM to return
class Book(BaseModel):
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    year: int = Field(description="The year the book was published")
    genre: str = Field(description="The genre of the book")
    summary: str = Field(description="A brief summary of the book's plot")


def main():
    # Create a router with the model(s) we want to use
    router = get_router(models=("gemini-2.0-flash",))

    # Create a system message that instructs the model
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a literature expert who provides information about books."
            )
        ]
    )

    # Create a user message asking for information
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content="Tell me about the book '1984' by George Orwell.")]
    )

    # Send the messages and request a structured response
    response = llm_structured_sync(
        messages=[system_message, user_message],
        router=router,
        response_model=Book,
    )

    # Access the structured data
    book = response.structured_response
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"Published: {book.year}")
    print(f"Genre: {book.genre}")
    print(f"Summary: {book.summary}")
    
    # You can also access token usage statistics
    print(f"\nToken usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
