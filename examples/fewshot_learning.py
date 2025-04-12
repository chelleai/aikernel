"""Example of using fewshot learning with LLMs.

This example demonstrates how to create fewshot examples 
and use them to guide the model's responses.
"""

from pydantic import BaseModel, Field

from aikernel import (
    LLMMessagePart,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_structured_sync,
)
from aikernel._internal.prompts.fewshot import FewshotExample, FewshotPrompt


# Define input and output types for our examples
class SentimentInput(BaseModel):
    text: str = Field(description="The text to analyze for sentiment")


class SentimentOutput(BaseModel):
    sentiment: str = Field(description="The sentiment of the text (positive, negative, or neutral)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    keywords: list[str] = Field(description="Key words that influenced the sentiment analysis")


def main():
    # Create a router with the model(s) we want to use
    router = get_router(models=("gemini-2.0-flash",))

    # Create system message for the prompt
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a sentiment analysis assistant. Given a piece of text, "
                "determine if the sentiment is positive, negative, or neutral."
            )
        ]
    )

    # Create fewshot examples to guide the model
    examples = [
        # Positive example
        FewshotExample(
            input=SentimentInput(
                text="I absolutely loved the new restaurant downtown. The food was amazing and the service was top-notch!"
            ),
            output=SentimentOutput(
                sentiment="positive",
                confidence=0.95,
                keywords=["loved", "amazing", "top-notch"]
            )
        ),
        # Negative example
        FewshotExample(
            input=SentimentInput(
                text="This movie was a complete waste of time. The plot made no sense and the acting was terrible."
            ),
            output=SentimentOutput(
                sentiment="negative",
                confidence=0.9,
                keywords=["waste of time", "no sense", "terrible"]
            )
        ),
        # Neutral example
        FewshotExample(
            input=SentimentInput(
                text="The conference starts at 9am and ends at 5pm. Lunch will be provided at noon."
            ),
            output=SentimentOutput(
                sentiment="neutral",
                confidence=0.85,
                keywords=["conference", "starts", "ends", "lunch"]
            )
        )
    ]

    # Create the fewshot prompt with our system message and examples
    fewshot_prompt = FewshotPrompt(
        system=system_message,
        examples=examples
    )

    # Render the prompt into messages
    prompt_messages = fewshot_prompt.render()

    # Add the actual query at the end
    user_query = LLMUserMessage(
        parts=[
            LLMMessagePart(
                content=SentimentInput(
                    text="I just finished the exam and I think I did okay, but I'm not sure if I passed all sections."
                ).model_dump_json()
            )
        ]
    )
    
    messages = prompt_messages + [user_query]

    # Get a structured response using the fewshot examples as guidance
    response = llm_structured_sync(
        messages=messages,
        router=router,
        response_model=SentimentOutput
    )

    # Access the structured output
    result = response.structured_response
    print("Text: I just finished the exam and I think I did okay, but I'm not sure if I passed all sections.")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Keywords: {', '.join(result.keywords)}")
    
    # Try another example without having to redefine everything
    another_query = LLMUserMessage(
        parts=[
            LLMMessagePart(
                content=SentimentInput(
                    text="We just won the championship! I can't believe it, we're the best team ever!"
                ).model_dump_json()
            )
        ]
    )
    
    messages = prompt_messages + [another_query]
    
    another_response = llm_structured_sync(
        messages=messages,
        router=router,
        response_model=SentimentOutput
    )
    
    another_result = another_response.structured_response
    print("\nText: We just won the championship! I can't believe it, we're the best team ever!")
    print(f"Sentiment: {another_result.sentiment}")
    print(f"Confidence: {another_result.confidence:.2f}")
    print(f"Keywords: {', '.join(another_result.keywords)}")


if __name__ == "__main__":
    main()
