"""Example of using tool calls with an LLM.

This example demonstrates how to define tools that the LLM can call,
and how to process the tool calls made by the LLM.
"""

from pydantic import BaseModel, Field

from aikernel import (
    LLMMessagePart,
    LLMSystemMessage,
    LLMTool,
    LLMUserMessage,
    get_router,
    llm_tool_call_sync,
)


# Define the parameters for each tool
class WeatherParams(BaseModel):
    location: str = Field(description="The city and country to get weather for")
    unit: str = Field(default="celsius", description="The temperature unit (celsius or fahrenheit)")


class RestaurantParams(BaseModel):
    cuisine: str = Field(description="The type of cuisine")
    location: str = Field(description="The city to find restaurants in")
    price_range: str = Field(default="moderate", description="The price range (budget, moderate, expensive)")


def main():
    # Create a router with the model(s) we want to use
    router = get_router(models=("gemini-2.0-flash",))

    # Define the tools that the LLM can use
    weather_tool = LLMTool(
        name="get_weather",
        description="Get the current weather for a specific location",
        parameters=WeatherParams,
    )

    restaurant_tool = LLMTool(
        name="find_restaurants",
        description="Find restaurants of a specific cuisine in a location",
        parameters=RestaurantParams,
    )

    # Create messages for the conversation
    system_message = LLMSystemMessage(
        parts=[
            LLMMessagePart(
                content="You are a helpful assistant that provides information about weather and restaurants."
            )
        ]
    )

    # Example 1: User query that likely needs the weather tool
    user_message1 = LLMUserMessage(
        parts=[LLMMessagePart(content="What's the weather like in Tokyo today?")]
    )

    # Send the messages with the tools, allowing the model to decide if it wants to use a tool
    print("Example 1: Weather query")
    response1 = llm_tool_call_sync(
        messages=[system_message, user_message1],
        model="gemini-2.0-flash",
        tools=[weather_tool, restaurant_tool],
        tool_choice="auto",  # Let the model decide whether to use a tool
        router=router,
    )

    # Check if the model decided to call a tool
    if response1.tool_call:
        print(f"Tool called: {response1.tool_call.tool_name}")
        print(f"Arguments: {response1.tool_call.arguments}")
        
        # In a real application, you would now call your actual weather service
        # with the parameters provided by the model
        if response1.tool_call.tool_name == "get_weather":
            location = response1.tool_call.arguments["location"]
            unit = response1.tool_call.arguments.get("unit", "celsius")
            print(f"Would now fetch weather for {location} in {unit}")
    else:
        print(f"Model chose to respond with text: {response1.text}")

    # Example 2: User query that likely needs the restaurant tool
    user_message2 = LLMUserMessage(
        parts=[LLMMessagePart(content="Can you suggest some Italian restaurants in New York?")]
    )

    print("\nExample 2: Restaurant query")
    response2 = llm_tool_call_sync(
        messages=[system_message, user_message2],
        model="gemini-2.0-flash",
        tools=[weather_tool, restaurant_tool],
        tool_choice="auto",
        router=router,
    )

    # Check if the model decided to call a tool
    if response2.tool_call:
        print(f"Tool called: {response2.tool_call.tool_name}")
        print(f"Arguments: {response2.tool_call.arguments}")
        
        # In a real application, you would now call your restaurant finder service
        if response2.tool_call.tool_name == "find_restaurants":
            cuisine = response2.tool_call.arguments["cuisine"]
            location = response2.tool_call.arguments["location"]
            price_range = response2.tool_call.arguments.get("price_range", "moderate")
            print(f"Would now search for {price_range} {cuisine} restaurants in {location}")
    else:
        print(f"Model chose to respond with text: {response2.text}")
    
    # Example 3: Require the model to call a tool
    user_message3 = LLMUserMessage(
        parts=[LLMMessagePart(content="How's the weather in Paris?")]
    )
    
    print("\nExample 3: Requiring a tool call")
    response3 = llm_tool_call_sync(
        messages=[system_message, user_message3],
        model="gemini-2.0-flash",
        tools=[weather_tool, restaurant_tool],
        tool_choice="required",  # Require the model to call a tool
        router=router,
    )
    
    print(f"Tool called: {response3.tool_call.tool_name}")
    print(f"Arguments: {response3.tool_call.arguments}")


if __name__ == "__main__":
    main()
