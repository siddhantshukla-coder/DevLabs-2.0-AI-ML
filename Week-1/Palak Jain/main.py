from smolagents import (CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool)
import requests

# Tool 1: Tourists attractions search
search_tool = DuckDuckGoSearchTool()


# Tool 2: Weather check krega
@tool
def get_weather(city: str) -> str:
    """
    This tool gets current weather.
    Args:
        city: Name of city.
    """
    url = f"https://wttr.in/{city}?format=3"
    return requests.get(url).text


# Tool 3:This tool will check the weather first, then suggest the clothes which are comfortable for that weather condition
@tool
def packing_advisor(weather_condition: str) -> str:
    """
    This tool suggest which items to pack.
    Args:
        weather_condition: Weather description.
    """
    weather_condition = weather_condition.lower()

    if "rain" in weather_condition:
        return "Carry umbrella and raincoat."
    if "cold" in weather_condition:
        return "Carry warm clothes and jacket."
    if "hot" in weather_condition:
        return "Carry cap and light clothes."
    return "Carry comfortable clothes and walking shoes."

# Model
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key="token",
    max_tokens=2096,
    temperature=0.5,
)

agent = CodeAgent(
    model=model,
    tools=[
        search_tool,
        get_weather,
        packing_advisor
    ],
    max_steps=3,
    verbosity_level=0,
)

query = input("\nYou: ")
answer = agent.run(query)
print(answer)
