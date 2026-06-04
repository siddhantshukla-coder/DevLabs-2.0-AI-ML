# Week 2 — Tool Use & Function Calling

**Dates:** Jun 2 – Jun 8

---

## What We're Building This Week

Last week you built an agent with 3 fake tools and hardcoded data.

This week you'll understand **how tool calling actually works under the hood**, wire up **real APIs**, use **typed inputs with Pydantic** so your tools don't break silently, and build an agent that calls tools in the right order based on what the user asks.

By the end of this week, you'll understand the thing that separates a toy agent from a real one.

---

## 1. How Tool Calling Actually Works

Here's the thing most tutorials skip: **the LLM doesn't actually call your function.**

You do.

Here's the real flow:

```
You → "What's the weather in Mumbai?" → Claude

Claude → thinks → "I should call get_weather. Here's the input:"
        {
          "tool": "get_weather",
          "input": { "city": "Mumbai" }
        }

You → see that Claude wants to call a tool
You → actually run get_weather("Mumbai")
You → send the result back to Claude

Claude → reads the result → responds to the user
```

The model never runs any code. It just says "I want to call X with these inputs." Your code executes X and passes the result back.

This is why the ReAct loop from Week 1 has a `while True` — you keep looping until the model stops asking for tool calls.

---

## 2. Why Pydantic?

Last week your tools looked like this:

```python
def get_product_price(product_name: str) -> str:
    ...
```

That's fine. But what happens when Claude sends:

```json
{ "product_name": 42 }
```

Or:

```json
{ "productName": "iPhone" }
```

Your function silently gets the wrong input. It fails with a cryptic error, or worse, returns wrong data.

**Pydantic** solves this. It's a library that:
- Validates input types at runtime (not just at type-check time)
- Coerces values where possible (e.g. `"42"` → `42` if you asked for an int)
- Throws a clear error with the exact field and reason when something is wrong

Think of it as a bouncer for your function's inputs.

```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    unit: str = "celsius"   # optional with a default

# This works
WeatherInput(city="Mumbai")

# This also works — pydantic coerces
WeatherInput(city="Mumbai", unit="fahrenheit")

# This FAILS with a clear error
WeatherInput(city=123)  # city must be a string
```

---

## 3. Structured Outputs — Getting JSON Back From Claude

Sometimes you don't want a conversational response. You want **structured data**.

Example: you're extracting product info from a review and need it as a Python dict with guaranteed fields.

Without structured output:
```
"The product is an iPhone 15, costs around ₹80,000 and has 4.5 stars."
```

With structured output:
```json
{
  "product": "iPhone 15",
  "price": 80000,
  "rating": 4.5
}
```

How? You define the shape with Pydantic and tell Claude to match it:

```python
from pydantic import BaseModel
import anthropic
import json

class ProductReview(BaseModel):
    product: str
    price: int
    rating: float
    sentiment: str   # "positive", "negative", "neutral"

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    system="""Extract product info and return ONLY a JSON object with these fields:
    product (str), price (int in INR), rating (float 0-5), sentiment (str).
    Return raw JSON, no markdown, no explanation.""",
    messages=[{
        "role": "user",
        "content": "I bought the iPhone 15 for about 80k. It's amazing, easily 4.5/5."
    }]
)

raw_json = response.content[0].text
data = ProductReview.model_validate_json(raw_json)  # validates AND parses
print(data.product)    # "iPhone 15"
print(data.price)      # 80000
print(data.rating)     # 4.5
```

`model_validate_json` will throw a clear `ValidationError` if Claude returned garbage. You catch that and retry or log it.

---

## 4. Connecting to Real APIs

Last week: hardcoded dictionaries.
This week: real HTTP calls.

Here's the pattern. We'll use [wttr.in](https://wttr.in) — a free weather API, no key needed:

```python
import requests
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    unit: str = "celsius"

def get_weather(city: str, unit: str = "celsius") -> str:
    """Fetch real weather data from wttr.in."""
    fmt = "j1"  # JSON format
    response = requests.get(f"https://wttr.in/{city}?format={fmt}", timeout=5)
    response.raise_for_status()

    data = response.json()
    current = data["current_condition"][0]
    temp = current["temp_C"] if unit == "celsius" else current["temp_F"]
    desc = current["weatherDesc"][0]["value"]
    return f"{temp}°{'C' if unit == 'celsius' else 'F'}, {desc} in {city}"
```

Key things to notice:
- `timeout=5` — never make an HTTP call without a timeout. Your agent will hang forever otherwise.
- `raise_for_status()` — throws an exception if the API returns 4xx/5xx. Catch this in your tool and return a useful error string to the agent.
- The function still returns a `str` — that's what you send back to Claude as the tool result.

---

## 5. Typed Tool Registry — The Right Way to Build This

In Week 1, tools were scattered — the function here, the definition there. Let's clean that up.

Here's a pattern that keeps everything in one place and uses Pydantic for validation:

```python
from pydantic import BaseModel
from typing import Callable, Any
import anthropic

# 1. Input schema as a Pydantic model
class WeatherInput(BaseModel):
    city: str
    unit: str = "celsius"

class NewsInput(BaseModel):
    topic: str
    max_results: int = 3

# 2. One class per tool
class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        func: Callable[..., str],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.func = func

    def run(self, raw_input: dict[str, Any]) -> str:
        # Pydantic validates here — bad input = clear error, not silent failure
        validated = self.input_model(**raw_input)
        return self.func(**validated.model_dump())

    def to_claude_definition(self) -> dict:
        # Convert Pydantic model → Claude tool definition automatically
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }
```

Now you define tools like this:

```python
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    input_model=WeatherInput,
    func=get_weather,   # your actual function
)

news_tool = Tool(
    name="get_news",
    description="Get recent news headlines on a topic.",
    input_model=NewsInput,
    func=get_news,
)

TOOLS: list[Tool] = [weather_tool, news_tool]
TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}
```

And the agent loop becomes clean:

```python
tool_definitions = [t.to_claude_definition() for t in TOOLS]

# In the ReAct loop:
for block in response.content:
    if block.type == "tool_use":
        tool = TOOL_MAP[block.name]
        result = tool.run(block.input)   # Pydantic validates here
```

---

## 6. Full Example — Research Agent With Real APIs

```python
"""
Week 2 — Research Agent
Uses 2 real APIs: wttr.in (weather) + DuckDuckGo search (no key needed)
"""

import requests
from pydantic import BaseModel
from typing import Callable, Any
import anthropic


# ── Tool inputs ──────────────────────────────────────────────

class WeatherInput(BaseModel):
    city: str

class SearchInput(BaseModel):
    query: str
    max_results: int = 3


# ── Real API functions ────────────────────────────────────────

def get_weather(city: str) -> str:
    try:
        res = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        res.raise_for_status()
        return res.text.strip()
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"


def search_web(query: str, max_results: int = 3) -> str:
    try:
        res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=5,
        )
        res.raise_for_status()
        data = res.json()
        results = data.get("RelatedTopics", [])[:max_results]
        if not results:
            return "No results found."
        lines = [r.get("Text", "") for r in results if "Text" in r]
        return "\n".join(f"• {line}" for line in lines)
    except requests.RequestException as e:
        return f"Search failed: {e}"


# ── Tool class ────────────────────────────────────────────────

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        func: Callable[..., str],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.func = func

    def run(self, raw_input: dict[str, Any]) -> str:
        validated = self.input_model(**raw_input)
        return self.func(**validated.model_dump())

    def to_claude_definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


# ── Registry ──────────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool("get_weather", "Get current weather for a city.", WeatherInput, get_weather),
    Tool("search_web", "Search the web for a query.", SearchInput, search_web),
]
TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


# ── Agent ─────────────────────────────────────────────────────

class ResearchAgent:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.tool_definitions = [t.to_claude_definition() for t in TOOLS]

    def run(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        print(f"\nUser: {user_message}")

        while True:
            response = self.client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                system="You are a research assistant. Use tools to find real information.",
                tools=self.tool_definitions,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return block.text

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → {block.name}({block.input})")
                    result = TOOL_MAP[block.name].run(block.input)
                    print(f"  ← {result[:80]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    agent = ResearchAgent()

    queries = [
        "What's the weather in Delhi and Mumbai right now?",
        "Search for recent AI safety news and summarise what you find.",
    ]

    for q in queries:
        print(agent.run(q))
        print("-" * 60)
```

---

## 7. When Does Claude Call a Tool vs Just Reply?

This is one of the most important things to understand.

Claude uses your **tool description** to decide. Vague descriptions → wrong decisions.

| Bad description | Good description |
|---|---|
| `"Get info"` | `"Fetch real-time stock price for a ticker symbol from Yahoo Finance"` |
| `"Search"` | `"Search the web for recent news articles on a topic. Use this when the user asks about current events or recent information."` |
| `"Weather"` | `"Get current temperature and conditions for a city. Always use this tool — do not guess weather."` |

**Always use this when** in your description is a powerful phrase. It tells Claude: don't try to answer from memory, use this tool.

---

## Resources

### Course
- [DeepLearning.AI – AI Agents in LangGraph](https://deeplearning.ai/short-courses/ai-agents-in-langgraph) ← Weeks 2–3 content here

### Docs
- [Anthropic Tool Use Docs](https://docs.anthropic.com/en/docs/tool-use) ← read the "How it works" section
- [Pydantic Docs – BaseModel](https://docs.pydantic.dev/latest/concepts/models/)

### Videos
- [Prompt Engineering – Function Calling Crash Course](https://youtube.com/@engineerprompt)
- [AI Anytime – Build an Agent with Tools from Scratch](https://youtube.com/@AIAnytime)

### APIs to try (all free, no credit card)
- [wttr.in](https://wttr.in) — weather, no key
- [DuckDuckGo Instant Answer API](https://api.duckduckgo.com/?q=test&format=json) — search, no key
- [Tavily](https://tavily.com) — better search, free tier
- [Groq](https://groq.com) — free, fast LLM inference (alternative to Anthropic)

---

## Week 2 Deliverable

**Harder than Week 1.**

Build a **typed research or productivity agent** that:

1. Uses **2 real APIs** (not hardcoded data)
2. All tool inputs validated with **Pydantic BaseModel**
3. Uses the **Tool class pattern** from Section 5 (not scattered functions)
4. Handles **API errors gracefully** — tool should return an error string, not crash
5. Add **type hints everywhere** — function args, return types, class attributes

**Bonus challenge:** Add a third tool that chains off the first two. Example: search for a city → get weather for it → summarise both.

Submit in `submissions/your-name/` with a README showing 3 sample runs and their outputs.
