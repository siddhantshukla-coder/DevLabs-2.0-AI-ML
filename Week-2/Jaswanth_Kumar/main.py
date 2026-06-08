import datetime
import json
import requests

from pydantic import BaseModel

from google import genai

#Query = "Give me News on Topic Hollywood Movies"
#Query = "Give me News on Topic Technology on Date 04-06-2020"
Query = "Give me News about India on Today"

client = genai.Client(api_key="API_KEY")

API_KEY_NEWS = ""

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"{Query}",
        config={
            "system_instruction": """
            Extract topic info and return ONLY a JSON object with these fields:
            topic (str), date (object).
            Return raw JSON, no markdown, no explanation.
            """
        }
    )
    print(response.text)
except Exception as e:
    print("Credits are Completed")



class NewsInput(BaseModel):
    topic: str
    date: str = f"{datetime.date}"

raw_json = response.text.strip()

if raw_json.startswith("```"):
    raw_json = raw_json.replace("```json", "")
    raw_json = raw_json.replace("```", "")
    raw_json = raw_json.strip()

data = json.loads(raw_json)

print(data)
print(type(data))

print(data["topic"])
print(data["date"])
print(type(data["date"]))

def get_news(topic:str,date:str) -> str:
    response = requests.get(f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY_NEWS}")
    response.raise_for_status()

    data = response.json()
    print(len(data))
    print(data['articles'][0]["description"])

get_news(data["topic"],data["date"])
