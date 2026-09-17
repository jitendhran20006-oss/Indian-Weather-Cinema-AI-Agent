import os
import uvicorn
import requests

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# MOVIE TOOL
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


# ============================================================
# TEMPERATURE CONVERSION TOOL
# ============================================================

@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


# ============================================================
# WEATHER TOOL - OPEN-METEO
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using Open-Meteo."""

    try:
        # Step 1: Find city coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find the city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location["name"]

        # Step 2: Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data["current"]

        temperature = current["temperature_2m"]
        weather_code = current["weather_code"]

        return (
            f"The current temperature in {city_name} is "
            f"{temperature}°C. "
            f"Weather code: {weather_code}."
        )

    except requests.exceptions.RequestException as e:
        return f"Weather service error: {str(e)}"

    except Exception as e:
        return f"Weather error: {str(e)}"


# ============================================================
# TOOLS
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# GEMINI API
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# AI AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized AI agent restricted to Indian weather "
        "and Indian cinema-related questions. "
        "Use the available tools when appropriate. "
        "For questions outside Indian weather and cinema, say: "
        "'I am not authorized to answer questions outside of Indian "
        "weather and cinema.'"
    )
)


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# FORMAT INPUT
# ============================================================

def format_for_agent(x):
    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# EXTRACT RESPONSE
# ============================================================

def extract_text_response(agent_output):
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            str(last_message)
        )

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))

            return "".join(text_parts)

        return str(content)

    return str(agent_output)


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Indian Weather & Cinema AI Agent",
    description="AI agent for Indian weather and cinema queries",
    version="1.0.0"
)


add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
