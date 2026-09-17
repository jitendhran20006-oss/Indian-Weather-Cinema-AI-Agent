import os
import uvicorn
import requests

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


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
    """Convert Celsius to Fahrenheit."""

    return temp_c * 1.8 + 32


# ============================================================
# WEATHER FUNCTION - OPEN-METEO
# ============================================================

def get_weather(city: str) -> str:
    """Get current weather using Open-Meteo."""

    try:

        # Find city coordinates
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
            timeout=15
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find the city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location["name"]

        # Get weather
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
            timeout=15
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data["current"]

        temperature = current["temperature_2m"]
        weather_code = current["weather_code"]

        return (
            f"The current temperature in {city_name} is "
            f"{temperature}°C. Weather code: {weather_code}."
        )

    except Exception as e:

        return f"Weather service error: {str(e)}"


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# AI AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        search_movies,
        change_to_f
    ],
    system_prompt=(
        "You are a specialized AI agent for Indian cinema. "
        "Answer Indian movie-related questions using the available tools. "
        "For questions outside Indian weather and cinema, say: "
        "'I am not authorized to answer questions outside of Indian weather and cinema.'"
    )
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the AI agent"
    )


# ============================================================
# MAIN ROUTER
# ============================================================

def process_request(x):

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    text = user_input.lower()

    # --------------------------------------------
    # WEATHER REQUEST
    # --------------------------------------------

    weather_words = [
        "weather",
        "temperature",
        "climate",
        "forecast"
    ]

    if any(word in text for word in weather_words):

        # Try to extract a city after "in"
        city = None

        if " in " in text:
            city = text.split(" in ")[-1].strip()

        if not city:
            return "Please specify a city. Example: What is the weather in Hyderabad?"

        return get_weather(city)


    # --------------------------------------------
    # MOVIE / OTHER REQUEST
    # --------------------------------------------

    try:

        result = agent.invoke({
            "messages": [
                ("user", user_input)
            ]
        })

        messages = result.get("messages", [])

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

        return str(result)

    except Exception as e:

        return f"AI agent error: {str(e)}"


# ============================================================
# LANGSERVE
# ============================================================

chain = RunnableLambda(
    process_request
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
    chain,
    path="/agent"
)


# ============================================================
# RUN
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
