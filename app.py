import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. DEFINE TOOLS
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


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""

    # --------------------------------------------------------
    # Step 1: Find city coordinates
    # --------------------------------------------------------

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find weather data for city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Step 2: Get weather information
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Step 3: Check weather response
        # ----------------------------------------------------

        if "current" not in weather_data:
            return f"Could not retrieve weather data for {city}"

        current_weather = weather_data["current"]

        # ----------------------------------------------------
        # Step 4: Create result
        # ----------------------------------------------------

        result = {
            "resolved_city": location["name"],
            "temperature_celsius": current_weather["temperature_2m"],
            "weather_code": current_weather["weather_code"]
        }

        return json.dumps(result)

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

    except Exception as e:
        return f"Could not retrieve weather data: {str(e)}"


# List of tools available to the AI agent
tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 2. INITIALIZE GEMINI MODEL
# ============================================================

# Get Gemini API key from environment variable
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
# 3. CREATE AI AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian "
        "weather and cinema. "

        "For weather questions, use the weather tool. "

        "For Indian movie questions, use the movie search tool. "

        "For Celsius to Fahrenheit conversion, use the temperature "
        "conversion tool. "

        "For any other roles, topics, questions, or general knowledge "
        "outside of Indian weather and movies, you must say exactly: "

        "'I am not authorized to answer questions outside of Indian "
        "weather and cinema.'"
    )
)


# ============================================================
# 4. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the AI agent"
    )


# ============================================================
# 5. FORMAT USER INPUT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 6. EXTRACT AI RESPONSE
# ============================================================

def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Check top-level messages
    messages = agent_output.get("messages")

    # Check nested messages
    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    # Return final message
    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            str(last_message)
        )

        return content

    return str(agent_output)


# ============================================================
# 7. CREATE LANGCHAIN CHAIN
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
# 8. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather & Cinema AI Agent",
    description="AI agent for Indian weather and cinema queries",
    version="1.0.0"
)


# Add LangServe route
add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 9. RUN SERVER
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
