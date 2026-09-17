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
# 1. MOVIE TOOL
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Jab We Met, Veer-Zaara, Sita Ramam",
        "thriller": "Drishyam, Andhadhun, Ratsasan"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


# ============================================================
# 2. TEMPERATURE CONVERSION TOOL
# ============================================================

@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


# ============================================================
# 3. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""

    weather_api_key = os.environ.get("WEATHER_API_KEY")

    if not weather_api_key:
        return "Weather API key is not configured."

    weather_url = "https://api.weatherapi.com/v1/current.json"

    weather_params = {
        "key": weather_api_key,
        "q": city,
        "aqi": "no"
    }

    try:

        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        response.raise_for_status()

        weather_data = response.json()

        if "current" not in weather_data:
            return f"Could not retrieve weather data for {city}"

        current = weather_data["current"]

        result = {
            "resolved_city": weather_data["location"]["name"],
            "country": weather_data["location"]["country"],
            "temperature_celsius": current["temp_c"],
            "condition": current["condition"]["text"],
            "humidity": current["humidity"],
            "wind_kph": current["wind_kph"]
        }

        return json.dumps(result)

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

    except Exception as e:
        return f"Could not retrieve weather data: {str(e)}"


# ============================================================
# 4. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 5. GEMINI MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 6. AI AGENT
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are an Indian Weather and Cinema AI Agent. "

        "You can answer questions about Indian weather and Indian movies. "

        "For weather questions, use the get_weather tool. "

        "For movie questions, use the search_movies tool. "

        "For Celsius to Fahrenheit conversion, use the change_to_f tool. "

        "Do not answer unrelated general knowledge questions. "

        "For questions outside Indian weather and cinema, say exactly: "

        "'I am not authorized to answer questions outside of Indian "
        "weather and cinema.'"
    )
)


# ============================================================
# 7. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the AI agent"
    )


# ============================================================
# 8. PROCESS USER REQUEST
# ============================================================

def process_request(x):

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    text = user_input.lower().strip()

    # --------------------------------------------------------
    # WEATHER REQUEST
    # --------------------------------------------------------

    weather_words = [
        "weather",
        "temperature",
        "forecast",
        "climate"
    ]

    if any(word in text for word in weather_words):

        # Try to identify the city from common phrases

        city = None

        phrases = [
            "weather in ",
            "weather of ",
            "temperature in ",
            "temperature of ",
            "forecast in ",
            "forecast of "
        ]

        for phrase in phrases:

            if phrase in text:

                city = text.split(phrase, 1)[1].strip()

                # Remove question marks
                city = city.replace("?", "").strip()

                break

        # If city was not detected
        if not city:

            return {
                "messages": [
                    (
                        "assistant",
                        "Please specify the city you want the weather for."
                    )
                ]
            }

        # Call weather tool directly
        weather_result = get_weather.invoke(city)

        # ----------------------------------------------------
        # Convert JSON weather result into readable response
        # ----------------------------------------------------

        try:

            weather_data = json.loads(weather_result)

            city_name = weather_data.get(
                "resolved_city",
                city.title()
            )

            temperature = weather_data.get(
                "temperature_celsius",
                "N/A"
            )

            condition = weather_data.get(
                "condition",
                "N/A"
            )

            humidity = weather_data.get(
                "humidity",
                "N/A"
            )

            wind = weather_data.get(
                "wind_kph",
                "N/A"
            )

            answer = (
                f"The current weather in {city_name} is "
                f"{temperature}°C with {condition}. "
                f"Humidity is {humidity}% and wind speed is "
                f"{wind} km/h."
            )

            return {
                "messages": [
                    ("assistant", answer)
                ]
            }

        except Exception:

            return {
                "messages": [
                    ("assistant", weather_result)
                ]
            }

    # --------------------------------------------------------
    # OTHER REQUESTS → AI AGENT
    # --------------------------------------------------------

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 9. EXTRACT FINAL RESPONSE
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

        return content

    return str(agent_output)


# ============================================================
# 10. CREATE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(process_request)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 11. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather & Cinema AI Agent",
    description="AI agent for Indian weather and cinema queries",
    version="1.0.0"
)


# ============================================================
# 12. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 13. RUN SERVER
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
