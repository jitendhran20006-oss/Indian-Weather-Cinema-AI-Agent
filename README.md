# Indian Weather & Cinema AI Agent

An AI-powered agent built with Python, LangChain, Gemini, and FastAPI.
The agent is designed to handle Indian weather and cinema-related
queries using custom tools and external APIs.

## 🚀 Features

- 🌤️ Get current weather information for cities
- 🎬 Search Indian movies by genre
- 🌡️ Convert Celsius to Fahrenheit
- 🤖 AI agent with tool-calling capabilities
- 🔌 FastAPI API endpoint
- 🌐 Weather data using Open-Meteo API

## 🛠️ Technologies Used

- Python
- LangChain
- Google Gemini / Generative AI
- FastAPI
- LangServe
- Pydantic
- Requests
- Open-Meteo API

## 🏗️ Project Architecture

```text
User
  ↓
FastAPI
  ↓
LangServe
  ↓
AI Agent
  ↓
Custom Tools
  ├── Weather Tool
  ├── Indian Movie Search
  └── Temperature Converter


## 🚀 How It Works

The user sends a query to the AI agent through the FastAPI application.

The AI agent analyzes the query and selects the appropriate tool:

- 🌤️ Weather Tool → Gets current weather information for a city
- 🎬 Movie Search Tool → Searches Indian movies by genre
- 🌡️ Temperature Converter → Converts Celsius to Fahrenheit

## 📌 Project Status

This project is currently being improved as part of my AI/ML learning journey.

## 👨‍💻 Author

**Jitendhra**

B.Tech – Artificial Intelligence & Machine Learning
