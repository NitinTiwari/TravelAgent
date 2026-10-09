# pip install mcp requests

from mcp.server.fastmcp import FastMCP
import requests
import os

try:
    from settings import OPENWEATHER_API_KEY
except ImportError:
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

###############################################################################
# FastMCP Custom Weather Server
#
# Functional Details:
# - Exposes OpenWeather API capabilities as a Model Context Protocol (MCP) server.
# - get_current_weather: Returns real-time temperature, humidity, condition, and wind speed.
# - get_forecast: Provides a 5-day / 3-hour interval weather forecast for any specified city.
# - Enables seamless tool invocation for LLM agents via FastMCP stdio transport.
###############################################################################

mcp = FastMCP("Weather Server")

@mcp.tool()
def get_current_weather(city: str):

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
    )

    data = response.json()

    if response.status_code != 200:
        return data

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }



@mcp.tool()
def get_forecast(city: str):

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    forecast = []

    # Return first 5 forecast entries
    for item in data["list"][:5]:

        forecast.append(
            {
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"]
            }
        )

    return {
        "city": city,
        "forecast": forecast
    }




if __name__ == "__main__":
    mcp.run()