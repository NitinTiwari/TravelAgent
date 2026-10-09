
# Example free API usage
# - AviationStack


# create api key
# https://aviationstack.com/ 
# pip install requests


    
import os
import requests
from dotenv import load_dotenv

###############################################################################
# Direct AviationStack Flight Search Tool
#
# Functional Details:
# - Directly communicates with the AviationStack REST API endpoint (/v1/flights).
# - Searches and parses active flight schedules, airline names, and departure/arrival airports.
# - Serves as a standalone tool module for flight information retrieval.
###############################################################################

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")


def search_flights(query):

    url = "http://api.aviationstack.com/v1/flights"

    params = {
        "access_key": API_KEY,
        "limit": 5
    }

    response = requests.get(url, params=params)

    data = response.json()

    flights = []

    if "data" in data:

        for flight in data["data"][:5]:

            airline = flight.get("airline", {}).get("name", "Unknown")

            departure = flight.get(
                "departure", {}
            ).get("airport", "Unknown")

            arrival = flight.get(
                "arrival", {}
            ).get("airport", "Unknown")

            status = flight.get("flight_status", "Unknown")

            flights.append(
                f"""
Airline: {airline}
Departure: {departure}
Arrival: {arrival}
Status: {status}
"""
            )

    return "\n".join(flights)