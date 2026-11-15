"""The trip planner of workbook 05 as one LangGraph graph, with three agents.

orchestrator agent (reads the request; the graph asks the user when people, budget or cities are missing)
  -> one city agent per city, in parallel (Send); a reducer (operator.add) joins their reports
  -> recommender agent (compares the reports and recommends one city)
"""
import json
import operator
import re
from datetime import date, timedelta
from typing import Annotated

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

MAX_CITIES = 4
model = ChatAnthropic(model="claude-haiku-5-5", max_tokens=4096)


# ---------- the four tools of workbook 05 ----------

@tool
def next_weekend() -> str:
    """Return today's date and the dates of the coming Saturday and Sunday. Call it when the user says "this weekend"."""
    today = date.today()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    sunday = saturday + timedelta(days=1)
    return json.dumps({"today": today.isoformat(), "saturday": saturday.isoformat(), "sunday": sunday.isoformat()})


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@tool(parse_docstring=True)
def weather_forecast(city: str, start_date: str, end_date: str) -> str:
    """Get the daily weather forecast for a city. Works only for dates up to 16 days from today.

    Args:
        city: City name in English, for example "Prague" (not "Praha").
        start_date: First day, as YYYY-MM-DD.
        end_date: Last day, as YYYY-MM-DD.
    """
    try:
        places = httpx.get(GEOCODING_URL, params={"name": city, "count": 1, "language": "en"}, timeout=10).json()
        if not places.get("results"):
            return f"Error: no city called {city!r} was found. Try the English name."
        place = places["results"][0]
        response = httpx.get(FORECAST_URL, timeout=10, params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
            "start_date": start_date, "end_date": end_date,
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_probability_max",
        })
        response.raise_for_status()
    except httpx.HTTPError as error:
        return f"Error: the weather service did not answer ({type(error).__name__}). It forecasts at most 16 days ahead."
    daily = response.json()["daily"]
    days = [
        {"date": day, "min_c": low, "max_c": high, "rain_chance_pct": rain}
        for day, low, high, rain in zip(daily["time"], daily["temperature_2m_min"],
                                        daily["temperature_2m_max"], daily["precipitation_probability_max"])
    ]
    return json.dumps({"city": place["name"], "country": place["country"], "days": days,
                       "source": "Weather data by Open-Meteo.com"})


# The travel guide of workbook 05: example prices written for the course, not real offers.
TRIP_GUIDE = [
    "Prague, Czechia. Currency: Czech koruna (CZK). Bus from Berlin: about 4.5 hours, return fare about 40 EUR per person. "
    "Hostel bed: about 25 EUR per night. Cheap food: a lunch menu costs about 8 EUR. "
    "Free: Charles Bridge early in the morning, and the view from Letna Park.",
    "Vienna, Austria. Currency: euro (EUR). Bus from Berlin: about 9.5 hours, return fare about 70 EUR per person. "
    "Hostel bed: about 35 EUR per night. Cheap food: a sausage stand costs about 5 EUR. "
    "Free: the flea market at the Naschmarkt on Saturday morning.",
    "Dresden, Germany. Currency: euro (EUR). Bus from Berlin: about 2.5 hours, return fare about 25 EUR per person. "
    "Hostel bed: about 25 EUR per night. Free: a walk along the Elbe meadows, and the Neustadt district.",
]


def words(text: str) -> set[str]:
    """The set of lower-case words in a text."""
    return set(re.findall(r"[a-z]+", text.lower()))


@tool
def search_trip_guide(query: str) -> str:
    """Search our student travel guide for bus fares, travel times, hostel prices and free things to do. Put the city name in the query."""
    query_words = words(query)
    ranked = sorted(TRIP_GUIDE, key=lambda note: len(query_words & words(note)), reverse=True)
    best_notes = [note for note in ranked[:2] if query_words & words(note)]
    if not best_notes:
        return "No note matches. The guide covers Prague, Vienna and Dresden."
    return "\n\n".join(best_notes)


@tool(parse_docstring=True)
def budget_per_person(total_budget_eur: float, people: int, nights: int,
                      return_fare_eur: float, bed_per_night_eur: float) -> str:
    """Compute what is left per person for food and activities after the fare and the beds.

    Args:
        total_budget_eur: Budget of the whole group, in euros.
        people: Number of travellers.
        nights: Number of nights away.
        return_fare_eur: Return fare for one person, in euros.
        bed_per_night_eur: Price of one bed for one night, in euros.
    """
    per_person = total_budget_eur / people
    left = per_person - return_fare_eur - nights * bed_per_night_eur
    return json.dumps({"budget_per_person_eur": round(per_person, 2),
                       "left_for_food_and_activities_eur": round(left, 2), "within_budget": left >= 0})


# ---------- the agents ----------

class TripRequest(BaseModel):
    """What the orchestrator agent reads from the user's text."""

    people: int | None = Field(None, description="Number of travellers; null if the user did not say it")
    budget_eur: float | None = Field(None, description="Total budget of the whole group in euros; null if the user did not say it")
    nights: int = Field(1, description="Number of nights; 1 for a weekend or if the user did not say it")
    cities: list[str] = Field(default_factory=list,
                              description="The destination cities the user names, in English; empty if the user named none")


ORCHESTRATOR_RULES = """You read a trip request from students who live in Berlin and fill in the plan.
Rules:
- Fill in only what the user said. Use null for the number of people or the budget if the user did not give it.
- Put only the cities that the user names into cities. If the user names no city, leave cities empty,
  also when the user asks for ideas: the graph will ask the user.
- If the user says "all of them" or "all three" about the travel guide, the cities are Prague, Vienna and Dresden.
- You do not research anything."""

CITY_RULES = """You research one city for a short, cheap trip from Berlin.
Rules:
- Research only the one city you are given.
- Use the tools for every fact: dates, weather, fares, prices and sums. Never guess a number.
- If the travel guide has no note for the city, say that the prices are unknown. Do not guess them.
- Prices from the travel guide are course examples. Say so in your report.
- Report in at most 8 lines: the dates, the travel, the bed, the money left per person, the weather and what to pack.
- Do not compare with other cities."""

RECOMMENDER_RULES = """You compare trip options for students who live in Berlin.
Use only the facts in the reports. Never recommend a plan over the budget; if nothing fits, say so.
If there is only one report, say whether that trip fits the budget.
Recommend one city in at most 6 lines."""

orchestrator_agent = create_agent(model=model, tools=[], system_prompt=ORCHESTRATOR_RULES, response_format=TripRequest)
city_agent = create_agent(model=model,
                          tools=[next_weekend, weather_forecast, search_trip_guide, budget_per_person],
                          system_prompt=CITY_RULES)
recommender_agent = create_agent(model=model, tools=[], system_prompt=RECOMMENDER_RULES)


# ---------- the graph ----------

class CityReport(BaseModel):
    city: str
    report: str


class PlannerState(BaseModel):
    request: str
    trip: TripRequest | None = None
    # The reducer: the city agents write here in the same step, and operator.add joins their lists.
    reports: Annotated[list[CityReport], operator.add] = []
    recommendation: str = ""


# The payload of one Send: the input of one city agent.
class CityJob(BaseModel):
    city: str
    trip: TripRequest


# What the graph asks for each missing item. The rules are in code, not in a prompt, so they always hold.
QUESTIONS = {
    "people": "how many people are travelling",
    "budget": "the total budget for the whole group, in euros",
    "cities": "which cities to compare: our travel guide has prices for Prague, Vienna and Dresden "
              "(name one or more, or say \"all three\")",
    "too_many_cities": f"at most {MAX_CITIES} cities to compare",
}


def missing_items(trip: TripRequest) -> list[str]:
    """The items that the plan still needs: people, budget, and 1 to MAX_CITIES cities."""
    missing = []
    if not trip.people:
        missing.append("people")
    if not trip.budget_eur:
        missing.append("budget")
    if not trip.cities:
        missing.append("cities")
    elif len(trip.cities) > MAX_CITIES:
        missing.append("too_many_cities")
    return missing


def run_orchestrator_agent(state: PlannerState) -> dict:
    """Node orchestrator_agent: the orchestrator agent turns the user's text into a TripRequest."""
    result = orchestrator_agent.invoke({"messages": [("user", state.request)]})
    return {"trip": result["structured_response"]}


def route_after_orchestrator(state: PlannerState) -> str | list[Send]:
    """Conditional edge after the orchestrator agent: ask the user if something is missing, otherwise one Send per city."""
    if missing_items(state.trip):
        return "ask_user"
    return [Send("city_agent", CityJob(city=city, trip=state.trip)) for city in state.trip.cities]


def ask_user(state: PlannerState) -> dict:
    """Node: pause the graph with interrupt() and wait for the user's answer."""
    missing = missing_items(state.trip)
    question = "Please tell me " + "; and ".join(QUESTIONS[item] for item in missing) + "."
    answer = interrupt({"question": question, "missing": missing})
    return {"request": f"{state.request}\nAnswer to \"{question}\": {answer}"}


def run_city_agent(job: CityJob) -> dict:
    """Node city_agent, once per city: the city agent researches one city."""
    trip = job.trip
    brief = (f"Research {job.city} for this weekend: from Berlin, {trip.people} people, "
             f"{trip.budget_eur} EUR in total, {trip.nights} night(s).")
    result = city_agent.invoke({"messages": [("user", brief)]})
    # A list with ONE report; the reducer joins the lists of all city agents.
    return {"reports": [CityReport(city=job.city, report=result["messages"][-1].text)]}


def run_recommender_agent(state: PlannerState) -> dict:
    """Node recommender_agent: the recommender agent compares the reports and recommends one city."""
    reports = "\n\n".join(f"## {report.city}\n{report.report}" for report in state.reports)
    result = recommender_agent.invoke({"messages": [("user", f"Request: {state.request}\n\nReports:\n{reports}")]})
    return {"recommendation": result["messages"][-1].text}


def build_graph():
    builder = StateGraph(PlannerState)
    # Each agent runs in a node of the same name.
    builder.add_node("orchestrator_agent", run_orchestrator_agent)
    builder.add_node("ask_user", ask_user)
    builder.add_node("city_agent", run_city_agent)
    builder.add_node("recommender_agent", run_recommender_agent)
    builder.add_edge(START, "orchestrator_agent")
    # The list names the nodes route_after_orchestrator MAY start: ask_user, or city_agent once per city.
    builder.add_conditional_edges("orchestrator_agent", route_after_orchestrator, ["ask_user", "city_agent"])
    builder.add_edge("ask_user", "orchestrator_agent")
    builder.add_edge("city_agent", "recommender_agent")
    builder.add_edge("recommender_agent", END)
    # The checkpointer keeps the state of each conversation, so a paused run can continue after the answer.
    # The serializer may read our own Pydantic classes back from the checkpoint.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        (__name__, name) for name in ("TripRequest", "CityReport", "CityJob", "PlannerState")])
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
