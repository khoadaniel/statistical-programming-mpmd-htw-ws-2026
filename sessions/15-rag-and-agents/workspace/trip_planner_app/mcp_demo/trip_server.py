"""A local MCP server for the trip planner: the same tools as the agents of the app, offered over MCP.

What the server offers (the three kinds of thing an MCP server can offer, plus skills):

- tools:      next_weekend, weather_forecast, search_trip_guide, budget_per_person
              and, for skill discovery, list_skills and load_skill
- resources:  trip://guide (the travel guide as text) and skill://<name> (the body of one skill)
- prompts:    plan_weekend_trip (a ready-made question the host can offer to the user)

Skills are folders in skills/, each with a SKILL.md: a short header (name and description) and a body
with instructions. The server shows only the headers first; an agent loads a body when it needs it.
MCP has no "skill" type of its own, so a skill is offered here as a tool pair and as a resource.

Run it over stdio (the default; a client starts it):

    uv run mcp_demo/trip_server.py

Run it over Streamable HTTP on this computer only:

    uv run mcp_demo/trip_server.py --http --port 8765

A stdio server must never print to standard output: that stream carries the protocol messages.
Log to standard error instead.
"""

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import httpx
from fastmcp import FastMCP

SKILLS_FOLDER = Path(__file__).with_name("skills")

mcp = FastMCP("trip-planner")

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


# ---------- tools ----------

@mcp.tool
def next_weekend() -> dict:
    """Return today's date and the dates of the coming Saturday and Sunday. Call it when the user says "this weekend"."""
    today = date.today()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    return {"today": today.isoformat(), "saturday": saturday.isoformat(),
            "sunday": (saturday + timedelta(days=1)).isoformat()}


@mcp.tool
def weather_forecast(city: str, start_date: str, end_date: str) -> dict:
    """Get the daily weather forecast for a city, for dates up to 16 days from today.

    city is the English name, for example "Prague" (not "Praha"). Dates are YYYY-MM-DD.
    Needs an internet connection (Open-Meteo, CC BY 4.0).
    """
    try:
        places = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1, "language": "en"}, timeout=10).json()
        if not places.get("results"):
            return {"error": f"No city called {city!r} was found. Try the English name."}
        place = places["results"][0]
        response = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
            "start_date": start_date, "end_date": end_date,
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_probability_max",
        })
        response.raise_for_status()
    except httpx.HTTPError as error:
        return {"error": f"The weather service did not answer ({type(error).__name__}). It forecasts at most 16 days ahead."}
    daily = response.json()["daily"]
    days = [{"date": day, "min_c": low, "max_c": high, "rain_chance_pct": rain}
            for day, low, high, rain in zip(daily["time"], daily["temperature_2m_min"],
                                            daily["temperature_2m_max"], daily["precipitation_probability_max"], strict=True)]
    return {"city": place["name"], "country": place["country"], "days": days, "source": "Open-Meteo.com"}


def words(text: str) -> set[str]:
    """The set of lower-case words in a text."""
    return set(re.findall(r"[a-z]+", text.lower()))


@mcp.tool
def search_trip_guide(query: str) -> str:
    """Search the student travel guide for bus fares, travel times, hostel prices and free things to do.

    Put the city name in the query. The guide covers Prague, Vienna and Dresden.
    """
    query_words = words(query)
    ranked = sorted(TRIP_GUIDE, key=lambda note: len(query_words & words(note)), reverse=True)
    best_notes = [note for note in ranked[:2] if query_words & words(note)]
    if not best_notes:
        return "No note matches. The guide covers Prague, Vienna and Dresden."
    return "\n\n".join(best_notes)


@mcp.tool
def budget_per_person(total_budget_eur: float, people: int, nights: int,
                      return_fare_eur: float, bed_per_night_eur: float) -> dict:
    """Compute what is left per person for food and activities after the fare and the beds.

    Amounts are in euros; total_budget_eur is the budget of the whole group.
    """
    if people < 1:
        return {"error": "people must be 1 or more."}
    per_person = total_budget_eur / people
    left = per_person - return_fare_eur - nights * bed_per_night_eur
    return {"budget_per_person_eur": round(per_person, 2),
            "left_for_food_and_activities_eur": round(left, 2), "within_budget": left >= 0}


# ---------- resources and a prompt ----------

@mcp.resource("trip://guide")
def trip_guide() -> str:
    """The whole travel guide as plain text."""
    return "\n\n".join(TRIP_GUIDE)


@mcp.prompt
def plan_weekend_trip(city: str, people: int, budget_eur: float) -> str:
    """A ready-made request: can this group afford a weekend trip to the city?"""
    return (f"We are {people} students in Berlin with {budget_eur:.0f} EUR in total. "
            f"Can we afford this coming weekend in {city}? Check the fare, the beds and the weather.")


# ---------- skills: discovery by header, loading by body ----------

def read_skill(folder: Path) -> dict:
    """Read one skills/<name>/SKILL.md: the header (name, description) and the body."""
    text = (folder / "SKILL.md").read_text(encoding="utf-8")
    _, header, body = text.split("---", 2)
    fields = dict(line.split(":", 1) for line in header.strip().splitlines())
    return {"name": fields["name"].strip(), "description": fields["description"].strip(), "body": body.strip()}


def all_skills() -> dict[str, dict]:
    """Every skill in the skills folder, by name."""
    skills = [read_skill(folder) for folder in sorted(SKILLS_FOLDER.iterdir()) if (folder / "SKILL.md").is_file()]
    return {skill["name"]: skill for skill in skills}


@mcp.tool
def list_skills() -> list[dict]:
    """List the skills this server offers: only the name and a one-line description of each.

    Call it first. Then call load_skill for the skill that fits the task.
    """
    return [{"name": skill["name"], "description": skill["description"]} for skill in all_skills().values()]


@mcp.tool
def load_skill(name: str) -> str:
    """Load the full instructions of one skill by name (a name from list_skills)."""
    skills = all_skills()
    if name not in skills:
        return f"No skill called {name!r}. Skills: {', '.join(skills)}."
    return skills[name]["body"]


@mcp.resource("skill://{name}")
def skill_resource(name: str) -> str:
    """The instructions of one skill, as a resource (the same text as load_skill)."""
    return load_skill(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip planner MCP server")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.http:
        # 127.0.0.1 means: only programs on this computer can connect.
        mcp.run(transport="http", host="127.0.0.1", port=arguments.port, show_banner=False)
    else:
        mcp.run(show_banner=False)
