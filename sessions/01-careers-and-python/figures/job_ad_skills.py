"""Which skills do Berlin job ads ask for, by role? Writes job_ad_skills.csv for make_figures.py.

The ads come from the Data Berlin job board (https://databerlin.net/jobs), which lists Berlin data and AI jobs.
An ad is assigned to a role by its title (seniority leads, heads and managers are left out); a skill group counts
as named when one of its keywords appears in the ad text. Keyword matching is crude: it shows what ads mention,
not how much of the job a skill is. The board changes daily, so the result describes one day.

Run from the repository root (about 250 page requests, a few minutes):

    uv run --with trafilatura python sessions/01-careers-and-python/figures/job_ad_skills.py
"""

import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
BOARD = "https://databerlin.net"
HEADERS = {"User-Agent": "Mozilla/5.0 (course material, HTW Berlin)"}

ROLES = {  # role: regular expression on the job title
    "Data analyst": r"\bdata analyst\b|\bproduct analyst\b|\bbi analyst\b|\banalytics analyst\b",
    "Data scientist": r"\bdata scientist\b",
    "Data engineer": r"\bdata engineer\b",
    "ML engineer": r"\b(machine learning|ml|mlops) engineer\b",
    "AI engineer": r"\b(ai|genai|llm|agentic ai|applied ai) engineer\b",
}
EXCLUDE = r"\b(head|director|vp|chief|lead|principal|staff|manager)\b"

SKILLS = {  # skill group: regular expression on the ad text (case-insensitive)
    "SQL": r"\bsql\b",
    "Statistics, experiments": r"statistic|a/b[- ]test|\bexperiment|hypothesis|causal infer",
    "Dashboards, BI": r"dashboard|tableau|power ?bi|looker|metabase|superset|business intelligence",
    "Python": r"\bpython\b",
    "Machine learning": r"machine learning|\bml\b|scikit|xgboost|lightgbm|pytorch|tensorflow|deep learning|predictive model",
    "Data pipelines": r"\betl\b|\belt\b|airflow|dagster|\bspark\b|\bdbt\b|kafka|data pipeline",
    "Cloud": r"\baws\b|\bgcp\b|google cloud|azure",
    "Containers, CI/CD": r"docker|kubernetes|\bk8s\b|ci/cd|continuous (integration|delivery|deployment)|terraform",
    "LLMs, agents": r"\bllms?\b|large language|gen ?ai|generative ai|agentic|\bai agents?\b|\brag\b|retrieval[- ]augmented|prompt",
    "Communication, stakeholders": r"communicat|stakeholder|storytelling",
    "German": r"\bgerman\b|deutsch",
}


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=25) as response:
        return response.read().decode("utf-8", "replace")


def collect() -> list[dict]:
    import trafilatura

    listing = get(f"{BOARD}/jobs")
    cards = re.split(r"(?=<(?:article|li|div)[^>]*data-category=)", listing)
    ads = []
    for card in cards:
        slug = re.search(r'href="/jobs/([^"]+)"', card)
        title = re.search(r'data-title="([^"]*)"', card[:1500])
        if not (slug and title):
            continue
        title = title.group(1)
        role = next((r for r, rx in ROLES.items() if re.search(rx, title, re.I)), None)
        if role is None or re.search(EXCLUDE, title, re.I):
            continue
        text = trafilatura.extract(get(f"{BOARD}/jobs/{slug.group(1)}")) or ""
        ads.append({"role": role, "title": title, "text": text})
        time.sleep(0.3)
    return ads


def ad_text(page_text: str) -> str:
    """The advertisement only: drop the board's navigation and its list of related roles (other companies' names)."""
    return page_text.split("Related roles")[0].replace("← All jobs", "")


def summarise(ads: list[dict], retrieved: str) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        texts = [ad_text(a["text"]) for a in ads if a["role"] == role]
        for skill, rx in SKILLS.items():
            share = sum(bool(re.search(rx, t, re.I)) for t in texts) / len(texts)
            rows.append({"role": role, "n_ads": len(texts), "skill": skill, "share": round(share, 3), "retrieved": retrieved})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # optional: a JSON file of already downloaded ads ({id: {"role", "title", "text"}}) instead of downloading
    if len(sys.argv) > 2:
        ads, retrieved = list(json.load(open(sys.argv[1])).values()), sys.argv[2]
    else:
        ads, retrieved = collect(), date.today().isoformat()
    table = summarise(ads, retrieved)
    table.to_csv(OUT / "job_ad_skills.csv", index=False)
    print(table.pivot(index="role", columns="skill", values="share").round(2))
