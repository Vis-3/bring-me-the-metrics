"""
Central config — loads .env and exposes typed constants to the whole codebase.
Every other module imports from here, never from os.environ directly.
This means if an env var name changes, we fix it in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Walk up from this file to find .env — works regardless of where scripts are run from
load_dotenv(Path(__file__).parent / ".env")


def _require(key: str) -> str:
    """Fail loudly at import time if a required secret is missing — better than
    a cryptic error mid-pipeline run."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# ── Spotify ───────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID = _require("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = _require("SPOTIFY_CLIENT_SECRET")

# ── Last.fm ───────────────────────────────────────────────────────────────────
LASTFM_API_KEY = _require("LASTFM_API_KEY")
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID = _require("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _require("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = _require("S3_BUCKET_NAME")
ATHENA_S3_OUTPUT = _require("ATHENA_S3_OUTPUT")

# ── S3 partition path builders ─────────────────────────────────────────────────
# Centralised here so partition strategy is defined in exactly one place.
# If we ever change the partition scheme, only this file changes.

def bronze_prefix(source: str, subgenre: str, date: str) -> str:
    """Returns: bronze/source=spotify/subgenre=deathcore/date=2024-01-15/"""
    return f"bronze/source={source}/subgenre={subgenre}/date={date}/"

def silver_prefix(table: str, subgenre: str) -> str:
    """Returns: silver/table=tracks/subgenre=deathcore/"""
    return f"silver/table={table}/subgenre={subgenre}/"

def gold_prefix(mart: str) -> str:
    """Returns: gold/mart=riff_economy/"""
    return f"gold/mart={mart}/"


# ── Subgenres ─────────────────────────────────────────────────────────────────
# Single source of truth — ingestion, silver, and dbt all iterate over this list.
# Adding a new subgenre means changing this one line.
SUBGENRES = [
    "metalcore",
    "deathcore",
    "melodic metalcore",
    "progressive metal",
    "symphonic metal",
    "djent",
    "nu-metal",
    "black metal",
    "melodic death metal",
]

# ── ML thresholds ─────────────────────────────────────────────────────────────
# Defined here so the labeling logic and the scoring logic use identical values.
BREAKOUT_LISTENER_THRESHOLD = 1_000_000   # positive class: crossed this
UNDERGROUND_LISTENER_CEILING = 200_000    # negative class: stayed below this
ENTITY_RESOLUTION_AUTO_ACCEPT = 90        # fuzzy match score, 0-100
ENTITY_RESOLUTION_REVIEW_FLOOR = 70       # below this → rejected, not queued
