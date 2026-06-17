"""
MusicBrainz API client — no API key needed, just a meaningful User-Agent.
Rate limit: 1 request/second enforced by _REQUEST_DELAY_SECONDS.

Provides: formed year, country, album release dates.
Replaces Spotify for structured metadata — fully open, no quota restrictions.
"""
import time
import requests
from config import ENTITY_RESOLUTION_AUTO_ACCEPT, ENTITY_RESOLUTION_REVIEW_FLOOR
from rapidfuzz import fuzz

_API_BASE = "https://musicbrainz.org/ws/2"
_REQUEST_DELAY_SECONDS = 1.1  # slightly over 1s to stay safely under rate limit

# MusicBrainz requires a descriptive User-Agent — anonymous requests get blocked
_HEADERS = {
    "User-Agent": "MetalIntelligencePipeline/0.1 (metal-intelligence-pipeline; data engineering portfolio)",
    "Accept": "application/json",
}


class MusicBrainzClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict:
        """GET with rate limiting and retry logic."""
        time.sleep(_REQUEST_DELAY_SECONDS)  # enforce 1 req/sec before every call

        for attempt in range(retries):
            response = self._session.get(f"{_API_BASE}/{endpoint}", params=params)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 503:
                # MusicBrainz returns 503 when overloaded — back off and retry
                wait = 5 * (attempt + 1)
                print(f"  ⚠ MusicBrainz 503 — waiting {wait}s")
                time.sleep(wait)
                continue

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"  ⚠ rate limited — waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            response.raise_for_status()

        raise RuntimeError(f"MusicBrainz API failed after {retries} retries: {endpoint}")

    # ── Public methods ────────────────────────────────────────────────────────

    def find_artist(self, artist_name: str) -> dict | None:
        """
        Search for an artist by name, return the best Group match.

        Matching strategy (designed in Phase 0):
          1. Filter candidates to type=Group — eliminates solo artists, orchestras
          2. Among Groups, pick highest MusicBrainz search score
          3. Validate with fuzzy name similarity — flag if below thresholds
        """
        data = self._get(
            "artist",
            params={"query": artist_name, "limit": 5, "fmt": "json"},
        )

        candidates = data.get("artists", [])
        if not candidates:
            return None

        # Filter to Groups only — every band in our pipeline is a Group
        groups = [a for a in candidates if a.get("type") == "Group"]
        if not groups:
            # Fall back to all candidates if no Groups found (some bands mislabelled)
            groups = candidates

        # Pick highest MusicBrainz search score
        best = max(groups, key=lambda a: a.get("score", 0))

        # Validate with fuzzy similarity between search query and matched name
        similarity = fuzz.token_sort_ratio(
            artist_name.lower().strip(),
            best.get("name", "").lower().strip()
        )

        if similarity >= ENTITY_RESOLUTION_AUTO_ACCEPT:
            status = "auto_accepted"
        elif similarity >= ENTITY_RESOLUTION_REVIEW_FLOOR:
            status = "review_required"
        else:
            status = "rejected"

        return {
            "mbid": best.get("id"),
            "mb_name": best.get("name"),
            "type": best.get("type"),
            "country": best.get("country"),
            "formed_year": self._extract_formed_year(best),
            "disambiguation": best.get("disambiguation", ""),
            "mb_score": best.get("score", 0),
            "similarity_score": round(similarity, 2),
            "resolution_status": status,
        }

    def _extract_formed_year(self, artist: dict) -> int | None:
        """Extract formed year from MusicBrainz life-span — structured, not regex."""
        begin = artist.get("life-span", {}).get("begin", "")
        if begin and len(begin) >= 4:
            try:
                year = int(begin[:4])
                return year if 1960 <= year <= 2025 else None
            except ValueError:
                return None
        return None

    def get_artist_albums(self, mbid: str) -> list[dict]:
        """
        Fetch all official albums and EPs for an artist by MBID.
        Uses release-groups endpoint — one entry per album, not per release edition.
        """
        albums = []
        offset = 0
        limit = 100  # MusicBrainz allows up to 100 per page

        while True:
            data = self._get(
                f"release-group",
                params={
                    "artist": mbid,
                    "type": "album|ep|single",
                    "limit": limit,
                    "offset": offset,
                    "fmt": "json",
                },
            )

            batch = data.get("release-groups", [])
            if not batch:
                break

            for rg in batch:
                # Only include official releases — excludes bootlegs, promos
                albums.append({
                    "mbid": rg.get("id"),
                    "title": rg.get("title"),
                    "type": rg.get("primary-type"),
                    "first_release_date": rg.get("first-release-date"),
                })

            total = data.get("release-group-count", 0)
            offset += len(batch)
            if offset >= total:
                break

        return albums
