"""
Last.fm API client — handles rate limiting, pagination, and retries.
No OAuth needed — every request includes the API key as a query parameter.
"""
import time
import requests
from config import LASTFM_API_KEY, LASTFM_BASE_URL

# Last.fm allows 5 requests/second for free tier — we stay under with a small delay
_REQUEST_DELAY_SECONDS = 0.25  # 4 requests/second, safely under the limit

# Number of artists per page — Last.fm max is 50
_PAGE_SIZE = 50


class LastFmClient:
    def __init__(self):
        self._session = requests.Session()

    def _get(self, method: str, params: dict = None, retries: int = 3) -> dict:
        """
        GET wrapper with retry logic. Last.fm always returns 200 even for errors —
        error cases come back as JSON with an 'error' key, so we check for that too.
        """
        base_params = {
            "method": method,
            "api_key": LASTFM_API_KEY,
            "format": "json",
        }
        if params:
            base_params.update(params)

        for attempt in range(retries):
            time.sleep(_REQUEST_DELAY_SECONDS)
            response = self._session.get(LASTFM_BASE_URL, params=base_params)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"  ⚠ rate limited — waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if response.status_code >= 500:
                wait = 2 ** attempt
                print(f"  ⚠ server error {response.status_code} — retrying in {wait}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            # Last.fm wraps API-level errors in the response body with status 200
            if "error" in data:
                # Error 6 = artist/tag not found — not worth retrying
                if data["error"] == 6:
                    return {}
                raise RuntimeError(f"Last.fm API error {data['error']}: {data.get('message')}")

            return data

        raise RuntimeError(f"Last.fm API failed after {retries} retries: {method}")

    # ── Public methods ────────────────────────────────────────────────────────

    def get_tag_artists(self, tag: str, max_pages: int = 10) -> list[dict]:
        """
        Fetch top artists for a genre tag across multiple pages.
        max_pages=10 gives us up to 500 artists per subgenre (10 × 50).
        Bronze gets all of them — Silver applies the 50K listener floor.
        """
        artists = []

        for page in range(1, max_pages + 1):
            data = self._get(
                "tag.gettopartists",
                params={"tag": tag, "limit": _PAGE_SIZE, "page": page},
            )

            if not data or "topartists" not in data:
                break

            batch = data["topartists"].get("artist", [])
            if not batch:
                break

            artists.extend(batch)
            total_pages = int(data["topartists"]["@attr"].get("totalPages", 1))
            print(f"  → page {page}/{min(total_pages, max_pages)}: {len(batch)} artists")

            if page >= total_pages:
                break

        return artists

    def get_artist_info(self, artist_name: str) -> dict:
        """
        Fetch full artist info: listener count, play count, biography, tags.
        This is the richest endpoint — drives both Silver enrichment and ML features.
        """
        data = self._get(
            "artist.getinfo",
            params={"artist": artist_name, "autocorrect": 1},
        )
        return data.get("artist", {})

    def get_artist_similar(self, artist_name: str, limit: int = 10) -> list[dict]:
        """
        Fetch similar artists — used as a feature signal in the Breakout Predictor.
        Cross-genre similarity (high tag_diversity) is a breakout signal.
        """
        data = self._get(
            "artist.getsimilar",
            params={"artist": artist_name, "limit": limit, "autocorrect": 1},
        )
        return data.get("similarartists", {}).get("artist", [])

    def get_weekly_chart_artists(self, tag: str, weeks_back: int = 52) -> list[dict]:
        """
        Fetch weekly artist charts for a tag going back N weeks.
        This is the historical time series data that drives listener velocity features.
        Last.fm weekly charts reset every Monday — we fetch from current week backwards.

        weeks_back=52 gives one year of weekly snapshots per subgenre.
        For full historical backfill (2015-present) pass weeks_back=520.
        """
        from datetime import date, timedelta

        charts = []
        # Last.fm weekly charts use Unix timestamps for from/to parameters
        # Start from today and walk backwards in 7-day increments
        end_date = date.today()

        for week in range(weeks_back):
            start_date = end_date - timedelta(days=7)
            from_ts = int(time.mktime(start_date.timetuple()))
            to_ts = int(time.mktime(end_date.timetuple()))

            data = self._get(
                "tag.getweeklychartlist",
                params={"tag": tag, "from": from_ts, "to": to_ts},
            )

            if data:
                # Embed the week period on the record for partitioning in Bronze
                data["_week_start"] = start_date.isoformat()
                data["_week_end"] = end_date.isoformat()
                charts.append(data)

            end_date = start_date

            if week % 10 == 0:
                print(f"  → fetched week {week + 1}/{weeks_back}: {start_date}")

        return charts
