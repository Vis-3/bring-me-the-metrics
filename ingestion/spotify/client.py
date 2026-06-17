"""
Spotify API client — handles auth, token refresh, rate limiting, and pagination.
All other Spotify modules import this client; none call requests directly.
"""
import time
import requests
from datetime import datetime, timedelta
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"

# Token is valid for 3600s; we refresh at 55 minutes to avoid mid-request expiry
_TOKEN_LIFETIME_SECONDS = 55 * 60

# Proactive delay between requests to stay under unpublished rate limits
_REQUEST_DELAY_SECONDS = 0.5


class SpotifyClient:
    def __init__(self):
        self._token: str | None = None
        self._token_expires_at: datetime = datetime.min  # force auth on first request

    def _authenticate(self) -> None:
        """Request a fresh client credentials token from Spotify."""
        response = requests.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        )
        response.raise_for_status()
        self._token = response.json()["access_token"]
        self._token_expires_at = datetime.now() + timedelta(seconds=_TOKEN_LIFETIME_SECONDS)
        print("  ↻ Spotify token refreshed")

    def _headers(self) -> dict:
        """Return auth headers, refreshing token proactively if near expiry."""
        if datetime.now() >= self._token_expires_at:
            self._authenticate()
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, url: str, params: dict = None, retries: int = 3) -> dict:
        """GET with retry logic for transient failures and rate limiting."""
        for attempt in range(retries):
            time.sleep(_REQUEST_DELAY_SECONDS)
            response = requests.get(url, headers=self._headers(), params=params)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"  ⚠ rate limited — waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if response.status_code == 401:
                self._token_expires_at = datetime.min
                continue

            if response.status_code >= 500:
                wait = 2 ** attempt
                print(f"  ⚠ server error {response.status_code} — retrying in {wait}s")
                time.sleep(wait)
                continue

            print(f"  ✗ {response.status_code} response: {response.text[:500]}")
            response.raise_for_status()

        raise RuntimeError(f"Spotify API failed after {retries} retries: {url}")

    # ── Public methods ────────────────────────────────────────────────────────

    def find_artist(self, artist_name: str) -> dict | None:
        """
        Look up an artist by name from Last.fm, return the best Spotify match.

        Strategy: search Spotify for the name, pick the candidate with the highest
        popularity score. The real artist will almost always outscore cover bands
        and namesakes on popularity.

        Returns a dict with 'matched' (the best candidate) and 'candidates' (all
        results) so Bronze preserves the full search context for Silver to audit.
        Returns None if Spotify returns no results at all.
        """
        data = self._get(
            f"{_API_BASE}/search",
            params={"q": artist_name, "type": "artist", "limit": 5, "market": "US"},
        )

        candidates = data.get("artists", {}).get("items", [])
        if not candidates:
            return None

        # Pick highest popularity — handles cover bands and namesakes
        best_match = max(candidates, key=lambda a: a.get("popularity", 0))

        # Search returns a simplified object — fetch full artist to get popularity + followers
        full_artist = self._get(f"{_API_BASE}/artists/{best_match['id']}")

        return {
            "matched": full_artist,
            "candidates": candidates,        # preserved for Silver auditing
            "search_query": artist_name,     # what we searched for
        }

    def get_artist_albums(self, artist_id: str) -> list[dict]:
        """
        Fetch all albums for an artist across all pages.
        Fetches all types then filters client-side — avoids Spotify's inconsistent
        handling of the include_groups parameter across API versions.
        """
        albums = []
        url = f"{_API_BASE}/artists/{artist_id}/albums"

        while url:
            data = self._get(url)
            filtered = [a for a in data["items"] if a.get("album_type") in ("album", "single")]
            albums.extend(filtered)
            url = data.get("next")

        return albums

    def get_tracks(self, album_id: str) -> list[dict]:
        """Fetch all tracks for an album."""
        tracks = []
        url = f"{_API_BASE}/albums/{album_id}/tracks"
        params = {"limit": 50}

        while url:
            data = self._get(url, params=params)
            tracks.extend(data["items"])
            url = data.get("next")
            params = None

        return tracks

    def get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """
        Fetch audio features for up to 100 tracks at once (Spotify's batch limit).
        Batching here is critical — one request per track would exhaust rate limits.
        """
        features = []
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i + 100]
            data = self._get(
                f"{_API_BASE}/audio-features",
                params={"ids": ",".join(batch_ids)},
            )
            # API returns null for tracks it cannot find — filter them out
            features.extend([f for f in data["audio_features"] if f is not None])

        return features
