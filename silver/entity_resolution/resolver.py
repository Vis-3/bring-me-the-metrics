"""
Entity resolution — matches cleaned Last.fm artist records to Spotify records
using fuzzy string similarity on artist names.

Three-tier strategy (decided in Phase 0):
  score >= 90  → auto_accepted
  score 70-89  → review_required (kept in Silver with flag, excluded from ML training)
  score < 70   → rejected (kept in Silver with flag, excluded from all analysis)
"""
from rapidfuzz import fuzz
from config import ENTITY_RESOLUTION_AUTO_ACCEPT, ENTITY_RESOLUTION_REVIEW_FLOOR


def _similarity(a: str, b: str) -> float:
    """
    Token sort ratio — handles word order differences better than simple ratio.
    "Bring Me the Horizon" vs "Bring Me The Horizon" scores 100.
    "The Acacia Strain" vs "Acacia Strain" scores higher than plain ratio would.
    """
    return fuzz.token_sort_ratio(a.lower().strip(), b.lower().strip())


def resolve(lastfm_record: dict, spotify_records: dict[str, dict]) -> dict:
    """
    Attempt to match a Last.fm artist to a Spotify artist record.

    lastfm_record    — cleaned Last.fm artist dict with 'lastfm_name' field
    spotify_records  — dict keyed by lastfm_name → Spotify Bronze record
                       (built once from the full Spotify Bronze file for this subgenre)

    Returns the lastfm_record enriched with Spotify fields and resolution status.
    """
    lastfm_name = lastfm_record["lastfm_name"]
    result = lastfm_record.copy()

    # Default — no Spotify data found
    result.update({
        "spotify_id": None,
        "spotify_name": None,
        "spotify_popularity": None,
        "spotify_followers": None,
        "spotify_genres": [],
        "entity_resolution_status": "rejected",
        "entity_resolution_score": 0.0,
    })

    # Look up Spotify record by Last.fm name (Bronze stored it keyed by lastfm_name)
    spotify_record = spotify_records.get(lastfm_name)
    if not spotify_record:
        # Try case-insensitive lookup
        lower_map = {k.lower(): v for k, v in spotify_records.items()}
        spotify_record = lower_map.get(lastfm_name.lower())

    if not spotify_record:
        return result

    spotify_artist = spotify_record.get("spotify_match", {})
    spotify_name = spotify_artist.get("name", "")

    # Compute similarity between Last.fm name and matched Spotify name
    score = _similarity(lastfm_name, spotify_name)

    # Assign resolution tier
    if score >= ENTITY_RESOLUTION_AUTO_ACCEPT:
        status = "auto_accepted"
    elif score >= ENTITY_RESOLUTION_REVIEW_FLOOR:
        status = "review_required"
    else:
        status = "rejected"

    result.update({
        "spotify_id": spotify_artist.get("id"),
        "spotify_name": spotify_name,
        "spotify_popularity": spotify_artist.get("popularity"),
        "spotify_followers": spotify_artist.get("followers", {}).get("total"),
        "spotify_genres": spotify_artist.get("genres", []),
        "entity_resolution_status": status,
        "entity_resolution_score": round(score, 2),
    })

    return result


def build_spotify_lookup(spotify_bronze_records: list[dict]) -> dict[str, dict]:
    """
    Build a name → record lookup dict from the Spotify Bronze file.
    Called once per subgenre run to avoid repeated list scans.
    """
    return {r["lastfm_name"]: r for r in spotify_bronze_records if r.get("lastfm_name")}
