"""
Bronze → Silver cleaning pipeline.
Steps in order: parse → filter → deduplicate → type cast.
Entity resolution happens after this, in entity_resolution/resolver.py.
"""
from datetime import date


def parse_listeners(value) -> int | None:
    """Last.fm returns listener counts as strings — cast safely."""
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def clean_lastfm_artist_info(raw_records: list[dict]) -> list[dict]:
    """
    Clean raw Last.fm artist info records.
    Returns one dict per artist with typed, normalised fields.
    """
    cleaned = []
    for record in raw_records:
        stats = record.get("stats", {})
        listeners = parse_listeners(stats.get("listeners"))
        play_count = parse_listeners(stats.get("playcount"))

        # Apply 50K listener floor — below this, signal is too weak for analysis.
        # Filter happens here in Silver, not Bronze, so Bronze stays complete.
        if listeners is None or listeners < 50_000:
            continue

        # Extract tags as a list of strings with the ingestion date as valid_from.
        # valid_to is null until the tag changes — SCD Type 2 pattern.
        tags = record.get("tags", {}).get("tag", [])
        if isinstance(tags, dict):
            tags = [tags]  # Last.fm returns a dict when there's only one tag
        subgenre_tags = [
            {
                "tag": t.get("name", "").lower().strip(),
                "valid_from": date.today().isoformat(),
                "valid_to": None,
            }
            for t in tags if t.get("name")
        ]

        # Extract formed year from bio summary if available
        bio = record.get("bio", {})
        formed_year = None
        content = bio.get("content", "") or bio.get("summary", "")
        # Simple heuristic: look for 4-digit year after "formed in" or "founded in"
        import re
        year_match = re.search(r"(?:formed|founded|started|began)[^\d]{0,10}(\d{4})", content, re.IGNORECASE)
        if year_match:
            year = int(year_match.group(1))
            if 1960 <= year <= date.today().year:
                formed_year = year

        cleaned.append({
            "lastfm_name": record.get("name", "").strip(),
            "lastfm_mbid": record.get("mbid", ""),
            "listeners": listeners,
            "play_count": play_count,
            "subgenre_tags": subgenre_tags,
            "source_tag": record.get("_source_tag", ""),
            "formed_year": formed_year,
            "bio_summary": bio.get("summary", "")[:500],  # truncate for storage
        })

    return cleaned


def deduplicate_artists(records: list[dict]) -> list[dict]:
    """
    Deduplicate by lastfm_name — same artist can appear in multiple subgenre tags.
    When duplicates exist, keep the record with the highest listener count
    (most complete data) and merge subgenre_tags from all occurrences.

    Deduplication must happen before entity resolution to avoid duplicate
    Spotify lookups producing conflicting matches for the same artist.
    """
    seen: dict[str, dict] = {}

    for record in records:
        name = record["lastfm_name"].lower().strip()

        if name not in seen:
            seen[name] = record
        else:
            existing = seen[name]
            # Keep higher listener count record as the base
            if (record["listeners"] or 0) > (existing["listeners"] or 0):
                record["subgenre_tags"] = existing["subgenre_tags"] + record["subgenre_tags"]
                seen[name] = record
            else:
                # Merge tags from this duplicate into the existing record
                existing["subgenre_tags"] = existing["subgenre_tags"] + record["subgenre_tags"]

    # Deduplicate tags within each artist (same tag can come from multiple subgenres)
    for record in seen.values():
        unique_tags = {t["tag"]: t for t in record["subgenre_tags"]}
        record["subgenre_tags"] = list(unique_tags.values())

    return list(seen.values())
