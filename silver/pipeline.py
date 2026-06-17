"""
Silver pipeline runner — Bronze → Silver for all subgenres.

Steps per subgenre (order matters — see cleaner.py):
  1. Read Last.fm artist info from Bronze
  2. Clean, type-cast, apply 50K listener floor
  3. Deduplicate artists across subgenre tags
  4. Read MusicBrainz metadata from Bronze
  5. Merge MusicBrainz fields (formed_year, country, mbid) into unified record
  6. Write unified Parquet to Silver

Separate run for Betrayal Tracker albums:
  --albums flag writes silver/table=albums/ from MusicBrainz Bronze

Run for a single subgenre:
    uv run python -m silver.pipeline --subgenre deathcore

Run for all subgenres:
    uv run python -m silver.pipeline --all

Run album table:
    uv run python -m silver.pipeline --albums
"""
import argparse
import io
import boto3
import pandas as pd
from datetime import date

from silver.cleaning.bronze_reader import (
    read_lastfm_artist_info,
    read_lastfm_weekly_charts,
    read_musicbrainz_metadata,
    read_musicbrainz_albums,
)
from silver.cleaning.cleaner import clean_lastfm_artist_info, deduplicate_artists
from silver.writer import write_silver_artists
from config import SUBGENRES, S3_BUCKET_NAME, AWS_REGION


def build_mb_lookup(mb_records: list[dict]) -> dict[str, dict]:
    """Index MusicBrainz records by lastfm_name for O(1) merge lookup."""
    return {r["lastfm_name"]: r for r in mb_records if r.get("lastfm_name")}


def merge_musicbrainz(lastfm_record: dict, mb_lookup: dict) -> dict:
    """
    Merge MusicBrainz fields into a cleaned Last.fm artist record.
    Only auto_accepted matches are merged — review/rejected records
    contribute no MusicBrainz data rather than potentially wrong data.
    """
    result = lastfm_record.copy()
    mb = mb_lookup.get(lastfm_record["lastfm_name"])

    if mb and mb.get("resolution_status") == "auto_accepted":
        result.update({
            "mbid": mb.get("mbid"),
            "mb_name": mb.get("mb_name"),
            "country": mb.get("country"),
            "formed_year": mb.get("formed_year"),  # replaces regex-extracted value
            "mb_resolution_score": mb.get("similarity_score"),
        })
    else:
        result.update({
            "mbid": None,
            "mb_name": None,
            "country": None,
            "mb_resolution_score": None,
            # keep formed_year from Last.fm bio regex as fallback
        })

    return result


def run_subgenre(subgenre: str, run_date: date) -> None:
    print(f"\n── Silver pipeline: {subgenre} ──")

    # Step 1+2+3 — read, clean, filter, deduplicate
    raw = read_lastfm_artist_info(subgenre, run_date)
    print(f"  Read {len(raw)} raw Last.fm records")
    cleaned = clean_lastfm_artist_info(raw)
    print(f"  {len(cleaned)} after 50K listener floor")
    deduped = deduplicate_artists(cleaned)
    print(f"  {len(deduped)} after deduplication")

    # Step 4+5 — merge MusicBrainz
    mb_raw = read_musicbrainz_metadata(subgenre, run_date)
    mb_lookup = build_mb_lookup(mb_raw)
    print(f"  Loaded {len(mb_lookup)} MusicBrainz records")

    unified = [merge_musicbrainz(record, mb_lookup) for record in deduped]

    mb_merged = sum(1 for r in unified if r.get("mbid"))
    print(f"  MusicBrainz merged: {mb_merged}/{len(unified)} artists")

    # Step 6 — write
    write_silver_artists(unified, subgenre, run_date)
    print(f"  ✓ {subgenre} complete")


def run_weekly_charts(subgenre: str, run_date: date) -> None:
    """
    Flatten Bronze weekly chart snapshots → Silver weekly_charts table.
    Bronze stores one record per week (the raw Last.fm response).
    Silver produces one row per artist per week — the temporal spine for dbt models.
    """
    print(f"\n── Silver weekly charts: {subgenre} ──")

    raw_weeks = read_lastfm_weekly_charts(subgenre, run_date)
    if not raw_weeks:
        print(f"  ✗ no weekly chart data found in Bronze for {subgenre}")
        return

    print(f"  Read {len(raw_weeks)} weekly snapshots from Bronze")

    rows = []
    for week in raw_weeks:
        week_start = week.get("_week_start")
        if not week_start:
            continue

        # Last.fm tag.gettopartists response nests artists under "topartists" > "artist"
        # tag.getweeklychartlist returns metadata only — handle both shapes defensively
        artists = []
        top = week.get("topartists", {})
        if isinstance(top, dict):
            artists = top.get("artist", [])
        elif isinstance(top, list):
            artists = top

        for artist in artists:
            if not isinstance(artist, dict):
                continue
            name = artist.get("name", "")
            listeners = artist.get("listeners") or artist.get("playcount")
            if not name or not listeners:
                continue
            try:
                rows.append({
                    "artist_name": name,
                    "week_start": week_start,
                    "listeners": int(listeners),
                    "subgenre": subgenre,
                })
            except (ValueError, TypeError):
                continue

    if not rows:
        print("  ✗ no artist-week rows extracted — check Bronze chart format")
        return

    print(f"  Extracted {len(rows)} artist-week rows")
    df = pd.DataFrame(rows)

    key = f"silver/table=weekly_charts/subgenre={subgenre}/date={run_date.isoformat()}/weekly_charts.parquet"
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
    parquet_bytes = buffer.getvalue()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=parquet_bytes)
    print(f"  ✓ wrote s3://{S3_BUCKET_NAME}/{key} ({len(df)} rows, {len(parquet_bytes):,} bytes)")


def run_albums(run_date: date) -> None:
    """Write Silver album table from MusicBrainz Bronze — feeds Betrayal Tracker."""
    print("\n── Silver pipeline: albums (betrayal tracker) ──")

    albums = read_musicbrainz_albums(run_date)
    if not albums:
        print("  ✗ no album data found in Bronze")
        return

    print(f"  Read {len(albums)} album records from Bronze")
    df = pd.DataFrame(albums)

    # Parse release year from first_release_date for easy filtering in dbt
    df["release_year"] = pd.to_datetime(
        df["first_release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    key = f"silver/table=albums/subgenre=betrayal_tracker/date={run_date.isoformat()}/albums.parquet"
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
    parquet_bytes = buffer.getvalue()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=parquet_bytes)
    print(f"  ✓ wrote s3://{S3_BUCKET_NAME}/{key} ({len(df)} records, {len(parquet_bytes):,} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--albums", action="store_true", help="Write Silver album table")
    parser.add_argument("--weekly-charts", action="store_true", help="Write Silver weekly charts table")
    parser.add_argument("--date", type=str)
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()

    if args.albums:
        run_albums(run_date)
    elif args.weekly_charts:
        if args.all:
            for subgenre in SUBGENRES:
                run_weekly_charts(subgenre, run_date)
        elif args.subgenre:
            run_weekly_charts(args.subgenre, run_date)
        else:
            parser.print_help()
    elif args.all:
        for subgenre in SUBGENRES:
            run_subgenre(subgenre, run_date)
    elif args.subgenre:
        run_subgenre(args.subgenre, run_date)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
