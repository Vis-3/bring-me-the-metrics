"""
MusicBrainz ingestion runner — enriches Last.fm artist list with:
  - formed_year (structured, from life-span)
  - country (geographic origin)
  - album release dates (for Betrayal Tracker)

Replaces Spotify as the second data source. Fully open, no API key needed.

Run for a single subgenre:
    uv run python -m ingestion.musicbrainz.ingest --subgenre deathcore

Run for all subgenres:
    uv run python -m ingestion.musicbrainz.ingest --all

Fetch albums for Betrayal Tracker artists:
    uv run python -m ingestion.musicbrainz.ingest --albums
"""
import argparse
import gzip
import json
import boto3
from datetime import date
from ingestion.musicbrainz.client import MusicBrainzClient
from ingestion.common.s3 import write_bronze
from config import SUBGENRES, S3_BUCKET_NAME, AWS_REGION


def load_lastfm_artists(subgenre: str, run_date: date) -> list[dict]:
    """Read Last.fm tag_artists from Bronze — MusicBrainz enriches the same list."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"bronze/source=lastfm/subgenre={subgenre}/date={run_date.isoformat()}/tag_artists.json.gz"
    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return json.loads(gzip.decompress(response["Body"].read()))
    except Exception:
        raise FileNotFoundError(
            f"Last.fm Bronze not found for {subgenre}/{run_date}. "
            "Run Last.fm ingestion first."
        )


def ingest_subgenre(client: MusicBrainzClient, subgenre: str, run_date: date) -> None:
    print(f"\n── MusicBrainz ingestion: {subgenre} ──")

    lastfm_artists = load_lastfm_artists(subgenre, run_date)
    print(f"  Loaded {len(lastfm_artists)} artists from Last.fm Bronze")

    enriched = []
    matched = 0
    review = 0
    rejected = 0

    for i, artist in enumerate(lastfm_artists):
        name = artist.get("name", "")
        if not name:
            continue

        result = client.find_artist(name)
        if not result:
            rejected += 1
            continue

        status = result["resolution_status"]
        if status == "auto_accepted":
            matched += 1
        elif status == "review_required":
            review += 1
        else:
            rejected += 1

        enriched.append({
            "lastfm_name": name,
            **result,
        })

        if (i + 1) % 50 == 0:
            print(f"  → {i + 1}/{len(lastfm_artists)} — matched: {matched}, review: {review}, rejected: {rejected}")

    write_bronze(
        enriched,
        source="musicbrainz",
        subgenre=subgenre,
        filename="artist_metadata.json.gz",
        run_date=run_date,
    )
    print(f"  ✓ {subgenre} complete — matched: {matched}, review: {review}, rejected: {rejected}")


def ingest_albums(client: MusicBrainzClient, run_date: date) -> None:
    """
    Fetch full discography for Betrayal Tracker focus artists.
    Stored under 'betrayal_tracker' subgenre partition — cross-genre concern.
    """
    betrayal_artists = [
        "Bring Me the Horizon",
        "Asking Alexandria",
        "In Flames",
        "Architects",
        "Spiritbox",
        "Bad Omens",
        "Motionless in White",
        "Parkway Drive",
    ]

    print("\n── MusicBrainz album ingestion: betrayal tracker ──")
    all_albums = []

    for name in betrayal_artists:
        result = client.find_artist(name)
        if not result or not result.get("mbid"):
            print(f"  ✗ not found: {name}")
            continue

        albums = client.get_artist_albums(result["mbid"])
        for album in albums:
            album["artist_name"] = name
            album["artist_mbid"] = result["mbid"]

        all_albums.extend(albums)
        print(f"  ✓ {name}: {len(albums)} releases")

    write_bronze(
        all_albums,
        source="musicbrainz",
        subgenre="betrayal_tracker",
        filename="artist_albums.json.gz",
        run_date=run_date,
    )
    print(f"  ✓ {len(all_albums)} total album records written to Bronze")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--albums", action="store_true", help="Fetch albums for Betrayal Tracker artists")
    parser.add_argument("--date", type=str)
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    client = MusicBrainzClient()

    if args.albums:
        ingest_albums(client, run_date)
    elif args.all:
        for subgenre in SUBGENRES:
            ingest_subgenre(client, subgenre, run_date)
    elif args.subgenre:
        ingest_subgenre(client, args.subgenre, run_date)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
