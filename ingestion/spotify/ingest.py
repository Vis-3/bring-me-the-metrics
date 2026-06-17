"""
Spotify ingestion runner — uses Last.fm artist list as source of truth for
which artists belong to each subgenre, then enriches with Spotify audio features.

Run for a single subgenre:
    uv run python -m ingestion.spotify.ingest --subgenre deathcore

Run for all subgenres:
    uv run python -m ingestion.spotify.ingest --all
"""
import argparse
import gzip
import json
import boto3
from datetime import date
from ingestion.spotify.client import SpotifyClient
from ingestion.common.s3 import write_bronze
from config import SUBGENRES, S3_BUCKET_NAME, AWS_REGION


def load_lastfm_artists(subgenre: str, run_date: date) -> list[dict]:
    """
    Read the Last.fm tag_artists file from Bronze for this subgenre/date.
    Spotify ingestion depends on Last.fm having run first — Last.fm is the
    spine of the pipeline, Spotify enriches it.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"bronze/source=lastfm/subgenre={subgenre}/date={run_date.isoformat()}/tag_artists.json.gz"

    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        compressed = response["Body"].read()
        raw = gzip.decompress(compressed)
        return json.loads(raw)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Last.fm Bronze not found for {subgenre}/{run_date}. "
            "Run Last.fm ingestion first: uv run python -m ingestion.lastfm.ingest"
        )


def ingest_subgenre(client: SpotifyClient, subgenre: str, run_date: date) -> None:
    print(f"\n── Spotify ingestion: {subgenre} ──")

    # Load Last.fm artist list — this drives which artists we look up on Spotify
    lastfm_artists = load_lastfm_artists(subgenre, run_date)
    print(f"  Loaded {len(lastfm_artists)} artists from Last.fm Bronze")

    matched = 0
    not_found = 0
    all_track_data = []

    for artist in lastfm_artists:
        artist_name = artist.get("name", "")
        if not artist_name:
            continue

        # Step 1 — find artist on Spotify by name, pick highest popularity match
        result = client.find_artist(artist_name)
        if not result:
            print(f"  ✗ not found on Spotify: {artist_name}")
            not_found += 1
            continue

        spotify_artist = result["matched"]
        artist_id = spotify_artist["id"]
        matched += 1

        # Step 2 — fetch full discography
        albums = client.get_artist_albums(artist_id)

        # Step 3 — fetch tracks for each album
        all_tracks = []
        for album in albums:
            tracks = client.get_tracks(album["id"])
            for track in tracks:
                # Embed album context on each track — avoids cross-file joins in Bronze
                track["_album"] = {
                    "id": album["id"],
                    "name": album["name"],
                    "release_date": album["release_date"],
                    "album_type": album["album_type"],
                }
            all_tracks.extend(tracks)

        # Step 4 — batch fetch audio features for all tracks
        track_ids = [t["id"] for t in all_tracks if t.get("id")]
        audio_features = client.get_audio_features(track_ids) if track_ids else []
        features_by_id = {f["id"]: f for f in audio_features}

        for track in all_tracks:
            track["_audio_features"] = features_by_id.get(track["id"])
            # Embed the Spotify artist + search match context for Silver auditing
            track["_artist"] = spotify_artist
            track["_lastfm_name"] = artist_name
            track["_match_candidates"] = result["candidates"]

        all_track_data.extend(all_tracks)

        if matched % 10 == 0:
            print(f"  → {matched} artists matched, {not_found} not found")

    # Write all tracks for this subgenre in one file
    write_bronze(
        all_track_data,
        source="spotify",
        subgenre=subgenre,
        filename="tracks.json.gz",
        run_date=run_date,
    )

    print(f"  ✓ {subgenre} complete — {matched} matched, {not_found} not found on Spotify")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", type=str, help="Single subgenre to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all subgenres")
    parser.add_argument("--date", type=str, help="Override run date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    client = SpotifyClient()

    if args.all:
        for subgenre in SUBGENRES:
            ingest_subgenre(client, subgenre, run_date)
    elif args.subgenre:
        ingest_subgenre(client, args.subgenre, run_date)
    else:
        parser.print_help()


if __name__ == "__main__":

    main()
