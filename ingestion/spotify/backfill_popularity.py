"""
Backfill missing popularity and followers fields in Spotify Bronze.

The search endpoint returns a simplified artist object without popularity/followers.
This script reads existing Bronze, fetches full artist objects by ID for any
records missing these fields, and overwrites Bronze with enriched data.

Run for a single subgenre:
    uv run python -m ingestion.spotify.backfill_popularity --subgenre deathcore

Run for all subgenres:
    uv run python -m ingestion.spotify.backfill_popularity --all
"""
import argparse
import gzip
import json
import boto3
from datetime import date
from ingestion.spotify.client import SpotifyClient
from config import SUBGENRES, S3_BUCKET_NAME, AWS_REGION


def backfill_subgenre(client: SpotifyClient, subgenre: str, run_date: date) -> None:
    print(f"\n── Backfill popularity: {subgenre} ──")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"bronze/source=spotify/subgenre={subgenre}/date={run_date.isoformat()}/artist_popularity.json.gz"

    # Read existing Bronze
    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        records = json.loads(gzip.decompress(response["Body"].read()))
    except Exception:
        print(f"  ✗ no Bronze found for {subgenre}/{run_date}")
        return

    print(f"  Read {len(records)} existing records")

    enriched = 0
    for record in records:
        matched = record.get("spotify_match", {})
        # Only fetch if popularity is missing — avoids unnecessary API calls
        if matched.get("popularity") is None and matched.get("id"):
            full_artist = client._get(f"https://api.spotify.com/v1/artists/{matched['id']}")
            if full_artist:
                record["spotify_match"] = full_artist
                enriched += 1

    print(f"  Enriched {enriched} records with popularity + followers")

    # Overwrite Bronze with enriched records — same key, same format
    compressed = gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"))
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=compressed,
        ContentEncoding="gzip",
        ContentType="application/json",
    )
    print(f"  ✓ Bronze overwritten with enriched data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--date", type=str)
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    client = SpotifyClient()

    if args.all:
        for subgenre in SUBGENRES:
            backfill_subgenre(client, subgenre, run_date)
    elif args.subgenre:
        backfill_subgenre(client, args.subgenre, run_date)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
