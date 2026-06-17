"""
Last.fm ingestion runner — fetches tag artists, artist info, and weekly charts
per subgenre and writes raw JSON to S3 Bronze.

Run for a single subgenre:
    uv run python -m ingestion.lastfm.ingest --subgenre deathcore

Run for all subgenres:
    uv run python -m ingestion.lastfm.ingest --all

Historical backfill (5 years of weekly charts):
    uv run python -m ingestion.lastfm.ingest --all --weeks-back 260
"""
import argparse
from datetime import date
from ingestion.lastfm.client import LastFmClient
from ingestion.common.s3 import write_bronze
from config import SUBGENRES


def ingest_subgenre(client: LastFmClient, subgenre: str, run_date: date, weeks_back: int) -> None:
    print(f"\n── Last.fm ingestion: {subgenre} ──")

    # Step 1 — top artists for this tag (up to 500, Bronze gets all of them)
    print("  Fetching tag artists...")
    artists = client.get_tag_artists(tag=subgenre, max_pages=10)
    write_bronze(artists, source="lastfm", subgenre=subgenre, filename="tag_artists.json.gz", run_date=run_date)
    print(f"  ✓ {len(artists)} artists written to Bronze")

    # Step 2 — full artist info for each artist (listeners, playcounts, tags, bio)
    print("  Fetching artist info...")
    artist_infos = []
    for artist in artists:
        name = artist.get("name", "")
        if not name:
            continue
        info = client.get_artist_info(name)
        if info:
            # Embed the subgenre tag we used to find this artist —
            # Silver entity resolution needs this to assign subgenre correctly
            info["_source_tag"] = subgenre
            artist_infos.append(info)

    write_bronze(artist_infos, source="lastfm", subgenre=subgenre, filename="artist_info.json.gz", run_date=run_date)
    print(f"  ✓ {len(artist_infos)} artist info records written to Bronze")

    # Step 3 — weekly charts (historical time series for velocity features)
    print(f"  Fetching {weeks_back} weeks of charts...")
    charts = client.get_weekly_chart_artists(tag=subgenre, weeks_back=weeks_back)
    write_bronze(charts, source="lastfm", subgenre=subgenre, filename="weekly_charts.json.gz", run_date=run_date)
    print(f"  ✓ {len(charts)} weekly chart snapshots written to Bronze")

    print(f"  ✓ {subgenre} complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", type=str, help="Single subgenre to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all subgenres")
    parser.add_argument("--date", type=str, help="Override run date (YYYY-MM-DD), default=today")
    parser.add_argument("--weeks-back", type=int, default=52, help="Weeks of chart history to fetch (default=52)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    client = LastFmClient()

    if args.all:
        for subgenre in SUBGENRES:
            ingest_subgenre(client, subgenre, run_date, args.weeks_back)
    elif args.subgenre:
        ingest_subgenre(client, args.subgenre, run_date, args.weeks_back)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()