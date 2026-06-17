"""Quick Silver quality check — unified artist record inspection."""
import io
import boto3
import pandas as pd
from config import S3_BUCKET_NAME, AWS_REGION

s3 = boto3.client("s3", region_name=AWS_REGION)

# ── Artists table ─────────────────────────────────────────────────────────────
key = "silver/table=artists/subgenre=deathcore/date=2026-06-17/artists.parquet"
df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)["Body"].read()))

print(f"Columns: {list(df.columns)}")
print(f"\nTotal records: {len(df)}")
print(f"MusicBrainz merged: {df['mbid'].notna().sum()}/{len(df)}")
print(f"Has country: {df['country'].notna().sum()}")
print(f"Has formed_year: {df['formed_year'].notna().sum()}")

print("\nSample records:")
print(df[["lastfm_name", "listeners", "country", "formed_year", "mb_name"]].head(10).to_string())

# ── Albums table ──────────────────────────────────────────────────────────────
key2 = "silver/table=albums/subgenre=betrayal_tracker/date=2026-06-17/albums.parquet"
df2 = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=S3_BUCKET_NAME, Key=key2)["Body"].read()))

print("\n── Albums table ──")
print(f"Total albums: {len(df2)}")
print(df2[["artist_name", "title", "type", "first_release_date", "release_year"]].head(15).to_string())
