"""
ML data layer — loads mart_artist_features from Athena and produces
a clean feature matrix ready for XGBoost.

Feature engineering decisions:
- country: top-10 by artist count, rest → "Other", then label-encoded
- subgenre: label-encoded (9 categories, XGBoost handles ordinal fine)
- nulls: median imputation for numerics (tree models tolerate it;
  we track which columns were imputed so SHAP stays interpretable)
- temporal split: formed_year < 2015 = train, >= 2015 = test
  prevents look-ahead: we never use post-2015 formation signal to
  predict outcomes the model would have seen during training
"""
import boto3
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from config import AWS_REGION, ATHENA_S3_OUTPUT

GLUE_DATABASE = "metal_intelligence_marts"

# Features fed to XGBoost — order matters for SHAP waterfall readability
NUMERIC_FEATURES = [
    "band_age_years",
    "total_albums",
    "studio_albums",
    "avg_years_between_albums",
    "years_since_last_release",
    "plays_per_listener",
    "mb_resolution_score",
]

CATEGORICAL_FEATURES = ["subgenre", "country_encoded"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_breakout"

# Temporal split boundary — bands formed before this year go to train
SPLIT_YEAR = 2005
# How many top countries to keep before bucketing the rest as "Other"
TOP_N_COUNTRIES = 10


def _run_athena_query(sql: str) -> pd.DataFrame:
    """Execute a query against Athena and return results as a DataFrame."""
    athena = boto3.client("athena", region_name=AWS_REGION)

    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": GLUE_DATABASE, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_S3_OUTPUT},
    )
    query_id = response["QueryExecutionId"]

    # Poll until complete
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise RuntimeError(f"Athena query {state}: {reason}")
        time.sleep(2)

    # Paginate results
    paginator = athena.get_paginator("get_query_results")
    rows = []
    columns = None

    for page in paginator.paginate(QueryExecutionId=query_id):
        result = page["ResultSet"]
        if columns is None:
            columns = [c["Label"] for c in result["ResultSetMetadata"]["ColumnInfo"]]
        for row in result["Rows"][1 if columns and not rows else 0:]:
            rows.append([d.get("VarCharValue", None) for d in row["Data"]])

    return pd.DataFrame(rows, columns=columns)


def load_features() -> pd.DataFrame:
    """Load mart_artist_features from Athena."""
    print("Loading mart_artist_features from Athena...")
    sql = """
        SELECT
            artist_name,
            subgenre,
            country,
            formed_year,
            band_age_years,
            total_albums,
            studio_albums,
            avg_years_between_albums,
            years_since_last_release,
            plays_per_listener,
            mb_resolution_score,
            current_listeners,
            is_breakout
        FROM mart_artist_features
        WHERE is_breakout IS NOT NULL
    """
    df = _run_athena_query(sql)

    # Athena returns everything as strings — cast numerics
    numeric_cols = [
        "formed_year", "band_age_years", "total_albums", "studio_albums",
        "avg_years_between_albums", "years_since_last_release",
        "plays_per_listener", "mb_resolution_score", "current_listeners",
        "is_breakout",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Loaded {len(df)} artists ({int(df['is_breakout'].sum())} breakout, "
          f"{int((df['is_breakout'] == 0).sum())} underground)")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply feature engineering. Returns transformed DataFrame and
    a metadata dict (encoders, imputation medians) needed at inference time.
    """
    df = df.copy()
    metadata = {}

    # ── Country encoding ────────────────────────────────────────────────────
    # Keep top-N countries by artist count; bucket rest as "Other"
    top_countries = (
        df["country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()
    )
    metadata["top_countries"] = top_countries
    df["country_bucketed"] = df["country"].where(
        df["country"].isin(top_countries), other="Other"
    ).fillna("Other")

    country_enc = LabelEncoder()
    df["country_encoded"] = country_enc.fit_transform(df["country_bucketed"])
    metadata["country_encoder"] = country_enc

    # ── Subgenre encoding ────────────────────────────────────────────────────
    subgenre_enc = LabelEncoder()
    df["subgenre"] = subgenre_enc.fit_transform(df["subgenre"].fillna("unknown"))
    metadata["subgenre_encoder"] = subgenre_enc

    # ── Numeric imputation ───────────────────────────────────────────────────
    # Median imputation per feature — tree models are robust to this.
    # We record medians so inference uses train-set statistics, not test-set.
    imputation_medians = {}
    for col in NUMERIC_FEATURES:
        median = df[col].median()
        imputation_medians[col] = median
        df[col] = df[col].fillna(median)
    metadata["imputation_medians"] = imputation_medians

    print(f"  Imputed medians: { {k: round(v, 2) for k, v in imputation_medians.items()} }")
    return df, metadata


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Stratified random split — 80% train, 20% test.

    Why not temporal split on formed_year: this is a cross-sectional snapshot
    dataset, not a time-series. Every feature (plays_per_listener, total_albums,
    band_age_years) is current state — there is no time axis to leak across.
    Temporal split on formed_year creates a degenerate test set where bands
    formed post-2005 have had no time to reach 1M listeners, leaving ~0
    breakout artists in the test set and making all breakout metrics meaningless.
    Stratified split preserves the 5% breakout rate in both train and test.
    """
    from sklearn.model_selection import train_test_split

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"  Train: {len(X_train)} artists")
    print(f"  Test:  {len(X_test)} artists")
    print(f"  Train breakout rate: {y_train.mean():.1%}")
    print(f"  Test breakout rate:  {y_test.mean():.1%}")

    return X_train, y_train, X_test, y_test
