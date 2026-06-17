# Metal Music Intelligence Pipeline — Interview Study Guide

## Project Overview

A production-grade batch ELT pipeline ingesting Last.fm API (listener counts, scrobble trends, genre tags, weekly charts) and MusicBrainz API (formed year, country, album release dates) data into AWS S3 (medallion architecture), transformed with dbt on Athena, validated with Great Expectations, orchestrated with Airflow. ML system predicts which underground metal bands are about to break through to mainstream using XGBoost.

**Three analytical pillars:**
1. **Scene Lifecycle** — how metal subgenres evolved in popularity 2015-2025 using Last.fm listener velocity and tag diversity
2. **Betrayal Tracker** — which bands changed sound and did it help or hurt them (listener trajectory correlated with album releases)
3. **Breakout Predictor** — XGBoost classifier predicting which underground bands will cross 1M listeners

**Tech stack:** Python 3.12, uv, Last.fm API, MusicBrainz API, AWS S3, AWS Athena, dbt Core, Great Expectations, Apache Airflow, Parquet, XGBoost, Scikit-learn, MLflow, Docker, GitHub Actions, Medallion Architecture (Bronze/Silver/Gold)

---

## Phase 0 — Architecture & Design Decisions

### Medallion Architecture

Three layers: Bronze (raw) → Silver (cleaned) → Gold (analytical marts)

- **Bronze** = raw gzipped JSON, immutable, recovery-only, never queried analytically
- **Silver** = cleaned Parquet, query-optimized, feeds dbt
- **Gold** = dbt analytical marts, business-ready aggregations

> **Q: What is medallion architecture and why did you choose it?**
> Separates storage concerns (where data lives) from transformation concerns (SQL logic). Bronze is the source of truth for recovery — if parsing logic has a bug, reprocess from Bronze without re-calling APIs. Silver is query-optimized Parquet that dbt reads. Gold is the final analytical layer. Each layer has a clear owner and responsibility.

---

### Partition Strategy

**Bronze:** `source=lastfm/subgenre=deathcore/date=2024-01-15/`
- Source first — recovery by source when parser breaks
- Subgenre second — subgenre-specific parsing bugs
- Date last — surgical precision for specific run dates

**Silver:** `table=artists/subgenre=deathcore/date=2024-01-15/`
- Subgenre first — most analytical queries filter by subgenre (Scene Lifecycle, Betrayal Tracker)
- Athena skips 8 of 9 subgenre folders immediately

> **Q: Why does partition order matter in S3/Athena?**
> Athena prunes partitions left-to-right. If date comes first, a query filtering by subgenre must open every date folder (thousands) before pruning. Subgenre first means Athena opens 1 of 9 folders and skips the rest. This is partition pruning — directly affects query cost (Athena is billed per byte scanned).

> **Q: Why is Bronze partitioned differently than Silver?**
> Bronze is write-optimized and recovery-optimized, not query-optimized. The most likely recovery operation is "reprocess everything from one broken source" — so source comes first. Silver is query-optimized because that's what dbt and analysts read.

---

### Bronze Format

**Choice:** Gzipped JSON (`.json.gz`)

> **Q: Why keep raw JSON in Bronze instead of converting to Parquet immediately?**
> "Raw" means untransformed content, not uncompressed file. Bronze is the audit trail — APIs change their response format over time. If Silver parsing logic had a bug six months ago, Bronze lets you reprocess with corrected logic without re-calling the API (which may be rate-limited or returning different data now). gzip reduces storage 60-80% while keeping JSON intact and Athena-readable. The immutability principle: Bronze is never modified, only appended.

---

### Data Modeling

**Chose:** Star schema for analytical marts, One Big Table (OBT) for ML feature table

| Model | Structure | Best for |
|---|---|---|
| Star schema | Fact + dimension tables | SQL analysts, aggregations |
| Snowflake schema | Normalized dimensions | Rarely worth it — storage is cheap |
| One Big Table | Everything denormalized | ML — XGBoost needs one flat row per observation |
| Data Vault | Hubs/links/satellites | Regulatory compliance only |

> **Q: Why didn't you use Data Vault given you needed historical tag tracking?**
> Data Vault solves regulatory audit requirements — proving to a regulator what a record said on a specific date. Our requirement was analytical: routing artists to the right subgenre bucket based on when they were tagged. SCD Type 2 with valid_from/valid_to on the subgenre_tags array handles this with a fraction of the complexity.

---

### Slowly Changing Dimensions (SCD)

**Problem:** BMTH was tagged "deathcore" in 2010, "metalcore/pop-rock" by 2018. Simple row updates destroy historical tag data needed for Scene Lifecycle analysis.

**Solution:** SCD Type 2 on `subgenre_tags` only — stored as array of structs:
```python
subgenre_tags = [
    {"tag": "deathcore", "valid_from": "2008-01-01", "valid_to": "2015-01-01"},
    {"tag": "metalcore", "valid_from": "2015-01-01", "valid_to": None}
]
```

**Why only subgenre tags?** Other dimensions (artist name, formed year, country) can be SCD Type 1 (overwrite) — their history doesn't drive any analytical question.

> **Q: What are the three SCD types?**
> - Type 1 = overwrite (simple, loses history, use for fixing typos)
> - Type 2 = new row with valid_from/valid_to (full history, use when history drives analysis)
> - Type 3 = add previous_value column (tracks one change only, lazy compromise)

---

### Entity Resolution

**Problem:** Last.fm and MusicBrainz are independent databases with no shared IDs. "Bring Me the Horizon" vs "Bring Me The Horizon" — different string, same artist.

**Three-tier strategy:**
- Score ≥ 90% → `auto_accepted`
- Score 70–89% → `review_required` (kept in Silver with flag, excluded from ML training)
- Score < 70% → `rejected` (kept in Silver with flag, excluded from analysis)

**Tool:** rapidfuzz `token_sort_ratio` — handles word order differences ("The Acacia Strain" vs "Acacia Strain")

> **Q: How did you handle entity resolution?**
> Three-tier fuzzy matching. Auto-accept near-exact matches, flag ambiguous ones with a status column, reject poor matches but never delete them. The flag column is critical — silently dropping records makes data quality problems invisible. A data engineer six months later can audit every rejected match and understand why. Null is better than wrong data.

---

### ML Design Decisions

**Breakout label definition:**
- Positive class (breakout): crossed 1M listeners
- Negative class (stayed underground): stayed below 200K for 3+ years
- **Rejection region: 200K–1M excluded from training entirely**

> **Q: Why exclude the 200K–1M range from training?**
> Artists in this range are ambiguous — slowly breaking out, peaked early, or mid-tier forever. Training on ambiguous labels teaches the model noise. The rejection region ensures only clearly-labeled examples enter training, producing a cleaner decision boundary.

**Temporal train/test split (not random):**
- Train: pre-2022 data
- Test: 2022–2024

> **Q: Why temporal split instead of random?**
> Prevents temporal leakage. With random split, the model sees 2021 listener velocity during training while already knowing the 2022 breakout outcome (it's in the test set). This inflates accuracy — the model "predicts" outcomes it already knew. Temporal split ensures the model never sees the future during training.

---

## Phase 1 — Data Ingestion (Bronze Layer)

### API Client Design

**Client credentials vs OAuth (Spotify, before restrictions):**
OAuth = "act on behalf of a specific user" (browser popup required). Client credentials = "I am an application accessing public catalogue data." Audio features and artist metadata are public — no user context needed, simpler to automate.

**Proactive token refresh (chose approach B):**
Spotify tokens expire after 60 minutes. Refresh at 55 minutes rather than reacting to a 401. Reactive approach loses the request that triggered the 401. Proactive approach prevents the problem entirely.

**Rate limiting strategy:**
Proactive delay (0.5s between requests) + reactive Retry-After header on 429. APIs don't publish exact limits — small delay keeps us under, Retry-After tells us exactly how long to wait when we do get limited.

**Retry logic:**
- 5xx (server errors) → exponential backoff, retry
- 4xx (client errors, our bug) → fail immediately, retrying won't help
- 429 → use Retry-After header value exactly

---

### Spotify API Restrictions — Critical Lesson

**What happened (November 2024 Spotify API changes):**
- `/audio-features` → 403 Forbidden
- Artist popularity, followers, genres → null even from full artist endpoint
- `/artists/{id}/albums` → 400 "Invalid limit" (include_groups comma URL-encoded as %2C)

**What we tried:**
1. `genre:deathcore` search → 400 (not in Spotify taxonomy)
2. include_groups with comma in params → URL-encoded, rejected
3. Embedding params in URL string → still rejected
4. Reducing limit to 20 → still 400
5. Full artist endpoint → stripped object, only id/name/uri returned

**Decision:** Remove Spotify from active pipeline entirely.
- Spotify provided zero analytical value after restrictions
- Last.fm + MusicBrainz cover all three pillars
- Maintaining dead infrastructure is worse than removing it
- Code kept as documentation of the attempt

> **Q: What happened when a key data source was restricted mid-build?**
> Audited what each analytical pillar actually required. Discovered Spotify's restricted endpoints provided nothing Last.fm and MusicBrainz didn't already give us. Removed it rather than over-engineering around the restriction. Mature engineering judgment — don't maintain dead infrastructure.

> **Q: What would you do differently?**
> Test API endpoint availability and response schema before committing to the data model. A 30-minute API exploration script at the start would have caught Spotify's restrictions and informed source selection earlier.

---

### Final Data Source Architecture

| Source | Provides | Rate Limit | Auth |
|---|---|---|---|
| Last.fm | Listeners, play counts, weekly charts, genre tags, artist similarity | 5 req/sec | API key |
| MusicBrainz | Formed year, country, album release dates | 1 req/sec | None (User-Agent required) |
| Spotify | Removed — restrictions eliminated all useful data | — | — |

**Last.fm as pipeline spine:** Last.fm's community-driven tag system is more reliable genre signal than Spotify's internal taxonomy. `genre:deathcore` on Spotify returns 400. Last.fm has thousands of fans who tagged Lorna Shore as deathcore — that's ground truth. Pipeline flow: Last.fm artist list → MusicBrainz enrichment → unified Silver record.

---

### MusicBrainz Artist Matching Strategy

1. Search by artist name, limit 5 candidates
2. Filter to `type=Group` — eliminates solo artists, orchestras, fictional characters
3. Among Groups, pick highest MusicBrainz search score
4. Validate with rapidfuzz — same three-tier thresholds

**Why MusicBrainz for albums instead of Spotify:**
- Spotify rate-limited album fetch after 6 artists (84,327 second Retry-After returned)
- MusicBrainz is fully open, 1 req/sec, no quota restrictions
- release-groups endpoint gives cleaner data (one entry per album, not per regional release edition)

**Deathcore results:** 452/500 auto-accepted (90.4%), 13 review, 35 rejected

---

### Backfill Pattern

When Spotify Bronze was missing popularity/followers fields: targeted enrichment rather than full re-ingestion. Read existing Bronze, fetch only missing fields by ID, overwrite Bronze with enriched records.

**Bronze immutability exception:** Justified when fixing a structural data gap discovered immediately after ingestion, before downstream processing consumed the data. In production: log as a data quality incident with before/after snapshots.

---

## Phase 2 — Silver Layer (Cleaning & Unification)

### Pipeline Order of Operations

Order matters — cannot be changed:
1. Parse and type-cast (Last.fm returns listener counts as strings)
2. Apply 50K listener floor filter
3. Deduplicate artists
4. Merge MusicBrainz metadata
5. Write unified Parquet

> **Q: Why deduplicate before merging MusicBrainz?**
> Same artist can appear in multiple subgenre Bronze files (Architects tagged as both metalcore and melodic metalcore). If you merge first, you run entity resolution twice and potentially get conflicting matches, producing two rows for the same artist. Deduplicate first — one canonical record — then one clean merge attempt.

---

### 50K Listener Floor

**Applied in Silver, not Bronze.**

- Well below 200K underground class boundary — full underground artist pool captured
- Above hobby-level artists (100–1,000 listeners) who add noise without signal
- Bronze stays complete — threshold wrong? Reprocess from Bronze, no API re-calls

> **Q: Where in the pipeline do you apply business filters?**
> Silver, not Bronze. Bronze is immutable — applying filters there means re-calling APIs if the threshold changes. Silver is where business logic lives. This lets you change analytical decisions (50K → 75K) without touching the data source layer.

---

### Deduplication Strategy

Same artist in multiple subgenre files: keep record with highest listener count as base, merge subgenre_tags arrays from all duplicates, deduplicate tags within merged array.

Tag merge preserves cross-genre information — tag_diversity_score is itself a breakout predictor feature.

---

### MusicBrainz Merge — Conditional

Only `auto_accepted` matches merged. `review_required` and `rejected` records retain Last.fm data only — MusicBrainz fields left null.

**Reasoning:** A wrong match corrupts formed_year and country. Null is better than wrong. Flag column lets analysts identify which records lack MusicBrainz enrichment.

---

### Silver Schema (Unified Artist Record)

| Field | Source | Notes |
|---|---|---|
| lastfm_name | Last.fm | Primary key |
| listeners, play_count | Last.fm | Core metrics |
| subgenre_tags | Last.fm | Array of structs with valid_from/valid_to |
| formed_year | MusicBrainz | Replaces fragile bio regex |
| country | MusicBrainz | Geographic analysis |
| mbid, mb_name | MusicBrainz | Stable identifier |
| mb_resolution_score | Computed | Match quality audit |

**Two Silver tables:**
- `artists` — one row per artist, all sources merged, partitioned by subgenre
- `albums` — one row per album (one-to-many), partitioned under betrayal_tracker

---

### Parquet Format Choice

| Reason | Detail |
|---|---|
| Columnar storage | Athena reads only queried columns — major cost saving |
| Snappy compression | Smaller than gzip for columnar data, faster decompression |
| Schema enforcement | Data types preserved, no string/int ambiguity |
| Nested type support | subgenre_tags array of structs via PyArrow |

---

### Silver Results

- **Deathcore artists:** 136 records (from 500 Bronze, 364 below 50K floor)
- **MusicBrainz merged:** 128/136 (94%)
- **Has country:** 92/136
- **Has formed_year:** 124/136
- **Albums:** 388 releases across 8 Betrayal Tracker bands

---

## Failure Modes & Fixes

| Failure | Root Cause | Fix | Lesson |
|---|---|---|---|
| `genre:deathcore` → 400 | Not in Spotify taxonomy | Switch to Last.fm tag.getTopArtists | Test API search assumptions early |
| include_groups → 400 | Comma URL-encoded as %2C | Filter album_type client-side | Special chars in params break some APIs |
| audio-features → 403 | Spotify Extended Quota restriction | Remove from pipeline, pivot pillars | Third-party APIs change — design for fallback sources |
| 84,327s rate limit | Rapid sequential album fetches | 0.5s delay, switch to MusicBrainz | Proactive delays aren't enough for large sequential batches |
| Silver writer 0 bytes | buffer.tell() returns 0 after seek(0) | Capture len(buffer.getvalue()) before seek | In-memory buffer position tracking |
| MusicBrainz 503 | Server overload (volunteer-operated) | 5s × attempt backoff in retry logic | Open source APIs have lower uptime SLA |
| dbt DataCatalog not found | `database: metal_intelligence` in profiles.yml | Change to `database: AwsDataCatalog` | Athena catalog ≠ Glue database — database field is the catalog name |
| dbt AccessDeniedException glue:GetTableVersions | IAM policy missing catalog-level Glue action | Add inline policy granting glue:GetTableVersions on * | AWSGlueConsoleFullAccess doesn't cover all catalog-level Glue calls |
| formed_year DOUBLE vs INT mismatch | pandas writes nullable Int64 as DOUBLE in Parquet | Change Athena schema to DOUBLE for nullable int columns | Always check actual Parquet schema, not assumed pandas dtype |
| release_year INT64 vs DOUBLE mismatch | Over-corrected — changed to DOUBLE but Parquet was INT64 | Change back to BIGINT | Type mismatches go both directions — read the error message carefully |
| al.years_since_last_release not found | Referenced a computed alias from a CTE that doesn't have it | Inline the expression in the CASE statement directly | SQL CTEs don't forward aliases — re-express or use a subquery |
| Breakout F1 ≈ 0 with temporal split | Bands formed post-2005 haven't had time to reach 1M listeners — test set had 2 breakouts | Switch to stratified random split — cross-sectional data has no time axis to leak across | Split strategy depends on data structure, not general rules |
| XGB eval metric "aucpr:nan" | Test set had 0 positive class — can't compute PR curve | Move split year earlier, then fix root cause with stratified split | Always check class distribution in train AND test before training |

---

## Phase 3 — Gold Layer (dbt + Athena)

### dbt Architecture

**Why dbt for the Gold layer?**
dbt solves three problems: version-controlled SQL (transformations are code, not ClickOps), dependency graph (dbt knows `mart_album_legacy` depends on `stg_artists` and `stg_albums` and runs them in order), and built-in testing (not_null, uniqueness tests run as part of `dbt test`).

**Materialization choices:**

| Layer | Materialization | Reason |
|---|---|---|
| Staging | View | No storage cost, always reflects Silver. Simple pass-through — no reason to pay for a copy |
| Intermediate | View | Joining two sources, but still a stepping stone. Views compose cheaply in Athena |
| Marts | Table | Business analysts and ML pipeline query these repeatedly. View would re-scan Silver Parquet on every query — tables are pre-materialized in Athena |

> **Q: Why not materialize everything as a table?**
> Tables cost storage and require refresh. A staging view that does nothing but rename columns should never be materialized — it changes every time Silver changes, and the transformation is trivial. Materialize only what's queried repeatedly or expensively.

---

### dbt Project Structure

```
dbt/metal_intelligence/models/
├── staging/          # Views — rename columns, cast types, filter obvious nulls
│   ├── sources.yml   # Declares Silver S3 Parquet as Athena external tables
│   ├── stg_artists.sql
│   └── stg_albums.sql
└── marts/            # Tables — business aggregations and ML feature store
    ├── mart_subgenre_health.sql
    ├── mart_album_legacy.sql
    └── mart_artist_features.sql
```

**Staging layer responsibility:** type safety only. `cast(listeners as bigint)`, rename `lastfm_name` → `artist_name`, filter `listeners > 0`. No business logic — business logic lives in marts.

---

### Athena External Tables

Silver Parquet on S3 is read via Athena external tables declared in `sources.yml`. Athena never moves the data — it reads the Parquet files in place from S3.

```yaml
external:
  location: "s3://metal-intelligence-pipeline/silver/table=artists/"
  file_format: parquet
  partitions:
    - name: subgenre
      data_type: string
    - name: date
      data_type: string
```

> **Q: What is an Athena external table?**
> A schema definition that points at S3 data — Athena reads it without copying. The table lives in the AWS Glue Data Catalog (metadata store). Drop the table → data still exists in S3. This is the difference between an external and internal table: external tables separate metadata from storage. Critical for a medallion architecture — Silver Parquet is managed by the Python pipeline, Gold by dbt. They don't step on each other.

---

### dbt + Athena Configuration

**profiles.yml fields:**
- `database: AwsDataCatalog` — the Athena Data Catalog (not the Glue database name)
- `schema: metal_intelligence` — this becomes the Glue database name
- `s3_staging_dir` — where Athena writes query results (required by every Athena query)
- `s3_data_dir` — where dbt writes Gold tables

**Critical mistake made:** Set `database: metal_intelligence` instead of `database: AwsDataCatalog`. Athena's default catalog is always called `AwsDataCatalog` — the Glue database name goes in `schema`, not `database`.

> **Q: How do you manage credentials in dbt?**
> `env_var()` in profiles.yml. dbt doesn't load `.env` files — credentials are injected via environment variables set in the shell session before running dbt. Never hardcode credentials in profiles.yml — it gets committed to git. On Windows: `$env:AWS_ACCESS_KEY_ID = "..."` in PowerShell before `dbt run`.

---

### Analytical Mart Design

**Three marts, three stories:**

**`mart_subgenre_health`** — Golden Era analysis
- Grain: one row per subgenre
- Key feature: `golden_era_decade` — which formation decade produced the highest-listener bands in each genre
- Key feature: `breakout_pct` — what fraction of artists crossed 1M listeners
- Designed for: radar charts, subgenre comparison, scene health dashboard

**`mart_album_legacy`** — The Legacy Tracker
- Grain: one row per artist (deduped to highest-listener subgenre)
- Key feature: `years_since_last_release` — how long since the band put out music
- Key feature: `legacy_quadrant` — Active Giants / Legends / Hustling / Fading
- Designed for: quadrant scatter plot (X = years dormant, Y = listener count)

**`mart_artist_features`** — ML Feature Store
- Grain: one row per artist
- Key feature: `plays_per_listener` — engagement depth signal (loyal small fanbase vs casual huge one)
- Key feature: `is_breakout` — binary label (null for rejection region 200K-999K)
- Designed for: XGBoost training input

---

### Gold Layer Results

`dbt run` — all 5 models green, 15 seconds total:

| Model | Type | Rows | Story |
|---|---|---|---|
| `stg_artists` | View | — | Type-cast Silver artists |
| `stg_albums` | View | — | Type-cast Silver albums |
| `mart_subgenre_health` | Table | 9 | One row per subgenre — golden era, breakout %, geographic spread |
| `mart_album_legacy` | Table | 1,169 | One row per artist — years dormant × listeners, quadrant assignment |
| `mart_artist_features` | Table | 873 | ML feature store — rejection region excluded, is_breakout label |

`dbt test` — 4/4 data tests passing (not_null on artist_name and listeners in both staging models).

---

### Pivot: Scene Lifecycle → Subgenre Health

**Original plan:** weekly listener time series per subgenre (2015-2025 trend lines)

**What we discovered:** Last.fm's `tag.getweeklychartlist` returns chart date metadata only — no per-artist listener counts. Historical weekly listener data is not available via the public API.

**How we pivoted:**
1. Audited what data we actually have: one listener snapshot per artist (current), formed_year from MusicBrainz, album release dates
2. Identified what analytical story is still valid: subgenre health cross-section, golden era analysis, album legacy quadrants
3. The result is **more honest** — time-series trends built on a single snapshot would have been fabricated. A cross-sectional snapshot with richly calculated features is defensible.

> **Q: What would you do differently?**
> Validate API response shapes before designing mart schemas. A 10-line test script calling `tag.getweeklychartlist` and printing the response would have revealed it returns metadata in 5 minutes. We designed three dbt models around data that didn't exist. The fix was fast (half a session), but earlier discovery is cheaper.

---

## Phase 4 — ML Pipeline (XGBoost Breakout Predictor)

### Problem Framing

**Binary classification:** predict whether an underground band will cross 1M listeners.
- Positive class (breakout): current listeners ≥ 1M
- Negative class (underground): current listeners < 200K
- **Rejection region (200K–999K): excluded from training entirely** — ambiguous label

**Class imbalance:** 44 breakout / 829 underground (5% positive rate). XGBoost `scale_pos_weight = 18.94` (829/44) compensates — tells the model to treat each breakout example as 19× more important during gradient updates.

---

### Feature Engineering

| Feature | Type | Notes |
|---|---|---|
| `plays_per_listener` | Numeric | Engagement depth — plays ÷ listeners |
| `band_age_years` | Numeric | 2026 − formed_year |
| `total_albums`, `studio_albums` | Numeric | Discography depth |
| `avg_years_between_albums` | Numeric | Release cadence |
| `years_since_last_release` | Numeric | Dormancy signal |
| `mb_resolution_score` | Numeric | Data quality proxy |
| `subgenre` | Categorical → label encoded | 9 genres |
| `country` | Categorical → top-10 + Other → label encoded | Too many unique values for one-hot |

**Null imputation:** median per feature. Tree models tolerate median imputation — splits still work, the imputed value just becomes a cluster. Recorded training medians so inference uses train-set statistics, not test-set (prevents subtle leakage).

**Dead features discovered via SHAP:** `years_since_last_release`, `studio_albums`, `avg_years_between_albums` all had 0 SHAP contribution. Root cause: `total_albums` median imputed to 0 for most artists (MusicBrainz only covered Betrayal Tracker bands in full), making all album-derived features meaningless for the majority. This is an honest data gap, not a model bug.

---

### Train/Test Split — Why We Changed

**Original plan:** temporal split on `formed_year < 2015` (train) vs `>= 2015` (test).

**What we discovered:** bands formed after 2005 haven't had enough time to accumulate 1M listeners — they're all underground or in the rejection region. The test set had 2 breakout artists out of 325 — making all breakout metrics statistically meaningless.

**Fix:** stratified random split (80/20, `stratify=y`, `random_state=42`).

**Why this is correct for our data:** temporal split prevents look-ahead leakage in time-series data. But our dataset is a **cross-sectional snapshot** — every feature is current state (plays_per_listener, band_age_years, total_albums). There is no time axis to leak across. Stratified split preserves the 5% breakout rate in both train (5.0%) and test (5.1%), producing meaningful metrics.

> **Q: Why did you change from temporal to random split?**
> The temporal split assumption was borrowed from time-series thinking, but this dataset has no time axis — all features are point-in-time snapshots. Applying it created a degenerate test set with essentially no positive class. In ML, the right split strategy depends on the structure of your data, not a general rule. Cross-sectional data gets stratified random split. Time-series data gets temporal split. Confusing the two is a common mistake.

---

### Model Results

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC | 0.980 | Near-perfect ranking — model correctly orders breakout vs underground |
| Avg Precision (PR-AUC) | 0.757 | Strong — concentrates breakout predictions correctly |
| Breakout F1 | 0.67 | Solid for 9 test positives |
| Breakout Recall | 0.56 | Catches 5 of 9 real breakouts |
| Brier (calibrated) | 0.022 | Probabilities are well-calibrated |
| Brier (raw XGBoost) | 0.024 | Calibration improved probability quality |

> **Q: Why use PR-AUC instead of ROC-AUC for imbalanced data?**
> ROC-AUC is optimistic on imbalanced datasets — a model that scores every underground artist as 0.01 gets a high AUC because it correctly orders negatives. PR-AUC measures precision-recall tradeoff on the positive class only. With 5% breakout rate, PR-AUC is the honest metric. We report both.

---

### SHAP Feature Importance

| Feature | Mean |SHAP| | Story |
|---|---|---|
| `plays_per_listener` | 3.11 | Biggest signal by far — engagement depth beats raw audience size |
| `band_age_years` | 1.68 | Longevity — bands that survive long enough eventually build the audience |
| `country_encoded` | 1.41 | Geography determines the ceiling — US/Scandinavian bands have structural advantage |
| `subgenre` | 1.34 | Genre ceiling matters — metalcore breaks out, black metal rarely does |
| `mb_resolution_score` | 0.17 | Data quality proxy — well-matched bands have more complete feature vectors |
| Album features | 0.00 | Dead signal — imputation artifact from sparse MusicBrainz coverage |

**The `plays_per_listener` story:** a band with 50K fans who each play them 100 times is a stronger breakout candidate than one with 200K fans who play them 5 times. Loyal deep listeners signal genuine cultural resonance, not algorithmic discovery. This is the most interview-worthy finding in the project.

---

### Calibrated Classifier

**Why calibrate on top of XGBoost:**
XGBoost's `predict_proba` gives a discrimination score, not a probability. A score of 0.7 doesn't mean "70% chance of breaking out" — it means "more likely than 0.6." `CalibratedClassifierCV` with `method='sigmoid'` (Platt scaling) fits a logistic regression on top of XGBoost's raw scores to produce true probabilities.

**Brier score** measures calibration quality: 0 = perfect, 1 = worst. Our calibrated Brier (0.022) < raw XGBoost Brier (0.024) — calibration improved probability reliability.

**`cv=5`:** cross-validated calibration on training data — 5 folds, each fold calibrates on the held-out portion. More robust than `cv='prefit'` (which was removed in recent sklearn versions).

---

### MLflow Tracking

Local MLflow server (`mlflow server --host 127.0.0.1 --port 5000`) tracks:
- All hyperparameters (n_estimators, max_depth, learning_rate, scale_pos_weight, split strategy)
- All metrics (ROC-AUC, PR-AUC, Brier scores, classification report)
- SHAP importance CSV as an artifact
- Calibrated model registered as `metal-breakout-predictor` version 2

> **Q: What does MLflow give you over just printing metrics?**
> Reproducibility and comparison. Every run is logged with exact parameters, metrics, and the model artifact. Six months later you can load run ID `f25fe4ca` and reproduce the exact prediction. You can compare runs side-by-side in the UI to see which hyperparameter change improved PR-AUC. Without MLflow, you're relying on notes or memory.

---

## Key Interview Talking Points

**"Walk me through your dbt model structure."**
Three layers: staging (views, type casting and renaming), intermediate (removed — no joins needed after pivoting away from time-series), marts (tables, business aggregations). Each mart has a clear grain and feeds a specific analytical story. Staging models reference Silver Parquet via Athena external tables declared in sources.yml.

**"How did you handle the weekly chart data gap?"**
Discovered during Silver pipeline development that Last.fm's weekly chart endpoint returns date metadata, not artist listener counts. Audited what data we actually had (snapshot listeners, formed_year, album dates), designed analytical stories that were honest to the data, and rebuilt the marts around what we could actually answer. The result — golden era analysis, legacy tracker quadrants, ML feature store — is more defensible than fabricated time-series trends would have been.

**"What does dbt give you over raw SQL in Athena?"**
Three things: dependency management (dbt builds the DAG, runs in order, fails loudly when a dependency is missing), version control (SQL is code, not ClickOps in the Athena console), and testing (not_null and uniqueness tests run automatically — catches data quality regressions before analysts see them).

**"Why Last.fm over Spotify for genre classification?"**
Last.fm's tag system is community-driven — thousands of fans tag artists. Spotify's taxonomy is proprietary and doesn't cover niche subgenres like deathcore. When `genre:deathcore` returned 400, it confirmed Last.fm is the correct source.

**"How did you handle the Spotify API restrictions?"**
Audited what each pillar required. Determined Last.fm + MusicBrainz covered everything. Removed Spotify rather than maintaining dead infrastructure. Kept the code as documentation.

**"What is your data quality strategy?"**
Bronze immutability (never lose the original), 50K floor in Silver not Bronze (preserve reprocessability), entity resolution flag columns (never silently drop records), MusicBrainz merge only on auto_accepted (null > wrong). dbt tests enforce not_null and uniqueness at the Gold layer — catches schema regressions before analysts see them.

**"What was the hardest debugging session?"**
A chain of Athena type mismatches on `formed_year`. Pandas writes nullable integers as DOUBLE in Parquet, but the Athena CREATE TABLE defined it as INT. Fixed by changing the schema to DOUBLE. Then `release_year` failed in the opposite direction — it was genuinely INT64 in the Parquet but we'd over-corrected it to DOUBLE. The lesson: always read the actual error message. "Type X in Parquet incompatible with type Y in schema" tells you which direction the mismatch goes — don't guess, just read it and match the schema to what's in the file.

**"Describe a time a project didn't go as planned."**
Spotify API restrictions eliminated audio features and all artist metrics mid-build. We audited requirements, pivoted the Riff Economy pillar to Scene Lifecycle (subgenre evolution using listener data), added MusicBrainz as a replacement source, and ended up with a cleaner architecture. The final pipeline is more defensible than the original design.
