-- Artist Features: ML feature table for the Breakout Predictor (XGBoost).
-- One row per artist — point-in-time snapshot suitable for training and inference.
-- Label: breakout = 1 if listeners >= 1M, underground = 0 if listeners < 200K.
-- Artists 200K-1M are EXCLUDED — they are in the "rejection region" (ambiguous label).

{{ config(materialized='table') }}

with artists as (
    select * from {{ ref('stg_artists') }}
),

albums as (
    select * from {{ ref('stg_albums') }}
),

-- Album release cadence features
artist_albums as (
    select
        artist_name,
        count(*)                                        as total_albums,
        min(release_year)                               as debut_year,
        max(release_year)                               as latest_album_year,
        case
            when count(*) > 1
            then cast(max(release_year) - min(release_year) as double) / (count(*) - 1)
        end                                             as avg_years_between_albums,
        count(case when album_type = 'Album' then 1 end) as studio_albums
    from albums
    group by artist_name
),

-- One row per artist — highest-listener subgenre wins if duplicated
artist_primary as (
    select
        artist_name,
        musicbrainz_id,
        subgenre,
        country,
        formed_year,
        listeners,
        play_count,
        mb_resolution_score,
        row_number() over (
            partition by lower(trim(artist_name))
            order by listeners desc
        ) as rn
    from artists
),

final as (
    select
        a.artist_name,
        a.musicbrainz_id,
        a.subgenre,

        -- artist metadata features
        a.country,
        a.formed_year,
        a.mb_resolution_score,
        (2026 - a.formed_year)                          as band_age_years,

        -- discography features
        coalesce(al.total_albums, 0)                    as total_albums,
        coalesce(al.studio_albums, 0)                   as studio_albums,
        al.debut_year,
        al.latest_album_year,
        al.avg_years_between_albums,
        case
            when al.latest_album_year is not null
            then 2026 - al.latest_album_year
        end                                             as years_since_last_release,

        -- listener signal (label source)
        a.listeners                                     as current_listeners,
        a.play_count,

        -- play_count / listeners ratio — engagement depth signal
        case
            when a.listeners > 0
            then round(cast(a.play_count as double) / a.listeners, 1)
        end                                             as plays_per_listener,

        -- ML label
        -- NULL for rejection region (200K-999K) — excluded from training
        case
            when a.listeners >= 1000000 then 1  -- breakout
            when a.listeners < 200000   then 0  -- underground
        end                                             as is_breakout

    from artist_primary a
    left join artist_albums al
        on lower(trim(a.artist_name)) = lower(trim(al.artist_name))
    where a.rn = 1
)

-- Exclude rejection region from this table — ML pipeline reads this directly
select * from final
where is_breakout is not null
order by current_listeners desc
