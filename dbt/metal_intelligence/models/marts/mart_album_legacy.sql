-- Album Legacy: one row per artist — years since last release crossed with listener count.
-- Feeds the Legacy Tracker quadrant chart (the Betrayal Tracker reimagined).
-- Grain: one row per artist (using highest-listener subgenre record if duplicated).

{{ config(materialized='table') }}

with artists as (
    select * from {{ ref('stg_artists') }}
),

albums as (
    select * from {{ ref('stg_albums') }}
),

-- Album stats per artist
album_stats as (
    select
        artist_name,
        count(*)                                            as total_albums,
        min(release_year)                                  as debut_year,
        max(release_year)                                  as last_album_year,
        max(first_release_date)                            as last_release_date,

        -- cadence: avg years between albums (null for single-album artists)
        case
            when count(*) > 1
            then cast(max(release_year) - min(release_year) as double) / (count(*) - 1)
        end                                                as avg_years_between_albums,

        -- album types breakdown
        count(case when album_type = 'Album' then 1 end)  as studio_albums,
        count(case when album_type = 'EP' then 1 end)     as eps,
        count(case when album_type = 'Single' then 1 end) as singles

    from albums
    group by artist_name
),

-- One artist row — if in multiple subgenres, keep the one with highest listeners
-- (avoids double-counting the same artist for the scatter plot)
artist_deduped as (
    select
        artist_name,
        musicbrainz_id,
        listeners,
        play_count,
        subgenre,
        country,
        formed_year,
        row_number() over (
            partition by lower(trim(artist_name))
            order by listeners desc
        ) as rn
    from artists
),

artist_primary as (
    select * from artist_deduped where rn = 1
)

select
    a.artist_name,
    a.musicbrainz_id,
    a.subgenre,
    a.country,
    a.formed_year,
    a.listeners,
    a.play_count,

    -- album context
    al.total_albums,
    al.studio_albums,
    al.eps,
    al.singles,
    al.debut_year,
    al.last_album_year,
    al.last_release_date,
    al.avg_years_between_albums,

    -- the key Legacy Tracker dimension
    -- null if no albums in MusicBrainz (band may exist, just not resolved)
    case
        when al.last_album_year is not null
        then 2026 - al.last_album_year
    end                                                    as years_since_last_release,

    -- career length from formation to last release
    case
        when a.formed_year is not null and al.last_album_year is not null
        then al.last_album_year - a.formed_year
    end                                                    as active_career_years,

    -- ML label (reused from artist_features for consistency)
    case
        when a.listeners >= 1000000 then 'breakout'
        when a.listeners < 200000   then 'underground'
        else                             'rising'
    end                                                    as listener_tier,

    -- quadrant assignment for the Legacy Tracker visual
    -- X-axis: years_since_last_release (0=active, high=dormant)
    -- Y-axis: listeners (high=massive, low=small)
    case
        when (2026 - al.last_album_year) <= 3 and a.listeners >= 500000 then 'Active Giants'
        when (2026 - al.last_album_year) > 3  and a.listeners >= 500000 then 'Legends'
        when (2026 - al.last_album_year) <= 3 and a.listeners < 500000  then 'Hustling'
        when (2026 - al.last_album_year) > 3  and a.listeners < 500000  then 'Fading'
    end                                                    as legacy_quadrant

from artist_primary a
left join album_stats al
    on lower(trim(a.artist_name)) = lower(trim(al.artist_name))
order by a.listeners desc
