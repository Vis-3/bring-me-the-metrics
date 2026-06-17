-- Subgenre Health: one row per subgenre — the scene-level aggregates.
-- Feeds the Golden Era scatter and subgenre comparison visuals.
-- Grain: one row per subgenre.

{{ config(materialized='table') }}

with artists as (
    select * from {{ ref('stg_artists') }}
),

-- Per-subgenre aggregates
subgenre_stats as (
    select
        subgenre,

        -- scene size
        count(*)                                            as total_artists,
        sum(listeners)                                      as total_listeners,
        avg(listeners)                                      as avg_listeners,
        approx_percentile(listeners, 0.5)                  as median_listeners,
        approx_percentile(listeners, 0.9)                  as p90_listeners,
        max(listeners)                                      as max_listeners,

        -- breakout / underground split
        count(case when listeners >= 1000000 then 1 end)   as breakout_artists,
        count(case when listeners < 200000 then 1 end)     as underground_artists,
        count(case when listeners between 200000 and 999999 then 1 end) as rising_artists,

        -- scene era — what decade built this genre?
        avg(cast(formed_year as double))                   as avg_formed_year,
        approx_percentile(cast(formed_year as double), 0.5) as median_formed_year,
        min(formed_year)                                   as oldest_band_year,

        -- most prolific era (mode approximation via median)
        -- artists with no formed_year excluded from era calcs
        count(case when formed_year is not null then 1 end) as artists_with_formed_year,

        -- geographic diversity
        count(distinct country)                            as country_count,
        count(case when country is not null then 1 end)    as artists_with_country

    from artists
    group by subgenre
),

-- Dominant era: the formed_year decade with the highest total listeners
era_dominance as (
    select
        subgenre,
        (floor(formed_year / 10) * 10)                     as formation_decade,
        sum(listeners)                                     as decade_total_listeners,
        count(*)                                           as decade_artist_count,
        row_number() over (
            partition by subgenre
            order by sum(listeners) desc
        )                                                  as era_rank
    from artists
    where formed_year is not null
    group by subgenre, floor(formed_year / 10) * 10
),

golden_era as (
    select subgenre, formation_decade as golden_era_decade, decade_total_listeners, decade_artist_count
    from era_dominance
    where era_rank = 1
),

-- Top country per subgenre by listener count
country_dominance as (
    select
        subgenre,
        country,
        sum(listeners)                                     as country_listeners,
        count(*)                                           as country_artist_count,
        row_number() over (
            partition by subgenre
            order by sum(listeners) desc
        )                                                  as country_rank
    from artists
    where country is not null
    group by subgenre, country
),

top_country as (
    select subgenre, country as dominant_country, country_listeners as dominant_country_listeners
    from country_dominance
    where country_rank = 1
)

select
    s.subgenre,

    -- scale
    s.total_artists,
    s.total_listeners,
    s.avg_listeners,
    s.median_listeners,
    s.p90_listeners,
    s.max_listeners,

    -- tier breakdown
    s.breakout_artists,
    s.rising_artists,
    s.underground_artists,
    round(cast(s.breakout_artists as double) / s.total_artists * 100, 1) as breakout_pct,

    -- era
    s.avg_formed_year,
    s.median_formed_year,
    s.oldest_band_year,
    s.artists_with_formed_year,
    g.golden_era_decade,
    g.decade_total_listeners   as golden_era_total_listeners,
    g.decade_artist_count      as golden_era_artist_count,

    -- geography
    s.country_count,
    s.artists_with_country,
    t.dominant_country,
    t.dominant_country_listeners

from subgenre_stats s
left join golden_era g on s.subgenre = g.subgenre
left join top_country t on s.subgenre = t.subgenre
order by s.total_listeners desc
