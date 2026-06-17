-- Staging model for Silver artists table.
-- Renames columns to consistent snake_case, casts types, applies final filters.
-- One row per artist per subgenre partition.

with source as (
    select * from {{ source('silver', 'artists') }}
),

staged as (
    select
        -- identifiers
        lastfm_name                                     as artist_name,
        mbid                                            as musicbrainz_id,
        mb_name                                         as musicbrainz_name,

        -- listener metrics
        cast(listeners as bigint)                       as listeners,
        cast(play_count as bigint)                      as play_count,

        -- subgenre context
        subgenre                                        as subgenre,  -- partition key
        source_tag                                      as source_tag,
        subgenre_tags                                   as subgenre_tags,

        -- artist metadata
        country,
        cast(formed_year as integer)                    as formed_year,
        bio_summary,

        -- entity resolution quality
        cast(mb_resolution_score as double)             as mb_resolution_score,

        -- partition date for freshness tracking
        date                                            as ingestion_date

    from source
    -- exclude artists with no listener data (safety net on top of Silver 50K floor)
    where listeners is not null
      and listeners > 0
)

select * from staged
