-- Staging model for Silver albums table.
-- Betrayal Tracker source — album release dates per artist.

with source as (
    select * from {{ source('silver', 'albums') }}
),

staged as (
    select
        artist_name,
        artist_mbid,
        album_id                                        as musicbrainz_album_id,
        title                                           as album_title,
        type                                            as album_type,
        first_release_date,
        cast(release_year as integer)                   as release_year

    from source
    where title is not null
      -- exclude releases with no date — can't anchor to listener trajectory
      and first_release_date is not null
)

select * from staged
