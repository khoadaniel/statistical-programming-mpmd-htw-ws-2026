-- Schema of the course database for the Berlin Airbnb case study, with keys and constraints.
-- Works in PostgreSQL and in DuckDB. Run it once on an empty database, then load the rows
-- (see 10-case-study-postgres-listings.ipynb): parent table (listings) first, child tables after.
-- Author: course team, licence CC-BY-4.0

DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS reviews_monthly;
DROP TABLE IF EXISTS weather_daily;
DROP TABLE IF EXISTS listings;

CREATE TABLE listings (                                   -- one row per listing of the snapshot
    id                   BIGINT PRIMARY KEY,
    host_id              BIGINT NOT NULL,
    district             TEXT NOT NULL,
    neighbourhood        TEXT NOT NULL,
    latitude             DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN 52.33 AND 52.68),
    longitude            DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN 13.08 AND 13.77),
    room_type            TEXT NOT NULL
        CHECK (room_type IN ('Entire home/apt', 'Private room', 'Hotel room', 'Shared room')),
    accommodates         INTEGER NOT NULL CHECK (accommodates >= 1),
    bedrooms             INTEGER CHECK (bedrooms >= 0),
    price                DOUBLE PRECISION CHECK (price > 0),          -- NULL: no price shown
    minimum_nights       INTEGER CHECK (minimum_nights >= 1),
    -- 2 listings carry the placeholder 2,147,483,647: the loader sets it to NULL (Session 4)
    maximum_nights       INTEGER CHECK (maximum_nights < 2147483647),
    availability_365     INTEGER NOT NULL CHECK (availability_365 BETWEEN 0 AND 365),
    number_of_reviews    INTEGER NOT NULL CHECK (number_of_reviews >= 0),
    first_review         DATE,
    last_review          DATE,
    review_scores_rating DOUBLE PRECISION CHECK (review_scores_rating BETWEEN 0 AND 5),
    license_status       TEXT NOT NULL CHECK (license_status IN
        ('registration number', 'missing', 'legal entity name', 'private host name', 'other')),
    CHECK (minimum_nights <= maximum_nights),
    CHECK (first_review <= last_review)
);

CREATE TABLE calendar (                                   -- one row per listing and night, next 365 days
    listing_id     BIGINT NOT NULL REFERENCES listings (id),   -- foreign key
    date           DATE NOT NULL,
    available      BOOLEAN NOT NULL,                      -- false: booked or blocked by the host
    minimum_nights INTEGER CHECK (minimum_nights >= 1),
    PRIMARY KEY (listing_id, date)                        -- composite key: one row per listing and night
);

CREATE TABLE reviews_monthly (                            -- reviews per listing and month since 2009
    listing_id BIGINT NOT NULL REFERENCES listings (id),
    month      DATE NOT NULL CHECK (EXTRACT(DAY FROM month) = 1),
    n_reviews  INTEGER NOT NULL CHECK (n_reviews > 0),
    PRIMARY KEY (listing_id, month)
);

CREATE TABLE weather_daily (                              -- daily Berlin weather (Open-Meteo, Session 2)
    date                DATE PRIMARY KEY,
    temperature_2m_mean DOUBLE PRECISION,
    precipitation_sum   DOUBLE PRECISION CHECK (precipitation_sum >= 0),
    sunshine_hours      DOUBLE PRECISION CHECK (sunshine_hours BETWEEN 0 AND 24)
);

CREATE INDEX listings_district_idx ON listings (district);
CREATE INDEX reviews_monthly_month_idx ON reviews_monthly (month);
