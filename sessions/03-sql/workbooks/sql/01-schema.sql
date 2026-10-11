-- Schema of a course database for the Berlin Airbnb case study, with keys and constraints (SQLite).
-- Used by extras/01-case-study-loading-with-constraints.ipynb: run it on an empty database, switch
-- foreign keys on (PRAGMA foreign_keys = ON), then load the parent table (listings) before the child
-- tables (calendar, reviews). Dates are text 'YYYY-MM-DD'; available is 1 (available) or 0.
-- Author: course team, licence CC BY 4.0

DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS listings;

CREATE TABLE listings (                                   -- one row per listing of the snapshot
    id                   INTEGER PRIMARY KEY,
    host_id              INTEGER NOT NULL,
    district             TEXT NOT NULL,
    neighbourhood        TEXT NOT NULL,
    latitude             REAL NOT NULL CHECK (latitude BETWEEN 52.33 AND 52.68),
    longitude            REAL NOT NULL CHECK (longitude BETWEEN 13.08 AND 13.77),
    room_type            TEXT NOT NULL
        CHECK (room_type IN ('Entire home/apt', 'Private room', 'Hotel room', 'Shared room')),
    accommodates         INTEGER NOT NULL CHECK (accommodates >= 1),
    bedrooms             INTEGER CHECK (bedrooms >= 0),
    price                REAL CHECK (price > 0),                   -- NULL: no price shown
    minimum_nights       INTEGER CHECK (minimum_nights >= 1),
    -- 2 listings carry the placeholder 2,147,483,647: the loader must decide what to do (Session 4)
    maximum_nights       INTEGER CHECK (maximum_nights < 2147483647),
    availability_365     INTEGER NOT NULL CHECK (availability_365 BETWEEN 0 AND 365),
    number_of_reviews    INTEGER NOT NULL CHECK (number_of_reviews >= 0),
    first_review         TEXT CHECK (first_review IS date(first_review)),   -- NULL or a valid date
    last_review          TEXT CHECK (last_review IS date(last_review)),
    review_scores_rating REAL CHECK (review_scores_rating BETWEEN 0 AND 5),
    license_status       TEXT NOT NULL CHECK (license_status IN
        ('registration number', 'missing', 'legal entity name', 'private host name', 'other')),
    CHECK (minimum_nights <= maximum_nights),
    CHECK (first_review <= last_review)
);

CREATE TABLE calendar (                                   -- one row per listing and night, next 365 days
    listing_id INTEGER NOT NULL REFERENCES listings (id), -- foreign key
    date       TEXT NOT NULL CHECK (date IS date(date)),
    available  INTEGER NOT NULL CHECK (available IN (0, 1)),   -- 0: booked or blocked by the host
    PRIMARY KEY (listing_id, date)                        -- composite key: one row per listing and night
);

CREATE TABLE reviews (                                    -- reviews per listing and month since 2009
    listing_id INTEGER NOT NULL REFERENCES listings (id),
    month      TEXT NOT NULL CHECK (month IS date(month, 'start of month')),   -- first day of a month
    n_reviews  INTEGER NOT NULL CHECK (n_reviews > 0),
    PRIMARY KEY (listing_id, month)
);

CREATE INDEX listings_district_idx ON listings (district);
