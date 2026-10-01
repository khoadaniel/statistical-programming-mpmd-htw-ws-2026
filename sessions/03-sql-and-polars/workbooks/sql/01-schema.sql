-- Schema of the course database with keys and constraints.
-- Works in PostgreSQL and in DuckDB. Run it once on an empty database,
-- then load the rows (see 10-case-study-postgres-reviews.ipynb).
-- Author: course team, licence CC-BY-4.0

DROP TABLE IF EXISTS reviews_test;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    parent_asin      TEXT PRIMARY KEY,          -- one row per product
    main_category    TEXT,
    title            TEXT NOT NULL,
    features         TEXT,
    description      TEXT,
    price            DOUBLE PRECISION CHECK (price > 0),   -- NULL allowed: price unknown
    store            TEXT,
    categories       TEXT,
    details          TEXT,
    train_avg_rating DOUBLE PRECISION CHECK (train_avg_rating BETWEEN 1 AND 5),
    train_n_reviews  INTEGER CHECK (train_n_reviews >= 1)
);

CREATE TABLE reviews (
    review_id         TEXT PRIMARY KEY,
    rating            SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title             TEXT,
    text              TEXT,
    parent_asin       TEXT NOT NULL REFERENCES products (parent_asin),  -- foreign key
    user_id           TEXT NOT NULL,
    helpful_vote      INTEGER NOT NULL CHECK (helpful_vote >= 0),
    verified_purchase BOOLEAN NOT NULL,
    date              TIMESTAMP NOT NULL,
    n_images          INTEGER NOT NULL CHECK (n_images >= 0),
    label             TEXT NOT NULL CHECK (label IN ('neg', 'neu', 'pos'))
);

CREATE TABLE reviews_test (
    review_id         TEXT PRIMARY KEY,
    title             TEXT,
    text              TEXT,
    parent_asin       TEXT NOT NULL REFERENCES products (parent_asin),
    user_id           TEXT NOT NULL,
    helpful_vote      INTEGER NOT NULL CHECK (helpful_vote >= 0),
    verified_purchase BOOLEAN NOT NULL,
    date              TIMESTAMP NOT NULL,
    n_images          INTEGER NOT NULL CHECK (n_images >= 0)
);

CREATE INDEX reviews_parent_asin_idx ON reviews (parent_asin);
CREATE INDEX reviews_date_idx ON reviews (date);
