-- PostgreSQL only. The script case-study/prepare_airbnb.py --postgres writes the tables
-- listings, calendar, reviews_monthly and weather_daily with pandas.to_sql, which creates
-- no keys and no constraints. Run this file once afterwards (psql -f 02-add-constraints.sql,
-- or from Python) to add them.
-- A statement fails with an error if existing rows violate the rule: that error is a
-- data quality finding, not a bug in the script. Three rules are known to fail on the
-- snapshot of 26 June 2026 and are therefore added as NOT VALID: PostgreSQL then checks
-- every new or changed row, but not the rows that are already in the table.
-- Author: course team, licence CC-BY-4.0

ALTER TABLE listings
    ADD PRIMARY KEY (id),
    ALTER COLUMN room_type SET NOT NULL,
    ALTER COLUMN district SET NOT NULL,
    ADD CONSTRAINT listings_room_type_check
        CHECK (room_type IN ('Entire home/apt', 'Private room', 'Hotel room', 'Shared room')),
    ADD CONSTRAINT listings_accommodates_check CHECK (accommodates >= 1),
    ADD CONSTRAINT listings_price_check CHECK (price > 0),
    ADD CONSTRAINT listings_coordinates_check
        CHECK (latitude BETWEEN 52.33 AND 52.68 AND longitude BETWEEN 13.08 AND 13.77),
    ADD CONSTRAINT listings_nights_check CHECK (minimum_nights <= maximum_nights),
    ADD CONSTRAINT listings_availability_check CHECK (availability_365 BETWEEN 0 AND 365),
    ADD CONSTRAINT listings_license_status_check CHECK (license_status IN
        ('registration number', 'missing', 'legal entity name', 'private host name', 'other'));

-- 2 listings carry maximum_nights = 2,147,483,647, the largest 32-bit integer (Session 4).
ALTER TABLE listings
    ADD CONSTRAINT listings_maximum_nights_check CHECK (maximum_nights < 2147483647) NOT VALID;

ALTER TABLE calendar ADD PRIMARY KEY (listing_id, date);
ALTER TABLE reviews_monthly ADD PRIMARY KEY (listing_id, month);
ALTER TABLE weather_daily
    ADD PRIMARY KEY (date),
    ADD CONSTRAINT weather_precipitation_check CHECK (precipitation_sum >= 0);

-- The calendar covers 79 listings and reviews_monthly 57 listings that are not in the
-- listings table: Inside Airbnb scrapes the files separately. Find them with:
--   SELECT COUNT(DISTINCT listing_id) FROM calendar c
--   WHERE NOT EXISTS (SELECT 1 FROM listings l WHERE l.id = c.listing_id)
ALTER TABLE calendar
    ADD CONSTRAINT calendar_listing_fkey FOREIGN KEY (listing_id) REFERENCES listings (id) NOT VALID;
ALTER TABLE reviews_monthly
    ADD CONSTRAINT reviews_monthly_listing_fkey FOREIGN KEY (listing_id) REFERENCES listings (id) NOT VALID;

CREATE INDEX IF NOT EXISTS listings_district_idx ON listings (district);
CREATE INDEX IF NOT EXISTS reviews_monthly_month_idx ON reviews_monthly (month);
