-- PostgreSQL only. The script case-study/prepare_data.py --postgres writes the tables
-- with pandas.to_sql, which creates no keys and no constraints. Run this file once
-- afterwards (psql -f 02-add-constraints.sql, or from Python) to add them.
-- Every statement fails with an error if existing rows violate the rule:
-- that error is a data quality finding, not a bug in the script.
-- Author: course team, licence CC-BY-4.0

ALTER TABLE products
    ADD PRIMARY KEY (parent_asin),
    ADD CONSTRAINT products_price_check CHECK (price > 0);

ALTER TABLE reviews
    ADD PRIMARY KEY (review_id),
    ADD CONSTRAINT reviews_product_fkey FOREIGN KEY (parent_asin) REFERENCES products (parent_asin),
    ADD CONSTRAINT reviews_rating_check CHECK (rating BETWEEN 1 AND 5),
    ADD CONSTRAINT reviews_label_check CHECK (label IN ('neg', 'neu', 'pos')),
    ALTER COLUMN date SET NOT NULL;

ALTER TABLE reviews_test
    ADD PRIMARY KEY (review_id),
    ADD CONSTRAINT reviews_test_product_fkey FOREIGN KEY (parent_asin) REFERENCES products (parent_asin);

CREATE INDEX IF NOT EXISTS reviews_parent_asin_idx ON reviews (parent_asin);
CREATE INDEX IF NOT EXISTS reviews_date_idx ON reviews (date);
