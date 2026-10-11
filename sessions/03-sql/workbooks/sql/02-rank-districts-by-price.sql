-- Rank Berlin districts by the average price of a short stay, separately for entire homes and
-- private rooms (common table expressions and a window function). Runs on the course database
-- case-study/data/airbnb.sqlite (SQLite 3.25 or newer) and unchanged on PostgreSQL.
-- Short stays: minimum stay below 28 nights and a price shown (the course convention).
-- Author: course team, licence CC BY 4.0

WITH short_stays AS (                      -- CTE 1: the listings the question is about
    SELECT district, room_type, price
    FROM listings
    WHERE minimum_nights < 28 AND price IS NOT NULL
      AND room_type IN ('Entire home/apt', 'Private room')
), per_district AS (                       -- CTE 2: one row per room type and district
    SELECT room_type, district,
           COUNT(*) AS n_listings,
           ROUND(AVG(price), 1) AS avg_price
    FROM short_stays
    GROUP BY room_type, district
    HAVING COUNT(*) >= 30                  -- no ranking on a handful of listings
), ranked AS (                             -- CTE 3: rank within each room type
    SELECT *,
           RANK() OVER (PARTITION BY room_type ORDER BY avg_price DESC) AS rnk,
           avg_price - MAX(avg_price) OVER (PARTITION BY room_type) AS gap_to_top
    FROM per_district
)
SELECT room_type, rnk, district, n_listings, avg_price, gap_to_top
FROM ranked
ORDER BY room_type, rnk, district;
