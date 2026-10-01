-- Rank products by the number of reviews per year (window function).
-- Works in PostgreSQL and DuckDB on the tables of 01-schema.sql.
-- Author: course team, licence CC-BY-4.0

WITH per_year AS (                         -- CTE 1: one row per year and product
    SELECT EXTRACT(YEAR FROM date)::int AS year,
           parent_asin,
           COUNT(*) AS n_reviews
    FROM reviews
    GROUP BY year, parent_asin
), ranked AS (                             -- CTE 2: rank within each year
    SELECT *,
           RANK() OVER (PARTITION BY year ORDER BY n_reviews DESC) AS rnk,
           ROUND(100.0 * n_reviews / SUM(n_reviews) OVER (PARTITION BY year), 2) AS pct_of_year
    FROM per_year
)
SELECT r.year, r.rnk, r.n_reviews, r.pct_of_year, p.store, LEFT(p.title, 40) AS product
FROM ranked AS r
JOIN products AS p USING (parent_asin)
WHERE r.rnk <= 3 AND r.year >= 2017       -- a window result is filtered one level up
ORDER BY r.year, r.rnk;
