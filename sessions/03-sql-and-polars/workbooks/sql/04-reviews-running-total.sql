-- Reviews per month in Berlin (a proxy for demand): running total within each year and the
-- change against the same month one year earlier (window functions). Works in PostgreSQL
-- and DuckDB. June and July 2026 are incomplete (the listings were scraped between 26 June
-- and 3 July, and reviews appear with a delay) and are excluded.
-- Author: course team, licence CC-BY-4.0

WITH monthly AS (
    SELECT CAST(month AS DATE) AS month, SUM(n_reviews) AS n_reviews
    FROM reviews_monthly
    WHERE month >= '2018-01-01' AND month < '2026-06-01'
    GROUP BY 1
)
SELECT month,
       n_reviews,
       SUM(n_reviews) OVER (PARTITION BY EXTRACT(YEAR FROM month) ORDER BY month) AS running_total_year,
       n_reviews - LAG(n_reviews, 12) OVER (ORDER BY month)                       AS change_vs_last_year
FROM monthly
ORDER BY month;
