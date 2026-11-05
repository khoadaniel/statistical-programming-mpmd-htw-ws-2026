-- Reviews per month in Berlin (a proxy for demand): running total within each year and the
-- change against the previous month (window functions). Runs on the course database `airbnb`
-- on the local PostgreSQL server. `month` is a DATE, the first day of the month;
-- EXTRACT(YEAR FROM month) is the year.
-- June and July 2026 are incomplete (the listings were scraped between 26 June and 3 July 2026,
-- and reviews appear with a delay) and are excluded.
-- Author: course team, licence CC BY 4.0

WITH monthly AS (
    SELECT month, SUM(n_reviews) AS n_reviews
    FROM reviews
    WHERE month >= DATE '2018-01-01' AND month < DATE '2026-06-01'
    GROUP BY month
)
SELECT month,
       n_reviews,
       SUM(n_reviews) OVER (PARTITION BY EXTRACT(YEAR FROM month) ORDER BY month) AS running_total_year,
       n_reviews - LAG(n_reviews) OVER (ORDER BY month)                            AS change_vs_previous_month
FROM monthly
ORDER BY month;
