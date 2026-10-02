-- Rank headings by the number of decisions per issuing country and start year
-- (window function). Works in PostgreSQL and DuckDB on the tables of 01-schema.sql.
-- Author: course team, licence CC-BY-4.0

WITH per_year AS (                         -- CTE 1: one row per country, year and heading
    SELECT issuing_country,
           EXTRACT(YEAR FROM start_date)::int AS year,
           heading,
           COUNT(*) AS n_decisions
    FROM decisions
    GROUP BY issuing_country, year, heading
), ranked AS (                             -- CTE 2: rank within each country and year
    SELECT *,
           RANK() OVER (PARTITION BY issuing_country, year ORDER BY n_decisions DESC) AS rnk,
           ROUND(100.0 * n_decisions
                 / SUM(n_decisions) OVER (PARTITION BY issuing_country, year), 2) AS pct_of_year
    FROM per_year
)
SELECT r.issuing_country, r.year, r.rnk, r.heading, r.n_decisions, r.pct_of_year,
       LEFT(n.heading_description, 40) AS heading_description
FROM ranked AS r
LEFT JOIN nomenclature AS n ON n.heading = r.heading
WHERE r.rnk <= 3 AND r.issuing_country IN ('DE', 'FR', 'PL')   -- filter one level up
ORDER BY r.issuing_country, r.year, r.rnk, r.heading;
