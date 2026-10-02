-- PostgreSQL only. The script case-study/prepare_data.py --postgres writes the tables
-- decisions, decisions_test and nomenclature with pandas.to_sql, which creates no keys
-- and no constraints. Run this file once afterwards (psql -f 02-add-constraints.sql,
-- or from Python) to add them.
-- A statement fails with an error if existing rows violate the rule: that error is a
-- data quality finding, not a bug in the script. Two rules are known to fail on the
-- training data and are therefore added as NOT VALID: PostgreSQL then checks every new
-- or changed row, but not the rows that are already in the table.
-- Author: course team, licence CC-BY-4.0

ALTER TABLE nomenclature
    ADD PRIMARY KEY (heading),
    ADD CONSTRAINT nomenclature_heading_check CHECK (heading ~ '^[0-9]{4}$');

ALTER TABLE decisions
    ADD PRIMARY KEY (bti_reference),
    ADD CONSTRAINT decisions_heading_check CHECK (heading ~ '^[0-9]{4}$'),
    ADD CONSTRAINT decisions_chapter_check CHECK (chapter = substr(heading, 1, 2)),
    ADD CONSTRAINT decisions_status_check CHECK (status IN ('VALID', 'INVALID')),
    ALTER COLUMN start_date SET NOT NULL,
    ALTER COLUMN description SET NOT NULL;

-- 51 decisions use heading 8803, which the HS 2022 revision deleted (parts of aircraft are
-- now in 8807), and the nomenclature table is HS 2022. Find them with:
--   SELECT heading, COUNT(*) FROM decisions d
--   WHERE NOT EXISTS (SELECT 1 FROM nomenclature n WHERE n.heading = d.heading) GROUP BY heading
ALTER TABLE decisions
    ADD CONSTRAINT decisions_heading_fkey FOREIGN KEY (heading)
        REFERENCES nomenclature (heading) NOT VALID;

-- 510 annulled decisions (invalidation code 55) carry the placeholder end date 1900-01-01,
-- which lies before their start date (see Session 4).
ALTER TABLE decisions
    ADD CONSTRAINT decisions_dates_check CHECK (end_date >= start_date) NOT VALID;

ALTER TABLE decisions_test
    ADD PRIMARY KEY (id);

CREATE INDEX IF NOT EXISTS decisions_heading_idx ON decisions (heading);
CREATE INDEX IF NOT EXISTS decisions_start_date_idx ON decisions (start_date);
