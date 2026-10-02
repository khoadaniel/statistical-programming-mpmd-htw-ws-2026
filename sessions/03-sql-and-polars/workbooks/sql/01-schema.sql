-- Schema of the course database with keys and constraints.
-- Works in PostgreSQL and in DuckDB. Run it once on an empty database,
-- then load the rows (see 10-case-study-postgres-decisions.ipynb).
-- Author: course team, licence CC-BY-4.0

DROP TABLE IF EXISTS decisions_test;
DROP TABLE IF EXISTS decisions;
DROP TABLE IF EXISTS nomenclature;

CREATE TABLE nomenclature (
    heading             TEXT PRIMARY KEY CHECK (length(heading) = 4),   -- one row per HS heading
    heading_description TEXT NOT NULL,
    chapter             TEXT NOT NULL CHECK (length(chapter) = 2),
    chapter_description TEXT,
    section             TEXT,
    section_name        TEXT
);

CREATE TABLE decisions (
    bti_reference                TEXT PRIMARY KEY,
    issuing_country              TEXT NOT NULL CHECK (length(issuing_country) = 2),
    language                     TEXT NOT NULL CHECK (length(language) = 2),
    start_date                   DATE NOT NULL,
    end_date                     DATE,          -- end_date >= start_date fails for 510 rows (Session 4)
    date_of_issue                DATE,
    status                       TEXT NOT NULL CHECK (status IN ('VALID', 'INVALID')),
    invalidation_reason          TEXT,          -- NULL: the decision expired normally or is valid
    description                  TEXT NOT NULL,
    keywords                     TEXT,
    classification_justification TEXT,
    cn_code                      TEXT NOT NULL,  -- 8 digits, but 1,040 rows have only 4 or 6 (Session 4)
    heading                      TEXT NOT NULL REFERENCES nomenclature (heading),  -- foreign key
    chapter                      TEXT NOT NULL CHECK (chapter = substr(heading, 1, 2))
);

CREATE TABLE decisions_test (
    id              TEXT PRIMARY KEY,
    issuing_country TEXT NOT NULL CHECK (length(issuing_country) = 2),
    language        TEXT NOT NULL CHECK (length(language) = 2),
    start_date      DATE NOT NULL,
    description     TEXT NOT NULL
);

CREATE INDEX decisions_heading_idx ON decisions (heading);
CREATE INDEX decisions_start_date_idx ON decisions (start_date);
