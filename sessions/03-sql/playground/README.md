# Session 3 · SQL playground

[Open notebook](sql-playground.ipynb)

A small notebook to try SQL on your own. The first cell starts a new, empty PostgreSQL server in a temporary folder and creates three small tables:

- `students`: one row per student (name, programme, start year).
- `courses`: one row per course (title, credits).
- `scores`: one row per exam, with foreign keys to `students` and `courses`.

Every command sits in its own cell with a comment. Besides filtering, joins, grouping, CTEs and window functions, it shows text patterns with `LIKE` and regular expressions (`~`, `~*` and their negations `!~`, `!~*`), type conversion with `CAST` and `::`, dates with `EXTRACT`, `DATE_TRUNC`, `INTERVAL` and `AGE`, and how the notebook connects: the server, the database `postgres` and the connection string. The last cell stops the server and deletes the temporary folder, so each run starts from an empty database. To start again, restart the kernel and run all cells. The playground does not need the course database `airbnb` or the case-study download.

## Star against snowflake

[Open notebook](star-vs-snowflake-playground.ipynb)

The same students, courses and exam results stored twice: as a star schema (tables `star_…`, flat dimensions that repeat the programme, faculty and department names) and as a snowflake schema (tables `snowflake_…`, one table per level). Each section runs the same task on both: the same question with one join against three, a rename that changes four rows against one, an update that leaves the star inconsistent, a programme without students that only the snowflake can store. A table of pros and cons closes it. Like the playground, it starts its own empty server and needs no course data.
