# Session 3 · SQL playground

[Open notebook](sql-playground.ipynb)

A small notebook to try SQL on your own. The first cell starts a new, empty PostgreSQL server in a temporary folder and creates three small tables:

- `students`: one row per student (name, programme, start year).
- `courses`: one row per course (title, credits).
- `scores`: one row per exam, with foreign keys to `students` and `courses`.

Every command sits in its own cell with a comment. The last cell stops the server and deletes the temporary folder, so each run starts from an empty database. To start again, restart the kernel and run all cells. The playground does not need the course database `airbnb` or the case-study download.
