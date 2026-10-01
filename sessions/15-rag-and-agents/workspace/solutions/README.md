# Reference solutions (for self-checking)

Look here only after you have tried the exercise yourself. Each file shows the changed function; copy it
into the module named in the file and run `uv run pytest -q`: the `xfail` tests of `tests/test_exercises.py`
then report `XPASS`.

| Exercise | File | Replaces |
|---|---|---|
| 1 | [pgvector_filter.py](pgvector_filter.py) | `PgVectorStore.search` in `src/review_assistant/store.py` |
| 2 | [chunk_sentences.py](chunk_sentences.py) | `chunk_sentences` in `src/review_assistant/chunking.py` |
| 3 | [agent_budget.py](agent_budget.py) | `run_agent` in `src/review_assistant/agent.py` |
