"""Exercise 1: metadata filter in SQL (replace SQLiteVectorStore.search in src/bti_assistant/store.py)."""

import numpy as np

from bti_assistant.store import Hit, _rank


def search(self, vector: np.ndarray, k: int = 5, chapter: str | None = None) -> list[Hit]:
    sql, params = "SELECT chunk_id, doc_id, text, heading, embedding FROM chunks", []
    if chapter is not None:
        sql, params = sql + " WHERE substr(heading, 1, 2) = ?", [chapter]  # SQL selects, NumPy ranks
    rows = self.con.execute(sql, params).fetchall()
    return _rank(rows, vector, k, self.dim)
