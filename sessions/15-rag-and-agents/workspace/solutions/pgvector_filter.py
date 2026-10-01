"""Exercise 1: metadata filter in SQL (replace PgVectorStore.search in src/review_assistant/store.py)."""

import numpy as np

from review_assistant.store import Hit


def search(self, vector: np.ndarray, k: int = 5, parent_asin: str | None = None) -> list[Hit]:
    q = np.asarray(vector, dtype=np.float32)
    where, params = "", [q]
    if parent_asin is not None:
        where, params = "WHERE parent_asin = %s", [q, parent_asin]
    rows = self.con.execute(
        f"""SELECT chunk_id, doc_id, text, 1 - (embedding <=> %s) AS score, parent_asin
            FROM {self.table}
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s""",
        (*params, q, k),
    ).fetchall()
    return [Hit(r[0], r[1], r[2], float(r[3]), r[4]) for r in rows]
