"""Figures for the Session 15 theory pages. Run from the repository root:

    uv run --with sentence-transformers python sessions/15-rag-and-agents/theory/figures/make_figures.py

Uses the EBTI case-study data, intfloat/multilingual-e5-small (about 470 MB) and the English
cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 (about 90 MB), both downloaded on first use.
Corpus: 10,000 decisions of 2017-2021 from the training sample; queries: 500 decisions of 2022-2023.
Deterministic up to floating-point differences between machines. Runs about two minutes with a GPU.
"""

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e44ad"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})

d = pd.read_parquet(DATA / "train_sample.parquet")
year = d["start_date"].dt.year
corpus = d[year <= 2021].sample(10_000, random_state=0).reset_index(drop=True)
queries = d[year >= 2022].sample(2_000, random_state=0).iloc[:500].reset_index(drop=True)
model = SentenceTransformer("intfloat/multilingual-e5-small")
E = model.encode(("passage: " + corpus["description"]).tolist(), batch_size=64, normalize_embeddings=True)
Q = model.encode(("query: " + queries["description"]).tolist(), batch_size=64, normalize_embeddings=True)
H = corpus["heading"].to_numpy()

# ------------------------------------------------------------ figure 1: heading hit@k curves
TOKEN = re.compile(r"\w+")
tok = [Counter(TOKEN.findall(t.lower())) for t in corpus["description"]]
length = np.array([sum(c.values()) for c in tok])
df_count = Counter(w for c in tok for w in c)
N, avg = len(tok), length.mean()
postings: dict[str, list[tuple[int, int]]] = {}
for i, c in enumerate(tok):
    for w, n in c.items():
        postings.setdefault(w, []).append((i, n))


def bm25(q, depth=50):
    s = np.zeros(N)
    for w in set(TOKEN.findall(q.lower())):
        if w in df_count:
            idf = math.log(1 + (N - df_count[w] + 0.5) / (df_count[w] + 0.5))
            for i, tf in postings[w]:
                s[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length[i] / avg))
    return list(np.argsort(-s)[:depth])


def rrf(rankings, k=60):
    sc = {}
    for r in rankings:
        for rank, i in enumerate(r[:50], 1):
            sc[i] = sc.get(i, 0) + 1 / (k + rank)
    return sorted(sc, key=lambda i: -sc[i])


ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
ks = np.arange(1, 21)
names = ["dense (multilingual-e5)", "keyword (BM25)", "hybrid (RRF)", "hybrid + English cross-encoder"]
curves = {n: [] for n in names}
for j, q in enumerate(queries["description"]):
    dense = list(np.argsort(-(E @ Q[j]))[:50])
    kw = bm25(q)
    hy = rrf([dense, kw])
    top20 = hy[:20]
    reranked = [top20[i] for i in np.argsort(-ce.predict([(q, corpus["description"][i]) for i in top20]))]
    for n, r in zip(names, [dense, kw, hy, reranked]):
        curves[n].append([float(queries["heading"][j] in set(H[r[:k]])) for k in ks])

fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
for (name, vals), color in zip(curves.items(), [BLUE, ORANGE, AQUA, PURPLE]):
    m = np.mean(vals, axis=0)
    ax.plot(ks, m, color=color, lw=2, marker="o", ms=3.5, label=f"{name}: hit@5 = {m[4]:.2f}")
ax.axvline(5, color=INK2, lw=1, ls=":")
ax.text(5.2, 0.33, "k = 5", color=INK2, fontsize=9)
ax.set(xlabel="k (number of retrieved decisions)", ylabel="heading hit@k (500 queries)", ylim=(0.3, 0.9),
       xlim=(0.5, 20.5), xticks=[1, 5, 10, 15, 20])
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Is the true heading among the headings of the k retrieved decisions?", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "recall-at-k.png")
print({n: np.round(np.mean(v, axis=0)[[0, 4, 9, 19]], 3).tolist() for n, v in curves.items()})

# ------------------------------------------------------------ figure 2: embedding map
labels = {"6403": "6403 footwear, leather uppers", "6404": "6404 footwear, textile uppers",
          "9503": "9503 toys", "6307": "6307 made-up textile articles (masks, ...)"}
sel = corpus[corpus["heading"].isin(labels)].groupby("heading").head(60)
Es = E[sel.index]
query = "disposable face mask made of nonwoven fabric"
qv = model.encode("query: " + query, normalize_embeddings=True)
pca = PCA(2, random_state=0).fit(Es)
P, qp = pca.transform(Es), pca.transform(qv[None])[0]
top5 = np.argsort(-(Es @ qv))[:5]
fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
for (h, label), color in zip(labels.items(), [BLUE, ORANGE, AQUA, YELLOW]):
    m = (sel["heading"] == h).to_numpy()
    ax.scatter(P[m, 0], P[m, 1], s=36, color=color, edgecolor=SURFACE, lw=1.2, label=label, zorder=2)
ax.scatter(P[top5, 0], P[top5, 1], s=150, facecolor="none", edgecolor=INK, lw=1.5, zorder=3,
           label="5 nearest to the query")
for i in top5:
    ax.annotate(sel["language"].iloc[i], P[i], xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK2)
ax.scatter(*qp, marker="*", s=320, color=INK, zorder=4)
ax.annotate(f'English query: "{query}"', qp, xytext=(10, 10), textcoords="offset points", fontsize=9, color=INK)
ax.set(xlabel="first principal component", ylabel="second principal component", xticks=[], yticks=[])
ax.legend(frameon=False, fontsize=8.5, loc="best")
ax.set_title("Decisions of four headings in many languages, projected to two dimensions", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "embedding-map.png")
print("nearest:", sel.iloc[top5][["heading", "language"]].values.tolist())
print("saved figures")
