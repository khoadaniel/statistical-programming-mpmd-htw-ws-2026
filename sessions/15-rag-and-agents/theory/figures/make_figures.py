"""Figures for the Session 15 theory pages. Run from the repository root:

    uv run --with pandas --with pyarrow --with matplotlib --with scikit-learn --with sentence-transformers \
        python sessions/15-rag-and-agents/theory/figures/make_figures.py

Uses the case-study data and all-MiniLM-L6-v2 (downloaded on first use, about 90 MB). Deterministic.
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
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})

reviews = pd.read_parquet(DATA / "train_sample.parquet")
products = pd.read_parquet(DATA / "products.parquet", columns=["parent_asin", "title"])
top = reviews["parent_asin"].value_counts().index[:300]
docs = (reviews[reviews["parent_asin"].isin(top) & (reviews["text"].str.len() > 0)]
        .groupby("parent_asin").head(20)
        .merge(products.rename(columns={"title": "product"}), on="parent_asin").reset_index(drop=True))
docs["doc"] = docs["title"] + ". " + docs["text"]


def chunk(text, size=120, overlap=20):
    words = text.split()
    return [" ".join(words[s:s + size]) for s in range(0, max(len(words) - overlap, 1), size - overlap)]


chunks = docs.assign(chunk=docs["doc"].map(chunk)).explode("chunk", ignore_index=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(chunks["chunk"].tolist(), batch_size=64, normalize_embeddings=True)

# ------------------------------------------------------------ figure 1: recall@k curves
test_set = {
    "The smart scale will not connect to my WiFi network": {"r006688", "r006941", "r007834", "r009514"},
    "Does this fish oil give you fishy burps?": {"r006960", "r082069", "r102598", "r114099"},
    "Is the face mask big enough for an adult with a large face?": {"r325547", "r326996", "r331242", "r332009"},
    "The lids of the pill box open by themselves": {"r163757", "r169573", "r178628", "r180913", "r184429",
                                                    "r191360"},
    "The heating pad stopped working after a few months": {"r224359", "r237231", "r252220", "r259580",
                                                           "r343446", "r344999", "r356255"},
    "Is the GermGuardian UV sanitizer noisy?": {"r008885", "r009564", "r012771"},
    "Does Nerdwax keep glasses in place all day or does it wear off?": {"r062078", "r063993", "r064806",
                                                                        "r070674", "r067183"},
    "Can you adjust the volume of the Marpac Dohm white noise machine?": {"r003861", "r005523"},
    "Do Magic Eraser sponges fall apart quickly?": {"r035340", "r036315", "r038171", "r031760"},
    "Does the plug-in UV air sanitizer remove bad smells from a room?": {"r000987", "r004506", "r007112",
                                                                         "r008013", "r011183"},
}


def ranked_ids(scores):
    return chunks.assign(s=scores).sort_values("s", ascending=False).drop_duplicates("review_id")["review_id"].tolist()


tok = [Counter(re.findall(r"[a-z0-9]+", t.lower())) for t in chunks["chunk"]]
length = np.array([sum(c.values()) for c in tok])
df_count = Counter(w for c in tok for w in c)
N, avg = len(tok), length.mean()


def bm25(q):
    s = np.zeros(N)
    for w in set(re.findall(r"[a-z0-9]+", q.lower())):
        if w in df_count:
            idf = math.log(1 + (N - df_count[w] + 0.5) / (df_count[w] + 0.5))
            tf = np.array([c.get(w, 0) for c in tok])
            s += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / avg))
    return ranked_ids(s)


def rrf(rankings, k=60):
    sc = {}
    for r in rankings:
        for i, d in enumerate(r[:50], 1):
            sc[d] = sc.get(d, 0) + 1 / (k + i)
    return sorted(sc, key=lambda d: -sc[d])


ks = np.arange(1, 21)
curves = {"dense (MiniLM)": [], "keyword (BM25)": [], "hybrid (RRF)": []}
for q, rel in test_set.items():
    dense = ranked_ids(emb @ model.encode(q, normalize_embeddings=True))
    kw = bm25(q)
    for name, r in [("dense (MiniLM)", dense), ("keyword (BM25)", kw), ("hybrid (RRF)", rrf([dense, kw]))]:
        curves[name].append([len(set(r[:k]) & rel) / len(rel) for k in ks])

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
for (name, vals), color in zip(curves.items(), [BLUE, ORANGE, AQUA]):
    m = np.mean(vals, axis=0)
    ax.plot(ks, m, color=color, lw=2, marker="o", ms=4, label=name)
    ax.annotate(f"{name}  {m[-1]:.2f}", (ks[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=9)
ax.axvline(5, color=INK2, lw=1, ls=":")
ax.text(5.2, 0.04, "k = 5", color=INK2, fontsize=9)
ax.set(xlabel="k (number of retrieved reviews)", ylabel="mean recall@k (10 questions)", ylim=(0, 1),
       xlim=(0.5, 27), xticks=[1, 5, 10, 15, 20])
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Recall@k on ten labelled questions about product reviews", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "recall-at-k.png")
print({n: np.round(np.mean(v, axis=0)[[0, 4, 9, 19]], 3).tolist() for n, v in curves.items()})

# ------------------------------------------------------------ figure 2: embedding map
names = {"B0077L8YFI": "smart scale", "B0047VWYSO": "fish oil", "B07FTK5DWF": "white noise machine",
         "B00BR1FSU8": "cleaning sponge"}
sel = chunks[chunks["parent_asin"].isin(names)].drop_duplicates("review_id")
E = emb[sel.index]
query = "the wifi connection keeps failing"
qv = model.encode(query, normalize_embeddings=True)
pca = PCA(2, random_state=0).fit(E)
P, qp = pca.transform(E), pca.transform(qv[None])[0]
top5 = np.argsort(-(E @ qv))[:5]
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
for (asin, label), color in zip(names.items(), [BLUE, ORANGE, AQUA, YELLOW]):
    m = (sel["parent_asin"] == asin).to_numpy()
    ax.scatter(P[m, 0], P[m, 1], s=40, color=color, edgecolor=SURFACE, lw=1.5, label=label, zorder=2)
ax.scatter(P[top5, 0], P[top5, 1], s=150, facecolor="none", edgecolor=INK, lw=1.5, zorder=3,
           label="5 nearest to the query")
ax.scatter(*qp, marker="*", s=320, color=INK, zorder=4)
ax.annotate(f'query: "{query}"', qp, xytext=(10, 10), textcoords="offset points", fontsize=9, color=INK)
ax.set(xlabel="first principal component", ylabel="second principal component", xticks=[], yticks=[])
ax.legend(frameon=False, fontsize=9, loc="best")
ax.set_title("Review embeddings of four products, projected to two dimensions", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "embedding-map.png")
print("saved figures")
