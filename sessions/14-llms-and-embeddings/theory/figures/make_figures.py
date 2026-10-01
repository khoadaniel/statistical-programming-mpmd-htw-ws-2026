"""Figures for Session 14 (language models and embeddings).

Run from the repository root:
    uv run --with pandas --with pyarrow --with scikit-learn --with matplotlib \
        --with sentence-transformers --with tiktoken --with umap-learn \
        python sessions/14-llms-and-embeddings/theory/figures/make_figures.py

Writes tokenisation.png, attention.png and embedding_space.png next to this script.
embedding_space.png encodes 1,500 sample reviews with sentence-transformers/all-MiniLM-L6-v2
(about 90 MB download on first use). Everything is seeded and deterministic up to
floating-point differences between machines.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[4] / "case-study" / "data" / "train_sample.parquet"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2471a3", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#17a589"]


def draw_tokens(ax, y, tokens, label, colors=("#d6eaf8", "#fdebd0")):
    """Draw tokens as alternating coloured boxes in one row."""
    x = 0.0
    ax.text(-0.2, y, label, ha="right", va="center", fontsize=10, fontweight="bold")
    for i, tok in enumerate(tokens):
        shown = tok.replace(" ", "␣")          # make leading spaces visible
        w = 0.13 * len(shown) + 0.25
        ax.add_patch(plt.Rectangle((x, y - 0.3), w, 0.6, facecolor=colors[i % 2],
                                   edgecolor="#555555", linewidth=0.8))
        ax.text(x + w / 2, y, shown, ha="center", va="center", fontsize=9, family="monospace")
        x += w + 0.08
    ax.text(x + 0.15, y, f"{len(tokens)} tokens", va="center", fontsize=9, color="#555555")
    return x


def tokenisation() -> None:
    import tiktoken
    from transformers import AutoTokenizer

    bert = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    gpt = tiktoken.get_encoding("o200k_base")
    en = "Unbelievably overpriced toothbrush heads"
    de = "Leider überteuerte Zahnbürstenaufsätze"
    rows = [
        ("words", en.split()),
        ("WordPiece\n(MiniLM/BERT)", bert.tokenize(en)),
        ("BPE\n(o200k, GPT-4o)", [gpt.decode([i]) for i in gpt.encode(en)]),
        ("German, words", de.split()),
        ("German,\nWordPiece", bert.tokenize(de)),
        ("German, BPE", [gpt.decode([i]) for i in gpt.encode(de)]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=130)
    width = 0
    for k, (label, toks) in enumerate(rows):
        width = max(width, draw_tokens(ax, -k * 0.9 - (0.4 if k >= 3 else 0), toks, label))
    ax.set_xlim(-2.4, width + 1.6)
    ax.set_ylim(-5.4, 0.6)
    ax.axis("off")
    ax.set_title("The same text split into words and into subword tokens\n"
                 "(␣ = leading space; ## = continues the previous token)")
    fig.tight_layout()
    fig.savefig(OUT / "tokenisation.png")
    plt.close(fig)


def attention() -> None:
    """Illustrative attention weights: softmax of hand-set scores (not from a trained model)."""
    tokens = ["the", "pump", "stopped", "because", "it", "overheated"]
    scores = np.array([
        [2.0, 1.0, 0.2, 0.0, 0.1, 0.1],
        [0.5, 2.0, 1.0, 0.0, 0.6, 0.4],
        [0.1, 1.5, 2.0, 0.5, 0.4, 0.8],
        [0.0, 0.3, 1.2, 2.0, 0.3, 1.2],
        [0.2, 2.8, 0.6, 0.2, 1.0, 1.2],
        [0.0, 1.2, 0.8, 0.6, 1.6, 2.0],
    ])
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=130)
    ax.imshow(weights, cmap="Blues", vmin=0, vmax=weights.max())
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if weights[i, j] > 0.4 else "black")
    ax.set_xticks(range(len(tokens)), tokens, rotation=30, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("attends to (keys)")
    ax.set_ylabel("token being updated (queries)")
    ax.add_patch(plt.Rectangle((-0.5, 3.5), len(tokens), 1, fill=False, edgecolor="#c0392b", lw=2))
    ax.set_title("Attention weights (illustrative): each row sums to 1;\n"
                 "'it' takes most of its new vector from 'pump'")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "attention.png")
    plt.close(fig)


def embedding_space() -> None:
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    reviews = pd.read_parquet(DATA)
    sub = reviews.groupby("label").sample(n=500, random_state=0)
    texts = (sub["title"] + ". " + sub["text"]).tolist()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    E = model.encode(texts, batch_size=64, normalize_embeddings=True)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(E)
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(E)

    tv = TfidfVectorizer(min_df=5, stop_words="english")
    T = tv.fit_transform(texts)
    names = tv.get_feature_names_out()
    overall = np.asarray(T.mean(axis=0)).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=130)
    for k in range(6):
        m = km.labels_ == k
        lift = np.asarray(T[m].mean(axis=0)).ravel() / (overall + 1e-9)
        lift[np.asarray((T[m] > 0).sum(axis=0)).ravel() < 5] = 0
        top = ", ".join(names[np.argsort(lift)[::-1][:3]])
        axes[0].scatter(xy[m, 0], xy[m, 1], s=6, color=PALETTE[k], label=f"{k}: {top}")
    axes[0].set_title("Coloured by k-means cluster (k = 6)\nlegend: distinctive words")
    axes[0].legend(fontsize=7, markerscale=2, loc="best", frameon=False)
    for label, color in [("pos", "#2471a3"), ("neu", "#7f8c8d"), ("neg", "#c0392b")]:
        m = (sub["label"] == label).to_numpy()
        axes[1].scatter(xy[m, 0], xy[m, 1], s=6, color=color, label=label, alpha=0.7)
    axes[1].set_title("Coloured by sentiment label")
    axes[1].legend(markerscale=2, frameon=False)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle("1,500 reviews embedded with all-MiniLM-L6-v2 (384 dimensions), "
                 "projected to 2-D with UMAP")
    fig.tight_layout()
    fig.savefig(OUT / "embedding_space.png")
    plt.close(fig)


if __name__ == "__main__":
    tokenisation()
    attention()
    embedding_space()
    print("figures written to", OUT)
