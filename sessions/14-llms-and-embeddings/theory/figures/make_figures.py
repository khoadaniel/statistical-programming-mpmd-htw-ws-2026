"""Figures for Session 14 (language models and embeddings).

Run from the repository root:
    uv run --with sentence-transformers python sessions/14-llms-and-embeddings/theory/figures/make_figures.py

Writes tokenisation.png, attention.png and embedding_space.png next to this script.
embedding_space.png encodes 1,500 EBTI decisions of five chapters with intfloat/multilingual-e5-small
(about 470 MB download on first use). Everything is seeded and deterministic up to
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

    sp = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    gpt = tiktoken.get_encoding("o200k_base")
    en = "plastic toy car with wheels"
    de = "Kunststoffspielzeugauto mit Rädern"
    rows = [
        ("words", en.split()),
        ("SentencePiece\n(XLM-R, e5)", sp.tokenize(en)),
        ("BPE\n(o200k, GPT-4o)", [gpt.decode([i]) for i in gpt.encode(en)]),
        ("German, words", de.split()),
        ("German,\nSentencePiece", sp.tokenize(de)),
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
                 "(␣ = leading space in BPE; ▁ = start of a word in SentencePiece)")
    fig.tight_layout()
    fig.savefig(OUT / "tokenisation.png")
    plt.close(fig)


def attention() -> None:
    """Illustrative attention weights: softmax of hand-set scores (not from a trained model)."""
    tokens = ["the", "boot", "leaks", "because", "it", "cracked"]
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
                 "'it' takes most of its new vector from 'boot'")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "attention.png")
    plt.close(fig)


def embedding_space() -> None:
    import umap
    from sentence_transformers import SentenceTransformer

    d = pd.read_parquet(DATA)
    chapters = {"61": "61 knitted clothing", "64": "64 footwear", "85": "85 electrical machinery",
                "95": "95 toys, sports", "22": "22 beverages"}
    sub = d[d["chapter"].isin(list(chapters))].groupby("chapter").sample(n=300, random_state=0)
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    E = model.encode(("passage: " + sub["description"]).tolist(), batch_size=64, normalize_embeddings=True)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(E)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=130)
    for k, (ch, name) in enumerate(chapters.items()):
        m = (sub["chapter"] == ch).to_numpy()
        axes[0].scatter(xy[m, 0], xy[m, 1], s=6, color=PALETTE[k], label=name)
    axes[0].set_title("Coloured by HS chapter")
    axes[0].legend(fontsize=8, markerscale=2, loc="best", frameon=False)
    langs = sub["language"].where(sub["language"].isin(["de", "fr", "en", "nl"]), "other")
    for k, lang in enumerate(["de", "fr", "nl", "en", "other"]):
        m = (langs == lang).to_numpy()
        axes[1].scatter(xy[m, 0], xy[m, 1], s=6, color=PALETTE[k], label=lang, alpha=0.7)
    axes[1].set_title("Coloured by language of the description")
    axes[1].legend(markerscale=2, frameon=False)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle("1,500 decisions of five chapters embedded with multilingual-e5-small (384 dimensions), "
                 "projected to 2-D with UMAP", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "embedding_space.png")
    plt.close(fig)


if __name__ == "__main__":
    tokenisation()
    attention()
    embedding_space()
    print("figures written to", OUT)
