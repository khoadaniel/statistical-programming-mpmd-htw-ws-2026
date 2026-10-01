"""Make the figures of the Session 11 theory pages.

Run from the repository root:
    uv run --with numpy --with matplotlib --with scikit-learn --with scipy \
        python sessions/11-unsupervised-learning/theory/figures/make_figures.py

All figures use toy data or built-in scikit-learn data and a fixed random seed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=110)
    plt.close(fig)
    print("wrote", name)


def kmeans_iterations():
    """Lloyd's algorithm step by step on three blobs, with deliberately poor starting centres."""
    X, _ = make_blobs(n_samples=150, centers=[[0, 0], [4, 4], [0, 5]], cluster_std=0.9, random_state=1)
    centres = np.array([[-1.0, 1.0], [1.0, -0.5], [3.0, 2.0]])  # poor start: two centres in one group
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharex=True, sharey=True)
    for step, ax in enumerate(axes):
        labels = np.argmin(((X[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        for k in range(3):
            ax.scatter(*X[labels == k].T, s=14, color=SERIES[k], alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.scatter(*centres.T, marker="X", s=160, color=INK, edgecolor="white", linewidth=1.5, zorder=3)
        inertia = ((X - centres[labels]) ** 2).sum()
        ax.set_title(f"Iteration {step}: inertia {inertia:.0f}")
        new = np.array([X[labels == k].mean(axis=0) for k in range(3)])
        for old, nw in zip(centres, new):
            if step < 3:
                ax.annotate("", xy=nw, xytext=old, arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})
        centres = new
    axes[0].set_ylabel("feature 2")
    for ax in axes:
        ax.set_xlabel("feature 1")
    fig.suptitle("k-means: assign points to the nearest centre (X), then move each centre to the mean of its points",
                 fontsize=10.5, color=INK)
    save(fig, "kmeans_iterations")


def elbow_silhouette():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=7)
    ks = range(2, 10)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(list(ks), inertia, color=SERIES[0], lw=2, marker="o", ms=6)
    a1.annotate("elbow at k = 4", xy=(4, inertia[2]), xytext=(5.5, inertia[0] * 0.6),
                arrowprops={"arrowstyle": "->", "color": INK}, color=INK)
    a1.set(title="Elbow method: within-cluster sum of squares", xlabel="number of clusters k", ylabel="inertia")
    a2.plot(list(ks), sil, color=SERIES[1], lw=2, marker="o", ms=6)
    best = int(np.argmax(sil))
    a2.annotate(f"maximum {sil[best]:.2f} at k = {list(ks)[best]}", xy=(list(ks)[best], sil[best]),
                xytext=(5.5, sil[best] - 0.05), arrowprops={"arrowstyle": "->", "color": INK}, color=INK)
    a2.set(title="Mean silhouette coefficient", xlabel="number of clusters k", ylabel="silhouette")
    save(fig, "elbow_silhouette")


def dendrogram_figure():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.45, size=(5, 2)) for c in ([0, 0], [3, 0.5], [1.5, 3])])
    Z = linkage(X, method="ward")
    cut = (Z[-3, 2] + Z[-2, 2]) / 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    set_link_color_palette(SERIES[:3])
    tree = dendrogram(Z, ax=a2, color_threshold=cut, above_threshold_color=MUTED)
    colour = dict(zip(tree["leaves"], tree["leaves_color_list"]))  # same colour in both panels
    a1.scatter(*X.T, s=60, c=[colour[i] for i in range(len(X))], edgecolor="white", linewidth=0.8)
    for i, (x, y) in enumerate(X):
        a1.annotate(str(i), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    a1.set(title="15 points, three groups", xlabel="feature 1", ylabel="feature 2")
    a2.axhline(cut, color=INK, ls="--", lw=1.2)
    a2.text(a2.get_xlim()[1], cut, "  cut: 3 clusters", va="bottom", ha="right", color=INK)
    a2.set(title="Dendrogram (Ward linkage)", xlabel="point", ylabel="merge height (distance)")
    a2.grid(False)
    save(fig, "dendrogram")


def pca_scree_biplot():
    wine = load_wine(as_frame=True)
    Z = StandardScaler().fit_transform(wine.data)
    pca = PCA().fit(Z)
    scores = pca.transform(Z)[:, :2]
    ratio = pca.explained_variance_ratio_
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    pcs = np.arange(1, len(ratio) + 1)
    a1.bar(pcs, ratio, color=SERIES[0], width=0.7)
    a1.plot(pcs, ratio.cumsum(), color=SERIES[1], marker="o", ms=5, lw=2, label="cumulative")
    a1.axhline(0.8, color=MUTED, ls=":", lw=1)
    a1.text(13, 0.81, "80 %", ha="right", va="bottom", color=INK, fontsize=9)
    a1.set(title="Scree plot: explained variance ratio (wine, standardised)", xlabel="principal component",
           ylabel="share of variance", xticks=pcs)
    a1.legend(frameon=False, loc="center right")
    for k in range(3):
        m = wine.target == k
        a2.scatter(*scores[m].T, s=14, color=SERIES[k], alpha=0.7, label=f"cultivar {k}", edgecolor="white", linewidth=0.3)
    load = pca.components_[:2].T * 6.5  # scale arrows to the score range
    top = np.argsort(-np.hypot(*load.T))[:6]
    for j in top:
        a2.annotate("", xy=load[j], xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1})
        name = wine.data.columns[j].replace("od280/od315_of_diluted_wines", "od280/od315")
        a2.text(*(load[j] * 1.15), name, fontsize=8, color=INK, ha="center",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1})
    a2.set_ylim(-4.3, 4.3)
    a2.set(title="Biplot: scores (points) and loadings (arrows)",
           xlabel=f"PC1 ({ratio[0]:.0%})", ylabel=f"PC2 ({ratio[1]:.0%})")
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "pca_scree_biplot")


if __name__ == "__main__":
    kmeans_iterations()
    elbow_silhouette()
    dendrogram_figure()
    pca_scree_biplot()
