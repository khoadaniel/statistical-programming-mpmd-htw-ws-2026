# Dimensionality reduction: PCA, t-SNE and UMAP

Tables in practice have many columns, and people can only look at two or three at a time. **Dimensionality reduction** replaces many columns by a few new ones that keep as much of the relevant information as possible. This page covers the linear standard method, **principal component analysis** (PCA), with its two key outputs, explained variance and loadings, and then two non-linear methods made for pictures, **t-SNE** and **UMAP**. It ends with what such pictures can and cannot show.

The code blocks build on each other; run them in order. They use two built-in scikit-learn datasets: 178 wines with 13 chemical measurements, and 1,797 images of handwritten digits with 64 pixels each.

```mermaid
flowchart LR
    G{"Goal?"} -->|"compress, denoise,<br/>features for a model"| PCA["PCA<br/>(linear, reversible,<br/>keeps global distances)"]
    G -->|"a 2-D picture of<br/>neighbourhoods"| NL["t-SNE or UMAP<br/>(non-linear, keeps<br/>local neighbours)"]
    PCA --> R1["Report explained variance<br/>and loadings"]
    NL --> R2["Do not read distances<br/>between clusters or cluster sizes"]
    PCA -->|"many columns:<br/>first 30-50 PCs"| NL
```

## Principal component analysis: explained variance and loadings

### Concept

PCA finds new axes, the **principal components** (PCs). Each PC is a weighted sum of the original columns. The first PC is the direction along which the rows vary most (largest **variance**). The second PC is the direction of largest remaining variance that is uncorrelated with (at right angles to) the first, and so on. With p columns there are p components; usually the first few carry most of the variance.

Three outputs matter:

- **Scores**: the coordinates of each row on the new axes; PC1 and PC2 scores give a 2-D map of the data.
- **Explained variance ratio**: the share of the total variance carried by each PC. A **scree plot** shows these shares; the cumulative sum tells how many PCs keep, say, 80 % of the variance.
- **Loadings**: the weights of the original columns in each PC. They say what a component *means*: a PC with large weights on alcohol, proline and colour intensity describes "strong, dark wines".

Worked example with two standardised columns that have correlation 0.8. Their covariance matrix is [[1, 0.8], [0.8, 1]]. Its **eigenvectors** (the directions that the matrix only stretches) are (1, 1)/√2 and (1, −1)/√2, with **eigenvalues** (the stretch factors, here the variances along those directions) 1 + 0.8 = 1.8 and 1 − 0.8 = 0.2. So PC1 is the average direction of the two columns and explains 1.8 / (1.8 + 0.2) = 90 % of the variance; PC2, their difference, explains 10 %. Both loadings on PC1 are 1/√2 ≈ 0.71: the two columns count equally.

![Left: scree plot for the wine data with bars falling from 36 % for PC1 to under 1 % for PC13 and a cumulative line passing 80 % at five components. Right: biplot of PC1 against PC2 with the three cultivars as separated point clouds and arrows for the six largest loadings](figures/pca_scree_biplot.png)

The **biplot** on the right combines scores (points) and loadings (arrows). Points lie in the direction of the arrows of the columns on which they have high values. The three wine cultivars, which PCA did not see, separate clearly in the first two components.

### Why it matters

Correlated columns carry overlapping information. PCA summarises them in a few uncorrelated scores. This helps to plot high-dimensional data, to remove noise (dropping the last PCs), to speed up later methods and to deal with models that suffer from many correlated inputs. Because PCA uses variance, it must be run on **standardised** columns unless all columns share one unit; otherwise the column with the largest numbers becomes PC1 on its own.

### How it works in Python

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

# the worked example: eigenvalues and eigenvectors of a 2 x 2 correlation matrix
eigval, eigvec = np.linalg.eigh(np.array([[1.0, 0.8], [0.8, 1.0]]))
print(eigval.round(2), np.abs(eigvec[:, -1]).round(2))   # [0.2 1.8] [0.71 0.71]

wine = load_wine(as_frame=True)                            # 178 wines, 13 measurements
Z = StandardScaler().fit_transform(wine.data)
pca = PCA().fit(Z)
print(pca.explained_variance_ratio_[:3].round(3))          # [0.362 0.192 0.111]
print(pca.explained_variance_ratio_.cumsum()[[1, 4]].round(2))   # [0.55 0.8 ]: 2 PCs 55 %, 5 PCs 80 %
print(PCA().fit(wine.data).explained_variance_ratio_[0].round(3))   # 0.998 without standardising

loadings = pd.DataFrame(pca.components_[:2].T, index=wine.data.columns, columns=["PC1", "PC2"])
print(loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(4).round(2))
#                                PC1   PC2
# flavanoids                    0.42 -0.00
# total_phenols                 0.39  0.07
# od280/od315_of_diluted_wines  0.38 -0.16
# proanthocyanins               0.31  0.04

scores = pca.transform(Z)[:, :2]                           # coordinates for a scatter plot
print(scores.shape)                                         # (178, 2)
```

PC1 is a "phenol" axis (flavanoids, total phenols, proanthocyanins). Without standardising, proline (values in the hundreds and thousands) alone accounts for 99.8 % of the variance.

`PCA(n_components=0.8)` keeps as many components as needed for 80 % of the variance. The sign of a component is arbitrary: two libraries may return PC1 and −PC1.

### In practice

- **Population genetics.** Novembre et al. (2008, *Nature*): PC1 and PC2 of the genotypes of about 1,400 Europeans reproduce the map of Europe.
- **Finance.** Litterman and Scheinkman (1991, *Journal of Fixed Income*): three components, called level, slope and curvature, explain most movements of government bond yields; risk desks still report exposures on them.
- **Faces and images.** Turk and Pentland (1991) represented faces by their first principal components ("eigenfaces"); workbook 13 reproduces the idea.

> [!WARNING]
> **High explained variance is not high relevance.** PCA keeps the directions with most variance, not those most related to a target. A small component can be the one that predicts churn.

> [!CAUTION]
> **Fit PCA on training data only.** Inside a model pipeline, PCA is a preparation step like scaling: `make_pipeline(StandardScaler(), PCA(10), model)`. Fitting it on all data before a split leaks information from the test set (Session 7).

## t-SNE and UMAP for visualisation

### Concept

PCA is linear: it can only rotate and project. Data that lie on a curved surface (images of the same digit written in different ways, for example) are not separated by any flat projection. Two non-linear methods focus on **neighbourhoods** instead of variance:

- **t-SNE** (t-distributed stochastic neighbour embedding; van der Maaten and Hinton, 2008) turns distances into probabilities that two rows are neighbours, both in the original space and in a 2-D map, and moves the points in the map until the two sets of probabilities match. The **perplexity** (typically 5–50) sets roughly how many neighbours each point considers.
- **UMAP** (uniform manifold approximation and projection; McInnes, Healy and Melville, 2018) builds a graph of the **n_neighbors** nearest neighbours of every row and lays the graph out in 2-D. **min_dist** sets how tightly points may be packed. UMAP is usually faster than t-SNE, keeps somewhat more global structure and can transform new rows.

Both keep **local** structure: rows that are neighbours in the original space stay neighbours in the map. A measure of this is **trustworthiness** (between 0 and 1): how many of the 10 nearest neighbours of each point in the map are also among its true neighbours.

### Why it matters

A good 2-D map lets people see groups, outliers and mislabelled examples in data with dozens or thousands of columns (embeddings of reviews in Session 14, for instance). But the maps distort. Distances *between* groups, the *size* of groups and the empty space in a t-SNE or UMAP map have no reliable meaning, and different settings can produce different pictures from the same data.

### How it works in Python

```python
digits = load_digits()                                     # 1,797 images, 8 x 8 = 64 pixels
X = digits.data
pca_map = PCA(n_components=2).fit_transform(X)
tsne_map = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)   # about 10-30 s

for name, emb in {"PCA": pca_map, "t-SNE": tsne_map}.items():
    print(name, round(trustworthiness(X, emb, n_neighbors=10), 3))
# PCA 0.83
# t-SNE 0.993
print(PCA().fit(X).explained_variance_ratio_[:2].round(3))   # [0.149 0.136]: only 29 % in 2 PCs

# UMAP (package umap-learn, in the course environment)
import umap  # noqa: E402
umap_map = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
print("UMAP", round(trustworthiness(X, umap_map, n_neighbors=10), 3))   # UMAP 0.988

# plot: plt.scatter(*tsne_map.T, c=digits.target, s=4, cmap="tab10")
```

In the t-SNE and UMAP maps the ten digits form ten separate groups, although the labels were not used; in the PCA map they overlap. Workbook 17 shows how the t-SNE picture changes with the perplexity, workbook 18 how UMAP changes with n_neighbors and min_dist.

### In practice

- **Single-cell biology.** t-SNE and UMAP maps of gene expression in thousands of single cells are the standard figure for showing cell types; Becht et al. (2019, *Nature Biotechnology*) compared UMAP with t-SNE for this use.
- **Inspecting embeddings.** The TensorFlow Embedding Projector shows PCA, t-SNE and UMAP views of word or image embeddings to explore what a model has learned.
- **Data quality.** Mislabelled training examples often appear as points of one colour inside a group of another colour in a t-SNE map of image or text data.

> [!WARNING]
> **Do not cluster on a t-SNE map, and do not read distances between groups.** t-SNE inflates dense groups and shrinks sparse ones; distances between well-separated groups are arbitrary. Cluster on the original data (or on PCA scores) and use the map only to look.

> [!CAUTION]
> **Fix the random seed and try several settings.** t-SNE and UMAP are stochastic. A structure that appears for one perplexity or one seed only is not a finding. Wattenberg, Viégas and Johnson (2016) show striking examples.

> [!TIP]
> For data with hundreds of columns (text embeddings), first reduce to 30–50 principal components, then run t-SNE or UMAP on the scores. It is faster and removes noise.

## Check your understanding

1. Two standardised columns have correlation 0. What are the eigenvalues of their correlation matrix, and how much variance does PC1 explain?
2. In the wine data PC1 explains 36 % after standardising and 99.8 % without. Which result is misleading, and why?
3. What does a loading of 0.42 for flavanoids on PC1 tell you? What does it not tell you?
4. In a t-SNE map, group A is twice as large as group B and far away from it. Which of these two observations can you report?
5. When would you prefer PCA scores over a UMAP map as input to a later model?

## Further reading

- James, G., Witten, D., Hastie, T., Tibshirani, R. and Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, Chapter 12.2 "Principal Components Analysis". Springer. https://www.statlearning.com/
- Wattenberg, M., Viégas, F. and Johnson, I. (2016). How to use t-SNE effectively. *Distill*. https://doi.org/10.23915/distill.00002
- Coenen, A. and Pearce, A. (2019). *Understanding UMAP*. Google PAIR. https://pair-code.github.io/understanding-umap/
- McInnes, L., Healy, J. and Melville, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426. https://arxiv.org/abs/1802.03426
