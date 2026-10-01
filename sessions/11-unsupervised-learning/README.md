# Session 11 · Unsupervised learning: clustering, dimensionality reduction and anomaly detection

> [!NOTE]
> **Guiding question.** What structure is in the data when there is no target to predict?

**Learning outcomes.** Students are able to

- distinguish unsupervised from supervised learning and explain why its evaluation is harder
- apply and evaluate k-means, hierarchical and density-based clustering
- reduce dimensionality with PCA, visualise high-dimensional data and detect anomalies

## Session plan

**0:00–0:45 · Clustering and k-means** ([theory/01](theory/01-clustering-and-k-means.md))

- [Supervised and unsupervised learning](theory/01-clustering-and-k-means.md#supervised-and-unsupervised-learning)
- [Distance, similarity and scaling](theory/01-clustering-and-k-means.md#distance-similarity-and-scaling)
- [k-means computed by hand on a small example](theory/01-clustering-and-k-means.md#k-means-computed-by-hand)
- [Choosing the number of clusters (elbow method, silhouette)](theory/01-clustering-and-k-means.md#choosing-the-number-of-clusters-elbow-method-and-silhouette)
- *Practice:* segment the churn customers with k-means and describe the segments → [workbooks/04-case-study-telco-segments.ipynb](workbooks/04-case-study-telco-segments.ipynb)

**1:00–1:45 · Hierarchical clustering, DBSCAN and evaluation** ([theory/02](theory/02-hierarchical-dbscan-and-evaluation.md))

- [Hierarchical clustering and dendrograms](theory/02-hierarchical-dbscan-and-evaluation.md#hierarchical-clustering-and-dendrograms)
- [Density-based clustering (DBSCAN)](theory/02-hierarchical-dbscan-and-evaluation.md#density-based-clustering-dbscan)
- [Evaluating and interpreting clusters (silhouette, stability, cluster profiles)](theory/02-hierarchical-dbscan-and-evaluation.md#evaluating-and-interpreting-clusters)
- *Practice:* compare k-means, hierarchical clustering and DBSCAN on product-level features → Part A of [workbooks/24-case-study-product-clusters.ipynb](workbooks/24-case-study-product-clusters.ipynb)

**2:00–2:45 · Dimensionality reduction, anomalies and unsupervised features** ([theory/03](theory/03-dimensionality-reduction.md), [theory/04](theory/04-anomaly-detection-and-unsupervised-features.md))

- [Principal component analysis (explained variance, loadings)](theory/03-dimensionality-reduction.md#principal-component-analysis-explained-variance-and-loadings)
- [t-SNE and UMAP for visualisation](theory/03-dimensionality-reduction.md#t-sne-and-umap-for-visualisation)
- [Model-based anomaly detection with Isolation Forest and LOF, continuing the outliers of Session 4](theory/04-anomaly-detection-and-unsupervised-features.md#model-based-anomaly-detection-isolation-forest-and-local-outlier-factor)
- [Clusters and components as features](theory/04-anomaly-detection-and-unsupervised-features.md#clusters-and-components-as-features)
- *Practice:* case study: cluster products by their review statistics, visualise them with PCA and test whether cluster membership improves the tree-based model of Session 10 → Part B of [workbooks/24-case-study-product-clusters.ipynb](workbooks/24-case-study-product-clusters.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-clustering-and-k-means.md](theory/01-clustering-and-k-means.md) | Unsupervised learning, distances and scaling, k-means by hand, elbow and silhouette | 1 | core |
| [theory/02-hierarchical-dbscan-and-evaluation.md](theory/02-hierarchical-dbscan-and-evaluation.md) | Linkage and dendrograms, DBSCAN, silhouette, stability, profiles | 2 | core |
| [theory/03-dimensionality-reduction.md](theory/03-dimensionality-reduction.md) | PCA, scree plot, loadings and biplot, t-SNE, UMAP | 3 | core |
| [theory/04-anomaly-detection-and-unsupervised-features.md](theory/04-anomaly-detection-and-unsupervised-features.md) | Isolation Forest, LOF, clusters and components as features | 3 | core |
| [workbooks/01-pdsh-kmeans.ipynb](workbooks/01-pdsh-kmeans.ipynb) | k-means in depth (PDSH 5.11): EM view, limits, digits, colour compression | 1 | core |
| [workbooks/02-kmeans-assumptions.ipynb](workbooks/02-kmeans-assumptions.ipynb) | Four situations in which k-means fails | 1 | optional |
| [workbooks/03-kmeans-silhouette-analysis.ipynb](workbooks/03-kmeans-silhouette-analysis.ipynb) | Silhouette plots for k = 2 to 6 | 1 | core |
| [workbooks/04-case-study-telco-segments.ipynb](workbooks/04-case-study-telco-segments.ipynb) | **Practice 1:** Telco customer segments with k-means (own) | 1 | core |
| [workbooks/05-agglomerative-dendrogram.ipynb](workbooks/05-agglomerative-dendrogram.ipynb) | Dendrogram of an agglomerative clustering | 2 | core |
| [workbooks/06-linkage-comparison.ipynb](workbooks/06-linkage-comparison.ipynb) | Single, average, complete and Ward linkage compared | 2 | optional |
| [workbooks/07-dbscan.ipynb](workbooks/07-dbscan.ipynb) | DBSCAN: core points, noise, metrics | 2 | core |
| [workbooks/08-hdbscan.ipynb](workbooks/08-hdbscan.ipynb) | HDBSCAN for clusters of varying density | 2 | optional |
| [workbooks/09-clustering-algorithms-compared.ipynb](workbooks/09-clustering-algorithms-compared.ipynb) | Eleven algorithms on six toy datasets | 2 | core |
| [workbooks/10-pca-intuitions.ipynb](workbooks/10-pca-intuitions.ipynb) | Geometric intuition of PCA (INRIA MOOC) | 3 | core |
| [workbooks/11-pca-components.ipynb](workbooks/11-pca-components.ipynb) | Explained variance and the number of components (INRIA MOOC) | 3 | core |
| [workbooks/12-pca-as-preprocessing.ipynb](workbooks/12-pca-as-preprocessing.ipynb) | Scaling before PCA, PCA in a pipeline (INRIA MOOC) | 3 | optional |
| [workbooks/13-pca-in-depth.ipynb](workbooks/13-pca-in-depth.ipynb) | PCA in depth, noise filtering, eigenfaces (PDSH 5.09) | 3 | optional |
| [workbooks/14-pca-iris.ipynb](workbooks/14-pca-iris.ipynb) | PCA of the iris data in 3-D | 3 | optional |
| [workbooks/15-manifold-learning.ipynb](workbooks/15-manifold-learning.ipynb) | Manifold learning: MDS, LLE, Isomap (PDSH 5.10) | 3 | optional |
| [workbooks/16-manifold-methods-compared.ipynb](workbooks/16-manifold-methods-compared.ipynb) | Manifold methods including t-SNE on the S-curve | 3 | optional |
| [workbooks/17-tsne-perplexity.ipynb](workbooks/17-tsne-perplexity.ipynb) | How the perplexity changes a t-SNE map | 3 | core |
| [workbooks/18-umap-parameters.ipynb](workbooks/18-umap-parameters.ipynb) | UMAP parameters n_neighbors, min_dist, metric | 3 | optional |
| [workbooks/19-islp-unsupervised-lab.ipynb](workbooks/19-islp-unsupervised-lab.ipynb) | ISLP Chapter 12 lab: PCA and clustering (review of the session) | 1–3 | optional |
| [workbooks/20-anomaly-detection-compared.ipynb](workbooks/20-anomaly-detection-compared.ipynb) | Four anomaly detectors on toy data | 3 | core |
| [workbooks/21-isolation-forest.ipynb](workbooks/21-isolation-forest.ipynb) | Isolation Forest decision boundary and path lengths | 3 | core |
| [workbooks/22-local-outlier-factor.ipynb](workbooks/22-local-outlier-factor.ipynb) | Local outlier factor scores | 3 | core |
| [workbooks/23-outlier-detection-benchmark.ipynb](workbooks/23-outlier-detection-benchmark.ipynb) | Isolation Forest and LOF on real benchmark data | 3 | optional |
| [workbooks/24-case-study-product-clusters.ipynb](workbooks/24-case-study-product-clusters.ipynb) | **Practice 2 and 3:** product clusters, PCA, cluster feature in a model (own) | 2, 3 | core |

Sources and licences: [source.md](source.md). Workbooks 01, 13 and 15 have a non-commercial, no-derivatives text licence (CC-BY-NC-ND): use them unchanged.

## Before and after the session

**Preparation.** Read the first section of [theory/01](theory/01-clustering-and-k-means.md) and run the first cells of [workbook 04](workbooks/04-case-study-telco-segments.ipynb) to check that the Telco data load. Make sure `case-study/data/train.parquet` exists (see [case-study/README.md](../../case-study/README.md)).

**Team project until the next session.** Segmentation or anomaly detection where it supports the project.

**Further reading (optional).**

- James et al. (2023). *An Introduction to Statistical Learning with Applications in Python*, Chapter 12. https://www.statlearning.com/
- scikit-learn User Guide: [Clustering](https://scikit-learn.org/stable/modules/clustering.html), [Decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [Outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- Wattenberg, Viégas and Johnson (2016). [How to use t-SNE effectively](https://distill.pub/2016/misread-tsne/). *Distill*.
- Coenen and Pearce (2019). [Understanding UMAP](https://pair-code.github.io/understanding-umap/). Google PAIR.

## Setup

Everything except workbook 19 runs in the course environment (`uv sync` in the repository root, then `uv run jupyter lab`). It uses scikit-learn, scipy, matplotlib, seaborn and umap-learn.

- Workbook 19 (ISLP lab) needs the `ISLP` package: `uv run --with ISLP jupyter lab`.
- The Telco data are read from GitHub, and some scikit-learn examples download datasets (`fetch_openml`, `fetch_lfw_people`, `fetch_kddcup99`): an internet connection is needed the first time.
- Workbook 24 reads `case-study/data/train.parquet` and finds the repository root by itself.
- To regenerate the figures: `uv run python sessions/11-unsupervised-learning/theory/figures/make_figures.py` from the repository root.
