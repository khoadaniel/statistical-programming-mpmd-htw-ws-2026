# Session 11 · Unsupervised learning: clustering, dimensionality reduction and anomaly detection

> **Guiding question.** What structure is in the data when there is no target to predict?

## 1 · Clustering

[Open notebook](workbooks/01-lecture-clustering.ipynb)

- Supervised and unsupervised learning: no target, three kinds of evidence instead of a test score.
- Distance, similarity and scaling: Euclidean, Manhattan and cosine; why units decide the result.
- k-means computed by hand, k-means as expectation–maximisation, where it fails, digits and colour compression.
- Choosing the number of clusters: elbow, silhouette and silhouette plots on the Telco customers.
- Hierarchical clustering and dendrograms: linkages, cutting the tree, Ward on the Berlin listings.
- Density-based clustering: DBSCAN hot spots on the map, HDBSCAN, all methods on toy data.
- Evaluating and interpreting clusters: silhouette, bootstrap stability, adjusted Rand index, profiles.
- Worked example: what kinds of Airbnb offers exist in Berlin?

## 2 · Dimensionality reduction and anomaly detection

[Open notebook](workbooks/02-lecture-dimensionality-reduction-and-anomaly-detection.ipynb)

- Principal component analysis: explained variance, loadings, geometry, scaling and the number of components.
- t-SNE and UMAP for visualisation: perplexity, manifold methods, UMAP parameters, what maps cannot show.
- Worked example: PCA of the amenities of the Berlin listings.
- Model-based anomaly detection with Isolation Forest and local outlier factor, continuing Session 4.
- Implausible listings: detectors compared with a price ratio from a cross-validated model.
- Clusters and components as features inside a pipeline; location clusters in the price model.

## Team project until the next session

Open points from the interim review.

## Setup

- Prepare the Airbnb data once: `uv run python case-study/prepare_airbnb.py`. The Telco data are read from GitHub.
- The ISLP lab among the extras needs `uv run --with ISLP jupyter lab`; some other extras download data sets on first run.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
