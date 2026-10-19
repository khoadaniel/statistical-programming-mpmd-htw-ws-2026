# Session 11 · Unsupervised learning: clustering, dimensionality reduction and anomaly detection

> **Guiding question.** What structure is in the data when there is no target to predict?

## 1 · Clustering

[Open notebook](workbooks/01-lecture-clustering.ipynb)

- Supervised and unsupervised learning: no target, three kinds of evidence instead of a test score.
- Distance, similarity and scaling: Euclidean, Manhattan and cosine side by side; why units decide the result.
- k-means: computed by hand, as expectation–maximisation, where it fails, digits and colour compression.
- Choosing the number of clusters: elbow, silhouette and silhouette plots on the Telco customers.
- From customers to listings: the Berlin listings as six standardised features.
- Hierarchical clustering and dendrograms: linkages compared, cutting the tree, Ward on the Berlin listings.
- Density-based clustering: core, border and noise points, DBSCAN hot spots on the map, HDBSCAN, all methods on toy data.
- Evaluating and interpreting clusters: separation, bootstrap stability, the adjusted Rand index.
- Worked example: what kinds of Airbnb offers exist in Berlin?

## 2 · Dimensionality reduction and anomaly detection

[Open notebook](workbooks/02-lecture-dimensionality-reduction-and-anomaly-detection.ipynb)

- Principal component analysis: geometry on two columns, explained variance and loadings, scaling, the number of components, noise filtering.
- t-SNE and UMAP for visualisation: perplexity, UMAP parameters, trustworthiness, manifold methods, what maps cannot show.
- Worked example: PCA of the amenities of the Berlin listings.
- Model-based anomaly detection with Isolation Forest and local outlier factor, continuing Session 4: detectors on toy data, evaluation with known labels, and implausible Berlin listings compared with a price ratio from a cross-validated model.
- Clusters and components as features inside a pipeline: Telco churn, scaling before PCA in a classifier, location clusters in the price model.

## Team project until the next session

Open points from the interim review.

## Setup

- The Airbnb data: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)). The Telco data are read from GitHub.
- The ISLP lab among the extras needs the `ISLP` package. Open the extras with it from the repository root; some other extras download data sets on first run.

  ```bash
  uv run --with ISLP jupyter lab sessions/11-unsupervised-learning/workbooks/extras
  ```

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
