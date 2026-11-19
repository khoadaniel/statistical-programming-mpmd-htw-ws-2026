# Session 11 · Unsupervised learning: clustering, dimensionality reduction and anomaly detection

> **Guiding question.** What structure is in the data when there is no target to predict?

## 1 · Clustering

[Open notebook](workbooks/01-lecture-clustering.ipynb)

- Supervised and unsupervised learning: no target, no single right clustering; the same customers segmented two ways, and why the people who use the groups decide.
- Distance, similarity and scaling: Euclidean, Manhattan and cosine side by side; why units decide the result.
- k-means: computed by hand, as expectation–maximisation, where it fails, digits and colour compression.
- Choosing the number of clusters: elbow, silhouette and silhouette plots on the Telco customers; the iris flowers, two groups or three.
- Hierarchical clustering and dendrograms: linkages compared, Ward on the wine data, cutting the tree.
- Density-based clustering: core, border and noise points, HDBSCAN, all methods on toy data.
- The clustering families on the wine data and on two half-moons, in one loop.
- Evaluating and interpreting clusters: separation, bootstrap stability, the adjusted Rand index.
- Worked example: what kinds of Airbnb offers exist in Berlin? From the data to five named offer types, with Ward, k-means, hot spots on the map and DBSCAN.

**End-to-end: clustering.** Wines grouped by their lab analysis in the steps from the problem to the report: k-means for k = 2 to 8 with the elbow, the silhouette score and the clusters on a 2D map, agglomerative clustering compared, what the three groups mean, stability, 95% intervals of the silhouette score and of the differences, and HDBSCAN.

[Open notebook](workbooks/end-to-end-clustering.ipynb)

## 2 · Dimensionality reduction and anomaly detection

[Open notebook](workbooks/02-lecture-dimensionality-reduction-and-anomaly-detection.ipynb)

- Principal component analysis: geometry on two columns, explained variance and loadings, scaling, the number of components, noise filtering.
- t-SNE and UMAP for visualisation: perplexity, UMAP parameters, trustworthiness, manifold methods, what maps cannot show.
- Worked example: PCA of the amenities of the Berlin listings.
- Model-based anomaly detection with Isolation Forest and local outlier factor, continuing Session 4: detectors on toy data, evaluation with known labels, and implausible Berlin listings compared with a price ratio from a cross-validated model.
- Clusters and components as features inside a pipeline: Telco churn, scaling before PCA in a classifier, location clusters in the price model.

**End-to-end: dimensionality reduction.** Handwritten digits in fewer numbers: PCA with the number of components chosen by a 90% rule and checked with a digit reader, what each component means, and maps from PCA, t-SNE and UMAP, with 95% intervals.

[Open notebook](workbooks/end-to-end-dimensionality-reduction.ipynb)

**End-to-end: anomaly detection.** Unusual Berlin listings with planted typing errors: Isolation Forest, tuned, against the local outlier factor, with precision, recall and the interval of the difference.

[Open notebook](workbooks/end-to-end-anomaly-detection.ipynb)

## Team project until the next session

Open points from the interim review.

## Setup

- The Airbnb data: get the data with `git lfs pull` (see [case-study/README.md](../../case-study/README.md)). The Telco data are read from GitHub.
- The ISLP lab among the extras needs the `ISLP` package. Open the extras with it from the repository root; some other extras download data sets on first run.

  ```bash
  uv run --with ISLP jupyter lab sessions/11-unsupervised-learning/workbooks/extras
  ```

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
