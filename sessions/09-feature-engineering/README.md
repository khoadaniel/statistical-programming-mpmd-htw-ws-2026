# Session 9 · Advanced feature engineering and leakage

> **Guiding question.** Which additional inputs make a model better, and how do we build them without leakage?

## 1 · Feature construction without leakage

[Open notebook](workbooks/01-lecture-feature-construction-without-leakage.ipynb)

- Features from dates and times: date parts, elapsed time, cyclical and periodic spline encodings.
- Interactions between features: product columns, all pairwise products, splines and kernels.
- High-cardinality categories: grouping rare levels, unseen levels, count encoding, skrub's `TableVectorizer`.
- Target encoding: smoothing, the naive leak, cross-fitting with `TargetEncoder`, comparison of encoders.
- Aggregation features from related tables: point-in-time aggregates, lags, as-of joins, `AggJoiner`.
- Target leakage: derived and snapshot columns, leakage through preprocessing and groups, detection.
- Custom transformers in pipelines: `FunctionTransformer`, a transformer class, tuning preprocessing.
- Worked example: which feature groups help the Berlin price model.

## Team project until the next teaching day

How the team will deal with rare headings; inputs documented with the moment at which each is known.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
