# Session 9 · Advanced feature engineering and leakage

> **Guiding question.** Which additional inputs make a model better, and how do we build them without leakage?

## 1 · Feature construction without leakage

[Open notebook](workbooks/01-lecture-feature-construction-without-leakage.ipynb)

- Features from dates and times: date parts, elapsed time, seasonality, cyclical and periodic spline encodings, dates known at prediction time.
- Interactions between features: product columns, room type × guests, all pairwise products, splines and kernels.
- High-cardinality categories: grouping rare levels and choosing the threshold, unseen levels, count encoding, skrub's `TableVectorizer`.
- Target encoding and its leakage risk: smoothing, the naive leak, cross-fitting with `TargetEncoder`, its parameters, binary and multiclass targets, comparison of encoders.
- Aggregation features from related tables: point-in-time aggregates, an out-of-time check, lags and rolling windows, as-of joins, `AggJoiner`.
- Target leakage (an input that contains information about the target that is not known at prediction time): derived and snapshot columns, detection, leakage through preprocessing and groups, a checklist.
- Custom transformers in pipelines: `FunctionTransformer`, a transformer class and its contract, tuning preprocessing.
- Worked example: which constructed features help the Berlin price model?

## Team project until the next teaching day

How the team will deal with rare headings; inputs documented with the moment at which each is known.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
