# Model card: review sentiment classifier

Template after Mitchell et al. (2019), *Model Cards for Model Reporting*. Replace every `<...>` and keep
the card next to the model files: one card per released version. Values marked *example* come from the
course run of version 1.0.0 and must be replaced with your own.

## Model details

| | |
|---|---|
| Name and version | review-sentiment `<1.0.0>` (semantic versioning: major = new training data or method) |
| Date | `<YYYY-MM-DD>` |
| Developers | `<team, contact>` |
| Type | TF-IDF (word 1–2-grams, sublinear tf, min_df 2) + logistic regression (C = 4, balanced class weights) |
| Software | scikit-learn `<version>` (see `metadata.json`), Python `<version>`; locked in `uv.lock` |
| Files | `model.joblib` (SHA-256 in `metadata.json`), `metadata.json` |
| Licence | `<licence of the code and the model>` |
| Previous version | `<none / 1.0.0>`; rollback: redeploy the image tag of that version |

## Intended use

- **Primary use:** sort Amazon health and personal care product reviews into `neg` (1–2 stars), `neu` (3) and `pos` (4–5) for *aggregate* reporting: share of negative reviews per product and month.
- **Primary users:** product and quality teams; the course leaderboard.
- **Out of scope:** decisions about individual customers or reviewers; other product categories or languages without re-evaluation; medical claims in reviews; moderation (deleting reviews).

## Factors

Groups or conditions across which performance may differ, and which the evaluation reports separately:

- review length (short reviews carry little text),
- verified versus unverified purchases,
- product category and store,
- year of the review (drift; see below).

## Metrics

- **Macro-F1**: every class counts equally, so the rare neutral class matters (leaderboard metric).
- Accuracy and per-class F1 reported alongside.
- Uncertainty: bootstrap 95 % interval over the evaluation reviews (200 resamples).
- Decision threshold: the class with the highest predicted probability.

## Evaluation data

- `<e.g. newest 20 % of training reviews (time-based hold-out, up to 2021); 2022 feedback Oct–Dec>`
- Why: the model is used on *future* reviews, so evaluation data must be newer than training data.

## Training data

- Amazon Reviews 2023, Health and Personal Care (McAuley Lab; Hou et al. 2024), reviews up to `<date>`; `<n>` reviews (`train_sample.parquet`, SHA-256 in `metadata.json`) `<+ feedback_2022.csv>`.
- Label: star rating mapped to three classes; the rating is the author's, not an expert judgement.
- Class shares in training: `<neg 19.2 %, neu 7.5 %, pos 73.3 %>` *(example)*.

## Quantitative analyses

| Data | Macro-F1 | F1 neg | F1 neu | F1 pos | Accuracy |
|---|---|---|---|---|---|
| Validation, newest 20 % up to 2021 | `<0.694>` *(example)* | `<0.80>` | `<0.35>` | `<0.93>` | `<0.86>` |
| 2022 feedback | `<0.689>` *(example)* | | | | |
| Subgroup: verified purchases | `< >` | | | | |
| Subgroup: reviews under 50 characters | `< >` | | | | |

## Monitoring and maintenance

| Signal | Statistic | Threshold | Action |
|---|---|---|---|
| Input drift: text length | PSI over reference deciles; KS statistic | PSI > 0.25 | review, consider retraining |
| Prediction drift | predicted class shares; chi-square test | negative share ± 5 points | check with labelled sample |
| Label shift | true class shares when labels arrive | `<...>` | update reporting; retrain |
| Performance | macro-F1 on newly labelled reviews | below validation − 0.03 | retrain and compare |

- Observed so far: `<the negative share rose from 19 % (to 2021) to 26 % (2022); macro-F1 changed from 0.694 to 0.689>` *(example)*.
- Retraining schedule: `<e.g. when a new year of labels is released, or when a trigger fires>`.
- Version history: `<1.0.0: trained on reviews to 2021; 2.0.0: + 2022 feedback, ...>`.

## Ethical considerations

- Reviews are written by people and can contain personal and health information; the service logs only text length, label and probability, never the text.
- User ids are hashed in the data and not used by the model.
- The label is the star rating, not the text: sarcasm, mixed reviews and ratings that contradict the text are misread.

## Caveats and recommendations

- The neutral class is hard (F1 about 0.35): do not report neutral shares without their uncertainty.
- Re-evaluate before using the model on another category, language or platform.
- `<limitations you found in the error analysis>`
