# Model card: HS heading suggestions for customs decisions

Template after Mitchell et al. (2019), *Model Cards for Model Reporting*. Replace every `<...>` and keep
the card next to the model files: one card per released version. Values marked *example* come from the
course run of version 1.0.0 on the 50,000-decision sample and must be replaced with your own.

## Model details

| | |
|---|---|
| Name and version | tariff-heading `<1.0.0>` (semantic versioning: major = new training data or method) |
| Date | `<YYYY-MM-DD>` |
| Developers | `<team, contact>` |
| Type | word TF-IDF (unigrams, sublinear tf, min_df 2) + linear SVM (`SGDClassifier`, hinge loss, alpha 1e-5); coefficients with \|w\| < 0.05 pruned |
| Output | the three highest-scoring four-digit HS headings with decision scores (margins, not probabilities) and English heading texts |
| Software | scikit-learn `<version>` (see `metadata.json`), Python `<version>`; locked in `uv.lock` |
| Files | `model.joblib` (SHA-256 in `metadata.json`), `metadata.json` |
| Licence | `<licence of the code and the model>` |
| Previous version | `<none / 1.0.0>`; rollback: redeploy the image tag of that version |

## Intended use

- **Primary use:** suggest candidate HS headings for a description of goods, as a starting point for a customs officer or a trader who searches the tariff; course leaderboard.
- **Primary users:** classification experts who check every suggestion; students of the course.
- **Out of scope:** issuing or refusing a Binding Tariff Information decision; classification at subheading (6 digits) or CN (8 digits) level; duty calculation; descriptions outside the EU languages of the training data; use without a person in the loop.

## Factors

Groups or conditions across which performance may differ, and which the evaluation reports separately:

- language of the description (German 57 %, French 16 %; some languages have a few hundred training decisions only),
- issuing country (GB issued decisions only until 2020),
- description length (short descriptions score lower),
- descriptions that quote their own heading number (about a quarter; nearly always right) versus those that do not,
- frequency of the heading in training (long tail: about 900 headings, many with fewer than ten examples),
- start year (drift; HS 2022 revision).

## Metrics

- **Accuracy** (leaderboard metric) and **macro-F1** (every heading counts equally; long tail).
- Top-3 accuracy (is the right heading among the three suggestions?) and chapter accuracy.
- Uncertainty: bootstrap 95 % interval over the evaluation decisions.
- Decision rule: the highest score; suggestions with a top score below `<0>` are flagged for review.

## Evaluation data

- `<e.g. newest 20 % of the training decisions (time-based hold-out, 2022-2023); 2024 feedback labels>`
- Why: the model is used on *future* requests, so evaluation data must be newer than training data.

## Training data

- European Commission, EBTI database (BTI decisions with start of validity `<2017-2023>`), `<n>` decisions (`train_sample.parquet`, SHA-256 in `metadata.json`) `<+ feedback_2024.csv>`.
- Input: the description of goods only. Not used: justification, keywords, CN code, status, end date (decided with or after the classification).
- Label: the four-digit heading of the decision, taken by a customs authority; some decisions were later revoked.
- Chapter shares in training: `<85: 14.7 %, 84: 7.5 %, 39: 7.0 %, ...>` *(example)*.

## Quantitative analyses

| Data | Accuracy | Macro-F1 | Top-3 accuracy | Chapter accuracy |
|---|---|---|---|---|
| Validation, newest 20 % (2022-2023) | `<0.779>` *(example)* | `<0.519>` | `<0.848>` | `<0.844>` |
| 2024 feedback | `<0.804>` *(example)* | | | |
| Subgroup: descriptions in German | `< >` | | | |
| Subgroup: descriptions under 200 characters | `< >` | | | |
| Subgroup: no quoted heading number | `< >` | | | |

## Monitoring and maintenance

| Signal | Statistic | Threshold | Action |
|---|---|---|---|
| Input drift: description length | PSI over reference deciles; KS statistic | PSI > 0.25 | review, consider retraining |
| Input drift: language, issuing country | PSI over category shares | PSI > 0.25 | check new countries or languages |
| Prediction drift | PSI of predicted chapter shares | PSI > 0.1 | check with labelled sample |
| Label shift | PSI of true chapter shares when labels arrive; share of unseen headings | `<...>` | retrain |
| Performance | accuracy on newly labelled decisions | below validation − 0.03 | retrain and compare |

- Observed so far: `<country PSI 0.24 (no GB decisions since 2021); chapter label shift PSI 0.06; 0.4 % unseen headings; accuracy 0.80 on 2024 against 0.78 in validation>` *(example)*.
- Retraining schedule: `<e.g. once a year when the new year of decisions is labelled, and after each HS revision>`.
- Version history: `<1.0.0: decisions 2017-2023; 2.0.0: + 2024 feedback, ...>`.

## Ethical and legal considerations

- Published BTI decisions are public, but a pending request contains confidential business information; the service logs only length, language, top heading and score, never the description.
- The holders of decisions are not part of the data.
- A wrong heading can change the duty a trader pays. The model's output is a suggestion; the classification is the responsibility of the customs authority (governance questions: module 3.3).

## Caveats and recommendations

- Short descriptions often score below zero and miss the right heading; show the three suggestions with their scores, not one answer.
- Headings created by the HS 2022 revision (for example 8524, flat panel display modules) have few training examples.
- Re-evaluate after every nomenclature revision and before using the model on another tariff or language.
- `<limitations you found in the error analysis>`
