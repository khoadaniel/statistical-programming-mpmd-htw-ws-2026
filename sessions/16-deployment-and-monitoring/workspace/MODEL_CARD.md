# Model card: HS heading suggestions for customs decisions

Template after Mitchell et al. (2019), [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993). A model card is a short document that is published with a trained model and states what it is for, how well it works, for whom it was evaluated and where it should not be trusted.

**Read two real ones before you fill this in:**

- [GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md) (OpenAI, 2019): short and close to the structure of the paper (details, intended uses, out-of-scope uses, evaluation, limitations).
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), the embedding model of Session 14, and [BERT base](https://huggingface.co/google-bert/bert-base-uncased): model cards in the form most people see them, as the README of a model on the Hugging Face Hub. Look for "intended uses", "limitations" and "bias".
- How to write one: Hugging Face's [guide to model cards](https://huggingface.co/docs/hub/model-cards) and its [annotated model card](https://huggingface.co/docs/hub/model-card-annotated), which explains every section; more examples in Google's [model card gallery](https://modelcards.withgoogle.com/).

Replace every `<...>` and keep
the card next to the model files: one card per released version. Values marked *example* come from the
course run of version 1.0.0 (trained with `--before 2026-01-01` on `decisions.parquet`, workspace exercise 4) and must be replaced with your own.

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

- **Primary use:** suggest candidate HS headings for a description of goods, as a starting point for a customs officer or a trader who searches the tariff; the course's final project (independent evaluation).
- **Primary users:** classification experts who check every suggestion; students of the course.
- **Out of scope:** issuing or refusing a Binding Tariff Information decision; classification at subheading (6 digits) or CN (8 digits) level; duty calculation; descriptions outside the EU languages of the training data; use without a person in the loop.

## Factors

Groups of requests for which the model may work better or worse than its average. Name them here, then report accuracy for each group in *Quantitative analyses*, so that a reader sees where the model is weak.

| Factor | Groups to compare | Why the score may differ |
|---|---|---|
| Language | German, French, all others | most descriptions are German; some languages have only a few hundred examples |
| Description length | under 200 characters, longer | a short description carries less information |
| Heading frequency | headings with fewer than ten training examples, the rest | the model has seen few examples of the rare headings (about 400 of 1,000) |
| Start year | 2026, earlier | wording and products change over time (drift) |

## Metrics

- **Accuracy** (scored in the final evaluation) and **macro-F1** (every heading counts equally, which matters because of the long tail: a few headings are very frequent and hundreds are rare).
- Top-3 accuracy (is the right heading among the three suggestions?) and chapter accuracy.
- Uncertainty: bootstrap 95 % interval over the evaluation decisions.
- Decision rule: the highest score; suggestions with a top score below `<0>` are flagged for review.

## Evaluation data

- `<e.g. newest 20 % of the training decisions (time-based hold-out, from 2025-07); your validation decisions (random, stratified by heading and language); the decisions of 2026 (for v1, which never saw them)>`
- Why: the model is used on *future* requests, so evaluation data must be newer than training data.

## Training data

- European Commission, EBTI database (BTI decisions valid on 2026-10-01, start of validity `<2023-10 to 2025-12>`), `<n>` decisions (`decisions.parquet`, SHA-256 in `metadata.json`)
- Input: the description of goods only. Not used: justification, keywords, CN code, end date (decided with or after the classification).
- Label: the four-digit heading of the decision, taken by a customs authority; the data contain only decisions that were still valid on 2026-10-01.
- Chapter shares in training: `<85: 11.7 %, 84: 8.7 %, 39: 7.1 %, ...>` *(example)*.

## Quantitative analyses

| Data | Accuracy | Macro-F1 | Top-3 accuracy | Chapter accuracy |
|---|---|---|---|---|
| Validation, newest 20 % (from 2025-07) | `<0.803>` *(example)* | `<0.551>` | `<0.861>` | `<0.859>` |
| Decisions of 2026, for v1 trained before 2026 | `<0.780>` *(example)* | | | |
| Reported score: your validation design, with its 95 % interval (also in the README) | `< >` | `< >` | | |
| Factor: descriptions in German / French / other languages | `< >` | | | |
| Factor: descriptions under 200 characters / longer | `< >` | | | |
| Factor: headings with fewer than ten training examples / the rest | `< >` | | | |
| Factor: start year 2026 / earlier | `< >` | | | |

## Monitoring and maintenance

| Signal | Statistic | Threshold | Action |
|---|---|---|---|
| Input drift: description length | PSI over reference deciles; KS statistic | PSI > 0.25 | review, consider retraining |
| Input drift: language, issuing country | PSI over category shares | PSI > 0.25 | check new countries or languages |
| Prediction drift | PSI of predicted chapter shares | PSI > 0.1 | check with labelled sample |
| Label shift | PSI of true chapter shares when labels arrive; share of unseen headings | `<...>` | retrain |
| Performance | accuracy on newly labelled decisions | below validation − 0.03 | retrain and compare |

- Observed so far: `<country PSI 0.035 (France's share rises); chapter label shift PSI 0.046; 0.2 % unseen headings; accuracy 0.780 on 2026 against 0.803 in validation>` *(example)*.
- Retraining schedule: `<e.g. once a year when the new year of decisions is labelled, and after each HS revision>`.
- Version history: `<1.0.0: decisions before 2026; 2.0.0: all training decisions including 2026, ...>`.

## Ethical and legal considerations

- Published BTI decisions are public, but a pending request contains confidential business information; the service logs only length, language, top heading and score, never the description.
- The holders of decisions are not part of the data.
- A wrong heading can change the duty a trader pays. The model's output is a suggestion; the classification is the responsibility of the customs authority (governance questions: module 3.3).

## Caveats and recommendations

- Short descriptions often score below zero and miss the right heading; show the three suggestions with their scores, not one answer.
- Many headings have fewer than ten training examples (long tail): macro-F1 `<0.55>` lies far below accuracy `<0.80>` *(example)*.
- Re-evaluate after every nomenclature revision and before using the model on another tariff or language.
- `<limitations you found in the error analysis>`
