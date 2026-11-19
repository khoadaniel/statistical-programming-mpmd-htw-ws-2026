# Evaluate your API

`evaluate_api.py` sends decisions to your running service, one by one, and scores the answers. The lecturer runs the same script on the independent evaluation set. Run it on your own validation decisions first: it shows your score and **which `id` went wrong**.

It has its own `pyproject.toml`, so it runs in its own environment and needs nothing from your project.

## Run it

1. Save your validation decisions (rows your model was not trained on) as a CSV file **outside Git**. It needs three columns:

   | Column | Content |
   |---|---|
   | `id` | unique name of the decision, for example its `bti_reference` |
   | `description` | the text sent to the service: its only input |
   | `subheading` | the right answer: six digits, as text |

   ```python
   check = validation.rename(columns={"bti_reference": "id"})
   check[["id", "description", "subheading"]].to_csv("data/my_validation.csv", index=False)
   ```

   Do not open and save the file in Excel: it turns `090422` into `90422`. A real example: [`example_validation.csv`](example_validation.csv).

2. Start your service (port 8000), then, in another terminal:

   ```bash
   cd sessions/17-18-final-presentations/evaluate-api
   uv run evaluate_api.py --url http://localhost:8000 --data path/to/my_validation.csv
   ```

   Add `--limit 200` for a quick check, `--output name.csv` to name the results file, `--timeout 10` to change the seconds waited for one answer.

The practice sessions (13 to 16) work on the four-digit heading. For them, add `--label heading`: the file then has a column `heading` instead of `subheading`, and the service must answer `{"heading": "6403", ...}`.

## What you get

On the screen (for the example input and an invented service):

```text
decisions          8
failed requests    1
accuracy           0.6250
macro-F1           0.4545
heading accuracy   0.7500
latency P95        0.001 s

what happened to the requests:
ok          7
http 500    1

2 wrong answers; 50% of them have the right heading
most frequent confusions (true -> answered):
true    predicted
392410  392490       1
620342  620462       1
```

| Line | Meaning |
|---|---|
| `failed requests` | errors, timeouts or answers without a six-digit `subheading`; each counts as wrong. The next block says why (`timeout`, `http 422`, `no subheading in the answer`, ...) |
| `accuracy` | share of right subheadings |
| `macro-F1` | F1 averaged over subheadings, so rare subheadings count equally |
| `heading accuracy` | share of answers whose first four digits are right: how much of the error lies below the heading |
| `latency P95` | 95 % of the answers came faster than this; it must stay under two seconds |
| confusions | the subheadings your service mixes up most often |

And in `evaluation_results.csv`, one row per `id` ([`example_results.csv`](example_results.csv)): the true and the answered subheading, `status`, `correct`, `heading_correct`, seconds. Filter `correct == False` to read the descriptions your model gets wrong. Join the file to your own table on `id` to look at language, length or anything else.

## What your API must answer

`POST /predict` with `{"description": "..."}` must answer `{"subheading": "640359", ...}`, and `GET /health` must answer 200. The full interface is in [Appendix A of the brief](../final-project.md#a--the-api-interface).

## An example model folder

[`example_model/`](example_model/) shows what a trained model looks like on disk and where it goes. It is the only model in Git, because it is an example: **your own model is never committed.** It was trained on the 8 invented decisions above, so it knows nothing about customs.

| File | What it is |
|---|---|
| `model.joblib` | the fitted scikit-learn pipeline (TF-IDF vocabulary and a linear model) |
| `metadata.json` | version, library versions, the classes the model knows and the checksum of `model.joblib`; a convention of the Session 16 workspace, optional |

In your repository, the training script writes the model files to `models/`, the `Dockerfile` copies that folder into the image, and the service loads it at start-up from the folder named by `MODEL_DIR`. You commit the model files with Git LFS: see [Appendix C of the project brief](../final-project.md#c--starter-files-model-upload-and-docker) and the ready-made [`repository-starter/.gitattributes`](repository-starter/.gitattributes). The evaluation never reads these files: it only calls your running service.
