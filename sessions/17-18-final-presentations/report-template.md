# Report template · Final project

Copy this file to `docs/report.md` in your team repository and replace every *italic instruction* with your text. Keep the headings. Write in plain academic English: short sentences, past tense for what you did, numbers with their uncertainty, no marketing words.

**Length.** Part A (the team's work) about 4 pages; Part B (individual contributions) about half a page per member; Part C (AI use) a few lines. A short report is better than a long one with repetition: every sentence should tell the reader something they need.

**How it is used.** Part A is assessed per team, Part B per member (it feeds the criterion *collaboration and presentation*, 20 %, which is graded individually). Every member must be able to explain any part of Part A in the questions after the presentation.

---

**Title:** *a title that states the result, for example "Suggesting HS headings from multilingual product descriptions with a character TF-IDF model"*
**Team:** *team name* · **Members:** *name (GitHub handle), ...*
**Repository:** *URL* · **Submission tag:** `submission` (commit *first 7 characters of its hash*) · **Date:** *YYYY-MM-DD*

## Part A · The team's work

### 1. Problem and use

*Who would use the suggestions, for which decision, and what a wrong suggestion costs (3–5 sentences). State the metric you optimise and why it fits that use; if your service may abstain, say when.*

### 2. Data

*What you received (`decisions.parquet`, number of decisions, headings, languages), the main quality problems you found, and every cleaning decision with its reason. List the columns you do not use as inputs and why.*

| Decision | Reason | Decisions affected |
|---|---|---:|
| *for example: drop descriptions under 3 words* | *templates without content* | *n* |

### 3. Validation design

*How you split the decisions to develop, tune and judge your models, and why this split answers the question "how well will the service do on new requests?". Name the part you touched only once, at the end. Say how you measured uncertainty (for example a bootstrap interval or repeated splits).*

### 4. Models compared

*The approaches you compared, from the baseline upwards, all on the same validation design. One table, one line per approach; keep the weaker ones too, because they are evidence.*

| Model | Accuracy | Macro-F1 | 95 % interval (accuracy) | Notes |
|---|---:|---:|---|---|
| Most frequent heading | | | | baseline |
| *...* | | | | |

*Two or three sentences on what the table shows and which differences are larger than their uncertainty.*

### 5. Final design and reported score

*The design you submit, in a few sentences: features, model, settings, data used for the final fit.*

| Reported score of the final design | Value | 95 % interval | Measured on |
|---|---:|---|---|
| Accuracy | | | *which decisions, how many* |
| Macro-F1 | | | |

*Your expectation for the independent evaluation, in one sentence: equal to this, lower, or higher, and why.*

### 6. Error analysis

*Two or three findings from reading errors and per-group scores (by language, heading frequency, chapter). For each: what you saw, an example, and what you did about it or would do next.*

### 7. Service

*How to start it (`docker compose up`), the interface it answers, the model version, the response time you measured, and how you tested it (unit tests, `evaluate_api.py` on your validation decisions). One sentence on monitoring: what you would watch after release.*

### 8. Limitations and next steps

*Where the service should not be trusted (rare headings, languages, product types), and the two changes you would make with more time.*

### References

*Sources you used (papers, documentation, course notebooks), in one consistent style, for example: Author (Year). Title. Publisher or URL.*

## Part B · Individual contributions

*Each member writes their own entry, in the first person, and is responsible for it. Name concrete work and point to evidence. About 150 words each.*

### *Member name* (*GitHub handle*)

- **What I built or analysed:** *for example "the validation split and the bootstrap intervals (section 3); the error analysis by language (section 6)".*
- **Evidence:** *links to your main pull requests and the files you wrote.*
- **Reviews:** *pull requests of others that you reviewed, with one example of a change your review led to.*
- **My part of the presentation:** *the section you present.*
- **What I would do differently:** *one sentence.*

*(repeat for every member)*

## Part C · Use of AI tools

*Which tools you used, for what (code, debugging, text), and how you checked the output. AI may help to write code and this report; every member has read and reviewed the text and can explain all code. Attach the HTW declaration on the use of AI tools.*

| Tool | Used for | How we checked it |
|---|---|---|
| *for example: GitHub Copilot* | *test functions in `tests/`* | *ran the tests, read every assertion* |

*We confirm that each of us has reviewed this report and the submitted code.* *Names of all members.*
