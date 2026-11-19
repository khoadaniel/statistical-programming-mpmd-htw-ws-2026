# Individual paper · Final project

Every member writes their own paper, alone. Copy this template and replace every *italic instruction* with your text. Keep the headings. Write plainly: short sentences, past tense for what you did, numbers with their uncertainty.

**Length:** at most 10 pages, not counting the appendix (use of AI tools and references).

**Format:** submit the paper as a PDF.

The paper tells the project as a whole, from the problem to the results, as you understand it, and states your own contribution. The criteria are in the [final project brief](final-project.md#presentation-and-assessment).

---

**Title:** *a title that states the result, for example "Suggesting customs subheadings from multilingual descriptions with a grouped validation design"*
**Author:** *your name (GitHub account)* · **Team:** *team name* · **Other members:** *names*
**Repository:** *URL* · **Submission tag:** `submission` (commit *first 7 characters of its hash*) · **Date:** *YYYY-MM-DD*

## 1. Problem statement

*Frame the problem before any model: who would use the suggestions and for which decision; what a wrong suggestion costs; and how this leads to the metric you optimise, the baseline you compare against and the validation design.*

## 2. Approach

*How the team solved the problem, from the baseline to the final design. Include:*

- *the **data**: the quality problems found and how they were handled;*
- *the **validation design**: how the decisions were split for training, tuning and the final measurement, and why;*
- *the **models and features** tried, and why the final design was chosen;*
- *the **theory** of each method: what it does, its assumptions, where it fails, with a source;*
- *the main **decisions** in a table:*

| Decision | Options considered | Chosen | Reason |
|---|---|---|---|
| *for example: how to split the decisions* | *random, stratified by subheading, grouped by near-identical texts* | *grouped* | *near-duplicates in training and validation would make the score too optimistic* |

## 3. Results

*The reported score (the accuracy of the six-digit subheading with its interval, plus the metrics that support your story) against the baseline; two or three findings from the error analysis (by language, subheading frequency, chapter), each with an example; and your expectation for the independent evaluation, with the reason.*

## 4. The service

*How the service is built and started with Docker, how it was tested, and its limits (see the model card).*

## 5. My contribution

*Your part of the project, with the evidence: the pull requests you authored and the ones you reviewed (links).*

## Appendix

### A. Use of AI tools

*Which tools you used, for what, and how you checked the output. Attach the HTW declaration on the use of AI tools.*

| Tool | Used for | How I checked it |
|---|---|---|
| *for example: an AI coding agent* | *the tests in `tests/`* | *ran them, read every assertion* |

### B. References

*Sources you used (papers, documentation, course notebooks), in one consistent style, for example: Author (Year). Title. Publisher or URL.*

*I confirm that I wrote this paper myself and that I can explain every part of it and of my code.* *Your name.*
