# Individual paper · Final project

Every member writes their own paper, alone. Copy this template and replace every *italic instruction* with your text. Keep the headings. Write plainly: short sentences, past tense for what you did, numbers with their uncertainty.

The paper has two parts. **Part 1** tells the project as a whole, from the problem to the results, as you understand it. **Part 2** says what you did yourself. The criteria are in the [final project brief](final-project.md#presentation-and-assessment).

---

**Title:** *a title that states the result, for example "Suggesting customs headings from multilingual descriptions with a grouped validation design"*
**Author:** *your name (GitHub account)* · **Team:** *team name* · **Other members:** *names*
**Repository:** *URL* · **Submission tag:** `submission` (commit *first 7 characters of its hash*) · **Date:** *YYYY-MM-DD*

## Part 1 · The project

### 1. Problem

*Who would use the suggestions and for which decision; what a wrong suggestion costs; and how this leads to the metric, the baseline and the validation design.*

### 2. Approach

*How the team solved the problem, from the baseline to the final design. Include:*

- *the **data**: the quality problems found and how they were handled;*
- *the **validation design**: how the decisions were split for training, tuning and the final measurement, and why;*
- *the **models and features** tried, and why the final design was chosen;*
- *the **theory** of each method: what it does, its assumptions, where it fails, with a source;*
- *the main **decisions** in a table:*

| Decision | Options considered | Chosen | Reason |
|---|---|---|---|
| *for example: how to split the decisions* | *random, stratified by heading, grouped by near-identical texts* | *grouped* | *near-duplicates in training and validation would make the score too optimistic* |

### 3. Results

*The reported score (accuracy and macro-F1 with their intervals) against the baseline; two or three findings from the error analysis (by language, heading frequency, chapter), each with an example; and your expectation for the independent evaluation, with the reason.*

### 4. The service

*How the service is built and started with Docker, how it was tested, and its limits (see the model card).*

## Part 2 · My contribution

### 5. What I did

*Your part of the project, and the evidence: the pull requests you authored and the ones you reviewed (links), the main files you wrote.*

### 6. Challenges

*Two or three problems you met, what you tried and how you solved them, or why you could not.*

### 7. Lessons learned

*What you learned about the method and about working in a team on a shared code base, and what you would do differently in a data job.*

## Appendix

### A. Use of AI tools

*Which tools you used, for what, and how you checked the output. Attach the HTW declaration on the use of AI tools.*

| Tool | Used for | How I checked it |
|---|---|---|
| *for example: an AI coding agent* | *the tests in `tests/`* | *ran them, read every assertion* |

### B. References

*Sources you used (papers, documentation, course notebooks), in one consistent style, for example: Author (Year). Title. Publisher or URL.*

*I confirm that I wrote this paper myself and that I can explain every part of it and of my code.* *Your name.*
