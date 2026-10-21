# Paper template · Final project

Every member writes their own paper. Copy this template and replace every *italic instruction* with your text. Keep the headings. Write in plain academic English: short sentences, past tense for what you did, numbers with their uncertainty, no marketing words.

A short paper is better than a long one with repetition: every sentence should tell the reader something they need.

**How it is used.** Your paper and the code you wrote are assessed together, for you alone (80 % of your grade; the presentation is the other 20 %). The criteria are in the [final project brief](final-project.md#presentation-and-assessment). Other members may read and comment on your paper, but you write it yourself, and it describes your own work and your own thinking.

---

**Title:** *a title that states your contribution, for example "A grouped validation design for multilingual HS heading suggestions"*
**Author:** *your name (GitHub handle)* · **Team:** *team name* · **Other members:** *names*
**Repository:** *URL* · **Submission tag:** `submission` (commit *first 7 characters of its hash*) · **Date:** *YYYY-MM-DD*

## 1. The project in brief

*In your own words, about half a page: who would use the suggestions and for which decision, the team's final design (features, model, service), and the team's reported score (accuracy and macro-F1 with their intervals, as in the README). State your area of responsibility in one sentence.*

## 2. My part of the problem

*What your area had to solve, and why it matters for the use of the service. Name the question your part answers, for example "how well will the service do on new requests?" for a validation design.*

## 3. Choices and alternatives

*The main decisions in your area. For each: the options you considered, what you chose, and why. Keep the options you rejected: they are evidence of your reasoning. A small table helps.*

| Decision | Options considered | Chosen | Reason |
|---|---|---|---|
| *for example: how to split the decisions* | *random, stratified by heading, grouped by near-identical texts* | *grouped* | *near-duplicates in train and validation would make the score too optimistic* |

## 4. Theoretical foundations

*The methods behind your choices, explained so that a fellow student can follow: what the method does, which assumptions it makes, and where it fails. Cite your sources (course notebooks, textbooks, papers, documentation).*

## 5. Results and their reliability

*Your results with their uncertainty (for example a bootstrap interval or repeated splits), compared with the baseline. Two or three findings from reading errors and per-group scores (by language, heading frequency, chapter): what you saw, an example, and what you did about it. Your expectation for the independent evaluation, in one sentence: equal to the reported score, lower, or higher, and why.*

## 6. My code

*Where your work is in the repository, and how to rerun it from the README.*

- **Files and functions I wrote:** *paths, with one line each on what they do.*
- **Main pull requests:** *links.*
- **Tests:** *what your tests check, and what they do not check.*
- **Reviews:** *pull requests of others that you reviewed, with one example of a change your review led to.*

## 7. Lessons learned from the group work

*A reflection of about half a page: what worked in the team and what did not, how the team made decisions and handled disagreement, what you learned about the method and about working on a shared code base, and what you would do differently in a data job.*

## 8. Use of AI tools

*Which tools you used, for what (code, debugging, text), and how you checked the output. Attach the HTW declaration on the use of AI tools.*

| Tool | Used for | How I checked it |
|---|---|---|
| *for example: GitHub Copilot* | *test functions in `tests/`* | *ran the tests, read every assertion* |

## References

*Sources you used (papers, documentation, course notebooks), in one consistent style, for example: Author (Year). Title. Publisher or URL.*

*I confirm that I wrote this paper myself and that I can explain every part of it and of my code.* *Your name.*
