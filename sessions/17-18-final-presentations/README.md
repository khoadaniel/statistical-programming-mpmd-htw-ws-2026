# Sessions 17–18 · Final project presentations

The last two sessions are reserved for the presentations of the seven teams: four teams in Session 17, three in Session 18. All teams present their solution of the same project, a REST API that suggests the customs heading of a product. The brief, milestones, rules and assessment criteria are in [final-project.md](final-project.md).

> **Format.** 15 minutes of presentation with a live demonstration, then 10 minutes of questions, including individual questions to each member. Have a recorded fallback ready in case the live demonstration fails.

## Checklist one week before

- [ ] Repository complete: README with setup instructions, reproducible data loading, tests passing in CI
- [ ] Service with the top-3 headings and dashboard running together with Docker Compose, released through the CI/CD workflow with a version tag such as `v2.0.0` (Session 16); monitoring plan and model card in the repository
- [ ] Tag `submission` set on `main` (a separate tag from the version tag; the same commit may carry both) and pushed at the code freeze (the day before Session 17; [how](final-project.md#how-to-submit-the-submission-tag)); `docker compose up` from the tag starts the API on port 8000, and it answers the interface of the final project (`POST /predict`, `GET /health`), checked with `case-study/evaluate_api.py`
- [ ] Report in `docs/report.md`, written with the [report template](report-template.md): Part A with the reported score (accuracy and macro-F1 of the final design, how they were estimated, their uncertainty), Part B with an entry by every member, Part C on AI use
- [ ] Every member has a part of the presentation and has reviewed at least two pull requests of others
- [ ] Rehearsed once with a timer (15 minutes)
- [ ] HTW declaration on the use of AI tools added to the repository; every member has reviewed the report

## Session 17 · Final project presentations I

**Session plan** (first part of the last teaching day)

| Step | Activity |
|---:|---|
| 1 | Opening: order of presentations, assessment criteria, rules for questions |
| 2 | Team 1: 15 minutes presentation with live demonstration, 10 minutes of questions, including individual questions to each member |
| 3 | Team 2: same format |
| 4 | Written peer feedback on Teams 1–2 |
| 5 | Team 3: same format |
| 6 | Break |
| 7 | Team 4: same format |
| 8 | Written peer feedback on Teams 3–4 |
| 9 | Common observations from the four presentations |
| 10 | Teams 5–7 test their live demonstrations in the room |

## Session 18 · Final project presentations II and course conclusion

**Session plan** (second part of the last teaching day, after the lunch break)

| Step | Activity |
|---:|---|
| 1 | Opening: order of Teams 5–7 and assessment criteria |
| 2 | Team 5: 15 minutes presentation with live demonstration, 10 minutes of questions, including individual questions to each member |
| 3 | Team 6: same format |
| 4 | Written peer feedback on Teams 5–6 |
| 5 | Team 7: same format |
| 6 | Break |
| 7 | Written peer feedback on Team 7 |
| 8 | Independent evaluation results: every team's accuracy and macro-F1 on the final evaluation set next to its reported score; the teams explain the differences, starting with the closest and the most distant pairs |
| 9 | Course review |
| 10 | Course evaluation (anonymous, in class) |
