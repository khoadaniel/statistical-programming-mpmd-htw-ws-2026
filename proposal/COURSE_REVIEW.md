# Course review: Statistical Programming — Data Analytics and Machine Learning in Python

**Review date:** 9 October 2026  
**Context:** 22 students with mixed programming and statistical backgrounds; 5 ECTS; proposed 18 sessions of 135 teaching minutes.  
**Perspective:** Teaching Python for applied data analysis and ML, with an achievable common standard and worthwhile advanced work.

**Overall judgement:** The course has a strong practical identity and unusually substantial supporting materials. Its principal weakness is the amount students appear expected to master, rather than a shortage of relevant topics. Keep the ambition in the problems, but reduce the number of methods and tools that every student must implement. Before delivery, resolve the assessment inconsistencies and the accidental exposure of the hidden test labels.

This is a review and a proposed teaching design, not an edit to the approved syllabus. “Core” below means an activity every student practises and explains; “extension” means optional depth. Where the premise identifies required topics, retain their coverage while distinguishing introduction, guided application and independent competence.

## 1. Evidence and limits of this review

The review draws on [the module description](MODULE_DESCRIPTION.md), [the premise](PREMISE.md), [the curriculum](CURRICULUM.md), the [course hub](../README.md), session plans and material inventories across S1–S18, selected theory pages and case-study notebook contents, the S15–S16 workspace documentation, the [project brief](../sessions/17-18-final-presentations/final-project.md), and the data-preparation and scoring code.

I inspected representative exercises and implementation details rather than executing all notebooks or auditing every line of the repository. The inventory contains **209 notebooks** and approximately **166,000 whitespace-delimited words in session theory Markdown**, including code. These are measures of the resource collection, not of assigned reading. No standalone slide deck or slide-specific source surfaced in the repository inventory; the course hub describes theory pages as the materials used in class. Slide recommendations therefore concern delivery design, not the visual quality of an inspected deck.

The pedagogical proposals are reviewer judgements, supported by a few teaching references cited where relevant. Student readiness, actual completion times, hardware and accessibility needs remain unknown. Existing labour-market figures, legal statements, benchmark scores and dataset counts were not independently verified in this review.

## 2. What deserves to be preserved

| Strength | Why it matters for this cohort |
|---|---|
| Guiding questions and practical decisions | Students can see what an analysis is for, rather than learning disconnected APIs. |
| Repeated Berlin accommodation and Telco examples | Familiar data reduce the effort needed to understand each new method. |
| Baselines, leakage, grouped/time validation and uncertainty | These are durable skills and a stronger common standard than knowing many algorithms. |
| Course-written case-study notebooks | They connect methods to the actual course and already contain interpretation questions. S1 also includes worked examples followed by independent tasks. |
| Tested Python workspaces | They make the notebook-to-application transition concrete. Offline fixtures and fake clients in later workspaces are particularly useful. |
| A shared project with milestones | Seven teams can compare approaches, and feedback can be reused across teams. |
| Scores do not directly determine grades | This supports honest reporting, including a justified decision to retain a simple model. |
| Substantive limitations and responsible-use discussion | Forecasting pages discuss review-count proxies, survivor bias and future-weather availability; later materials include model cards, prompt injection and restricted tools. These topics are already present. |
| Optional material and source attribution | The resource library can support strong students and future independent learning. |

The course does not need another major topic such as deep learning from scratch, Kubernetes, or a broader catalogue of agent frameworks. The next investment should make the existing material easier to learn and teach.

## 3. Highest-priority findings

### P0 — Correct the hidden-test workflow before releasing student data

**Observed:** [prepare_data.py](../case-study/prepare_data.py) writes `data/instructor/solution.csv` when students run the documented preparation command. [score.py](../case-study/score.py) reads that file and prints both public and private scores. Several student notebooks conditionally score against it, including the L1, L2 and retraining notebooks. The word `instructor` in a directory name does not make its contents private.

**Impact:** A student following the instructions receives hidden labels without deliberately breaking a rule. Repeated private scoring also undermines the lesson about preserving a final test set. The project already acknowledges that source decisions are public; this is a separate, avoidable disclosure through the normal course workflow.

**Recommendation:** Separate instructor preparation from student preparation. Distribute a frozen student bundle containing training labels and unlabelled test rows, and keep the scoring solution outside the student bundle. Student-side checks should validate submission format and score a labelled development split only. Remove private scoring cells from the student route. Audit cached notebook outputs as well as code.

After releasing 2024 labels, explicitly retire that year's public score as an evaluation measure. The S16 notebook already explains why its near-perfect retraining score is uninformative; apply the same rule to the leaderboard interface and project instructions. Preserve the rule against looking up public-source test answers, while describing the exercise honestly as an educational holdout rather than a secure competition.

**Acceptance check:** A fresh student setup neither creates a solution file nor reports private-period scores; final private scoring is performed after the submission freeze.

### P0 — Publish one consistent assessment contract

**Observed:** [MODULE_DESCRIPTION.md](MODULE_DESCRIPTION.md) specifies a final presentation **and written report**, with individual grading. [CURRICULUM.md](CURRICULUM.md) and the [project brief](../sessions/17-18-final-presentations/final-project.md) describe presentation/repository evidence and an **80% team / 20% individual** formula. The written report has no corresponding submission specification there. The premise also describes a proposed change from the currently listed assessment format.

**Recommendation:** Reconcile the documents with the institutionally agreed format before student publication. Specify deliverables, deadlines, report format if applicable, individual attribution, grade calculation and the status of the repository. Make clear whether individual questioning can affect only the 20% individual criterion or also the judgement of technical competence. This review flags documentary inconsistency; it does not establish what the examination regulations permit.

The rule that an unreproducible submission “is not assessed” also needs a precise procedure: distinguish an unavailable external download or platform outage from missing student work, define the frozen environment/data, and state any permitted correction process in advance.

### P1 — Reduce the compulsory path and label it unambiguously

“Core” currently covers large collections. For example, the materials tables label **9 notebooks in S1, 12 in S5, 10 in S7 and 13 in S11** as core. These might be intended as sources or selected excerpts, but students cannot reliably infer that.

For each session, identify:

- **Do in class:** one main notebook or workspace, with exact sections and stopping points.
- **Complete afterwards:** a short named task with an estimated time.
- **Reference:** material to consult when needed; no expectation of completion.
- **Extension:** one optional investigation with a clear question and endpoint.

Keep the library. Change its navigation and expectations. The first screen of each README should tell a student exactly what to open today and what counts as done.

### P1 — Budget teaching and feedback in the actual timetable

The proposal supplies **40.5 contact hours**, including **4.5 hours for final presentations**, leaving **36 hours for S1–S16**. S10 also promises seven ten-minute interim reviews “in class”: **70 minutes**, before transitions, within an already full tree-model session.

Replace part of S10's content with the reviews, or explicitly schedule a clinic with its workload accounted for. Do not add the reviews to three full teaching blocks. A workable S10 is one 45-minute tree/ensemble block, one 45-minute review block for four teams, and one 45-minute block for three teams plus a shared debrief. Teams awaiting review complete an independent model-comparison task. This requires moving most package comparison and SHAP material to extensions.

The module description says 4 weekly semester hours while the proposal schedules 18 × 3 UE. Reconcile the published contact-time convention and actual semester dates with the programme timetable; the documents alone do not explain the relationship.

### P1 — Verify readiness and protect novice practice time

“No necessary requirements” and “builds on first-semester foundations” communicate different practical expectations. A 45-minute Python refresher cannot also repair substantial gaps, introduce the environment and teach pandas fluently.

Use a short, ungraded diagnostic before S1 and provide a bounded refresher route. Keep formal admission requirements separate from readiness advice. Allow students to move between support and extension tasks throughout the course; do not classify them permanently as weak or strong.

## 4. A common standard with room for advanced work

### The common standard

By the end, each student should be able to take a modest unfamiliar dataset and:

1. Identify the unit of observation, target, intended user and what information exists at prediction time.
2. Load and inspect it, check a join, handle a small number of documented quality problems and write a useful function and test.
3. Produce a chart and a short, appropriately qualified interpretation.
4. Compare a baseline with a trained model using a suitable split and preprocessing pipeline.
5. Explain a metric, inspect errors and justify whether the model is useful.
6. Rerun the analysis, contribute through a reviewed change, and use a provided service template to demonstrate the result.

This is already a demanding and valuable outcome for a mixed cohort. Forecasting, clustering, embeddings and RAG can also have guided application outcomes without promising independent mastery of every implementation.

### Three levels within the same task

| Level | Example: validation | Evidence |
|---|---|---|
| Supported core | Complete a supplied pipeline and compare random versus host-grouped splits, with hints | Correct scores and a written explanation of which split matches the intended use |
| Independent core | Make the split and pipeline decisions on a small changed dataset | Working code, a leakage check and a justified choice |
| Extension | Investigate dependence, uncertainty or an alternative validation design | A controlled comparison and a short argument about what the estimate means |

The supported route should lead towards the same essential understanding, not a separate easier qualification. Extension work should deepen reasoning, not merely increase the number of libraries used. State explicitly that sophisticated methods are unnecessary for an excellent grade when the common task is executed and defended exceptionally well.

### Readiness diagnostic and support

Offer six short tasks: trace a function; repair a small error; filter and group a table; identify a duplicate-producing join; interpret a distribution; explain training versus evaluation on a tiny example. Include an “I have not learned this yet” option for ML. Ask separately about setup, laptop capacity and confidence; confidence is not a competence score.

Budget approximately 25–30 minutes for the diagnostic and a suggested two-hour refresher within the course workload. Provide solutions after an attempt, one setup clinic and a rescue dataset. If many students cannot filter a table or write a function, use S1's application-conversion block for Python practice and introduce classes in S2 through supplied code.

For 22 students, use **11 pairs during short coding activities** and the existing **seven project teams** for semester work. Rotate driver and navigator roles. In teams of three, rotate implementation, review/testing and interpretation; in the team of four, rotate an additional reproducibility role. Every student must eventually do all of these. Strong students should not become permanent unpaid teaching assistants, and novices should not become permanent slide authors.

## 5. Proposed session-by-session balance

This preserves the 18-session outline. Items moved out of live implementation can remain in short conceptual demonstrations or required-topic reference pages where needed.

| Session | Common practical endpoint | Reduce or move to extension | A meaningful advanced challenge |
|---|---|---|---|
| **1: Python and careers** | Answer three data questions, write one function, restart and rerun the notebook | Shorten career browsing; defer independent package/class construction if the diagnostic indicates a need | Turn one answer into a tested reusable function; handle a malformed input |
| **2: Engineering** | Fix a parser, add a test, open and review one change | Supply CI configuration and API scaffolding; inheritance and implementing a server are not novice milestones | Test timeout/error behaviour using an offline response fixture |
| **3: SQL and Polars** | Filter, aggregate and join correctly; verify row counts and keys | Pre-provision the database; demonstrate Polars on the same query; windows/lazy plans become depth | Diagnose row multiplication and compare equivalent query results and costs |
| **4: Quality** | Produce checks and a cleaning log; distinguish a bad record from a valid extreme | One simple imputer and one robust outlier rule; iterative imputation and Mahalanobis as extensions | Test sensitivity to an alternative missingness or outlier assumption |
| **5: EDA and statistics** | Produce one useful chart, one group comparison and one paragraph about uncertainty and confounding | One plotting interface for student coding; one main inferential example; dashboard and wider test menu optional | Explain a disagreement between summaries/tests and its substantive meaning |
| **6: Regression** | Fit a baseline and linear model, interpret residuals and error in euros | Keep one robust alternative guided; give logistic regression a short preview, then revisit in S8 | Compare robust fitting with deletion of extreme values on a declared evaluation population |
| **7: Validation** | Repair a leaking workflow; choose an appropriate split and tune one parameter | Supply the pipeline; defer nested CV, broad search comparisons and multiple curve exercises | Compare row and group bootstrap estimates and state their assumptions |
| **8: Classification** | Build the Telco pipeline and justify a threshold from a confusion matrix and stated costs | Reuse logistic regression; k-NN and calibration fitting optional; show ROC/PR without demanding every API | Evaluate calibration or test how the decision changes when cost assumptions change |
| **9: Features and imbalance** | Test one feature and one imbalance response within the validation process | Demonstrate resampling risks; target encoding and SMOTE implementation optional | Audit a time-dependent aggregate, or compare class weights with threshold selection |
| **10: Trees and interim reviews** | Compare a supplied ensemble against the established baseline; review every project | Reserve 70 minutes for reviews; one implemented boosting library; others introduced in comparison | Explain a disagreement between permutation importance and another importance method |
| **11: Unsupervised** | Scale, cluster, profile and critique a result; interpret a supplied PCA view | Briefly show alternative clustering and anomaly methods; t-SNE/UMAP tuning and algorithm breadth optional | Assess whether clusters survive resampling, feature changes or scaling changes |
| **12: Forecasting** | Seasonal baseline, one fitted alternative and rolling-origin evaluation; interpret an interval | Supply lag construction; ARIMA and multiple forecasting libraries are depth | Audit forecast-origin information or empirical interval coverage by horizon |
| **13: Text models** | Revisit an earlier supplied TF-IDF baseline; change one setting and analyse errors | Limit live model searches; use cached long-running results; SVD optional | Compare word and character features under language and frequency slices |
| **14: Embeddings and LLMs** | Compare one embedding/retrieval approach with TF-IDF on a fixed small evaluation set | Architecture detail, clustering and full-test encoding outside the required live route | Evaluate reranking or abstention under a fixed cost/latency budget |
| **15: RAG and agents** | Evaluate a small retrieval pipeline; explain retrieval versus answer failure | Supplied vector store; tool loop shown through a trace; implementing an agent optional | Add one bounded tool and test failure cases, including an instruction embedded in retrieved text |
| **16: Delivery** | Run the supplied API, add one contract test, interpret one drift signal and complete the model card | Begin service integration earlier; demonstrate container delivery; multi-container/cloud construction optional depth | Test rollback, a batch endpoint, or a justified retraining rule |
| **17–18: Presentations** | Present evidence and limitations; every student answers a technical question | Avoid making a successful live network call essential to demonstrating competence | Defend a negative result or a trade-off, rather than adding last-minute features |

Avoid introducing the pipeline as a fresh concept repeatedly. Show the basic `fit`/`transform`/`predict` pattern during S4–S6, use it with support in S7, and require greater independence in S8. Likewise, label the target-encoding example in S7 as a supplied leakage demonstration; its full implementation belongs later.

## 6. Teaching media: give each medium a distinct job

### Slides: orient, contrast and ask

The long theory pages are valuable reference chapters. Projecting and scrolling through them would make it hard for students to identify the live learning goal. Create a small teaching layer on top, rather than duplicating the textbook into slides.

As a starting budget, use **12–18 slides per session**, with one idea per slide and only small code fragments. A typical sequence is: a practical question; what students should already know; a tiny example; a prediction question; a visual explanation; a worked result; a common mistake; the exercise and its success criteria; a debrief; an exit question. The number is a planning aid, not a target to fill.

For S7, show six listings belonging to three hosts. Ask students to assign train/test labels, reveal a repeated host, and ask what population each split represents. Only then introduce the splitter code. This gives the API a purpose.

Use readable code and labelled axes; do not rely on colour alone. Supply the slides before class, use meaningful alt text and provide a text explanation of animations. Keep optional mathematical derivations in an appendix. Establish one notation sheet connecting `X`, `y`, rows, columns, predictions, residuals and the actual Python objects.

### Live coding: model the process of thinking and debugging

Use short rehearsed sequences with a saved checkpoint after each. Narrate the prediction before running code, inspect the result, and demonstrate one realistic mistake and its diagnosis. Pause typing while students predict or discuss. Have a recovery copy so a package or network problem does not consume the exercise period.

Students need practice reading tracebacks, checking shapes and types, inspecting intermediate tables, and reducing a failing example. These ideas appear in existing materials, but deserve a recurring teaching routine: **predict → run → inspect → explain → change**. Repeat it across Python, SQL, pandas and ML.

These recommendations are consistent with the practical guidance on live coding and cognitive load in [Teaching Tech Together](https://teachtogether.tech/en/index.html). The particular timing and examples proposed here are tailored reviewer suggestions.

### Notebooks: make student decisions visible

Use the course-written notebook as the main route and third-party notebooks as references. Extend the S1 pattern of worked examples followed by independent tasks to later sessions:

- One worked example with an explained result.
- A similar task with a few key steps omitted.
- A transfer task that changes the data or the decision.
- A final explanation in the student's own words.

Each exercise needs a time estimate, expected output, a hint and a completion check. Include conceptual checks as well as executable checks: passing an assertion does not show that a student understands why a split is appropriate. Keep solutions after a deliberate stopping point or in a separate solution view.

Long-running examples need a quick mode and saved results. S13 documents roughly nine minutes for a core notebook; S14 documents 20–40 minutes of CPU encoding for the L2 notebook. Those runtimes consume a substantial part of a 45-minute block before interpretation or debugging. They should not sit on the critical path of live teaching.

### Interactive tools: require a prediction and an explanation

Build only the few interactions that expose a difficult concept. The course already contains Plotly/Streamlit material and links to interactive explanations; add structured tasks around those assets before building a new platform.

| Interaction | Student action | Evidence of understanding | Priority |
|---|---|---|---|
| **Leakage lab** | Switch between random/grouped/time splits and a leaking feature | Explain why the reported score changed and which estimate fits the use case | First |
| **Threshold explorer** | Adjust threshold and error costs | Choose an action using confusion counts and total assumed cost | First |
| **Forecast timeline** | Move the forecast origin and reveal available features | Identify which lag/weather inputs exist at that point | Next |
| **Overfitting explorer** | Change sample size and complexity | Predict training/validation behaviour before viewing it | Next |
| **Retrieval inspector** | Inspect retrieved passages before seeing the answer | Distinguish retrieval failure, unsupported generation and a correct refusal | Later |
| **Drift dashboard** | Reveal a changing input distribution, then delayed labels | Explain why input drift alone does not establish model deterioration | Later |

Use fixed seeds and precomputed results where possible. Every interaction should have a static screenshot/table equivalent and keyboard-accessible controls. Do not add account creation, paid access or unfamiliar deployment steps merely to use a teaching visual.

### In-class rhythm

For a 45-minute block, try: **5 minutes retrieval/prediction, 8 minutes explanation, 10 minutes live example, 17 minutes student work, 5 minutes debrief/checkpoint**. Reallocate towards practice when a topic is already familiar. Keep the scheduled breaks.

Collect one small individual exit response each session, such as “Which column is unavailable at prediction time, and why?” If a substantial minority misses the same prerequisite, shorten the next extension and reteach it. As an initial operational rule, treat six or more of 22 students missing the same essential check as a signal to intervene, not a research-derived threshold.

The value of active student work is supported broadly by [Freeman et al.'s STEM meta-analysis](https://doi.org/10.1073/pnas.1319030111); it does not establish a specific optimal schedule for this cohort. The [Carpentries preparation guidance](https://carpentries.github.io/instructor-training/instructor/18-preparation.html) is especially useful for planning checkpoints and deciding what to cut when teaching runs long.

## 7. Project design and assessment alignment

### Keep EBTI, but make the first successful result arrive earlier

The shared multilingual task is distinctive and has substantial room for growth. Its difficulty is not only modelling: students must also understand customs terminology, more than 1,000 classes, temporal change, sparse representations and an unfamiliar domain.

Retain the early data milestones, but give teams a **small, working TF-IDF starter by S6–S8**, clearly labelled as supplied machinery whose text representation will be explained in S13. Use it to practise validation and error interpretation. Students should have a credible end-to-end baseline well before the last four teaching sessions.

Provide a one-page domain glossary and a few illustrated goods examples. Include examples in different languages with reading support. Translated explanations and post-decision keywords can help interpret training errors, but must stay visibly separate from permitted model inputs.

Offer three dataset scales: a tiny fixture for debugging, a fixed small development sample for class, and the larger training set for optional experiments. Preserve chronological development splits and document the class coverage of each sample. Do not silently remove rare classes to make the task easier while continuing to claim full-task performance.

### Define a sufficient final project

The minimum complete route should be explicit: a documented data pipeline, majority baseline and TF-IDF model, a defensible time split, a controlled comparison, error analysis, a tested supplied service adapted to the team's model, and an honest model card. Set the number and scope of required comparisons; “several approaches” is open-ended.

Make one extension optional: character features, multilingual embeddings, abstention, retrieval/reranking, calibration, or a robustness investigation. Fine-tuning or an agent should not be an implicit route to a high grade. The current S15 project component is already optional; make that optionality equally clear across the rubric and student-facing hub.

Start the service skeleton at S13 and require a basic integration check in S14 or S15. S16 should consolidate and evaluate delivery, not introduce the entire engineering requirement immediately before the freeze.

### Map outcomes to evidence

| Outcome | Suggested evidence | Current concern |
|---|---|---|
| Python, tests and collaboration | Individual explained change, useful test and review | PR counts alone do not establish understanding |
| SQL and data preparation | A checked query plus a reproducible transformation | Final text-model performance does not show SQL competence |
| Statistics and communication | A short uncertainty/assumption explanation and one chart | Can disappear behind model comparison |
| Validation and classification | Split rationale, pipeline, baseline and error slices | Strong alignment; preserve it |
| Clustering and forecasting | Small individual in-class interpretation/application checks | The shared classifier project provides little direct evidence |
| Embeddings/RAG/agents | Guided evaluation or trace critique | Not every team needs to implement these in the project |
| Delivery | API contract test, demonstration and monitoring decision | Risk of late infrastructure work overwhelming interpretation |

These checks can remain formative if the approved assessment remains project-only. Be explicit that an outcome introduced and checked formatively is different from one independently assessed for marks. If every listed outcome must contribute to the grade, align the project evidence or revise the outcome wording through the normal course process.

### Strengthen individual evidence without creating many new assignments

Keep one examination component if that is the agreed format. Use short in-class explanations and milestone conversations to identify difficulties early. For the final defence, give every student a comparable question about their own work and a shared essential concept. Publish question types: explain a split, diagnose leakage, interpret a metric, or explain a test.

The current ten-minute Q&A gives roughly 2.5–3.3 minutes per member before general team questions. Plan the questions and recording accordingly. Avoid treating fluent English or presentation confidence as a proxy for technical competence. Retain the recorded-demo fallback already specified for S17–S18.

Add pass-level and strong-performance descriptions beside the existing “very good” column. A correct simple model with careful analysis should have a visible path to a strong mark. Credit negative findings when the comparison is valid.

### Fix feedback bottlenecks

Charter feedback is promised, but validation/model feedback is often “on request,” and later feedback is largely the leaderboard. Schedule a short review of **every** validation plan. Provide a one-page feedback form: valid inputs, split, baseline, metric, major risk, next action. A score cannot tell a novice why the experiment is invalid.

Use rotating peer review for charts and project explanations, with lecturer review of consequential validation decisions. Replace “finish everything not completed in class” with a bounded task; otherwise slower students accumulate an expanding homework deficit.

## 8. Technical and scientific details worth tightening

### Target-dependent text cleaning

The preparation script calls `mask_own_code(description, cn_code)` before splitting. It masks numbers conditional on matching the known answer. That is understandable as a way to remove quoted tariff codes from historical records, but the transformation cannot be reproduced in the same way for a new unlabelled request.

Treat this as a potential training/evaluation-to-serving mismatch that needs investigation, not as proof that every reported score is wrong. Prefer an input-only, consistently applied cleaning rule, or frame the task explicitly as classification of curated historical decision descriptions. Test how often the rule changes text and whether those cases have different performance.

### Clarify the prediction moment

The project says “what exists when a request arrives,” while the export supplies decision descriptions and `start_date` is the start of validity. Verify which fields actually correspond to information available at request time. A retrospective decision description may differ materially from a trader's original request. The honest supported claim may be prediction on future published decision descriptions, with deployment on incoming requests still requiring separate validation.

### Stable data, stable examples

The Airbnb loader uses a changing snapshot, and EBTI preparation downloads a changing full export. A fixed random seed does not freeze the source rows. Provide an instructor-selected snapshot/date, checksums, a data manifest and a documented update policy. Keep examples and reference metrics tied to that version. The 2026 data are necessarily partial at the review date; label their cutoff instead of suggesting a completed calendar year.

### Metrics and the task people will use

The service returns three suggestions; the leaderboard primarily measures top-1 accuracy. Top-k evaluation already exists in the S13 notebook. Promote it into the project evidence, alongside macro-F1, language/frequency slices and an explanation of unseen headings. If teams use abstention, report coverage and error among accepted predictions. Specify the macro-F1 label set and zero-division convention so comparisons have the same meaning.

Also strengthen submission validation: the scorer detects missing and duplicate IDs but does not reject extra IDs outside the solution. Require exact ID-set equality. This is a small implementation issue, but a useful example of testing an input contract.

### Make existing caveats part of the exercise answer

The time-series theory already explains survivor bias, reviews as a proxy, unavailable future weather and optimism from choosing models on the backtest. Require one of those limitations in the submitted forecast recommendation. Likewise, require students to state that advertised listing price is not realised revenue, and that a churn risk score alone does not prove a retention offer changes a customer's behaviour.

S7's price example excludes prices below €10 or above €1,000. Ask students to state the resulting evaluation population; an error estimate on that restricted subset is not a claim about every listing. This turns a cleaning choice into statistical reasoning.

### Reproducibility and AI use

The existing AI declaration and explain/test requirement are a good foundation. Add a short individual activity in which students critique plausible AI-generated code containing one leakage or join error. Require evidence of verification in the project, not a large prompt transcript. Provide a few brief no-assistant practice checks to distinguish independent understanding from successful tool use.

## 9. Workload, access and operational readiness

Use the proposal's **135 hours** as a planning assumption. One possible allocation is:

| Activity | Hours per student |
|---|---:|
| Contact time, including presentations | 40.5 |
| Diagnostic, setup and refresher | 4 |
| Preparation: 30 minutes × 16 teaching sessions | 8 |
| Consolidation: one hour × 16 sessions | 16 |
| Team project implementation and analysis, per member | 50 |
| Report/presentation preparation and rehearsal | 12 |
| Recovery and unexpected difficulties | 4.5 |
| **Total** | **135** |

This is a suggested budget, not measured completion time. Project milestones must draw from the project allocation rather than becoming additional weekly assignments. Optional work should replace an extension allocation or be clearly beyond the required workload. Gather anonymous time estimates after S2, S5 and S8 and adjust the core accordingly.

The root environment installs a broad ML stack immediately, including later-session packages. Consider a small baseline environment and staged additions, while keeping exact known-working versions and simple instructions. Supply a single setup check that loads the small dataset, imports the required early packages and runs a tiny test.

The root README suggests Codespaces as an equivalent fallback, but no course devcontainer configuration appeared in the inventory. Either provide and rehearse that environment or qualify the promise. Reuse one PostgreSQL instance: the root and S15 instructions currently use different container names while both bind port 5432, which can create a collision if students follow both literally.

Define a tested route for students without GPUs, paid API accounts, administrator rights or capacity for multi-gigabyte downloads. Cached embeddings and saved model responses are appropriate for interpretation exercises; scripted stand-ins should be clearly labelled and never presented as evidence of real model quality. Keep live provider use optional or provide equitable course-funded access with a declared limit.

Supply a consistent folder map, glossary, keyboard-friendly materials and visible instructions rather than spoken-only steps. Captions or short transcripts should accompany any setup videos. Pilot the core route with someone closer to the intended entry level, and record where they need help.

## 10. Implementation order

| When | Action | Completion criterion | Relative effort |
|---|---|---|---|
| **Before student release** | Separate student/instructor data and scoring | Fresh setup reveals no private labels or private scores | Medium |
| **Before student release** | Reconcile assessment documents | One deliverable list, grading formula and deadline schedule | Small–medium |
| **Before teaching** | Label one core route per session | Every README names exact tasks, time estimates and a stopping point | Medium |
| **Before teaching** | Run diagnostic and rehearse setup/fallback | All students have a viable route into the first exercise | Medium |
| **Before teaching** | Reserve project feedback time | S10's 70 review minutes and validation feedback have actual slots | Small |
| **Prepare S1–S4 first** | Create brief slides, scaffolded exercises and exit checks | Pilot students can complete and explain the core task | Medium |
| **By S6–S8** | Release a small project baseline | Every team has a rerunnable prediction/evaluation path | Medium |
| **Before S13** | Prepare quick modes and cached outputs | No required live task waits for full-data training or encoding | Medium |
| **By S14–S15** | Check service integration | Each team can call its model through the supplied interface | Medium |
| **After the first delivery** | Refine interactive tools and extensions | Additions respond to observed misconceptions and workload data | Variable |

If preparation time is limited, fix the two P0 issues, clarify the core path, protect practice time and provide early feedback before investing in elaborate slides or custom tools. The existing resources already provide enough advanced content; the central design task is to turn that collection into a sequence that all 22 students can navigate successfully.
