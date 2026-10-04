# Statistical Programming: premise of the course design

**Course proposal · MPMD elective WP 5 · HTW Berlin · Winter semester 2026/27 · Part 1 of 2**

This document sets out the analysis behind the course: the module as defined in the regulations, its position in the MPMD curriculum, the requirements of the labour market, comparable university courses, the order of topics in standard texts, the choice of dataset and of assessment format. Section 8 draws the design decisions. The resulting session plan is in Part 2, [CURRICULUM.md](CURRICULUM.md).

## Contents

- [1. The module as defined](#1-the-module-as-defined)
- [2. Positioning in the MPMD curriculum](#2-positioning-in-the-mpmd-curriculum)
- [3. Labour-market requirements](#3-labour-market-requirements)
- [4. Comparable university courses](#4-comparable-university-courses)
- [5. Order of topics in standard texts](#5-order-of-topics-in-standard-texts)
- [6. Choice of the course dataset](#6-choice-of-the-course-dataset)
- [7. Choice of the assessment format](#7-choice-of-the-assessment-format)
- [8. Resulting design decisions](#8-resulting-design-decisions)
- [9. Sources](#9-sources)

## 1. The module as defined

Statistical Programming is elective WP 5 of the MPMD programme (5 ECTS, 4 WSH). The study and examination regulations define its purpose (AMBl. 15/2025, WP 5): students *implement statistical methods professionally in a suitable programming language and environment and distribute programming tasks within programming teams*.

The module description lists the topics that must be taught: well-structured programming in Python, databases, visualisation, teamwork with Git, a recap of basic statistics, robust regression, outlier detection, transformations and imputation, dimensionality reduction, cross-validation and bootstrap, hyperparameter search, tree packages (XGBoost, LightGBM, CatBoost), deployment and monitoring, churn prediction, NLP, embeddings and LLMs, retrieval-augmented generation and time series forecasting. The curriculum covers every topic; the session plans in [CURRICULUM.md](CURRICULUM.md#3-sessions) show where each one is taught.

The list is broad for 54 UE. The analysis in this document determines how the time is divided: which topics are taught in depth, which are recapped because another module teaches their theory, and which are added because the labour market requires them.

## 2. Positioning in the MPMD curriculum

Statistical Programming is elective WP 5 and can be taken in semester 2 or 3. Students take 2.2 Data Mining before or alongside it, and semester-3 students take 3.4 NLP and Neural Networks at the same time. The module does not repeat the theory of these modules; it adds the practical skills around them.

|  | 1.2 Foundations | 2.2 Data Mining | WP 5 Statistical Programming | 3.4 NLP and Neural Networks |
|---|---|---|---|---|
| Central question | How do we describe and test data? | Which methods exist, and how do they work? | **How do we turn data into reliable reports, tested findings and working models, in a team?** | How do we process language with neural networks? |
| Focus | Statistics, scripts, charts | Algorithms, model building, case study | **SQL, data quality, reporting, inference and A/B tests; the ML workflow from regression to LLMs; validation, deployment; Git** | Text representations, NLP tasks, neural architectures |
| Relation to WP 5 | Statistics recapped in applied form (Python, interpretation, A/B tests) | Algorithms used as tools and validated | — | Text treated as features and LLMs as components |

The regulations define the purpose of the module (AMBl. 15/2025, WP 5): students *implement statistical methods professionally in a suitable programming language and environment and distribute programming tasks within programming teams*. The design meets this through tested team code from Session 2 onwards and statistical methods at the core of both the analytics and the machine learning part.

**Recommended semester: 2.** Taken alongside Data Mining, the module gives students the SQL, analysis and validation skills they need for 3.1 Lab, 3.4 and the thesis. Semester-3 entry remains possible; the only prerequisite is 1.2 Foundations.

**Title.** The title in the regulations stays unchanged; the subtitle *Data Analytics and Machine Learning in Python* can be used in the module description and the course announcement without an amendment.

<details open>
<summary>Agreements proposed with other modules</summary>

- **1.2 Foundations:** WP 5 recaps descriptive statistics, tests, contingency tables and correlation in applied form, in Python; the theory stays in 1.2.
- **2.2 Data Mining:** WP 5 uses algorithms as tools and focuses on validation, features and deployment; their theory stays in 2.2.
- **3.2 Big Data:** WP 5 applies database concepts in SQL and uses Polars for large tables on a single machine; database technology beyond relational databases, data warehousing and big-data technology (Hadoop, distributed processing) stay in 3.2.
- **3.4 NLP and Neural Networks:** WP 5 uses text as features and pre-trained models and LLMs as components under evaluation; linguistic methods and neural networks stay in 3.4.
- **3.3 Responsible Data Management in Practice and WP 4 Data Ethics:** WP 5 applies documentation of data and models (model card) as engineering practice; governance, ethics and regulation, including the EU AI Act, stay in 3.3 and WP 4.
- **WP 7 Current Topics in Data Science:** forecasting there focuses on financial markets; WP 5 covers general forecasting and its evaluation.

</details>

### 2.1 The MPMD curriculum 2026

Modules of the curriculum from intake 2026 (MPMD website and study and examination regulations, AMBl. 15/2025). **CM** compulsory module, **EM** elective module, **WSH** weekly semester hours. Statistical Programming is elective WP 5 and can be taken as elective module 2.4 (semester 2) or 3.5 (semester 3). Module 3.1 is listed as an elective in the regulations and the curriculum table, but as compulsory on its module page.

| Sem. | Module | Type | WSH | ECTS | Examination | Relevance to this module |
|---|---|---|---|---|---|---|
| 1 | [1.1 International Project Management](https://mpmd.htw-berlin.de/studying/international-project-management) | CM | 6 | 10 | Project work 60 %, reflection paper 40 % | Project planning and management methods |
| 1 | [1.2 Foundations of Data Analytics and Statistical Programming](https://mpmd.htw-berlin.de/studying/foundations-of-data-analytics-and-statistical-programming-1) | CM | 6 | 10 | Case study 50, project work 10, oral examination 40 points | Statistics, data pre-processing, scripts of low to medium complexity, error-free programming, charts and reports |
| 1 | [1.3 Contract and International Business Law](https://mpmd.htw-berlin.de/studying/contract-and-international-business-law) | CM | 4 | 5 | Written examination 60 %, assignment 40 % | — |
| 1 | [1.4 Change Management and Leadership](https://mpmd.htw-berlin.de/studying/change-management-and-leadership) | CM | 4 | 5 | Written term paper | — |
| 2 | [2.1 Agile Project Management](https://mpmd.htw-berlin.de/studying/agile-project-management) | CM | 6 | 10 | Presentation 70, reflection paper 20, exercises 10 points | Agile team methods (used in the team projects) |
| 2 | [2.2 Data Mining](https://mpmd.htw-berlin.de/studying/data-mining) | CM | 6 | 10 | Case study 40 %, presentation 60 % | Data scrubbing, data mining models, evaluation of model quality |
| 2 | [2.3 Emerging Technologies and Artificial Intelligence](https://mpmd.htw-berlin.de/studying/emerging-technologies-and-artificial-intelligence-1) | CM | 4 | 5 | Case study documentation 60, presentation 40 points | AI methods and cloud solutions at an overview level |
| 3 | [3.1 Project Management and Data Analytics Lab](https://mpmd.htw-berlin.de/studying/project-management-and-data-analytics-lab-1) | EM / CM | 2 | 10 | Project contribution 20 %, presentation 30 %, assignment 50 % | Classification, segmentation, data quality in case studies |
| 3 | [3.2 Advanced Data Mining Techniques, Databases and Big Data](https://mpmd.htw-berlin.de/studying/advanced-data-mining-techniques-databases-and-big-data) | CM | 4 | 5 | Assignment | Database technology, ETL, data warehouse, text and web mining |
| 3 | [3.3 Responsible Data Management in Practice](https://mpmd.htw-berlin.de/studying/responsible-data-management-in-practice) | CM | 3 | 5 | Assignment 75, presentation 25 points | Data governance, ethics, security and compliance |
| 3 | [3.4 Natural Language Processing (NLP) and Neural Networks](https://mpmd.htw-berlin.de/studying/natural-language-processing-nlp-and-neural-networks) | CM | 4 | 5 | To be announced | Machine learning for text, entity recognition, text classification, relation extraction, NLP metrics |
| 4 | [4.1–4.2 Master's Thesis; Thesis Seminar and Final Oral Examination](https://mpmd.htw-berlin.de/studying/masters-thesis) | CM | 1 | 30 | Thesis; seminar and oral examination | Independent application of the methods |
| 2/3 | [WP 4 Data Ethics and Responsible Data Science](https://mpmd.htw-berlin.de/studying/data-ethics-and-responsible-data-science) | EM | 4 | 5 | Projects 80 %, participation 20 % | Bias, explainability, AI risk, guardrails for AI systems |
| 2/3 | **[WP 5 Statistical Programming (this module)](https://mpmd.htw-berlin.de/studying/statistical-programming)** | EM | 4 | 5 | Current: quiz 30 %, take-home coding assignment 40 %, oral examination 30 % | See this proposal |
| 2/3 | [WP 7 Current Topics in Data Science](https://mpmd.htw-berlin.de/studying/current-topics-in-data-science-real-world-financial-market-problems) | EM | 4 | 5 | Three assignments 20/20/60 % | Python for financial data: visualisation, time series, machine learning, causal analysis (Bloomberg terminal) |
| 2/3 | [WP 1–3, 6, 8 Negotiation; Group Facilitation; Technology Management; Managing International Projects; Current Topics in PM](https://mpmd.htw-berlin.de/studying) | EM | 2–4 | 5 | Various | — |

## 3. Labour-market requirements

The market analysis asks which skills employers in Berlin and Europe expect from graduates in data roles in 2026, and which of them are durable enough to justify teaching time.

| Indicator | Value | Source |
|---|---|---|
| Entry-level job postings in Germany vs 2019 | **−46 %** | [Indeed Hiring Lab DE, Sep 2026](https://hiringlab.indeed.com/de/blog/2026/09/23/berufseinstieg-stellenangebot-sinkt-ausbildungssystem-bietet-einen-anker/) |
| Net entry-level tech hiring in Europe (rest of world +14 %) | **−3 %** | [Linux Foundation, Tech Talent Europe 2026](https://www.linuxfoundation.org/hubfs/Research%20Reports/State-of-Tech-Talent-Europe-2026-REV-1.pdf) |
| German firms recruiting ICT specialists that find it hard (EU 57 %) | **72 %** | [Eurostat](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=ICT_specialists_-_statistics_on_hard-to-fill_vacancies_in_enterprises) |
| Berlin data & analytics postings, Sep 2025 → Mar 2026 | **+17.6 %** | [Indeed Hiring Lab DE, Apr 2026](https://hiringlab.indeed.com/de/blog/2026/04/02/tech-aufschwung-in-berlin/) |
| Data analysis among skills recruiters value in graduates (was 10th) | **4th** | [GMAC Corporate Recruiters Survey 2026](https://blog.efmdglobal.org/2026/06/30/corporate-recruiters-survey-ai/) |
| Data teams prioritising AI testing, vs 72 % prioritising AI coding | **24 %** | [dbt State of Analytics Engineering 2026](https://www.getdbt.com/blog/new-dbt-labs-report-finds-ai-driven-acceleration-is-outpacing-trust-and-governance) |

- **Entry-level hiring has decreased.** Employers still report difficulty in filling experienced positions, but entry-level postings have fallen in Germany and other European countries. Graduates therefore need demonstrable project experience in addition to their degree.
- **Evaluation and judgement gain importance.** Where AI tools automate programming tasks, the ability to judge whether a result is correct becomes more important. Employment of 22–25-year-olds in AI-exposed occupations has fallen mainly where AI automates tasks, not where it supports judgement ([Brynjolfsson, Chandar & Chen, Stanford Digital Economy Lab, update Aug 2026](https://digitaleconomy.stanford.edu/news/canariesaug26/)).
- **Communication is a frequently named gap.** In the Bitkom IT skills study 2025 (n = 855), missing soft skills (38 %) are the most frequent reason why IT positions in Germany remain unfilled, ahead of missing German (35 %) ([Bitkom 2026](https://www.bitkom.org/sites/main/files/2026-01/bitkom-studienbericht-it-fachkraefte-2025.pdf)).
- **Foundational skills are more durable than tools.** The skill content of technical jobs changes fastest ([Deming & Noray, QJE 2020](https://academic.oup.com/qje/article-abstract/135/4/1965/5858010)). Statistics, SQL, evaluation and communication remain relevant longer than individual frameworks, which the course treats as examples of general methods.

<details open>
<summary>Skills named in 1,019 Berlin data and AI job advertisements, and the sessions of the curriculum that teach them</summary>

| Skill | Share of advertisements | Sessions |
|---|---|---|
| LLM APIs (structured output, tool calling, cost) | 52 % | S14–S15 |
| Python for data (packaging, tests) | 48 % | S1–S2 |
| Analytical SQL | 34 % | S3 |
| Communication | 32 % | S5, S17–S18 |
| Stakeholder management | 31 % | S5, project charter |
| Agentic AI / tool use | 29 % | S15 |
| Machine learning (scikit-learn) | 27 % | S6–S11 |
| Cloud (AWS / GCP / Azure) | 22 % | S16 |
| APIs and deployment | 19 % | S2, S16 |
| CI/CD | 16 % | S2, S16 |
| Observability | 14 % | S16 |
| Data quality | 11 % | S4 |
| A/B testing | 10 % | S5 |
| Product analytics | 10 % | S5, S11 |
| dbt | 9 % | optional |
| MLOps | 8 % | S16 |
| Data modelling | 8 % | S3 |
| Evaluation and error analysis | n/a | S7–S8, S13–S15 |

Source: [Data Berlin skills snapshot, 1,019 live Berlin data/AI job ads, 24 Sep 2026](https://databerlin.net/skills); one aggregator on one day, indicative only.

</details>

### 3.1 Entry positions

The advertisements fall into three entry positions with a shared base and different emphases. The curriculum follows this structure: a shared base, an analytics part and a machine learning part. The final project is a machine learning project for all teams; its first steps (loading, data quality, exploration, baselines) practise the shared base.

| Entry position | Typical tasks | Most relevant sessions |
|---|---|---|
| Data or business analyst | SQL queries, reports and dashboards, statistical comparisons, A/B tests, segmentation | S1–S5, S11 |
| Data scientist | Models for prediction and their validation, experiments, communication | S1–S16 |
| ML or AI engineer | Model services, deployment, monitoring, LLM applications and agents | S1–S4, S6–S16 |
| Working student (Werkstudent) in these areas | The most common entry route: 9–12 % of the Berlin data and AI advertisements, compared with 1–2 % for junior positions (Data Berlin snapshot) | depends on the role |

## 4. Comparable university courses

- **Common content.** Leading courses teach transformers, LLMs, RAG and the evaluation of language models as separate topics. The team project usually accounts for 35–50 % of the grade; several courses added in-person or oral components in 2025–26 (MIT 6.7960, Stanford CS329H, Harvard AC215).
- **Recent additions (2025–26).** AI-assisted programming, evaluation of LLM applications as an engineering task, and the operation of LLM applications (LLMOps).
- **Less covered, and addressed here.** Computer science courses rarely combine SQL, data quality, reporting and statistical inference with the machine learning workflow and deployment. This module combines them for students without a computer science background.

The review covered 14 courses at Stanford, MIT, Harvard, CMU and UC Berkeley from 2025–26, among them Stanford CS224N, MIT Missing Semester, Harvard AC215 (MLOps and LLMOps), CMU 17-445 (ML in Production) and Berkeley Data 100.

## 5. Order of topics in standard texts

The order of the sessions follows standard introductory texts, so that each method builds on the previous one and students can use these texts alongside the course.

| Source | Order of topics | Use in the curriculum |
|---|---|---|
| Diez, Çetinkaya-Rundel & Barr, [OpenIntro Statistics](https://www.openintro.org/book/os/), 4th ed. | Summarising data → inference for categorical and numerical data → linear regression, which opens with correlation | Session 5 follows this order: describe, compare groups, contingency tables, correlation; correlation then leads into regression in Session 6 |
| Çetinkaya-Rundel & Hardin, [Introduction to Modern Statistics](https://openintro-ims.netlify.app/), 2nd ed. | Exploration → regression modelling → inference with randomisation and bootstrap | Exploration before modelling; the bootstrap is taught with validation in Session 7 |
| James et al., [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/) (ISLP) | Linear regression → classification (logistic regression, k-NN) → resampling → model selection and regularisation → tree-based methods → unsupervised learning (ch. 12) | Sessions 6–11: regression, validation, classification, features, tree-based models, then unsupervised learning, contrasted with the supervised methods taught before |
| UC Berkeley, [Data 100 course notes](https://ds100.org/course-notes/), Fall 2025 | pandas and SQL → EDA and visualisation → OLS → cross-validation and regularisation → logistic regression → clustering → PCA | Data access and preparation before modelling (Sessions 3–4); validation directly after the first models (Session 7) |

Two deliberate deviations: validation (Session 7) is taught before classification, as in Data 100, because every later session reports validated scores; and logistic regression is introduced together with linear regression (Session 6), so that Session 8 can concentrate on evaluation metrics.

## 6. Choice of the course dataset

The curriculum uses one running case study so that every method is practised on the same data and compared on a common leaderboard. Selection criteria: real, openly reusable public-sector data; text and a label for a leaderboard; metadata for the statistics sessions; more than one table for SQL; a time dimension for validation, forecasting and drift.

**EBTI for the text and language-model sessions.** Customs authorities of the member states issue binding decisions on how a described product is classified in the customs tariff, and the European Commission publishes all of them. The course task is to predict the four-digit HS heading from the description of goods. An earlier draft used Amazon product reviews; EBTI was preferred because its reuse terms are clear, it is a European public-sector source, and the classification of goods is a real task in customs administrations and trade.

**Pilot analysis.** The official full export was downloaded and the splits and reference models were computed:

- **Size and structure.** 1.05 million decisions from 2004 to 2026 in 23 languages (57 % German, 16 % French in the course period), with English keywords, validity dates, status and the customs' justification. The course uses 309,529 decisions from 2017–2023 for training and 113,188 from 2024–2026 as a hidden test set. A second table holds the HS nomenclature; a third the monthly counts since 2004.
- **Reference scores (accuracy, public test 2024).** Most frequent heading 4 %; simple features 8 %; word TF-IDF with a linear model 81 % on a 50,000-decision sample and 87 % on the full training set; character TF-IDF 88 %. The range leaves room for every method of the course to show a measurable effect.
- **A built-in leakage example.** The justification names the heading in 70 % of the training decisions but exists only after classification; it is the course's example of a feature that is not available at prediction time.
- **A built-in drift example.** Scores fall by about three points from the public to the private test years; the United Kingdom stops issuing decisions after Brexit, and the HS revision of 2022 changed headings. Session 16 uses these to teach monitoring and retraining.
- **Terms.** Reuse is permitted with acknowledgement of the source (Commission Decision 2011/833/EU); the holders of the decisions are not published. Each student downloads the data with a provided script.

**Datasets per part of the course.** A pilot of all sessions on EBTI alone showed that every part outside text and language models then relies on contrived questions. Sessions 1–12 therefore use the Berlin listings of Inside Airbnb (CC BY 4.0), where questions such as "what drives nightly prices?" are practical and familiar to the students, together with Berlin weather from the Open-Meteo API; EBTI is used in Sessions 13–16, where predicting the heading from a description is a natural text task, and is the task of the final project for all teams. Churn prediction, named in the module description, is taught on the IBM Telco sample data (7,043 customers).

## 7. Choice of the assessment format

The current module page lists a quiz (30 %), a take-home coding assignment (40 %) and an oral examination in the form of a job-interview simulation (30 %). The curriculum replaces these with one component: the final project, graded in its presentation with individual questions.

- **Attribution of take-home code.** With AI coding assistants, a take-home assignment no longer shows reliably what a student can do. Several comparable courses have moved weight to in-person and oral components (Section 4).
- **One project, assessed on its method, not its score.** All teams work on the leaderboard task, which makes the projects comparable and lets students discuss their solutions with each other. Decisions can be looked up in the public EBTI database, so a high score could be obtained without the methods of the course; the score is therefore not converted into marks, the submission must be reproducible from the team's repository, and the grade rests on how the team frames, handles, solves and explains the problem.
- **What the market values.** Project experience, judgement about results and communication are the skills employers name most often (Section 3); a presented project with individual questions assesses all three.
- **Individual attribution.** Group work must be attributable to each member. Each member presents a part, answers individual questions and is visible in the Git history; one criterion (20 %) is assessed per member.
- **Risk of a single component.** A single component concentrates the risk. It is mitigated by ungraded project milestones with feedback (charter, interim review in Session 10, three leaderboard rounds).

The change requires approval and must be announced at the start of the semester (RStPO §§ 9–14).

## 8. Resulting design decisions

The findings of Sections 2–7 lead to the following design decisions, which the curriculum implements.

| Finding | Design decision |
|---|---|
| Most graduates start in analyst roles; SQL is the usual first technical test | Shared base with SQL and data quality for all; an analytics part that completes the analyst track |
| Data science roles require modelling skills that 2.2 introduces conceptually; LLM skills are the most requested AI skills | Machine learning built up from regression to unsupervised learning, language models and agents, following the order of standard textbooks (ISLP, Data 100); two sessions on LLMs |
| The module description names the statistics only as a recap; the theory is taught in 1.2 | The statistics recap is applied, in one session, ordered as in standard introductions (OpenIntro, Introduction to Modern Statistics): describe, compare groups, relate variables, then regression |
| Employers expect results that work outside notebooks | Reports and dashboards in Session 5; model deployment and monitoring in Session 16 |
| Evaluation and judgement gain importance as AI tools automate coding | The statistics recap (S5) leads into regression (S6); validation (S7) is taught before the more complex models; RAG and agents are evaluated with test sets |
| Teamwork and communication are frequently named gaps; data roles increasingly require software engineering skills | Object-oriented programming, APIs and Git in Session 2; reviewed pull requests; presentation with individual questions |
| Participants have different programming backgrounds | Python for analysis and for applications in Session 1, with self-study notebooks; every topic starts from its central concept and a small example |
| Take-home work can no longer be attributed reliably; the leaderboard can be gamed | One graded component: one final project for all teams on the leaderboard task, presented with individual questions; the score is evidence, not a grade (Section 7) |
| The plan has to fit a semester of 18 sessions on any weekday | Time series forecasting is kept short, so that it can become self-study if a weekday has only 17 teaching weeks |

## 9. Sources

- MPMD website: curriculum from intake 2026 and module pages, retrieved 30 September 2026
- Study and examination regulations of the MPMD, AMBl. 15/2025, in force since 1 April 2026
- Labour market: Indeed Hiring Lab, Linux Foundation, Eurostat, GMAC, dbt Labs, Bitkom, Stanford Digital Economy Lab, Deming & Noray (2020), Data Berlin skills snapshot (links in Section 3)
- Comparable courses: course websites, retrieved September 2026
- Order of topics: Diez, Çetinkaya-Rundel & Barr, [OpenIntro Statistics](https://www.openintro.org/book/os/), 4th ed.; Çetinkaya-Rundel & Hardin, [Introduction to Modern Statistics](https://openintro-ims.netlify.app/), 2nd ed.; James et al., [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/); UC Berkeley, [Data 100 course notes](https://ds100.org/course-notes/), Fall 2025
- Course dataset: European Commission, [European Binding Tariff Information (EBTI) database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en); HS nomenclature from [datasets/harmonized-system](https://github.com/datasets/harmonized-system) (ODC-PDDL)

Detailed research notes are available on request.
