# Research notes — Module naming, HTW regulations, pedagogy evidence

Researched 2026-09-26. The key PDFs (MPMD StPO, HTW RStPO) were downloaded and read. Anything unconfirmed is marked.

## 1. Facts from HTW's own documents (these change the brief)
- **Legal basis:** the MPMD StPO (study and examination regulations), AMB 15/25, in force from **1 Apr 2026**: https://www.htw-berlin.de/fileadmin/HTW/Zentral/Rechtsstelle/Amtliche_Mitteilungsblaetter/2025/15_25.pdf
  - MPMD is a fee-paying, continuing-education Master (120 ECTS, 4 semesters, English).
  - §7(2) calls it a "Präsenzstudiengang" (on-campus programme), so **hybrid delivery needs confirming**.
  - **Admission:** a Bachelor's degree plus ≥1 year of work experience, **no programming or maths prerequisite** (AMB 14/25: https://www.htw-berlin.de/fileadmin/HTW/Zentral/Rechtsstelle/Amtliche_Mitteilungsblaetter/2025/14_25.pdf).
- **The module:** "MPMD WP 6 Statistical Programming", a practical exercise class (PÜ), 4 SWS, **5 LP**, level 2a, taken in semester 2 or 3.
  - Its learning outcomes (StPO Annex 3) are broad: sound knowledge of statistical programming, professional implementation of statistical procedures, splitting programming tasks across a team.
- **Workload:** 1 LP = 27 h, so **135 h**. Contact time is 40.5 h, leaving **≈94.5 h of self-study** (≈6–7 h/week).
- **An elective container already exists:** "WP 7 Current Topics in Data Science", with generic learning outcomes. The programme spokesperson chooses which electives run (§7(3)).
- **Overlap with compulsory modules:**
  - 1.2 Foundations of Data Analytics & Statistical Programming (Python basics, DataCamp)
  - 2.2 Data Mining
  - 2.3 Emerging Technologies & AI
  - 3.2 Advanced Data Mining, Databases & Big Data
  - 3.3 Responsible Data Management
  - 3.4 NLP & Neural Networks
  - 3.1 PM & Data Analytics Lab
  - WP 4 Data Ethics

  Because the elective can be taken in semester 2 **or** 3, prior knowledge in the cohort will be mixed.
- **Current public module description** (https://mpmd.htw-berlin.de/studying/statistical-programming/):
  - Already broad: Python, databases, visualisation, Git, tests, robust regression, outliers, dimensionality reduction, CV, deployment, churn, NLP, forecasting, intro to LLMs, gradient boosting.
  - **Assessment: quiz 30 % / take-home coding assignment 40 % / oral "job-interview simulation" 30 %.**

## 2. Naming patterns (verified)
| Institution | Title | Pattern |
|---|---|---|
| CMU | Machine Learning in Production / 11-695 AI Engineering | "in Production", "Engineering" |
| Stanford | CS 329S ML Systems Design | "Systems Design" |
| Berkeley | Data 100 Principles & Techniques of Data Science | foundational |
| TU Delft | Release Engineering for ML Applications | "Engineering" |
| UvA | Applied Machine Learning | "Applied" |
| Uni Mannheim | Data Science in Action; LLMs and Agents; AI Applications in Industry; Hot Topics in ML | "in Action", "Hot Topics" |
| ETH Zurich | Applied Statistical Regression | "Applied Statistical …" |
| KU Leuven | Advanced Analytics in a Big Data World | business analytics |
| UCL | Applied Data Science | "Applied" |
| Hertie | MSc Data Science **and AI** for Public Policy (renamed) | "+ AI" |
| HTW MPMD | Current Topics in Data Science; Emerging Technologies and AI | "Current Topics" |

**Takeaways:**
- "Applied …" is the most common employability signal in Europe and ages well.
- "in Production" / "Engineering" fit when there is real deployment.
- Buzzwords such as "Agents" or "GenAI" date quickly in a legal title.

## 3. Formal constraints (HTW)
- **Renaming:** RStPO §4(2) sentence 1 puts the module **title and learning outcomes in the StPO**, so a formal rename means an amendment ordinance: Faculty Council → internal approvals → official bulletin (Amtliches Mitteilungsblatt). RStPO: https://www.htw-berlin.de/fileadmin/HTW/Zentral/Rechtsstelle/Amtliche_Mitteilungsblaetter/Rahmenordnungen/RStPO_23_25_EN.pdf
- **What can change each semester (§4(2) sentence 2):** the **module description**, meaning contents, exam type/form/weighting, literature and responsible lecturer. A content refresh plus a descriptive subtitle is possible without changing the StPO.
- **Who owns the description (§5):** a full-time professor is module coordinator. An external lecturer proposes changes; the coordinator and programme spokesperson decide.
- **Accreditation:** HTW is system-accredited (AQAS, valid until 2029). Changing electives is "not substantial" (https://akkreditierungsrat.de/faq-kategorie/wesentliche-aenderungen/), so this is a governance matter, not an accreditation problem.
- **Exam rules:**
  - **Max 3 components** in a combined exam (§12(3)).
  - Must-pass per component only if the StPO says so (§17(3)).
  - Group work must allow individual grading (§16(1)).
  - **Orals need two examiners, or one examiner plus an observer**, 15–60 min each; group orals of up to 4 students are allowed (§14).
  - Modalities are fixed at the start of the semester (§16(3)).
- **HTW AI rules:**
  - AI tools are **allowed by default** unless excluded with a didactic justification (https://www.htw-berlin.de/lehre/lehre-gestalten/ki-in-lehre-und-pruefungen/rahmen-fuer-ki-tools).
  - There are four declaration templates (https://www.htw-berlin.de/lehre/lehre-gestalten/ki-in-lehre-und-pruefungen/eigenstaendigkeit-erklaeren-ki-verzeichnis).
  - HTW has no rule on paid LLM APIs.
- **German context:** Hochschulforum Digitalisierung (2025) recommends more oral and process-based exams (https://hochschulforumdigitalisierung.de/pruefen-mit-ki/).

## 4. Pedagogy evidence
- **Depth beats coverage:** Schwartz et al. 2009, *Sci Educ* (n = 8,310): https://onlinelibrary.wiley.com/doi/10.1002/sce.20328
- **Heavy perceived workload leads to surface learning:** Kember 2004: https://www.tandfonline.com/doi/abs/10.1080/0307507042000190778
- **Curriculum frameworks favour data acumen and the full workflow over tool coverage:**
  - ACM DS 2021: https://www.acm.org/binaries/content/assets/education/curricula-recommendations/dstf_ccdsc2021.pdf
  - NASEM 2018: https://nap.nationalacademies.org/catalog/25104/data-science-for-undergraduates-opportunities-and-options
  - EDISON: https://zenodo.org/records/7506445
- **Keep infrastructure friction low** (Çetinkaya-Rundel & Ellison 2021). **Git works as a scaffolded learning objective** (Beckman et al. 2021).
- **Project-based learning:** meta-analysis d = 0.71 (Chen & Yang 2019): https://www.sciencedirect.com/science/article/abs/pii/S1747938X19300211
- **Active learning:** +0.47 SD, and failure is 1.5× more likely under lecturing (Freeman et al. 2014): https://www.pnas.org/doi/10.1073/pnas.1319030111
- **High-impact practices** (capstones, community-based projects): Kuh 2008.
- **Prior-knowledge gaps persist:** pre/post correlation r = .53 (Simonsmeier 2022). The expertise-reversal effect argues for tiered tasks. A remedial maths course raised pass probability by ~35 % (Büchele 2020).
- **Generative AI and learning:**
  - "The Widening Gap": AI helps strong novices and hurts weak ones (Prather et al. 2024): https://dl.acm.org/doi/10.1145/3632620.3671116
  - Unrestricted GPT-4 tutoring lowered performance once AI was removed (Bastani et al. 2025, *PNAS*): https://www.pnas.org/doi/10.1073/pnas.2422633122
  - Anthropic RCT, 2026 (n = 52): the AI group scored 17 % lower on understanding and debugging a new library: https://www.anthropic.com/research/AI-assistance-coding-skills
  - Microsoft/CMU, CHI 2025: more confidence in AI went with less critical thinking.
  - Markers can't reliably detect AI-assisted work (Kofinas 2025).
  - **Two-lane assessment** (University of Sydney): one secure lane plus one AI-open lane.

## 5. Flagship-course features (indirect evidence)
A real client or real data, an individually attributable portfolio artefact, an interview-like defence, and industry or public-sector involvement.

Models:
- DSSG Europe / DSSGx Munich: https://dssgfellowship.org/europe/
- CorrelAid (70+ pro-bono projects): https://correlaid.org/en/projects/
- **CityLAB Berlin Summer School, with HTW as a regular partner**: https://citylab-berlin.org/veranstaltungsreihen/summer-school/

## Recommendations taken into the review
1. Keep the official title and add a subtitle through the semester description update.
2. Split the course: statistics/evaluation/engineering spine in WP 6, LLM/RAG/agents/evals content in WP 7 "Current Topics" (if the programme can staff it).
3. Cut to about 7 core units; make the rest optional project choices.
4. Align with the compulsory modules through the module coordinator.
5. Add a week-0 diagnostic and a bridge.
6. Assessment with a maximum of 3 components, two-lane, with the existing "job-interview" oral kept.
