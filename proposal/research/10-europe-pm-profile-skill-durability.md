# Research notes — Europe, the PM+Data profile, graduate gaps, language, skill durability

Research date 2026-09-26.

**Evidence labels:**
- **[S]** official or large-sample source
- **[M]** industry survey with a stated method
- **[W]** blog or vendor

## 1. Europe beyond Germany
**EU ICT specialist workforce:**
- [S] Eurostat (27 May 2026): 10.45 M ICT specialists (5.0 % of employment). Growth is slowing: +4.5 % in 2024, +2.6 % in 2025. https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260527-2
- [S] State of the Digital Decade 2026: 10.5 M against the 2030 target of 20 M; attracting global talent is named as a lever. https://digital-skills-jobs.europa.eu/en/latest/news/state-digital-decade-2026-without-urgent-action-eu-risks-missing-out-20-million-ict
- [M] Eurofound (via secondary source): 11.9 % of ICT specialists were born outside the EU, and migration is a primary recruitment channel. Hiring difficulty is mostly for **senior and specialised** roles.
- [S] Eurostat hard-to-fill ICT vacancies: EU 57.5 %, **DE 72.4 %**, ES 30.2 %.

**Shortage signals:**
- [S] ELA/EURES 2025 shortages report: ICT professionals have the most severe shortage, but AI may reduce shortages. https://www.ela.europa.eu/sites/default/files/2026-06/labour-shortages-report-ela-2025.pdf
- [S] Eurostat: 20 % of EU firms used AI in 2025 (13.5 % in 2024). https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20251211-2
- [M] Cedefop forecast: ICT jobs grow +2.5 %/yr in the EU but +1.3 %/yr in DE and IT.

**Entry-level signals:**
- [M] Linux Foundation, State of Tech Talent Europe 2026: net −3 % in Europe vs +14 % in the rest of the world; firms upskill 3.7× more than they hire. https://www.linuxfoundation.org/hubfs/Research%20Reports/State-of-Tech-Talent-Europe-2026-REV-1.pdf
- [S/M] France (INSEE via secondary source): IT salaried employment −3 % from 2023 to 2025; under-30 employment −7.4 % year on year.
- [M] UK (Adzuna): graduate vacancies about −46 % year on year.
- [M] Poland: junior roles are ≈5 % of IT ads.
- [S] Netherlands (UWV): the ICT market is cooling but still very tight.

**AI demand and Berlin:**
- [M] Indeed (Jul 2026): 54–59 % of AI-touched job titles are **outside tech** (DE/NL/FR/UK), including "AI enablement & consulting". https://hiringlab.indeed.com/uk/blog/2026/07/08/ai-is-spreading-across-job-titles-in-us-europe/
- [M] **Indeed DE (Apr 2026): Berlin "IT applications, data & analytics" postings +17.6 %, while the rest of Germany is falling.** https://hiringlab.indeed.com/de/blog/2026/04/02/tech-aufschwung-in-berlin/
- [M] LinkedIn Jobs on the Rise 2026: AI Engineer is #1 in FR/NL/ES/IT/UK, but it typically needs 2–3 years' experience.

**Verdict:** the shortage is senior and specialist; the junior squeeze is Europe-wide; Berlin data and analytics is a bright spot.

## 2. The PM+Data profile
**Demand:**
- [M] Axipro 2026 (3,519 EU AI ads): 7 builder roles per governance role; **fewer than 30 % of governance ads mention the AI Act**. https://axipro.co/eu-ai-act-hiring-gap-study/
- [S] AI Omnibus (Reg. 2026/1744): high-risk obligations moved to 2 Dec 2027, so governance demand ramps up in 2027.
- [M] IAPP 2025: governance teams are built by re-tasking existing staff, not by hiring juniors.
- [M] AI Product Manager (US, 12.4k ads): **2 % junior**; the top skill is use-case selection. https://axialsearch.com/insights/ai-product-jobs
- [M/W] Forward-deployed engineers: growing fast, mostly US.

**Why hybrid, evaluation-heavy profiles hold up:**
- [S] McKinsey State of AI 2025: engineers are the most-hired AI roles; AI specialists are getting easier to hire.
- [S] Canaries, Aug 2026: the gap is concentrated where AI **automates**; management and augmentative work is stable. https://digitaleconomy.stanford.edu/news/canariesaug26/
- [M] Gan 2026 (arXiv 2607.20807): **execution-heavy jobs grow more slowly than evaluation/judgment-heavy jobs** (correlational). https://arxiv.org/abs/2607.20807
- [M] GMAC Corporate Recruiters Survey 2026: **data analysis rose from the 10th to the 4th most-valued skill**; AI skills are what graduates are least prepared to demonstrate; Western European employers are more willing to sponsor visas. https://blog.efmdglobal.org/2026/06/30/corporate-recruiters-survey-ai/

**Realistic entry routes for MPMD graduates:** data or product analyst, analytics engineer, associate PM, AI enablement / implementation consultant, governance-adjacent analyst.

## 3. What employers say graduates lack
- [S] Bitkom IT skills study 2025 (n=855): unfilled roles blamed on soft skills (38 %), German (35 %), under-qualification (34 %); career changers are now hired as often as IT graduates. https://www.bitkom.org/sites/main/files/2026-01/bitkom-studienbericht-it-fachkraefte-2025.pdf
- [S] Stifterverband/McKinsey 2025: 79 % of firms lack AI competencies; only 1 in 5 cooperates with universities. https://www.stifterverband.org/medien/studie-ki-kompetenzen-unternehmen
- [M] dbt 2026: 72 % prioritise AI coding but only 24 % prioritise AI testing and observability. **This quality gap is where graduates can fill a need.**
- [M] Anaconda 2025: 43 % feel unprepared for AI tools and regulation.
- [M] CoderPad 2026: 46 % allow AI in technical assessments and judge candidates on catching AI errors.
- Prompt engineering: **the title is dead, the skill is table stakes.**

## 4. Language, first jobs, Werkstudent
- [M, indicative, own sample of 1,492 Arbeitnow ads, skewed towards English-friendly employers]: an ad counted as German if written in German or asking for German.

  | Role group | Ad in German or asks for German |
  |---|---|
  | **Project/programme manager** | **82 %** |
  | Data/AI roles | 48 % |
  | Software developer | 34 % |
  | Product manager | 30 % |

- [M] Indeed DE 2024: only 2.7 % of ads say "no German required" (NL 7.8 %, ES 5.8 %).
- [S] DAAD 2025 (21k students): two-thirds want to stay in Germany; **only a third feel ready for the job market**; German courses are decisive.
- Salaries: career starters' median €46k (Stepstone); junior data analyst ~€42k; Berlin data scientist ~€59k (all levels).
- Werkstudent-to-job conversion: no rigorous statistic exists (the ">60 %" figure comes from blogs).

## 5. Skill durability
- [S] Deming & Noray, QJE 2020: in tech-intensive jobs the skill content changes fastest and the earnings premium erodes, so teach durable foundations. https://academic.oup.com/qje/article-abstract/135/4/1965/5858010
- [W] The "skill half-life 2.5–5 years" figure has no traceable source; don't quote it.
- [M] Datadog, State of AI Engineering 2026: >70 % of organisations use 3+ LLM providers; agent-framework use 9 %→18 %. Being provider-agnostic is the durable skill. https://www.datadoghq.com/state-of-ai-engineering/
- Frameworks: teams are moving from LangChain to direct SDK calls [W]. MCP now sits under a Linux Foundation foundation, a multi-vendor standard [M/W].
- **Long half-life:** statistics, SQL, experimentation, evaluation, communication. **Short half-life:** specific orchestration frameworks.
