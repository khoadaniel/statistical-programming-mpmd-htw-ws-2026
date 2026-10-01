# Research notes — MLOps & deployment, LLMOps, pedagogy, project datasets

Links checked on 2026-09-24 (HTTP 200 unless noted; "403 to bots" = publisher blocks automated checks, likely fine in a browser).

## Changes in 2025–26 that affect the course design
1. **GitHub Classroom retired 28 Aug 2026.** Use Classroom 50 (https://classroom50.org/, teacher guide https://github.com/foundation50/classroom50/wiki/Teacher-Guide).
2. **Hugging Face Spaces no longer offers free hosting for Gradio or Docker apps.** Free accounts get static Spaces plus 2 Gradio Spaces on ZeroGPU (https://huggingface.co/docs/hub/spaces-overview). Don't make it the default deployment target.
3. **Free hosting that still works for students:** Streamlit Community Cloud (free), Render free tier (sleeps after 15 min idle), Google Cloud Run free tier, Modal Starter ($30/month credit). Fly.io is trial-only.
4. **MLOps Zoomcamp is self-paced only in 2026** (2025 materials). LLM Zoomcamp ran a 2026 cohort.
5. **Common Voice** moved to Mozilla Data Collective (Oct 2025).
6. **EU AI Act**: the Annex III high-risk obligations were delayed to 2 Dec 2027.

## 1. MLOps & deployment
| Title | URL | Notes |
|---|---|---|
| MLOps Zoomcamp | https://github.com/DataTalksClub/mlops-zoomcamp | MLflow, orchestration, deploy, Evidently; homework |
| Made With ML | https://madewithml.com/ | One project from design to serving |
| Full Stack Deep Learning 2022 | https://fullstackdeeplearning.com/course/2022/ | Partly dated |
| CMU ML in Production (Kästner) + open book | https://mlip-cmu.github.io/ · https://mlip-cmu.github.io/book/ | Most current university course; Fall 2026 |
| Stanford CS 329S / Designing ML Systems | https://stanford-cs329s.github.io/ · https://github.com/chiphuyen/dmls-book | |
| Rules of ML | https://developers.google.com/machine-learning/guides/rules-of-ml | |
| Hidden Technical Debt in ML Systems | https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html | |
| ML Test Score | https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/ | Use as a readiness rubric |
| Evidently ML observability course | https://www.evidentlyai.com/ml-observability-course · https://docs.evidentlyai.com/quickstart_ml | Drift; pair with NannyML |
| MLflow / DVC / W&B | https://mlflow.org/docs/latest/ml/getting-started/ · https://doc.dvc.org/start · https://docs.wandb.ai/models/quickstart | |
| sklearn Pipelines / Prefect 3 / Dagster University | https://scikit-learn.org/stable/modules/compose.html · https://docs.prefect.io/v3/get-started · https://courses.dagster.io/ | |
| FastAPI tutorial / FastAPI in Docker / Docker Get Started | https://fastapi.tiangolo.com/tutorial/ · https://fastapi.tiangolo.com/deployment/docker/ · https://docs.docker.com/get-started/ | |
| Hosting | https://docs.streamlit.io/deploy/streamlit-community-cloud · https://render.com/docs/free · https://cloud.google.com/run/pricing · https://modal.com/pricing | |
| Testing ML | https://www.jeremyjordan.me/testing-ml/ · https://cookiecutter-data-science.drivendata.org/ · https://book.the-turing-way.org/reproducible-research/reproducible-research/ | |

## 2. LLM apps & LLMOps
| Title | URL | Notes |
|---|---|---|
| LLM Zoomcamp (2026) | https://github.com/DataTalksClub/llm-zoomcamp | RAG, eval, monitoring |
| Gradio quickstart / sharing | https://www.gradio.app/guides/quickstart | |
| Streamlit chat & LLM apps | https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps | |
| Langfuse (OSS, self-hostable) | https://langfuse.com/docs | Tracing, cost, evals; EU-friendly |
| Hamel Husain — evals | https://hamel.dev/blog/posts/evals/ | |
| applied-llms.org / Eugene Yan patterns | https://applied-llms.org/ · https://eugeneyan.com/writing/llm-patterns/ | |
| Evidently LLM courses | https://www.evidentlyai.com/courses | |
| promptfoo / DeepEval | https://www.promptfoo.dev/docs/intro/ · https://github.com/confident-ai/deepeval | Prompt regression tests in CI |
| OWASP Top 10 for LLM apps | https://genai.owasp.org/llm-top-10/ | Prompt injection etc. |
| LiteLLM / Building effective agents | https://docs.litellm.ai/docs/ · https://www.anthropic.com/engineering/building-effective-agents | |
| EU AI Act timeline / AI literacy Q&A / compliance checker | https://artificialintelligenceact.eu/implementation-timeline/ · https://digital-strategy.ec.europa.eu/en/faqs/ai-literacy-questions-answers · https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/ | |

**AI Act dates** (timeline page, updated 31 Aug 2026): 2 Feb 2025 prohibitions and AI literacy (Art. 4) · 2 Aug 2025 general-purpose AI (GPAI) obligations · 2 Aug 2026 most provisions, including Art. 50 transparency (e.g. a chatbot must disclose that it is AI) · 2 Dec 2027 Annex III high-risk (delayed) · 2 Aug 2028 Annex I.

## 3. Pedagogy & assessment
| Title | URL | Notes |
|---|---|---|
| CRISP-DM / TDSP / lifecycle comparisons | https://www.datascience-pm.com/crisp-dm-2/ · https://www.datascience-pm.com/tdsp/ | Relevant to the PM side of MPMD |
| TDSP project template | https://github.com/Azure/Azure-TDSP-ProjectTemplate | Charter, data report, exit report |
| Deon ethics checklist | https://deon.drivendata.org/ | Graded deliverable |
| DS project rubric (M. Brett) / CMU rubrics | https://matthew-brett.github.io/dsfe/projects/rubric · https://www.cmu.edu/teaching/assessment/assesslearning/rubrics.html | |
| Peer assessment for reproducibility | https://zenodo.org/records/14650384 | |
| Grading for Growth | https://gradingforgrowth.com/ | Specs grading |
| Teaching Tech Together / Carpentries instructor training | https://teachtogether.tech/ · https://carpentries.github.io/instructor-training/ | Live coding, inclusivity |
| Inclusive group work | https://www.cmu.edu/teaching/designteach/teach/instructionalstrategies/groupprojects/index.html · https://teaching.cornell.edu/teaching-resources/building-inclusive-classrooms · https://learning.northeastern.edu/high-performing-teams-through-diversity-and-inclusivity/ | |
| Group formation evidence / CATME | https://arxiv.org/abs/2202.07439 · https://info.catme.org/ | |

## 4. Project datasets (verified)
| Idea | Data |
|---|---|
| Berlin Airbnb pricing & regulation | https://insideairbnb.com/get-the-data/ |
| Electricity price/load forecasting | https://www.smard.de/home/downloadcenter/download-marktdaten/ · https://smard.api.bund.dev/ · https://api.energy-charts.info/ |
| Weather covariates | https://opendata.dwd.de/ · https://brightsky.dev/ · https://github.com/wetterdienst/wetterdienst |
| Deutsche Bahn delays | https://github.com/piebro/deutsche-bahn-data · https://v6.db.transport.rest/ |
| Bundestag speeches | https://opendiscourse.de/ · https://github.com/open-discourse/open-discourse · https://www.bundestag.de/services/opendata |
| MP votes | https://www.abgeordnetenwatch.de/api |
| Job-ad skills | https://jobsuche.api.bund.dev/ |
| Berlin air quality | https://luftdaten.berlin.de/ · https://www.umweltbundesamt.de/en/data/air/air-data |
| Bike counters | https://daten.berlin.de/datensaetze/radzahldaten-in-berlin |
| Crime atlas | https://daten.berlin.de/datensaetze/kriminalitatsatlas-berlin |
| VBB / GTFS reliability | https://unternehmen.vbb.de/digitale-services/datensaetze/ · https://gtfs.de/en/ |
| Fuel prices (Tankerkönig) | https://creativecommons.tankerkoenig.de/ |
| Eurostat / Destatis | https://ec.europa.eu/eurostat/data/database · https://genesis.destatis.de/datenbank/online/ |
| Common Voice (moved) | https://mozilladatacollective.com/ |
| European Parliament | https://data.europarl.europa.eu/en/home |

Excluded (404/unresolvable): inclusiveteaching.org, Yale Poorvu page, Michigan CRLT (403), open-discourse.github.io, bundesAPI/deutschebahn-api, drivendata/deon repo, old Microsoft TDSP docs.
