# Research notes — Benchmark against top MIT / Stanford / Harvard courses (2024–26)

URLs checked 2026-09-24 (HTTP 200 unless noted). Detailed syllabi of CS229, 6.390 and 6.5940 require a login.

## Course inventory
| Course | Inst. | Latest term | URL | Core topics | Assessment / pedagogy | AI policy | New in 2025–26 |
|---|---|---|---|---|---|---|---|
| CS224N NLP with DL | Stanford | W26 | https://web.stanford.edu/class/cs224n/ | transformers, pretraining, post-training (SFT/RLHF/DPO), LoRA, agents/tool use/RAG, evals, reasoning | 4 psets 48 %; project 49 % (proposal 8, milestone 6, poster 3, report 32); teams 1–3 with mentor | AI as a collaborator; no substantial AI-written work | agents/RAG, evals, reasoning lectures |
| CS336 LM from Scratch | Stanford | Sp26 | https://cs336.stanford.edu/ | tokenizer→transformer→training, FlashAttention, scaling laws, data filtering, SFT + RL for reasoning | 5 from-scratch assignments | LLMs only for conceptual help; recommends turning off autocomplete | RL for reasoning |
| CS229 ML | Stanford | Su26 | https://cs229.stanford.edu/ | GLMs, kernels, NNs, PCA, RL | psets + exam | – | stable |
| CS230 Deep Learning | Stanford | 2025-26 | https://cs230.stanford.edu/ | CNN/RNN, optimisation | **flipped** (Coursera modules); project 40 % (proposal→milestone→report→poster); 2 mandatory mentor meetings | AI allowed with acknowledgement | guest lectures on LLMs |
| CS231n | Stanford | Sp26 | https://cs231n.stanford.edu/ | CV, diffusion, VLMs | from-scratch assignments 45 %, midterm 20 %, project 35 % | – | diffusion, VLMs |
| CME 295 Transformers & LLMs | Stanford | Au26 | https://cme295.stanford.edu/ | transformer internals, training, fine-tuning, efficiency, deployment | exams only | – | new in 2025; public cheatsheet |
| CS25 Transformers United V6 | Stanford | Sp26 | https://web.stanford.edu/class/cs25/ | frontier seminar | attendance | – | YouTube talks |
| CS329A Self-Improving AI Agents | Stanford | Au25 | https://cs329a.stanford.edu/ | verifiers, test-time compute, tool use, memory, agent evaluation | HW 50 %, research project (proposal, midterm presentation, final, poster) | – | whole course new |
| CS329H ML from Human Preferences | Stanford | Au26 | https://web.stanford.edu/class/cs329h/ | RLHF, choice models, fairness | project with **pre-analysis plan** 50 %; quizzes 30 %; **2 × 12-min oral exams 20 %**; required **AI-use reflection** | AI allowed, must be disclosed and reflected on | oral exams + AI disclosure |
| CS246 Mining Massive Datasets | Stanford | W26 | https://web.stanford.edu/class/cs246/ | Spark, LSH, recommender systems, GNNs | Colab + Spark HW, exam | – | GNNs |
| CS153 Frontier Systems | Stanford | Sp26 | https://cs153.stanford.edu/ | AI stack, policy | project "One-Person Frontier Lab" | AI use expected | AI-leveraged solo building |
| 6.390 Intro to ML | MIT | F26 | https://introml.mit.edu/ | regression → transformers, RL | same-week exercises, **in-person pair labs with check-off**, CAT-SOOP autograder | behind login | transformer week |
| 6.S191 Intro to Deep Learning | MIT | 2026 | https://introtodeeplearning.com/ | DL, generative models, LLMs | 3 Colab labs (incl. LLM fine-tuning); **project pitch competition with industry judges** | – | LLM fine-tuning lab; all material MIT-licensed |
| 6.7960 Deep Learning | MIT | F26 | https://deeplearning6-7960.github.io/ · OCW F24 https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/ | architectures, generalisation, scaling | 2024: psets + blog-post project; **2026: 80 % closed-book exams; AI-graded psets** | disclose the AI used and how | back to in-person exams |
| 6.5940 TinyML & Efficient AI | MIT | F26 | https://efficientml.ai/ | pruning, quantisation, efficient LLM serving | 5 labs + project | – | LLM quantisation |
| Missing Semester 2026 | MIT | IAP 26 | https://missing.csail.mit.edu/2026/ | shell, git, packaging, **agentic coding** (Claude Code, Codex, AGENTS.md, MCP, sandboxing), code quality | lectures + exercises | teaches agentic coding directly | full redesign |
| 6.C01/C51 Modeling with ML | MIT | Sp25 | https://computing.mit.edu/cross-cutting/common-ground-for-computing-education/modeling-machine-learning/ | ML core + discipline module | core + domain track | – | – |
| CS50x / CS50 AI | Harvard | 2026 | https://cs50.harvard.edu/x/ · https://cs50.harvard.edu/ai/ | CS basics, SQL, AI week; search → LLMs | autograders (check50), final project gallery | **only CS50's own Duck tutor allowed** | Duck tutor |
| CS1090A/B (ex-CS109) Data Science | Harvard | F26 | https://qrd.college.harvard.edu/directory/compsci-1090a/ · archive https://github.com/Harvard-IACS/2021-CS109A | wrangling, EDA, regression, bootstrap/CV, **imputation**, ensembles | weekly HW, midterm, group project | – | now on Canvas |
| AC215 MLOps & LLMOps | Harvard IACS | Sp26 | https://harvard-iacs.github.io/2026-AC215/ · https://github.com/Harvard-IACS/2026-AC215 | Docker, cloud, pipelines, RAG/agents/fine-tuning, deploy, K8s, CI/CD, monitoring | teams 3–5 build an app; **5 milestones 82 %**, team presentations; **every member must explain every part** | AI as a learning aid | LLMOps at the centre |
| CS 2881 AI Safety | Harvard | F25 | https://boazbk.github.io/mltheoryseminar/ | alignment, evals, reward hacking | seminar + project | – | first AI safety course |
| HBS DSAIL (Data Science & AI for Leaders) | HBS | 2025-26 (required) | https://www.thecrimson.com/article/2025/4/4/hbs-makes-ai-class-required/ | inference, ML, AI governance; **students build their own agents** | case method, 24/7 tutor bot | AI-native | replaced Data Science for Managers |
| CS294/194 Agentic AI | Berkeley | F25 | https://rdi.berkeley.edu/agentic-ai/f25 | reasoning, agent evaluation, multi-agent systems, safety | MOOC; project: build an evaluator agent, then a competitor agent | – | agent benchmarks |
| 17-445/645 ML in Production / AI Engineering | CMU | S26 | https://mlip-cmu.github.io/s2026/ | requirements, risk, drift, testing, MLOps, **RAG/agents, MCP labs, agent security**, fairness | **one evolving system over 4 milestones** (deploy → infra quality → monitoring/CD → fairness/security); specs-based pass/fail with resubmission | unrestricted; students own correctness | MCP labs, "AI Engineering" |
| Data 100 | Berkeley | Sp26 | https://ds100.org/sp26/ | pandas, SQL, EDA, OLS, CV, PCA | 11 autograded labs, 7 HW, 2 projects, computer-based exams | – | stable |

Also: CMU 11-711 ANLP F25 (https://phontron.com/class/anlp-fall2025/), MLSysBook (https://mlsysbook.ai/), MIT MicroMasters SDS. CS329S has not run since 2022; CMU MLiP and AC215 took its place.

## Synthesis
**Common core (2025–26):** Python/PyTorch · supervised basics + CV · transformers everywhere (even intro courses) · LLM lifecycle (pretrain → SFT → preference tuning → PEFT) · RAG + tool use/agents · **evaluation as its own lecture** · responsible AI · team project worth 35–50 % of the grade.

**Emerging topics:** agentic coding (Missing Semester 2026) · agents + **MCP** + agent security (CMU, Berkeley, AC215, HBS) · reasoning models / test-time compute · evals as engineering · efficient inference · LLMOps · AI-native teaching (tutor bots, AI-graded psets).

**Teaching patterns worth copying:**
1. Proposal → milestone → final, with mentor check-ins (CS224N, CS230).
2. **One evolving system graded at several milestones** (CMU M1–M4, AC215 MS1–5).
3. **Individual accountability**: oral exams (CS329H), "every member explains every part" (AC215).
4. Short-cycle autograded practice (6.390, Data 100, CS50).
5. In-person labs with check-offs.
6. Specs grading with resubmission (CMU).
7. Public showcase with external judges (6.S191, CS224N poster).
8. **Pre-analysis plan** (CS329H).
9. An explicit AI policy. Assessment is moving towards evidence that holds up when AI writes the code (6.7960 moved 80 % of the grade to exams).

**Gaps in elite CS courses, where our course stands out:** SQL/DuckDB · robust statistics · missing data · conformal prediction · time-series forecasting · visual communication · accessible MLOps · professional engineering practice · PM framing (requirements, risk, cost, stakeholders) · LLM cost/latency trade-offs.

**Recommendations taken into the redesign:** keep the arc. Add agents + MCP + evals as first-class topics. Make the final project **one evolving system graded at milestones** with a deployed MVP early. Add individual oral checks and a per-submission AI-use statement. Add PM framing (risk register, cost budget) in each block. Drop from-scratch deep learning, GPU work and Kubernetes.
