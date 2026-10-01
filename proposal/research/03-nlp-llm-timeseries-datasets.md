# Research notes — NLP & LLMs, time series forecasting, Berlin/German datasets

Links checked on 2026-09-24 (HTTP 200 unless noted).

## Findings that change the plan
- **"Forecasting: Principles and Practice, the Pythonic Way"** (https://otexts.com/fpppy/) is live and updated continuously (22 Sep 2026). It uses the Nixtla ecosystem and has chapters on neural nets and **foundation models**. It is the time-series backbone.
- **Jurafsky & Martin SLP3** draft updated 19 Aug 2026; **CS224N** Winter 2026 materials public.
- **TimesFM 3.0 weights are non-commercial**; use TimesFM 2.5 (Apache-2.0) or Chronos-2 (Apache-2.0) in class.
- **Inside Airbnb Berlin is CC BY 4.0**; latest snapshot 2026-06-26. Suitable for the red thread.
- Nixtla doc roots `/statsforecast/` and `/hierarchicalforecast/` loop on redirects. Always deep-link.
- `huggingface.co/datasets/gnad10` is 404 → use `community-datasets/gnad10`.

## NLP / LLMs
| Title | URL | Free | Notes |
|---|---|---|---|
| Hugging Face LLM Course | https://huggingface.co/learn/llm-course · BPE https://huggingface.co/learn/llm-course/chapter6/5 | yes | Spine; Colab per chapter |
| Speech and Language Processing 3e draft | https://web.stanford.edu/~jurafsky/slp3/ | yes | Theory reference |
| Stanford CS224N | https://web.stanford.edu/class/cs224n/ | yes | W26 slides & assignments |
| Hands-On Large Language Models — notebooks | https://github.com/HandsOnLLM/Hands-On-Large-Language-Models | notebooks free, book paid | Maps 1:1 to syllabus (classification, BERTopic, RAG, fine-tuning) |
| Karpathy — Zero to Hero / GPT tokenizer | https://karpathy.ai/zero-to-hero.html · https://github.com/karpathy/minbpe | yes | BPE from scratch |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | yes | Attention intuition |
| Raschka — LLMs from scratch | https://github.com/rasbt/LLMs-from-scratch | code free | |
| spaCy course | https://course.spacy.io/en/ | yes | In-browser exercises; German models |
| Sentence-Transformers | https://sbert.net/ · MTEB https://huggingface.co/spaces/mteb/leaderboard | yes | Multilingual embeddings |
| BERTopic | https://maartengr.github.io/BERTopic/ | yes | |
| DuckDB vss / LanceDB / Chroma / pgvector | https://duckdb.org/docs/stable/core_extensions/vss · https://lancedb.com/docs/ · https://cookbook.chromadb.dev/ · https://github.com/pgvector/pgvector | yes | Embedded = no server |
| Claude Cookbooks / OpenAI Cookbook | https://github.com/anthropics/claude-cookbooks · https://cookbook.openai.com/ | yes | Structured output, tool use, RAG, judge |
| Hamel Husain on evals | https://hamel.dev/blog/posts/evals/ · https://hamel.dev/blog/posts/llm-judge/ | yes | Error analysis, calibrated judges |
| RAGAS | https://docs.ragas.io/ | yes | RAG metrics |
| Chip Huyen — AI Engineering repo | https://github.com/chiphuyen/aie-book | repo free | Instructor reading |
| HF Agents Course / smol-course / Cookbook | https://huggingface.co/learn/agents-course · https://github.com/huggingface/smol-course · https://huggingface.co/learn/cookbook | yes | LoRA/SFT on Colab |
| DataTalksClub LLM Zoomcamp | https://github.com/DataTalksClub/llm-zoomcamp | yes | RAG + eval homework |
| Ollama | https://ollama.com/ | yes | Local models |
| German models/data | https://huggingface.co/deepset/gbert-base · https://huggingface.co/datasets/deepset/germanquad | yes | |

## Time series
| Title | URL | Free | Notes |
|---|---|---|---|
| FPP3 (R) | https://otexts.com/fpp3/ | yes | Canonical |
| FPP — Pythonic Way | https://otexts.com/fpppy/ | yes | **Backbone** |
| StatsForecast — getting started / CV / conformal | https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_short.html · https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html · https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html | yes | |
| MLForecast walkthrough | https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/end_to_end_walkthrough.html | yes | LightGBM + lags |
| utilsforecast losses | https://nixtlaverse.nixtla.io/utilsforecast/losses.html | yes | MASE, sMAPE, pinball |
| HierarchicalForecast | https://github.com/Nixtla/hierarchicalforecast | yes | |
| skforecast backtesting | https://skforecast.org/latest/user_guides/backtesting | yes | |
| sktime / Darts | https://www.sktime.net/ · https://unit8co.github.io/darts/ | yes | |
| Kaggle Learn Time Series | https://www.kaggle.com/learn/time-series | yes | Warm-up only |
| M5 accuracy paper | https://doi.org/10.1016/j.ijforecast.2021.11.013 | open access | LightGBM wins |
| Chronos / Chronos-2 | https://github.com/amazon-science/chronos-forecasting · https://arxiv.org/abs/2510.15821 · https://huggingface.co/amazon/chronos-2 | Apache-2.0 | Zero-shot, covariates |
| TimesFM | https://github.com/google-research/timesfm · https://arxiv.org/abs/2310.10688 | ≤2.5 Apache; 3.0 non-commercial | |
| Moirai 2.0 / TimeGPT | https://github.com/SalesforceAIResearch/uni2ts · https://github.com/Nixtla/nixtla | open / API | |
| GIFT-Eval | https://arxiv.org/abs/2410.10393 · https://huggingface.co/spaces/Salesforce/GIFT-Eval | yes | FM vs baselines, leakage |
| Prophet + Nixtla ARIMA-vs-Prophet benchmark | https://facebook.github.io/prophet/ · https://github.com/Nixtla/statsforecast/tree/main/experiments/arima_prophet_adapter | yes | AutoARIMA ~15–17 % more accurate, 37× faster on >100k series |

## Datasets
| Dataset | URL | License | Use |
|---|---|---|---|
| Inside Airbnb Berlin | https://insideairbnb.com/get-the-data/ · https://insideairbnb.com/data-policies/ | CC BY 4.0 | Red thread: listings, reviews (DE/EN), calendar |
| Berlin bike counters | https://daten.berlin.de/datensaetze/radzahldaten-in-berlin | dl-de/zero-2.0 | Hourly multi-series TS |
| SMARD | https://www.smard.de/en/downloadcenter/download-market-data/ | CC BY 4.0 ("Bundesnetzagentur \| SMARD.de") | Load, generation, prices |
| DWD Climate Data Center | https://opendata.dwd.de/climate_environment/CDC/ | Free w/ attribution | Weather covariates |
| Berlin Open Data | https://daten.berlin.de/ | mostly dl-de | Various |
| 10kGNAD | https://github.com/tblock/10kGNAD · https://huggingface.co/datasets/community-datasets/gnad10 | CC BY-NC-SA 4.0 | German news classification |
| GermEval 2017 ABSA (Deutsche Bahn) | https://huggingface.co/datasets/uhhlt/GermEval2017 | research use | Aspect sentiment |
| Multilingual Amazon Reviews | https://huggingface.co/datasets/mteb/amazon_reviews_multi | unclear — teaching only | DE vs EN sentiment |
| GermanQuAD | https://huggingface.co/datasets/deepset/germanquad | CC BY 4.0 | RAG eval set |
