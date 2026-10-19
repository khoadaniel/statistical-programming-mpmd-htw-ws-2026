# Session 16 · Extras and notebook provenance

These are the downloaded source notebooks behind the two [block notebooks](../../README.md), kept as unchanged reference copies, followed by the provenance of the data, figures and helper files of the session.

## Downloaded notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [01-skops-text-classification-model-card.ipynb](01-skops-text-classification-model-card.ipynb) | skops: train a text classifier (20 newsgroups), save it, and build a model card with metrics, a confusion-matrix plot and a classification report | 1, 2 | [skops documentation, *Text Classification with scikit-learn*](https://skops.readthedocs.io/en/stable/_downloads/6e78fb4720162fcd3ed20928cbf19d7c/plot_text_classification.ipynb) (sphinx-gallery notebook of `examples/plot_text_classification.py`, skops 0.16 docs) | [MIT](https://github.com/skops-dev/skops/blob/main/LICENSE) | 2026-10-14 | Renamed from `plot_text_classification.ipynb`; content unchanged. Curated into `02-lecture-architecture-monitoring-and-maintenance.ipynb`, section "Documenting a model in a model card" (the `card.Card`, `add`, `add_metrics`, `add_table` pattern, applied to the tariff model) |
| [02-evidently-data-drift-report.ipynb](02-evidently-data-drift-report.ipynb) | Evidently: data drift and data summary reports (`DataDriftPreset`, `DataSummaryPreset`) on the Adult dataset; downloads the data from OpenML. Needs its own environment (command in the session README) | 2 | [evidentlyai/evidently](https://raw.githubusercontent.com/evidentlyai/evidently/main/examples/classic_ml_validation.ipynb) | [Apache-2.0](https://github.com/evidentlyai/evidently/blob/main/LICENSE) | 2026-10-01 | Renamed from `examples/classic_ml_validation.ipynb`; content unchanged. Its idea (a per-column drift report) is rebuilt on the case-study data in `02-lecture-...`, section "Other drift measures and drift reports", with the defaults taken from the Evidently documentation (linked, not copied) |

## Authors and licences

- **skops developers (Hugging Face and contributors):** extra 01, MIT. Code adapted in Block 2 with an attribution line.
- **Evidently AI:** extra 02, Apache-2.0.
- **Course team:** the block notebooks, `weather.py`, the figures and the workspace. Notebooks and text CC BY 4.0; code MIT.

FastAPI documentation (MIT), HTTPX documentation, MDN Web Docs, Made With ML (MIT), the Evidently, Streamlit, Docker, GitHub Actions, SQLite and AWS documentation are referenced as links only; no files were copied.

## Course-written files of the session

| File | Content | Licence |
|---|---|---|
| [../01-lecture-web-apis-and-a-released-service.ipynb](../01-lecture-web-apis-and-a-released-service.ipynb) | Block 1: how web APIs work, requesting data with httpx, from notebook to service, a prediction API with FastAPI and pydantic and its tests, containers with Docker, continuous delivery with GitHub Actions, worked example and exercises | CC BY 4.0 |
| [../02-lecture-architecture-monitoring-and-maintenance.ipynb](../02-lecture-architecture-monitoring-and-maintenance.ipynb) | Block 2: system design, cloud services on AWS, data drift (KS test, PSI), prediction drift and label shift, retraining triggers and versioning, a model card, worked example (version 2 and the reported score) and exercises | CC BY 4.0 |
| [../weather.py](../weather.py) | Open-Meteo client (`WeatherClient`, `DailyWeather`, `parse_daily`, `yearly_chunks`) imported by Block 1 and exercise 1.7. | MIT |
| [../../figures/make_figures.py](../../figures/make_figures.py) with `drift-histogram.png` and `psi-explained.png` | Figures from the EBTI case-study data (issuing-country drift between the training decisions of 2023–2025 and the requests of 2026, chapter label shift from 2025 to 2026 with the headings of the training decisions of 2026, PSI per country), shown in Block 2. Regenerate with `uv run python sessions/16-deployment-and-monitoring/figures/make_figures.py` | CC BY 4.0 |
| [../../workspace/](../../workspace/README.md) | Package `tariff_service`, tests, Dockerfile, `compose.yaml`, workflows, model card template, dashboard and its Dockerfile | Code: MIT; text: CC BY 4.0 |

## Data

| Data | Used in | Source | Licence |
|---|---|---|---|
| Daily Berlin weather (temperature, precipitation, sunshine), requested live | `weather.py`, Block 1, exercises 1.1 and 1.7 (printed and mocked responses) | [Open-Meteo](https://open-meteo.com/) archive and forecast APIs | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); free API for non-commercial use (fewer than 10,000 calls per day) |
| Stored daily weather (`case-study/data/airbnb/weather_daily.parquet`) | Block 1 (comparison with the live download, offline fallback) | Open-Meteo, fetched by the lecturer's preparation script; part of `airbnb-v1.zip` on the course drive | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Live departures at S+U Alexanderplatz | Block 1, section "Requesting data with httpx" (nested JSON) | [v6.vbb.transport.rest](https://v6.vbb.transport.rest/), community-run, data of VBB | Not stored; requested live only |
| Customs decisions and nomenclature texts | Blocks 1 and 2, exercises, workspace | European Commission, *European Binding Tariff Information (EBTI)* database, full export; reuse with acknowledgement under Commission Decision 2011/833/EU. Nomenclature texts: European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN.xlsx`), same reuse conditions | Decision 2011/833/EU |

## References cited in the block notebooks

- Breck, E., Cai, S., Nielsen, E., Salib, M. and Sculley, D. (2017). The ML test score: a rubric for ML production readiness and technical debt reduction. *IEEE Big Data 2017*. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- Heaven, W. D. (2020). Our weird behavior during the pandemic is messing with AI models. *MIT Technology Review*, 11 May 2020.
- Kästner, C. *Machine Learning in Production: From Models to Products*. https://mlip-cmu.github.io/book/
- Lipton, Z. C., Wang, Y.-X. and Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML 2018*. https://arxiv.org/abs/1802.03916
- Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\* 2019*, 220–229. https://doi.org/10.1145/3287560.3287596
- Rabanser, S., Günnemann, S. and Lipton, Z. C. (2019). Failing loudly: an empirical study of methods for detecting dataset shift. *NeurIPS 2019*. https://arxiv.org/abs/1810.11953
- Ribeiro, M. T., Wu, T., Guestrin, C. and Singh, S. (2020). Beyond accuracy: behavioral testing of NLP models with CheckList. *ACL 2020*. https://arxiv.org/abs/2005.04118
- Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Wiggins, A. (2017). *The Twelve-Factor App*. https://12factor.net/
- Yurdakul, B. (2018). *Statistical properties of population stability index*. PhD thesis, Western Michigan University.
