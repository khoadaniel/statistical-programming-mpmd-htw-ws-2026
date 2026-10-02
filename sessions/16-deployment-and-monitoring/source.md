# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks
| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/02-evidently-data-drift-report.ipynb](workbooks/02-evidently-data-drift-report.ipynb) | Data drift and data summary reports (Evidently `DataDriftPreset`, `DataSummaryPreset`) on the Adult dataset | [evidentlyai/evidently](https://raw.githubusercontent.com/evidentlyai/evidently/main/examples/classic_ml_validation.ipynb) | [Apache-2.0](https://github.com/evidentlyai/evidently/blob/main/LICENSE) | 2026-10-01 | Renamed from `examples/classic_ml_validation.ipynb`; content unchanged |

FastAPI documentation (MIT), Made With ML (MIT), Streamlit documentation and the Docker documentation are referenced as links only; no files were copied.

## Own material
| File | Covers | Licence |
|---|---|---|
| [theory/01-from-notebook-to-service.md](theory/01-from-notebook-to-service.md) | Block 1 theory | CC-BY-4.0, course team |
| [theory/02-containers-and-continuous-delivery.md](theory/02-containers-and-continuous-delivery.md) | Block 2 theory | CC-BY-4.0, course team |
| [theory/03-monitoring-and-maintenance.md](theory/03-monitoring-and-maintenance.md) | Block 3 theory | CC-BY-4.0, course team |
| [theory/figures/make_figures.py](theory/figures/make_figures.py), `drift-histogram.png`, `psi-explained.png` | Figures from the EBTI case-study data (issuing-country drift, chapter label shift with the released 2024 labels, PSI per country) | CC-BY-4.0, course team |
| [workbooks/01-case-study-drift-and-retraining.ipynb](workbooks/01-case-study-drift-and-retraining.ipynb) | Practice block 3: drift, label shift, retraining candidates, final submission (L3) | Author: course team, licence CC-BY-4.0 |
| [workbooks/make_feedback_2024.py](workbooks/make_feedback_2024.py) | Lecturer script: releases the 2024 labels (`--out` for another path) | Author: course team, licence MIT |
| [workspace/](workspace/README.md) | Package `tariff_service`, tests, Dockerfile, workflows, model card template, dashboard | Code: MIT; text: CC-BY-4.0, course team |

Parts of the theory pages reuse the earlier course notes (course team).

## Citations
- Breck, E., Cai, S., Nielsen, E., Salib, M. and Sculley, D. (2017). The ML test score: A rubric for ML production readiness and technical debt reduction. *IEEE International Conference on Big Data 2017*. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- Dong, E., Du, H. and Gardner, L. (2020). An interactive web-based dashboard to track COVID-19 in real time. *The Lancet Infectious Diseases* 20(5), 533–534. https://doi.org/10.1016/S1473-3099(20)30120-1
- Heaven, W. D. (2020). Our weird behavior during the pandemic is messing with AI models. *MIT Technology Review*, 11 May 2020. https://www.technologyreview.com/2020/05/11/1001563/covid-pandemic-broken-ai-machine-learning-amazon-retail-fraud-humans-in-the-loop/
- European Commission. *European Binding Tariff Information (EBTI)* database, full export (case-study data). Reuse with acknowledgement under Commission Decision 2011/833/EU. https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en
- European Commission. *Harmonized System*. https://taxation-customs.ec.europa.eu/customs/common-customs-tariff-cct/tariff-classification-goods/harmonized-system_en (linked only)
- datasets/harmonized-system: *Harmonized System nomenclature (HS 2022)*, ODC-PDDL. https://github.com/datasets/harmonized-system
- Kästner, C. *Machine Learning in Production: From Models to Products*. Open textbook. https://mlip-cmu.github.io/book/
- Lazer, D., Kennedy, R., King, G. and Vespignani, A. (2014). The parable of Google Flu: Traps in big data analysis. *Science* 343(6176), 1203–1205. https://doi.org/10.1126/science.1248506
- Lipton, Z. C., Wang, Y.-X. and Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML 2018*. https://arxiv.org/abs/1802.03916
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D. and Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\*)*, 220–229. https://doi.org/10.1145/3287560.3287596
- Mohandas, G. *Made With ML*. https://madewithml.com/ (MIT licence)
- Ribeiro, M. T., Wu, T., Guestrin, C. and Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. *ACL 2020*. https://arxiv.org/abs/2005.04118
- Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*. https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Wiggins, A. (2017). *The Twelve-Factor App*. https://12factor.net/
- FastAPI documentation (Ramírez, S. and contributors), MIT licence. https://fastapi.tiangolo.com/
- skops documentation. https://skops.readthedocs.io/
