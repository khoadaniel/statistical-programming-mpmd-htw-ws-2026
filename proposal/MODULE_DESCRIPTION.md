# Statistical Programming: Data Analytics and Machine Learning with Python

| Module information | Details |
| :--- | :--- |
| **Duration** | 1 semester |
| **Status of the module** | Elective module |
| **ECTS** | 5 |
| **Weekly semester hours** | 4 |

## Learning outcomes and competencies

Building on the foundational knowledge acquired in the first semester, this module develops students' practical skills in data analytics and machine learning using Python. Students work through the full process of a data project: accessing and preparing data, exploring and interpreting patterns, building and evaluating models, and deploying and monitoring a model as an application.

The module combines statistical programming with professional software development practices. It prepares students to analyse data independently, make justified methodological choices and contribute effectively to programming teams. No prior knowledge of machine learning or natural language processing is assumed.

A substantial part of the module is dedicated to live coding, hands-on exercises and practical case studies using public, real-world datasets. Students work in Jupyter notebooks and structured Python projects, using tools such as SQL, pandas, Polars and scikit-learn. Applications include analysis of Berlin accommodation data, customer churn prediction, time series forecasting and multilingual text classification. A team project on customs classification brings together data preparation, model development, evaluation and deployment.

By the end of the module, students will be able to:

- Write well-structured, documented and tested Python code, and collaborate using Git, code review and continuous integration.
- Load, query and prepare data reproducibly from relational databases, APIs and large files, and assess and document data quality.
- Apply suitable statistical methods and visualisations to explore data, compare groups and communicate findings with their uncertainty and limitations.
- Build, validate and tune supervised and unsupervised machine learning models, using appropriate metrics and preventing data leakage (information reaching a model during training that it would not have at prediction time).
- Develop time series forecasts and assess their performance through backtesting and prediction intervals.
- Build text classifiers, use embeddings and language models, and evaluate retrieval-augmented generation and simple agents against baselines.
- Serve a tested model through an API in a container, explain basic cloud deployment architecture, and plan model monitoring and maintenance.
- Translate a practical problem into a data task, justify technical decisions, and present and defend project results and individual contributions.

## Topics

### Python programming and collaborative development

- Reproducible Python environments, Jupyter notebooks, scripts, modules and packages.
- Object-oriented programming, error handling, documentation and automated testing.
- Web APIs for accessing data and serving applications.
- Teamwork with Git and GitHub, pull requests, code review and continuous integration.

### Data access and preparation

- Relational databases and SQL: filtering, joins, aggregation and window functions.
- Data processing with pandas and Polars, including large datasets.
- Data quality checks, missing value imputation, outlier detection and variable transformations.
- Reproducible preprocessing pipelines and documentation of data decisions.

### Exploratory analysis and statistical methods

- Descriptive statistics, data visualisation and communication through reports and dashboards.
- Statistical tests, contingency tables, correlation, confidence intervals and effect sizes.
- Practical applications of statistical comparisons, including A/B testing.

### Machine learning and forecasting

- Linear, logistic and robust regression; regularisation, underfitting and overfitting.
- Training, validation and test sets; k-fold, grouped and time-based cross-validation; bootstrap methods and prevention of data leakage.
- Hyperparameter tuning with grid and randomised search.
- Classification, evaluation metrics, decision thresholds and probability calibration.
- Feature engineering and methods for handling imbalanced data.
- Decision trees, random forests and gradient boosting with XGBoost, LightGBM and CatBoost; model interpretation.
- Clustering, dimensionality reduction, visualisation of high-dimensional data and anomaly detection.
- Time series forecasting with baseline methods, exponential smoothing and machine learning; backtesting and prediction intervals.

### Natural language processing and language models

- Text preprocessing, bag-of-words, TF-IDF and multilingual text classification.
- Transformer concepts, embeddings, semantic search and generative language models.
- Retrieval-augmented generation (RAG) and simple agents with tool calling.
- Evaluation against baselines, error analysis, and consideration of quality, cost, latency and risks.

### Deployment, monitoring and maintenance

- Moving from notebooks to tested model services with FastAPI and Docker.
- Continuous delivery and an introduction to cloud architecture for data applications.
- Monitoring data drift and model performance, retraining decisions and model documentation.

## Requirements

| Requirement | Details |
| :--- | :--- |
| **Necessary requirements** | None |
| **Recommended requirements** | Foundational knowledge of Python and statistics, as covered in the first-semester module Foundations of Data Analytics and Statistical Programming. |

## Method of examination

*Applicable provisions: §§ 9–14 RStPO.*

- **Final project: 100 points**, assessed individually through an individual presentation (20 points) and an individual paper with the student's own code (80 points).

Students work on the project in teams, but each student is responsible for one area of it and is graded individually, on their own presentation, paper and code; no grade is given on the basis of group work. Assessment considers problem definition, data preparation, methodological choices, validation, implementation, communication of results and reflection on the group work. Weekly exercises are not graded, and the model's evaluation score is not directly converted into marks.
