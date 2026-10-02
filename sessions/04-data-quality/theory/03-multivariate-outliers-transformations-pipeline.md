# Multivariate outliers, transformations and a documented cleaning pipeline

This page covers the third block. Some observations are unusual only in the *combination* of their values; the **Mahalanobis distance** finds them. Skewed variables are made more symmetric with the logarithm, Box–Cox or Yeo–Johnson transformations, and variables on different scales are brought to a common scale. Finally, all decisions are combined into a **cleaning pipeline** that is code, runs in one go and writes a log of every decision. The practice task produces the cleaned table of BTI decisions with a log of every cleaning decision ([workbook 14](../workbooks/14-case-study-cleaned-decision-table.ipynb)). Model-based outlier detection (Isolation Forest, local outlier factor) follows in Session 11.

```mermaid
flowchart LR
    RAW["train.parquet<br/>309,529 rows"] --> S1["1 remove<br/>template texts"]
    S1 --> S2["2-3 placeholder dates,<br/>normalise text"]
    S2 --> S3["4-7 flag codes,<br/>duplicates, languages"]
    S3 --> S4["8-9 indicators<br/>for missing values"]
    S4 --> S5["10-12 transform,<br/>flag outliers"]
    S5 --> V{"validate"}
    V -->|pass| OUT["decisions_clean.parquet<br/>+ cleaning_log.csv"]
    V -->|fail| S1
```

## Multivariate outliers with the Mahalanobis distance

### Concept

A **multivariate outlier** is unusual in the combination of variables, not necessarily in any single one. A person who is 192 cm tall is not unusual, nor is a weight of 55 kg; a person who is both is.

The **Mahalanobis distance** measures how far a point x lies from the centre μ of the data, in units that take into account the spread of each variable and the correlation between them (the **covariance matrix** Σ):

d(x) = √((x − μ)ᵀ Σ⁻¹ (x − μ))

If the variables are uncorrelated and standardised, it is the ordinary (Euclidean) distance in standard deviations. If they are correlated, moving *along* the correlation is cheap and moving *against* it is expensive: the points of equal distance form an ellipse aligned with the data. For roughly normal data with p variables, d² follows a **chi-square distribution** with p degrees of freedom, which gives a cut-off: for p = 2 and a 99.9 % quantile, d² > 13.8.

The mean and covariance are themselves distorted by outliers (masking, as for the z-score). The **minimum covariance determinant** estimator (MCD, `sklearn.covariance.MinCovDet`, Rousseeuw 1984) estimates them from the most concentrated subset of about half the data, so that a group of outliers cannot pull the centre and stretch the ellipse towards itself.

A small example by hand with two standardised, uncorrelated variables: the point (2, 2) has Euclidean distance √8 = 2.83. If the variables have correlation 0.8, the same point lies *along* the correlation, and its Mahalanobis distance is √(8 / 1.8) = 2.11; the point (2, −2), *against* the correlation, has √(8 / 0.2) = 6.32.

![Classical and robust Mahalanobis ellipses for height and weight with a single outlier and a contaminating group](figures/mahalanobis-ellipses.png)

### Why it matters

Many data errors and many interesting cases show up only in combinations: a plausible age with an implausible diagnosis, a normal transaction amount at an unusual time, a long technical description without a single number. Univariate rules miss them. The robust version matters because real data often contain a *group* of anomalies (a batch of mis-coded records), which would otherwise widen the classical ellipse until it no longer flags them.

### How it works in Python

```python
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet

rng = np.random.default_rng(0)
height = rng.normal(172, 9, 500)                                 # cm
weight = 0.9 * height - 85 + rng.normal(0, 6, 500)               # kg, correlated with height
X = np.vstack([np.column_stack([height, weight]),
               [[192, 55]],                                       # row 500: tall and light
               np.column_stack([rng.normal(150, 3, 25), rng.normal(95, 4, 25)])])  # a contaminating group

z = (X[500] - X.mean(axis=0)) / X.std(axis=0)
print(z.round(1))                                                # [ 2.1 -1.4]: each variable alone |z| < 3

cut = chi2.ppf(0.999, df=2)                                      # 13.8
for est in [EmpiricalCovariance().fit(X), MinCovDet(random_state=0).fit(X)]:
    d2 = est.mahalanobis(X)                                      # squared distances
    print(f"{type(est).__name__:19s} d(row 500) = {np.sqrt(d2[500]):.1f}  "
          f"group flagged: {(d2[501:] > cut).sum()}/25  all flagged: {(d2 > cut).sum()}")
# EmpiricalCovariance d(row 500) = 3.2  group flagged: 17/25  all flagged: 17
# MinCovDet           d(row 500) = 6.4  group flagged: 25/25  all flagged: 28
```

With the classical estimate, the contaminating group pulls the centre and inflates the covariance: the tall, light person has d = 3.2, below the cut-off √13.8 = 3.7, and is not flagged, and 8 of the 25 group points also escape. With the MCD estimate, the person (d = 6.4) and the whole group are flagged, plus two ordinary points in the tails. The figure above shows the two ellipses.

On the decision data, workbook 14 computes the robust distance on two description features (Box–Cox length, log of 1 + number of digits) and flags 5,299 decisions (1.7 %), for example long descriptions without any number. A first attempt with four features, adding the number of lines and the share of upper-case letters, flagged 23.7 % of all decisions: the share of upper-case letters is bimodal, because about 7.6 % of the descriptions are written in capitals, and MCD then treats the whole minority group as outliers. The distance assumes one elliptical cloud; check that before you trust it.

### In practice

- **Industrial process monitoring** uses Hotelling's T² chart, a Mahalanobis distance from the in-control centre, to detect machine states where every sensor is within its limits but the combination is not.
- **Payment fraud detection** screens card transactions for combinations of amount, merchant type, place and time that are unusual for the cardholder; distance-based scores are one of the classical building blocks.
- **Chemometrics**: near-infrared spectroscopy calibrations in the food and pharmaceutical industries use Mahalanobis distances to detect samples that lie outside the calibration set before a prediction is trusted.

> [!WARNING]
> The Mahalanobis distance assumes one roughly elliptical cloud. With strongly skewed variables, transform them first (next section); with several clusters or many zeros, use the model-based methods of Session 11. With many variables relative to rows, the covariance estimate becomes unstable.

> [!CAUTION]
> MCD needs a covariance matrix that is not singular. A variable that is constant in more than half of the rows (for example the validity duration, which is 1,095 days for most decisions) makes MCD fail or give meaningless distances. Leave such variables out or transform them first.

## Transformations (logarithm, Box–Cox, Yeo–Johnson, scaling)

### Concept

A **transformation** applies the same function to every value of a variable. Two goals are common: making a skewed distribution more symmetric, and bringing variables to comparable scales.

**Logarithm.** log(x) turns multiplicative differences into additive ones: 10 → 100 → 1,000 become equally spaced. It compresses the long right tail of prices, incomes and text lengths. It requires x > 0; for counts with zeros, log(1 + x) (`np.log1p`) is used.

**Box–Cox** (Box and Cox, 1964) is a family of power transformations with a parameter λ, estimated from the data so that the result is as close to normal as possible:

y = (x^λ − 1) / λ for λ ≠ 0, and y = log(x) for λ = 0.

λ = 1 leaves the shape unchanged, λ = 0.5 is close to a square root, λ = 0 is the logarithm. Box–Cox requires x > 0.

**Yeo–Johnson** (Yeo and Johnson, 2000) extends the idea to zero and negative values, which suits counts and differences.

**Scaling** changes the unit but not the shape:

| Scaler | Formula | Use |
|---|---|---|
| `StandardScaler` | (x − mean) / SD | default for linear models, PCA, k-nearest neighbours |
| `MinMaxScaler` | (x − min) / (max − min) | bounded inputs, e.g. for neural networks |
| `RobustScaler` | (x − median) / IQR | data with outliers |

A worked example by hand: description lengths 10, 100 and 1,000 characters. Their mean is 370, dominated by the longest. Their base-10 logarithms are 1, 2 and 3, with mean 2, corresponding to a typical length of 10² = 100 characters (the geometric mean).

```mermaid
flowchart TD
    A{"Goal?"} -->|"common scale"| SC{"Outliers?"}
    SC -->|no| STD["StandardScaler"]
    SC -->|yes| ROB["RobustScaler"]
    A -->|"less skew"| V{"Values?"}
    V -->|"all > 0"| L["log, or Box–Cox<br/>(estimates λ)"]
    V -->|"zeros or negative"| Y["log1p for counts,<br/>or Yeo–Johnson"]
    L --> I["Interpret on the<br/>original scale"]
    Y --> I
```

### Why it matters

Many methods work better, or only as intended, with roughly symmetric variables on similar scales: linear regression assumes errors of constant spread (Session 6), distance-based methods treat one unit of each variable as equally important, and PCA (Session 11) is dominated by the variable with the largest variance. Outlier rules and the Mahalanobis distance also assume a symmetric centre. Transforming before analysing often turns "outliers" into ordinary tail values.

### How it works in Python

```python
import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
length = decisions["description"].str.len()                   # all > 0: Box-Cox is possible

transformed, lam = boxcox(length)
print(round(lam, 2), round(skew(length), 2), round(skew(np.log(length)), 2), round(skew(transformed), 2))
# 0.35 1.32 -0.75 -0.0   <- the log over-corrects (skew to the left); Box-Cox finds lambda = 0.35

# the same with scikit-learn, which stores lambda for new data
pt = PowerTransformer(method="box-cox").fit(length.to_frame())
print(pt.lambdas_.round(2))                                     # [0.35]

digits = decisions["description"].str.count(r"[0-9]").to_frame("n_digits")   # 13 % zeros: Yeo-Johnson or log1p
yj = PowerTransformer(method="yeo-johnson").fit(digits)
print(round(skew(digits["n_digits"]), 1), round(skew(np.log1p(digits["n_digits"])), 2),
      round(skew(yj.transform(digits).ravel()), 2))
# 4.4 -0.59 -0.06

X = pd.DataFrame({"n_chars": length, "n_digits": digits["n_digits"],
                  "n_lines": decisions["description"].str.count("\n") + 1})
print(StandardScaler().fit_transform(X).std(axis=0).round(2))   # [1. 1. 1.]
print(RobustScaler().fit(X).scale_)                             # IQR of each column: [468.  17.   7.]
```

![Description length before and after the Box–Cox transformation, with normal Q–Q plots](figures/box-cox-before-after.png)

After the transformation the histogram is close to symmetric and the points of the Q–Q plot lie close to the line, except in the extreme tails. Description length is a case where the plain logarithm is too strong (the skewness changes sign, from 1.32 to −0.75); Box–Cox with λ = 0.35, between a square root and a cube root, fits better. For the number of digits, with 13 % zeros, Yeo–Johnson and log1p both remove most of the skew.

### In practice

- **Economics**: wages, firm sizes and house prices are usually analysed on the log scale; a coefficient on log wages reads as an approximate percentage difference, which is the convention of the Mincer earnings equation.
- **Laboratory medicine**: reference intervals for skewed analytes, such as many hormone and enzyme concentrations, are often computed after a log or Box–Cox transformation, as described in the CLSI guideline EP28.
- **Machine-learning pipelines**: `PowerTransformer` and `StandardScaler` are fitted on training data inside a scikit-learn `Pipeline` and stored with the model (workbooks 10 and 11).

> [!WARNING]
> Fit transformations (λ, mean, SD, median, IQR) on the training data and apply them to the test data unchanged. Estimating λ on all data is a mild form of data leakage.

> [!CAUTION]
> Results on a transformed scale must be translated back for the reader. The back-transformed mean of log values is the geometric mean, not the arithmetic mean: exp(mean(log length)) is not the average length. Say which one you report.

## A documented cleaning pipeline

### Concept

A **cleaning pipeline** is a fixed sequence of steps that turns the raw table into the analysis table. It is **documented** when every step records:

- the **rule** (what was looked for),
- the **number of rows affected**,
- the **action** (remove, correct, flag, transform, impute, keep),
- the **reason** (why this action and not another).

The record of all steps is the **cleaning log**. Together with the code, it lets anyone reproduce the cleaned table from the raw data and understand every difference between them.

Typical decisions for the BTI data, with the counts from workbook 14:

| Step | Rule | Rows | Action | Reason |
|---|---|---|---|---|
| 1 | description is a database template (`SQL{...}`) | 73 | remove | says nothing about the goods |
| 2 | end date 1900-01-01 (annulled, code 55) | 510 | set to missing, flag `annulled` | a placeholder is not a date |
| 3 | Windows line breaks, no-break spaces, repeated spaces | 181,359 | correct | makes lengths comparable; line breaks kept |
| 4–5 | heading 8803 (deleted in HS 2022); CN code with 4 or 6 digits | 51 / 1,040 | flag | the label is still valid for its time |
| 6 | description identical to another decision | 23,606 | flag | renewals are real; they matter for validation (Session 7) |
| 7 | language neither official in the country nor English | 9 | flag | probably a wrong language code |
| 8–9 | keywords missing; invalidation reason missing | 1,273 / 264,254 | keep NULL, add indicators | not MCAR; structural |
| 10 | skewed length and digit counts | all | add Box–Cox / log1p columns | raw columns kept |
| 11–12 | univariate / Mahalanobis outliers | 169 / 5,299 | flag | inspect, do not delete |

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Found: check finds rows
    Found --> Error: data error?
    Found --> Genuine: genuine value?
    Error --> Removed: cannot be corrected
    Error --> Corrected: rule known
    Genuine --> Flagged: may matter later
    Genuine --> Transformed: skewed scale
    Removed --> Logged
    Corrected --> Logged
    Flagged --> Logged
    Transformed --> Logged
    Logged --> [*]
```

Good practice:

- **Raw data are never changed.** The pipeline reads raw files and writes new ones.
- **Flag rather than delete** when in doubt; a later analysis can filter on the flag.
- **One function per step**, chained with `DataFrame.pipe`, so that steps can be tested and reordered.
- **Validate the result** at the end with the rules of block 1, then save table and log together.

### Why it matters

Cleaning decisions change results. Removing duplicates changes counts; removing extreme descriptions changes average lengths; imputing a value changes every statistic computed from it. If decisions are not recorded, results cannot be reproduced or defended, and different team members clean the same data differently. A documented pipeline is also what reviewers, supervisors and future colleagues ask for first.

### How it works in Python

The pattern of workbook 14, reduced to two steps:

```python
import pandas as pd

raw = pd.read_parquet("case-study/data/train.parquet")
log = []


def record(step, rule, affected, action, reason, n_rows):
    log.append({"step": step, "rule": rule, "rows_affected": affected,
                "action": action, "reason": reason, "rows_after": n_rows})


def drop_template_descriptions(df):
    template = df["description"].str.upper().str.contains("SQL{", regex=False)
    out = df[~template]
    record("1", "description is a database template", int(template.sum()), "remove",
           "the text says nothing about the goods", len(out))
    return out


def fix_placeholder_end_date(df):
    out = df.copy()
    placeholder = out["end_date"].dt.year.eq(1900)
    out.loc[placeholder, "end_date"] = pd.NaT
    record("2", "end_date 1900-01-01 (annulled)", int(placeholder.sum()), "set to missing",
           "a placeholder is a hidden missing value, not a date", len(out))
    return out


clean = raw.pipe(drop_template_descriptions).pipe(fix_placeholder_end_date)
assert not clean["end_date"].dt.year.eq(1900).any()                # validate before saving
print(pd.DataFrame(log)[["step", "rows_affected", "action", "rows_after"]].to_string(index=False))
# step  rows_affected         action  rows_after
#    1             73         remove      309456
#    2            510 set to missing      309456
```

### In practice

- **Clinical trials** keep a *data management plan* and an audit trail: every change to a trial database is logged with who, when and why, as required by good clinical practice (ICH E6).
- The **Reinhart–Rogoff** case (2010/2013) is a frequently cited example of undocumented spreadsheet processing: a replication by Herndon, Ash and Pollin found excluded rows and a spreadsheet error that changed a widely quoted result on public debt and growth.
- **Data engineering teams** keep cleaning steps in version-controlled code (for example dbt models or Python modules) and run the validation tests after each step, so that each published table has a known lineage.

> [!IMPORTANT]
> The cleaned table can be the input of later analyses. Keep the pipeline and the log in your project repository, and regenerate the table with the code rather than copying it around.

> [!TIP]
> Write the reason as if for a colleague who disagrees: "flag renewed decisions but keep them, because each is a real decision; keep copies on one side of a validation split" can be discussed; "removed duplicates" cannot.

## Check your understanding

1. Why can a point be a Mahalanobis outlier although its z-score is below 3 in every variable?
2. Why does the robust (MCD) ellipse in the figure flag the contaminating group while the classical ellipse does not?
3. Box–Cox estimates λ = 0.35 for the description length. What does that tell you about the transformation compared with the logarithm, and why can it not be applied to the number of digits?
4. Which scaler would you choose for a variable with a few extreme values, and why?
5. For each of the following, decide between remove, correct, flag and keep, and write the log entry: (a) a decision whose description is "TEST"; (b) a description of 8,621 characters for a conveyor system; (c) a decision whose start date is 06/07/2200.

## Further reading

- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics*, 41(3), 212–223. https://doi.org/10.1080/00401706.1999.10485670
- Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211–252. https://doi.org/10.1111/j.2517-6161.1964.tb00553.x
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*, chapter 6 "Engineering numeric predictors" and chapter 8 "Handling missing data". Free online: https://feat.engineering/
- scikit-learn developers. *Preprocessing data* (user guide, section 7.3). https://scikit-learn.org/stable/modules/preprocessing.html
