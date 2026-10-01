# Choosing and designing charts

A chart is an argument made with position, length and colour. This page covers how to choose a chart for a question, what perception research says about reading charts, and how to use colour, labels and annotation so that every reader, including readers with a colour vision deficiency, sees the point. The code uses the three plotting libraries of the course: **matplotlib** (the base library, full control), **seaborn** (statistical charts from a DataFrame) and **Plotly** (interactive charts in the browser). The page ends with a chart critique, which is the practice task of the first block.

```mermaid
flowchart TD
    Q{"What is the question?"} -->|"which values occur?"| D["distribution:<br/>histogram, box plot"]
    Q -->|"which group is larger?"| C["comparison:<br/>bars from zero, dot plot"]
    Q -->|"do two numbers move together?"| R["relationship:<br/>scatter plot"]
    Q -->|"how did it change?"| T["change over time:<br/>line chart"]
    Q -->|"what share of a whole?"| P["composition:<br/>stacked bar; pie only for 2–3 parts"]
    D & C & R & T & P --> L["then: library<br/>static report: matplotlib/seaborn<br/>exploration or dashboard: Plotly"]
```

## Choosing a chart for the question

### Concept

Start from the question, not from the chart type. Four questions cover most analyses:

- **Distribution**: which values occur and how often? Histogram, density plot, box plot.
- **Comparison** of a quantity between groups: bars that start at zero, or dots.
- **Relationship** between two numeric variables: scatter plot.
- **Change over time**: line chart with time on the horizontal axis.

Worked example: "Has the share of negative reviews changed since 2014?" is a change-over-time question about one share per year, so a line chart with one point per year answers it. A pie chart per year would force the reader to compare angles across eight pies.

### Why it matters

The same data can answer different questions. A chart that fits the question makes the answer visible in seconds; a poorly chosen one hides it. Exploratory charts are also the first data check: gaps and spikes show up before any model is fitted.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["n_words"] = reviews["text"].str.split().str.len()

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
sns.histplot(reviews, x="n_words", log_scale=True, ax=axes[0, 0])        # distribution
share_neg = reviews.groupby("verified_purchase")["rating"].apply(lambda r: (r <= 2).mean())
share_neg.plot.bar(ax=axes[0, 1], ylabel="share of 1–2 star reviews")   # comparison
sns.scatterplot(reviews.sample(3000, random_state=1), x="n_words", y="helpful_vote",
                alpha=0.3, ax=axes[1, 0]).set(xscale="log", yscale="symlog")  # relationship
monthly = reviews.set_index("date").resample("MS").size()
monthly.plot(ax=axes[1, 1], ylabel="reviews per month")                 # change over time

print(share_neg.round(3).to_dict())      # {False: 0.179, True: 0.194}
print(monthly.idxmax().date(), monthly.max())   # 2020-01-01 884
```

### In practice

- The Financial Times *Visual Vocabulary* poster organises chart types by exactly these questions (deviation, correlation, ranking, distribution, change over time, part-to-whole).
- The UK Government Analysis Function publishes guidance on choosing charts for official statistics.
- Epidemic dashboards during COVID-19 (for example by the Robert Koch Institute and Our World in Data) showed daily cases as lines over time and compared regions with small multiples.

> [!TIP]
> Write the question as a sentence above the chart before you code it. If you cannot, you are not ready to choose a chart.

## Perception and the data–ink ratio

### Concept

A chart **encodes** numbers as visual properties: position, length, angle, area, colour. Cleveland and McGill (1984) measured how accurately people decode them. **Position on a common scale** is read most accurately, then length, then angle and area; colour saturation is least accurate. A bar chart uses position and length; a pie chart uses angle and area.

```mermaid
flowchart LR
    A["position on a<br/>common scale"] --> B["length"] --> C["angle, slope"] --> D["area"] --> E["colour<br/>saturation"]
    style A fill:#0072B2,color:#fff
    style E fill:#eeeeee
```

Tufte's **data–ink ratio** is the share of ink that shows data. Gridlines, borders, shadows and 3D effects that carry no information (**chartjunk**) should be reduced, and values can often be labelled directly instead of in a legend. Wilke's **principle of proportional ink**: the area of a shaded region must be proportional to the value it represents. This is why bars start at zero.

### Why it matters

Readers decide from what they see, not from the underlying table. Encodings that are decoded inaccurately lead to wrong comparisons, and clutter slows the reader down.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
share = reviews["rating"].value_counts(normalize=True).sort_index()
print(share.round(3).to_dict())   # {1: 0.133, 2: 0.059, 3: 0.075, 4: 0.118, 5: 0.615}

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
left.pie(share, labels=share.index)                         # angles and areas: hard to compare
right.barh(share.index.astype(str), share, color="0.35")    # positions on a common scale
right.bar_label(right.containers[0], labels=[f"{v:.0%}" for v in share], padding=3)
right.spines[["top", "right"]].set_visible(False)           # remove non-data ink
right.set(xlabel="share of reviews", ylabel="stars", xticks=[])
right.invert_yaxis()                                        # 1 star at the top
```

### In practice

- The UK Government Analysis Function guidance recommends bar charts instead of pie charts when categories are similar in size.
- Clinical trial reports use forest plots, which place every effect estimate on a common scale.
- Survey reports usually show agreement scales as stacked or diverging bars rather than as a series of pies.

> [!WARNING]
> **3D charts and dual axes.** A 3D bar chart distorts length through perspective. A chart with two y-axes lets the author choose the scales so that any two lines appear to cross. Use two charts side by side instead.

## matplotlib: figures, axes, labels and annotation

### Concept

matplotlib is the base plotting library of Python; pandas and seaborn draw with it. A **Figure** is the whole canvas; an **Axes** is one plot inside it, with its own x- and y-axis. `fig, ax = plt.subplots()` creates both, and all drawing is done with methods of `ax` (`ax.bar`, `ax.set`, `ax.annotate`). This **object-oriented interface** says explicitly which plot is changed.

Every chart for a reader needs axis labels with units, a readable number format and a title. A title that states the finding ("13 % of reviews give a single star") tells the reader what to look for. `ax.annotate` points to a data point with text and an arrow; one highlighted colour draws attention to the category discussed.

### Why it matters

Charts in reports are read without the analyst present. Labels, a finding title and a targeted annotation make them self-explanatory, and the explicit interface makes figures reproducible.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
counts = reviews["rating"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 3.5))                 # Figure = canvas, Axes = one plot
bars = ax.bar(counts.index, counts, color="0.35", width=0.7)
bars[0].set_color("#D55E00")                              # highlight the category discussed
ax.set(xlabel="Star rating", ylabel="Number of reviews",
       title="13 % of reviews give a single star")
ax.annotate(f"{counts[1]:,} one-star reviews", xy=(1, counts[1]), xytext=(1.5, 20000),
            arrowprops={"arrowstyle": "->"})
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter("{x:,.0f}")                  # 30,000 instead of 30000
fig.savefig("ratings.png", dpi=200, bbox_inches="tight")  # fixed size and resolution
print(type(fig).__name__, type(ax).__name__)             # Figure Axes
print(counts[1], round(counts[1] / counts.sum(), 3))     # 6669 0.133
```

### In practice

- Scientific journals require labelled axes with units and often vector formats (PDF, SVG), which `fig.savefig` produces.
- Data journalism annotates charts to guide readers, for example by marking a policy change on a time axis.
- Analysis reports generated from notebooks save matplotlib figures at a fixed size so that they look the same in every version of the report.

> [!TIP]
> Use the implicit `plt.plot(...)` style only for a quick look. As soon as there is more than one panel, switch to `fig, ax = plt.subplots(...)`.

## Colour and accessibility

### Concept

Colour has three jobs:

- a **qualitative** palette distinguishes unordered categories with clearly different hues;
- a **sequential** palette (light to dark, for example viridis) shows ordered magnitudes;
- a **diverging** palette uses two hues around a neutral midpoint for values above and below a reference.

About 8 % of men and 0.5 % of women of Northern European descent have a red–green colour vision deficiency. **Colour-blind-safe** palettes such as Okabe–Ito (seaborn's `"colorblind"` palette is similar) and the viridis colour maps stay distinguishable for them. Colour should never be the only carrier of meaning: add direct labels, position, marker shape or line style, and keep one fixed colour per category across all charts.

### Why it matters

A chart that part of the audience cannot read fails its purpose. Accessibility rules apply to many public-sector publications, and consistent colours across a dashboard let readers learn the mapping once.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print(sns.color_palette("colorblind").as_hex()[:3])   # ['#0173b2', '#de8f05', '#029e73']
OKABE_ITO = {"neg": "#D55E00", "neu": "#999999", "pos": "#0072B2"}   # one fixed colour per label

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["year"] = reviews["date"].dt.year
share = pd.crosstab(reviews["year"], reviews["label"], normalize="index").loc[2014:]
print(share.loc[[2014, 2021]].round(3))
# label    neg    neu    pos
# year
# 2014   0.159  0.085  0.756
# 2021   0.235  0.075  0.690

fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
share[["neg", "neu", "pos"]].plot.bar(stacked=True, ax=ax, width=0.8, color=OKABE_ITO,
                                      edgecolor="white", legend=False)
for label, y in [("negative", 0.1), ("neutral", 0.25), ("positive", 0.6)]:
    ax.text(7.6, y, label, va="center")              # direct labels instead of a legend
ax.tick_params(axis="x", rotation=0)
ax.set(xlabel="", ylabel="share of reviews",
       title="Negative reviews rose from 16 % (2014) to 23.5 % (2021)")
```

### In practice

- The Web Content Accessibility Guidelines (WCAG 2.1, success criterion 1.4.1) require that colour is not the only visual means of conveying information.
- matplotlib replaced the rainbow colour map "jet" by viridis as its default in version 2.0 (2017), because jet distorts perceived differences.
- Weather services use sequential palettes for precipitation and diverging palettes for temperature anomalies.

> [!CAUTION]
> **Rainbow colour maps** (jet, rainbow) create apparent boundaries where the data change smoothly and are unreadable in greyscale. Use viridis, cividis or a single-hue ramp for magnitudes. Workbooks [06](../workbooks/06-seaborn-color-palettes.ipynb) and [07](../workbooks/07-matplotlib-colormaps.ipynb) explain why.

## seaborn: statistical plots and small multiples

### Concept

**seaborn** draws statistical charts from a DataFrame in **long format** (one row per observation). You name the columns for x, y, colour (`hue`) and panels (`col`, `row`). Figure-level functions such as `displot`, `relplot` and `catplot` return a grid of Axes.

**Small multiples** are a series of small charts with the same axes, one per group. They avoid overlapping lines or histograms and let the eye compare shapes across panels. Shared axes make panels comparable; unshared axes (`sharey=False`) show the shape within each panel but must be pointed out to the reader.

### Why it matters

Comparing groups is the most frequent analytical task. Small multiples scale to many groups, where colour alone fails beyond four or five categories.

### How it works in Python

```python
import pandas as pd
import seaborn as sns

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["n_words"] = reviews["text"].str.split().str.len().clip(lower=1)

g = sns.displot(reviews, x="n_words", col="label", col_order=["neg", "neu", "pos"],
                log_scale=True, height=2.8, aspect=1.1, color="0.35")   # one panel per label
g.set_axis_labels("words per review", "reviews")
print(g.axes.shape)                                                   # (1, 3)
print(reviews.groupby("label")["n_words"].median().to_dict())       # {'neg': 23.0, 'neu': 26.0, 'pos': 19.0}

monthly = (reviews.set_index("date").groupby("verified_purchase")
           .resample("MS").size().rename("n").reset_index())
sns.relplot(monthly, x="date", y="n", col="verified_purchase", kind="line",
            height=2.8, aspect=1.6, color="0.35", facet_kws={"sharey": False})
```

### In practice

- Public health reports show infection curves per region as small multiples.
- Climate science uses panels of temperature anomalies by month or region.
- Product analytics compares conversion funnels per market in a grid of identical charts.

> [!NOTE]
> The seaborn tutorials in workbooks [04](../workbooks/04-seaborn-distributions.ipynb) and [05](../workbooks/05-seaborn-categorical.ipynb) cover histograms, kernel density estimates, box plots, violin plots and point plots with confidence intervals.

## Interactive charts with Plotly Express

### Concept

**Plotly** draws charts in the browser. **Plotly Express** (`plotly.express as px`) creates a complete figure in one call from a long-format DataFrame, with arguments similar to seaborn: `x`, `y`, `color`, `facet_col`. The reader can hover to read exact values, zoom and hide series by clicking the legend.

A figure consists of **traces** (one per series, in `fig.data`) and a **layout** (axes, title, legend). `fig.update_layout` changes the layout; `fig.write_html` saves a self-contained file that anyone can open in a browser without Python.

### Why it matters

Interactive charts let readers explore detail without the analyst producing dozens of static charts. They are the building blocks of dashboards, including the Streamlit app on [page 4](04-correlation-and-communication.md#communicating-findings-a-short-report-or-a-streamlit-dashboard).

### How it works in Python

```python
import pandas as pd
import plotly.express as px

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
OKABE_ITO = {"neg": "#D55E00", "neu": "#999999", "pos": "#0072B2"}

by_label = (reviews.set_index("date").groupby("label").resample("MS").size()
            .rename("reviews").reset_index().query("date >= '2015-01-01'"))
fig = px.line(by_label, x="date", y="reviews", color="label",
              color_discrete_map=OKABE_ITO, category_orders={"label": ["neg", "neu", "pos"]},
              labels={"date": "", "reviews": "reviews per month"},
              title="Monthly review volume by sentiment label")
fig.update_layout(hovermode="x unified")
print(len(fig.data), [trace.name for trace in fig.data])   # 3 ['neg', 'neu', 'pos']
fig.write_html("reviews_per_month.html", include_plotlyjs="cdn")   # open in a browser
```

### In practice

- Statistical offices such as Eurostat publish interactive charts so that users can read the value for their own country.
- Our World in Data publishes every chart as an interactive graphic with a downloadable table.
- Research groups share interactive HTML figures as supplementary material.

> [!WARNING]
> Interactivity is not a substitute for a clear static view. Hover text is invisible in a printed report or a screenshot, so the default view must already show the finding.

## Critique and improve a chart

### Concept

A chart critique asks four questions:

1. **Question**: which comparison should the reader make?
2. **Encoding**: is that comparison shown by position or length on a common scale?
3. **Accuracy**: do the axes start where they should; are units and population stated?
4. **Clarity**: is there anything the reader must decode that could be labelled directly?

![A poorly designed bar chart with a truncated axis and rainbow colours next to a line chart of the share of negative reviews with a finding as its title](figures/good-vs-poor-chart.png)

The left chart truncates the axis at 3.85, so a drop of 0.3 stars looks like a collapse; the rainbow colours encode nothing; the title says nothing. The right chart shows the measure a product manager cares about (share of 1–2 star reviews), starts at zero, uses one colour and states the finding in the title.

### Why it matters

Most analyses reach decision makers as one chart and a few sentences. Spotting and repairing misleading charts, including one's own, is a core professional skill.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["year"] = reviews["date"].dt.year
yearly = reviews.query("year >= 2014").groupby("year")["rating"].agg(
    avg="mean", share_neg=lambda r: (r <= 2).mean())
print(yearly.loc[[2014, 2021]].round(3))
#         avg  share_neg
# year
# 2014  4.099      0.159
# 2021  3.878      0.235

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(yearly.index, yearly["share_neg"], marker="o", color="#D55E00")
ax.set_ylim(0, 0.3)                                   # full axis from zero
ax.yaxis.set_major_formatter("{x:.0%}")
ax.set(ylabel="share of 1–2 star reviews",
       title="Negative reviews rose from 16 % (2014) to 23.5 % (2021)")
ax.spines[["top", "right"]].set_visible(False)
```

A line chart need not start at zero when the reader compares changes rather than magnitudes, but then the axis label must make the range obvious. Bars must always start at zero.

### In practice

- The UK Government Analysis Function advises against breaking the numerical axis of bar charts because it distorts the proportions between bars.
- The Financial Times' *Chart Doctor* column publishes before-and-after redesigns of published charts.
- The CONSORT guidelines for reporting clinical trials require absolute numbers alongside relative effects, for the same reason that bars start at zero.

> [!IMPORTANT]
> **Practice (block 1).** Plot the rating distribution and the number of reviews per month for the sample. Then take the poor chart above (code in [`figures/make_figures.py`](figures/make_figures.py)) and improve it with the four critique questions. The case-study notebook [18-case-study-verified-purchases-and-helpful-votes.ipynb](../workbooks/18-case-study-verified-purchases-and-helpful-votes.ipynb) starts with this task.

## Check your understanding

1. Which chart would you choose for "Do verified buyers write longer reviews?" and why?
2. Rank position, angle, area and colour saturation by how accurately people decode them.
3. Name two ways to make a chart readable without relying on colour.
4. When is it acceptable for a y-axis not to start at zero?
5. What is lost when a dashboard chart is printed, and how do you design for it?

## Further reading

- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. Free online: <https://clauswilke.com/dataviz/>
- Healy, K. (2018). *Data Visualization: A Practical Introduction*, chapter 1 "Look at data". Princeton University Press. <https://socviz.co/lookatdata.html>
- Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531–554. <https://doi.org/10.1080/01621459.1984.10478080>
- Government Analysis Function (2023). *Data visualisation: charts*. UK Government. <https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-charts/>
