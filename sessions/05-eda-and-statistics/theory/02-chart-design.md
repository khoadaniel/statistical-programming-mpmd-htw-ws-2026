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

Worked example: "Has the share of English-language decisions changed since 2017?" is a change-over-time question about one share per year, so a line chart with one point per year answers it. A pie chart per year would force the reader to compare angles across seven pies.

### Why it matters

The same data can answer different questions. A chart that fits the question makes the answer visible in seconds; a poorly chosen one hides it. Exploratory charts are also the first data check: gaps and spikes show up before any model is fitted.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
decisions["n_chars"] = decisions["description"].str.len()
decisions["n_keywords"] = decisions["keywords"].str.split(",").str.len()

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
sns.histplot(decisions, x="n_chars", log_scale=True, ax=axes[0, 0])           # distribution
median_len = decisions.groupby("language")["n_chars"].median().loc[["de", "fr", "en", "nl", "pl"]]
median_len.plot.bar(ax=axes[0, 1], ylabel="median characters per description")   # comparison
sns.scatterplot(decisions.sample(3000, random_state=1), x="n_chars", y="n_keywords",
                alpha=0.3, ax=axes[1, 0]).set(xscale="log")                   # relationship
monthly = decisions.set_index("start_date").resample("MS").size()
monthly.plot(ax=axes[1, 1], ylabel="decisions per month")                     # change over time

print(median_len.to_dict())   # {'de': 740.0, 'fr': 272.0, 'en': 309.0, 'nl': 544.0, 'pl': 455.0}
print(monthly.idxmax().date(), monthly.max())   # 2017-03-01 881
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

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
share = decisions["language"].value_counts(normalize=True)
share = pd.concat([share.head(5), pd.Series({"other": share.iloc[5:].sum()})])
print(share.round(3).to_dict())
# {'de': 0.573, 'fr': 0.162, 'en': 0.052, 'nl': 0.048, 'pl': 0.036, 'other': 0.129}

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
left.pie(share, labels=share.index)                         # angles and areas: hard to compare
right.barh(share.index, share, color="0.35")                # positions on a common scale
right.bar_label(right.containers[0], labels=[f"{v:.0%}" for v in share], padding=3)
right.spines[["top", "right"]].set_visible(False)           # remove non-data ink
right.set(xlabel="share of decisions", ylabel="language", xticks=[])
right.invert_yaxis()                                        # largest at the top
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

Every chart for a reader needs axis labels with units, a readable number format and a title. A title that states the finding ("57 % of decisions are written in German") tells the reader what to look for. `ax.annotate` points to a data point with text and an arrow; one highlighted colour draws attention to the category discussed.

### Why it matters

Charts in reports are read without the analyst present. Labels, a finding title and a targeted annotation make them self-explanatory, and the explicit interface makes figures reproducible.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
counts = decisions["language"].value_counts().head(6)

fig, ax = plt.subplots(figsize=(6, 3.5))                 # Figure = canvas, Axes = one plot
bars = ax.bar(counts.index, counts, color="0.35", width=0.7)
bars[0].set_color("#D55E00")                              # highlight the category discussed
ax.set(xlabel="Language of the description", ylabel="Number of decisions",
       title="57 % of decisions are written in German")
ax.annotate(f"{counts['de']:,} German decisions", xy=(0, counts["de"]), xytext=(1.2, 22000),
            arrowprops={"arrowstyle": "->"})
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter("{x:,.0f}")                  # 30,000 instead of 30000
fig.savefig("languages.png", dpi=200, bbox_inches="tight")  # fixed size and resolution
print(type(fig).__name__, type(ax).__name__)             # Figure Axes
print(counts["de"], round(counts["de"] / len(decisions), 3))   # 28656 0.573
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
OKABE_ITO = {"de": "#0072B2", "fr": "#E69F00", "en": "#D55E00", "other": "#BBBBBB"}   # one fixed colour per group

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
decisions["year"] = decisions["start_date"].dt.year
group = decisions["language"].where(decisions["language"].isin(["de", "fr", "en"]), "other")
share = pd.crosstab(decisions["year"], group, normalize="index")[["de", "fr", "en", "other"]]
print(share.loc[[2017, 2023]].round(3))
# language     de     fr     en  other
# year
# 2017      0.574  0.129  0.085  0.212
# 2023      0.598  0.167  0.013  0.222

fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
share.plot.bar(stacked=True, ax=ax, width=0.8, color=OKABE_ITO, edgecolor="white", legend=False)
for label, y in [("German", 0.3), ("French", 0.66), ("English", 0.76), ("other", 0.9)]:
    ax.text(6.6, y, label, va="center")              # direct labels instead of a legend
ax.tick_params(axis="x", rotation=0)
ax.set(xlabel="", ylabel="share of decisions",
       title="English-language decisions fell from 8.5 % (2017) to 1.3 % (2023)")
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

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
decisions["n_chars"] = decisions["description"].str.len()

g = sns.displot(decisions, x="n_chars", col="language", col_order=["de", "fr", "en"],
                log_scale=True, height=2.8, aspect=1.1, color="0.35")   # one panel per language
g.set_axis_labels("characters per description", "decisions")
print(g.axes.shape)                                                   # (1, 3)

monthly = pd.read_parquet("case-study/data/monthly_counts.parquet")
by_country = (monthly[monthly["issuing_country"].isin(["DE", "FR", "GB"])
                      & monthly["month"].between("2010-01-01", "2023-12-01")]
              .groupby(["issuing_country", "month"])["n_decisions"].sum().reset_index())
sns.relplot(by_country, x="month", y="n_decisions", col="issuing_country", kind="line",
            height=2.8, aspect=1.4, color="0.35", facet_kws={"sharey": False})
print(by_country.groupby("issuing_country")["month"].max().dt.date.to_dict())
# {'DE': datetime.date(2023, 12, 1), 'FR': datetime.date(2023, 12, 1), 'GB': datetime.date(2020, 12, 1)}
```

With `sharey=False` each panel has its own scale: Germany issues several times more decisions than the United Kingdom did, which the panels no longer show. Say so in the caption, or keep the shared axis.

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

Interactive charts let readers explore detail without the analyst producing dozens of static charts, for example the curve of their own country among 29. They are the building blocks of dashboards, including the Streamlit app on [page 4](04-correlation-and-communication.md#communicating-findings-a-short-report-or-a-streamlit-dashboard).

### How it works in Python

```python
import pandas as pd
import plotly.express as px

monthly = pd.read_parquet("case-study/data/monthly_counts.parquet")
countries = ["DE", "FR", "NL", "GB"]
by_country = (monthly[monthly["issuing_country"].isin(countries)
                      & monthly["month"].between("2004-01-01", "2025-12-01")]
              .groupby(["issuing_country", "month"])["n_decisions"].sum().reset_index())
fig = px.line(by_country, x="month", y="n_decisions", color="issuing_country",
              category_orders={"issuing_country": countries},
              color_discrete_sequence=["#0072B2", "#E69F00", "#009E73", "#D55E00"],
              labels={"month": "", "n_decisions": "decisions per month", "issuing_country": "country"},
              title="Binding Tariff Information decisions per month, four issuing countries")
fig.update_layout(hovermode="x unified")
print(len(fig.data), [trace.name for trace in fig.data])   # 4 ['DE', 'FR', 'NL', 'GB']
fig.write_html("decisions_per_month.html", include_plotlyjs="cdn")   # open in a browser
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

![A poorly designed bar chart of decisions per year with a truncated axis and rainbow colours next to a stacked bar chart from zero that separates the United Kingdom from the other countries, with a finding as its title](figures/good-vs-poor-chart.png)

The left chart shows the number of EBTI decisions per year (from `monthly_counts`). It truncates the axis at 38,000, so a fall of a fifth looks like a collapse to a quarter; the rainbow colours encode nothing; the title says nothing. The right chart starts at zero, uses colour for one distinction that explains part of the fall (decisions issued by the United Kingdom, which stopped after Brexit), and states the finding in the title.

### Why it matters

Most analyses reach decision makers as one chart and a few sentences. Spotting and repairing misleading charts, including one's own, is a core professional skill.

### How it works in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

monthly = pd.read_parquet("case-study/data/monthly_counts.parquet")
monthly["year"] = monthly["month"].dt.year
monthly["uk"] = monthly["issuing_country"].eq("GB").map({True: "United Kingdom", False: "other countries"})
yearly = (monthly[monthly["year"].between(2015, 2025)]
          .pivot_table(index="year", columns="uk", values="n_decisions", aggfunc="sum", fill_value=0))
print(yearly.loc[[2017, 2021]])
# uk    United Kingdom  other countries
# year
# 2017            3333            48149
# 2021               0            40897

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.bar(yearly.index, yearly["other countries"], color="0.6", label="other countries")
ax.bar(yearly.index, yearly["United Kingdom"], bottom=yearly["other countries"],
       color="#D55E00", label="United Kingdom")
ax.yaxis.set_major_formatter("{x:,.0f}")              # full axis from zero (bar default)
ax.legend(frameon=False, loc="lower left")
ax.set(ylabel="decisions per year",
       title="Decisions fell by a fifth from 2017 to 2021;\na third of the fall is the United Kingdom leaving")
ax.spines[["top", "right"]].set_visible(False)
```

A line chart need not start at zero when the reader compares changes rather than magnitudes, but then the axis label must make the range obvious. Bars must always start at zero.

### In practice

- The UK Government Analysis Function advises against breaking the numerical axis of bar charts because it distorts the proportions between bars.
- The Financial Times' *Chart Doctor* column publishes before-and-after redesigns of published charts.
- The CONSORT guidelines for reporting clinical trials require absolute numbers alongside relative effects, for the same reason that bars start at zero.

> [!IMPORTANT]
> **Practice (block 1).** Plot the language distribution, the distribution of description length and the number of decisions per month for the sample. Then take the poor chart above (code in [`figures/make_figures.py`](figures/make_figures.py)) and improve it with the four critique questions. The case-study notebook [18-case-study-ebti-exploration.ipynb](../workbooks/18-case-study-ebti-exploration.ipynb) starts with this task.

## Check your understanding

1. Which chart would you choose for "Are German descriptions longer than French ones?" and why?
2. Rank position, angle, area and colour saturation by how accurately people decode them.
3. Name two ways to make a chart readable without relying on colour.
4. When is it acceptable for a y-axis not to start at zero?
5. What is lost when a dashboard chart is printed, and how do you design for it?

## Further reading

- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. Free online: <https://clauswilke.com/dataviz/>
- Healy, K. (2018). *Data Visualization: A Practical Introduction*, chapter 1 "Look at data". Princeton University Press. <https://socviz.co/lookatdata.html>
- Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531–554. <https://doi.org/10.1080/01621459.1984.10478080>
- Government Analysis Function (2023). *Data visualisation: charts*. UK Government. <https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-charts/>
