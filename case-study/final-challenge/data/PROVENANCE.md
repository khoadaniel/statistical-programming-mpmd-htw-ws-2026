# EBTI data: the files

Data version **v1**. If the lecturer corrects the data, a higher version is pushed to the course repository:
run `git pull` to get it.

| Version | Changes |
|---|---|
| v1 | first release |

## What is in this folder

| File | Rows | One row is |
|---|---:|---|
| `decisions.parquet` | 103,656 | a decision of an EU customs authority: a description of goods and the code it belongs to |
| `nomenclature/Nomenclature EN.xlsx`, `DE`, `FR` | 25,866 each | a line of the EU customs tariff, in English, German or French |
| `nomenclature/Explanation for the Taric database extractions 11.pdf` | | the Commission's guide to the columns of the three Excel files |

The two link through the goods code. A code is read from the left, and each step down the tariff adds digits:

```
chapter   64          Footwear, gaiters and the like                          decisions: chapter
heading   6404        Footwear with uppers of textile materials               decisions: heading
subheading 6404 19    other footwear of this heading (six digits, HS6)        decisions: subheading  (the label)
CN code   6404 19 90  an eight-digit subdivision                              decisions: cn_code
TARIC     6404 19 90 00  (ten digits)                                         nomenclature: Goods code
```

So `subheading` in `decisions.parquet` is the first six digits of a `Goods code` in the nomenclature,
`heading` the first four and `cn_code` the first eight. The nomenclature gives the names of the codes; turning its lines into one row
per subheading is part of your work.

### `decisions.parquet`: one row per decision

A trader who is unsure how to classify a product asks the customs authority of a member state. The
authority answers with a Binding Tariff Information decision: the product, described in words, belongs
to this code. The decision is binding in the whole EU for three years. A shortened example is in
[the dataset introduction](../README.md#what-a-decision-looks-like).

| Column | Meaning |
|---|---|
| `bti_reference` | the decision's number; it starts with the issuing country |
| `issuing_country` | the member state that issued the decision (28 countries) |
| `language` | the language of the texts (23 languages; `de` is the most frequent) |
| `start_date`, `end_date` | the three years in which the decision is binding |
| `date_of_issue` | the day the decision was issued |
| `description` | the goods as described in the decision, in its language: the text your model reads |
| `keywords` | keywords that customs added to the decision, in English |
| `classification_justification` | why customs chose this code: rules, notes and texts of the tariff |
| `cn_code` | the eight-digit code of the decision, as text (`"84181080"`) |
| `subheading` | its first six digits: **the label to predict** of the final project (3,162 subheadings) |
| `heading` | its first four digits (1,012 headings) |
| `chapter` | its first two digits (95 chapters) |

Codes are stored as text, so leading zeros stay (`0901` is coffee). When a new request arrives, only
`description` and `language` exist; the other columns are written by customs during or after the
decision ([Things to keep in mind](../README.md#things-to-keep-in-mind)).

### `nomenclature/`: the EU customs tariff

The TARIC nomenclature lists every code of the EU customs tariff with its name, from the chapters
down to ten-digit codes. Each Excel file holds all lines in one language. The first rows of
`Nomenclature EN.xlsx`:

| Goods code | Start date | End date | Language | Hier. Pos. | Indent | Description | Descr. start date |
|---|---|---|---|---|---|---|---|
| `0100000000 80` | 31-12-1971 |  | EN | 2 |  | LIVE ANIMALS | 31-12-1971 |
| `0101000000 80` | 01-01-1972 |  | EN | 4 |  | Live horses, asses, mules and hinnies | 01-01-1972 |
| `0101210000 10` | 01-01-2012 |  | EN | 6 | -  | Horses | 01-01-2019 |
| `0101210000 80` | 01-01-2012 |  | EN | 6 | - -  | Pure-bred breeding animals | 01-01-2019 |

`Start date` and `End date` say when a line is valid (no end date: still valid). The guide
`Explanation for the Taric database extractions 11.pdf` explains all columns. The files are as published, unchanged.

## One reference date: 1 October 2026

Both sources describe the EU customs tariff on **1 October 2026**. A decision is binding for three years
and the tariff changes over time, so the date matters: on that day, every decision in
`decisions.parquet` is binding, and every code it uses is a valid line of the nomenclature.

## Where the files come from

| File | Source | Content as of |
|---|---|---|
| `decisions.parquet` | European Commission, EBTI database, full export of all decisions ever published: <https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_export_management.jsp?message=extractFull> | 1 October 2026 |
| `nomenclature/Nomenclature EN.xlsx`, `DE`, `FR` | European Commission, TARIC monthly extraction, CIRCABC group *TARIC & Quota Data and Information*, <https://circabc.europa.eu/ui/group/0e5f18c2-4b2f-42e9-aed4-dfe50ae1263b> (Library → TARIC data → 2026 → 10 - October) | 1 October 2026 |
| `nomenclature/Explanation for the Taric database extractions 11.pdf` | the same CIRCABC group | Rev. 11, 9 June 2026 |

**What the lecturer changed** in `decisions.parquet`:

1. Kept the decisions that were binding on 1 October 2026. They started between 2 October 2023 and
   1 October 2026.
2. Held back part of the decisions as the independent evaluation set of the final project. You never
   see them; the lecturer uses them to test your API.

The texts are as published; nothing in them was changed. The nomenclature files are unchanged.

## Licence and citation

Reuse is permitted with acknowledgement of the source (Commission Decision 2011/833/EU). Cite as:

- European Commission, *European Binding Tariff Information (EBTI)* database, decisions binding on
  1 October 2026.
- European Commission, *TARIC*, monthly extraction of 1 October 2026.
