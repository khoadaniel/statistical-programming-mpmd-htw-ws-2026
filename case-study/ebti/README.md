# EBTI: binding tariff decisions of the EU customs

The dataset of Sessions 13–16 and of the final project: about 120,000 decisions in which an EU customs authority states how a product is classified in the customs tariff. The task is to predict the four-digit **heading** of a product from its description.

## Where the data come from

Every product that crosses the border of the European Union must be classified in the customs nomenclature, and the classification decides the duty. When a trader is unsure, they can ask the customs authority of a member state for a **Binding Tariff Information (BTI)** decision: they describe the product, and customs states its code. According to the [European Commission](https://taxation-customs.ec.europa.eu/online-services/online-services-and-databases-customs/european-binding-tariff-information-ebti_en), a BTI decision is generally valid for three years throughout the EU, regardless of where it was issued, and binds all EU customs administrations and the holder. All decisions are entered into the **European Binding Tariff Information (EBTI)** database, which the Commission publishes; reuse is allowed with acknowledgement of the source (Commission Decision 2011/833/EU). The names of the holders are not published.

The lecturer downloaded the full export of the database on **15 October 2026** (1,051,034 decisions since 2004) and kept the **122,003 decisions that were valid on that day**, issued between October 2023 and October 2026. Decisions that repeat another one word for word (renewals) were kept once, leaving 119,514. Of these, one fold of seven (about 14 %, 17,074 decisions) was held back at random, stratified by heading and language together, as an independent evaluation set. You receive the other 102,440 and design your own split (see [case-study/README.md](../README.md#ebti-classifying-goods-in-the-eu-customs-tariff-sessions-1316) and the [final project](../../sessions/17-18-final-presentations/final-project.md)). The nomenclature, the list of all codes of the customs tariff, comes from the Commission's TARIC database (extraction of 1 October 2026).

| File | Rows | One row is |
|---|---:|---|
| `decisions.parquet` | 102,440 | a decision: country, language, dates, description, English keywords, justification, CN code, **heading** |
| `nomenclature/Nomenclature EN.xlsx` (and `DE`, `FR`) | 25,866 each | a line of the EU customs tariff (TARIC) in one language, with its validity dates |

## What a decision looks like

A shortened German decision of heading **6404** (footwear with outer soles of rubber or plastics and uppers of textile materials):

> Slipper, sog. Herren Freizeitschuhe, [...] Größe 42, Foto siehe Anlage,
> - mit Laufsohlen aus Kautschuk/Kunststoff,
> - mit Oberteil aus Spinnstoff (den größten Teil der Außenfläche bildend, damit keine Ware der Position 6402) und Kunststoff [...]
>
> keywords: OF PLASTICS, OF RUBBER, OF TEXTILE MATERIAL, OUTER SOLES, SHOES, UPPERS, FOOTWEAR

The heading is the first four digits of the code: `64` is the chapter (footwear), `6404` the heading. The description is written for an expert: it names the materials, which part forms the outer surface, and even rules out the neighbouring heading 6402. Learning to read such text, in many languages, is the task.

## Why this dataset

- **A real decision with real consequences.** The heading decides the duty a trader pays, and each decision is the judgement of a customs expert.
- **Text as data.** The input is free text, so Sessions 13–15 can compare word counts, embeddings and language models on the same task.
- **Many classes, a long tail, many languages.** Almost 1,000 headings, a few very frequent and hundreds rare, in 23 languages: accuracy alone is not enough, and the choice of method matters.
- **Leakage to find.** The training data contain columns that are written after the decision and name the answer (see below). A model that uses them shows data leakage: it uses information that a real request does not contain, so its score looks better than it is. Finding these columns is part of the work.

## A first look

### Many languages, one dominant

![Bar chart of the language of the description: German 61.0 percent, French 15.6, Dutch 4.0, Czech 3.7, Spanish 3.6, Polish 3.3, Italian 1.8, and 16 other languages 7.0 percent](figures/languages.png)

Each decision is written in the language of the issuing authority. Germany issues **60 %** of the valid decisions and France 15 %, so German and French descriptions dominate; the other 21 languages share the rest. The 28 issuing countries are the 27 member states plus Northern Ireland (code `XI`). Almost every training decision (99.7 %) also carries **English keywords**, which help you read a decision in a language you do not know.

### A long tail of headings

![Bar chart of 1,009 headings sorted by number of decisions on a log scale, falling from over 4,000 decisions to 1; a line marks that the 42 largest headings hold half of all decisions, and 377 headings have fewer than 10](figures/long-tail.png)

Your data use **1,009 headings**. The 42 largest hold half of all decisions, while 377 headings have fewer than ten decisions and 103 have exactly one. A model that learns only the frequent headings can reach a high accuracy and still fail most headings: that is why the course reports **macro-F1** (the average over headings) next to accuracy.

![Bar chart of the ten most frequent headings: 3926 other articles of plastics 4,311 decisions, 9503 toys 3,335, 6307 other made-up textile articles 3,047, 4202 bags and cases 2,901, 2106 food preparations 2,742, down to 7318 screws and bolts 1,567](figures/top-headings.png)

The most frequent headings are broad headings for everyday imported goods: plastic articles, toys, textile articles, bags, food preparations, steel articles. Together the ten largest account for 25 % of the decisions. Several are "not elsewhere specified" headings, which collect what fits nowhere else and are therefore hard to tell apart.

### Descriptions are short paragraphs

![Histogram of the description length in words: most descriptions have between about 40 and 160 words, median 89 words, with a long tail](figures/description-length.png)

The median description has **89 words**; 80 % have between 41 and 159. Long enough to carry the deciding details (material, function, how it is used), short enough for every method of the course.

## Things to keep in mind

- **The justification is the answer.** `classification_justification` explains the decision and names the heading in most cases; it is written after classification and does not exist for a new request. The same holds for `keywords`, `cn_code`, `chapter` and the dates. Never use them as model inputs; Session 13 measures how much they raise a score.
- **Quoted codes.** Customs often quote the tariff text, including codes. Numbers that repeat the decision's own full code are replaced by `<CODE>` (7 % of the descriptions).
- **Unseen headings.** 16 decisions of the evaluation set (15 headings) have a heading that never occurs in `decisions.parquet`; no model trained on the course data gets them right.
- **The nomenclature changes.** The Harmonized System is revised every five to six years: HS 2022 applies since 1 January 2022, and HS 2028 replaces it on 1 January 2028 ([World Customs Organization](https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2028-edition.aspx)). A revision creates, splits and deletes headings, so the set of classes a model knows is valid only until the next one.

## The nomenclature file

The folder `nomenclature/` holds the EU goods nomenclature as the European Commission publishes it, unchanged: the files `Nomenclature EN.xlsx`, `Nomenclature DE.xlsx` and `Nomenclature FR.xlsx` of the **TARIC monthly extraction of 1 October 2026**, from the Commission's CIRCABC group [TARIC & Quota Data and Information](https://circabc.europa.eu/ui/group/0e5f18c2-4b2f-42e9-aed4-dfe50ae1263b) (Library → TARIC data → 2026 → 10 - October). Each file lists every line of the customs tariff in one language, with its validity dates; the codes go down to ten digits, of which the first four are the heading. The files are raw: building a table of headings from them is part of your work.

## Licence and citation

European Commission, *European Binding Tariff Information (EBTI)* database, full export of 15 October 2026; reuse with acknowledgement of the source (Commission Decision 2011/833/EU). European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN/DE/FR.xlsx`), same reuse conditions. The figures are drawn by [`case-study/make_dataset_figures.py`](../make_dataset_figures.py) from the student data.
