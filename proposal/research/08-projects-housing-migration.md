# Research notes — Final-project problems: housing/renting & immigration/labour (Berlin/DE/EU)

Checked 2026-09-24. "Verified" means HTTP 200/206 and, where stated, the file was downloaded and opened.
Licences: dl-de/zero-2.0 needs no attribution; dl-de/by-2.0 and CC BY need attribution.

## Access facts that apply to several projects
1. daten.berlin.de returns 403 to scripts. Use the CKAN API instead (`https://datenregister.berlin.de/api/3/action/package_search?q=…`) together with the Berlin GDI WFS (`https://gdi.berlin.de/services/wfs/<name>`, `outputFormat=application/json`, EPSG:25833).
2. **Destatis GENESIS guest access is gone.** Students need a free token.
3. Some Amt für Statistik LOR CSV links return HTML. Get the data by email or from the XLSX reports.
4. The Eurostat API works without a key (`prc_hpi_q`, `migr_asyappctzm`, `migr_asydcfsta`, `migr_resfirst`, `migr_resocc`, `lfsa_ergaedn`; `migr_resbc1` is 404).
5. The UNHCR API needs `cf_type=ISO`.

## A. Housing
### H1 Rent checker (Mietspiegel 2026 / Mietpreisbremse) — Core → Stretch
- **Why now:** the DMB Mietenmonitor (Dec 2025) found 46 % of 25k Berlin listings over the Mietpreisbremse limit, and over a third at the rent-gouging threshold (https://mieterbund.de/app/uploads/2025/12/FactSheet_DMB_Mietenmonitor_2025.pdf). The Mietspiegel 2026 applies from May 2026, average 7.71 €/m² (https://www.berlin.de/sen/stadt/presse/pressemeldungen/pressemitteilung.1674998.php). IBB reports asking rents of 15.78 €/m².
- **Stakeholders:** Berliner Mieterverein, the Senate's Mietpreisprüfstelle (rent-check office), tenants.
- **Data:**
  - Mietspiegel PDF: https://mietspiegel.berlin.de/wp-content/uploads/2026/05/mietspiegel2026.pdf
  - Residential-location class per address: WFS `wohnlagenadr2026` (400,505 addresses, dl-de/zero; 2003–2024 editions also available)
  - Zensus 2022 100 m rent grid: https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Durchschn_Nettokaltmiete.zip (dl-de/by)
  - Mietpreisprüfstelle annual report 2025
  - Legal listing data: RWI-GEO-RED (ImmoScout24 listings) Campus Files via FDZ Ruhr: https://www.rwi-essen.de/en/research-advice/further/research-data-center-ruhr-fdz/data-sets/rwi-geo-red/x-real-estate-data-and-price-indices
- **Messy parts:** turning Mietspiegel PDF tables into rules; geocoding addresses; Zensus cells are perturbed/suppressed and on a different grid (EPSG:3035).
- **Methods:** rule engine for the reference rent; quantile/Huber regression; bootstrap CIs for the share of listings over the limit; validation against the official online calculator on a stratified sample of 100–200 addresses.
- **Risks:** it is **not legal advice**. **Do not scrape ImmoScout24 or WG-Gesucht (breaks their terms of service).**

### H2 Milieuschutz / displacement-pressure screening — Stretch
- **Why now:** districts must justify each social protection area with evidence; pre-emption rights were weakened by the 2021 BVerwG ruling; the Zensus finds 40,681 empty flats.
- **Data (WFS, dl-de/zero unless noted):**
  - Zensus 100 m grids: vacancy, owner-occupier rate, construction year, heating (dl-de/by)
  - `erhaltungsverordnungsgebiete:erhaltgeb_em`
  - `brw2026` land values
  - `kauffaelle_2025` (sale locations only, no prices)
  - Wohnatlas layers
  - `mss_2025`, `lor_2021`
- **Methods:** displacement-pressure index; Moran's I / LISA; robust spatial regression; spatially blocked CV; out-of-sample check (do existing protection areas score high?).

### H3 Short-term rentals under the EU regulation — Core
- **Why now:** EU Reg. 2024/1028 has applied since 20 May 2026, and Berlin amended its Zweckentfremdung law. Written parliamentary question Drs. 19/24602 has district-level enforcement numbers.
- **Data:** Inside Airbnb Berlin 2026-06-26 (CC BY 4.0). Check: 12,776 listings, **31 % with no registration number**, 8,443 with minimum stay under 30 nights.
- **Methods:** regex/NLP to validate registration numbers and detect commercial operators; detection of multi-listing hosts; comparison of snapshots before and after May 2026; robust estimate of housing withdrawn from the market.
- **Warning:** Berlin Airbnb price-prediction datasets exist on Kaggle, so **ban price prediction as the goal**. Aggregate by area; never name individual hosts.

### H4 Housing evidence tracker (written parliamentary questions) — Stretch, NLP-heavy
- **Data:** PARDOK XML, daily: https://www.parlament-berlin.de/opendata/pardok-wp19.xml, with links to the PDF answers.
- **Methods:** topic classification; table extraction into DuckDB; RAG with citations; gold set of 200 extracted numbers.
- Also covers immigration-office / naturalisation facts.

### H5 Housing supply pipeline and forecast — Core
- **Why now:** 11,027 completions in 2025 (−28 %), a pipeline of 48,394 approved flats, a target of ~20k/year (https://www.statistik-berlin-brandenburg.de/presse/2026/59-baufertigstellungen-2025-berlin/).
- **Data:** AfS F II 1 (monthly permits) / F II 2 (annual completions); GENESIS 311xx (token needed); `step_wo_2040`; Eurostat `prc_hpi_q`.
- **Methods:** distributed-lag model from permits to completions; ETS/ARIMA with hierarchical reconciliation; rolling-origin backtests.

### H6 Homeless accommodation demand — Stretch
- **Why now:** 53,600 people in homeless accommodation in 2025 (double 2022); projection of 85,600 by 2029.
- **Data:** berlin.de homelessness statistics; Destatis 22971; LAF refugee-accommodation WFS.
- **Risks:** sensitive population; thin data.

Rejected: Heizspiegel (not open), Studierendenwerk waiting lists (press figures only), scraping of listing portals.

## B. Immigration / integration / labour
### M1 Can you work in Berlin without German? (job-ad language requirements) — Core
- **Why now:** CHE DatenCHECK 11/2026 analysed 1.7 M graduate job ads: 39.9 % require German, 35.7 % English, 28.1 % both (https://www.che.de/2026/sprachkenntnisse-im-beruf-fuer-akademikerinnen-zaehlt-die-kombination-deutsch-und-englisch/). No Berlin-level, weekly-updated version exists.
- **Stakeholders:** Berlin Business Immigration Service, IQ Netzwerk, career services.
- **Data:**
  - BA Jobsuche API v6: `https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs?was=…&wo=Berlin` with header `X-API-Key: jobboerse-jobsuche`; details via v4 jobdetails
  - Docs: https://github.com/bundesAPI/jobsuche-api
- **Messy parts:** unofficial API (rate-limit politely); syndicated duplicates; mixed-language text; **coverage bias** (tech firms post on LinkedIn).
- **Methods:** language detection; CEFR-level extraction (rules + transformer); weekly vacancy-survival time series; logistic/mixed models; 500 hand-labelled ads with Cohen's κ; comparison with CHE.

### M2 Berlin service portal: English availability and plain-language access — Core, NLP
- **Why now:** long naturalisation times at the LEA (Landesamt für Einwanderung) and 40k inherited cases.
- **Data:** `https://service.berlin.de/export/dienstleistungen/json/` (1,140 services) and `/json/en/`. **Only 358 are really in English; 782 fall back to German.** No explicit licence, so ask the Senatskanzlei.
- **Methods:** German readability scores (LIX, Amstad); plain-language classifier; MT quality estimation; priority score; cited English Q&A evaluated with questions from migrant organisations.
- **Risks:** wrong legal information, so always cite the official page. **Never scrape or book appointments.**

### M3 BAMF asylum statistics: PDF → database with cross-source checks — Core
- **Data:**
  - Monthly BAMF country-of-origin PDFs (e.g. https://www.bamf.de/SharedDocs/Anlagen/DE/Statistik/Asylgeschaeftsstatistik/hkl-statistik-august-2026.pdf?__blob=publicationFile&v=4)
  - Eurostat `migr_asyappctzm` / `migr_asydcfsta`
  - UNHCR API
- **Methods:** table extraction validated by row/column totals; reconciliation with Eurostat; nowcasting; changepoint detection; Wilson CIs for protection rates.
- **Risks:** politically sensitive; neutral framing.

### M4 Integration courses after the 2026 budget cuts — Stretch
- **Why now:** ~20 % less funding in the 2026 budget (https://biaj.de/archiv-kurzmitteilungen/2203-integrationskurse-im-bundeshaushalt-2026-20-prozent-weniger-mittel-soll-als-2025-ausgegeben-wurden.html).
- **Data:** BAMF integration-course statistics by district (XLSX 2023–2025); GENESIS 12521; BA migration monitor.
- **Methods:** robust ratios; panel models; difference-in-differences once 2026 data arrive, with a pre-registered design.

### M5 Foreign-qualification recognition vs labour shortages — Stretch
- **Why now:** 86,600 positive recognitions in 2025 (https://www.destatis.de/DE/Presse/Pressemitteilungen/2026/08/PD26_295_212.html).
- **Data:** BA Migrationsmonitor Berlin XLSX (code 11); Destatis 21231; Eurostat `migr_resocc`.
- **Methods:** occupation crosswalks (KldB vs reference occupations); mismatch index with bootstrap.

### M6 Arrival neighbourhoods — Stretch, **highest ethics risk**
- **Data:** Zensus nationality grids, LAF accommodation, MSS, rent grid.
- **Methods:** segregation indices with bootstrap CIs.
- **Requirements:** ethics review and co-design with a migrant organisation.

### M7 Naturalisation throughput
- Weak data. Fold into H4. Never scrape the appointment booking system.

## Recommended top 6
M1 · H1 · M2 · H3 · H2 · M3 (H4 as a cross-topic option for strong NLP teams).
