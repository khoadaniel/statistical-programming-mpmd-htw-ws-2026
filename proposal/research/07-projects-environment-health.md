# Research notes — Final-project problems: environment/climate & health (Berlin/DE/EU)

Data URLs checked 2026-09-24 (HTTP 200 unless noted). Stakeholders are the institutions that would use the results; none has been contacted yet.
Kaggle check: only one exploration notebook exists, for the Berliner Feuerwehr data. There is nothing for AMELAG, Gieß den Kiez, the RKI emergency-department (ED) data or BfArM shortages.

## Ranking
| # | Project | Area | Level |
|---|---|---|---|
| 1 | H1 Heat–health early warning (heat → ED visits, excess deaths) | Health × climate | Core |
| 2 | H2 Ambulance response times: equity + demand forecast | Health | Core |
| 3 | E1 Heat vulnerability vs cool-room coverage gap | Env × health | Core |
| 4 | H3 Wastewater viral load → respiratory nowcast | Health | Core/Stretch |
| 5 | E2 Street-tree drought: who waters, which trees die | Env | Core |
| 6 | H4 Drug-shortage early warning + NLP on shortage notices | Health | Stretch |
| 7 | E3 Tempo-30 rollback evaluation (NO₂, traffic) | Env | Stretch |
| 8 | E4 Heat plan vs reality: energy transition by neighbourhood | Env | Core/Stretch |
| 9 | E5 Heavy-rain hazard map vs fire-brigade missions | Env | Stretch |
| 10 | E6 Spree low-water early warning | Env | Stretch |
| 11 | H5 Hospital-reform access atlas | Health | Stretch |
| 12 | H6 Outpatient doctor supply vs social need | Health | Stretch (data risk) |

## Health
### H1 Heat–health early warning
- **Why now:** Berlin adopted a Hitzeaktionsplan (heat action plan) with 72 measures in Nov 2025, and critics say it isn't being implemented (Tagesspiegel, BUND). The RKI estimates ~14,000 heat deaths in 2025.
- **Stakeholders:** SenWGP (Senate health department), district public-health offices, Aktionsbündnis Hitzeschutz Berlin.
- **Data:**
  - RKI ED surveillance: https://github.com/robert-koch-institut/Daten_der_Notaufnahmesurveillance. Daily TSV (last row 2026-09-23) with a **HEAT syndrome** plus ARI/SARI/ILI by age. CC BY 4.0.
  - RKI excess mortality: https://github.com/robert-koch-institut/Daten_des_Uebersterblichkeitsberichts. Weekly, by federal state and age, with nowcast flag.
  - DWD hourly/daily station data: https://opendata.dwd.de/climate_environment/CDC/. Felt temperature (gt.json) and Bright Sky API.
- **Messy parts:** ED data is relative and not Berlin-specific; weekly vs daily frequencies; nowcast revisions; COVID years as confounders.
- **Methods:** distributed-lag non-linear models (DLNM) with quasi-Poisson/negative binomial; bootstrap for attributable fractions; threshold detection; 1–7-day forecast (GBM vs SARIMAX) with rolling-origin CV and interval coverage.
- **Product:** "Hitze-Gesundheitsradar" API + dashboard.
- **Risks:** causal over-claiming; ecological fallacy; must not be presented as medical advice.

### H2 Ambulance response times
- **Why now:** repeated "Ausnahmezustand Rettungsdienst" (rescue-service emergency) declarations; the 10-minute target (Hilfsfrist) is often missed; the rescue-service law is being reformed.
- **Stakeholders:** Berliner Feuerwehr, SenInnSport (Senate interior department), MPs (written question S19-19457), aid organisations.
- **Data:**
  - https://github.com/Berliner-Feuerwehr/BF-Open-Data (CC BY 4.0, updated 2026-09-24): one row per mission 2018–2026 with `response_time` (seconds), dispatch category, district; daily aggregates; per-LOR planning-room data including target achievement; turnout times.
  - Joins: LOR 2021 WFS https://gdi.berlin.de/services/wfs/lor_2021, social index MSS 2023 https://gdi.berlin.de/services/wfs/mss_2023, DWD weather, holidays, respiratory waves.
- **Messy parts:** dispatch-code change leaves fields empty; missing response times for non-urgent missions; date but no hour; LOR boundary change in 2021; censoring/outliers.
- **Methods:** quantile regression of response time on social index; permutation tests; Moran's I; survival analysis; bootstrap CIs for target share; daily demand forecast (Poisson GBM vs SARIMAX).
- **Risks:** low privacy risk; politically sensitive.

### H3 Wastewater → respiratory nowcast
- **Why now:** AMELAG (national wastewater monitoring) funding was extended only year by year; the ministry now plans permanent operation, so its value must be shown.
- **Stakeholders:** RKI, UBA, LAGeSo, Berliner Wasserbetriebe (Berlin water utility).
- **Data:**
  - https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG, `amelag_einzelstandorte.tsv`: weekly per treatment plant (Ruhleben, Schönerlinde, Waßmannsdorf + Brandenburg plants); SARS-CoV-2, influenza A/B, RSV; lab-change and below-detection-limit flags.
  - Ground truth: RKI GrippeWeb, ARE, SARI, influenza/RSV repositories; https://github.com/EU-ECDC/Respiratory_viruses_weekly_data.
- **Messy parts:** lab changes cause level shifts; values below detection are censored; rain dilution; plant catchments don't match districts.
- **Methods:** censored (Tobit) regression; change-point detection; lead-lag analysis; nowcasting (dynamic regression / Kalman / GBM); evaluation with the weighted interval score (WIS).

### H4 Drug shortages
- **Why now:** 1,514 shortage reports in 2025; children's antibiotic syrups in shortage until March 2026.
- **Data:**
  - Live CSV: https://anwendungen.pharmnet-bund.de/lieferengpassmeldungen/public/csv (`;`-separated, Latin-1) with PZN, ATC, active substance, start/end, free-text **reason**.
  - It is a snapshot only, so history must come from the Wayback Machine plus a daily collector (GitHub Action → DuckDB).
- **Methods:** survival analysis of shortage duration (Kaplan–Meier, Cox); NLP classification of reasons (German BERT vs TF-IDF); forecasting reports per ATC class.
- **Risks:** the CSV contains manufacturer contact emails and phone numbers, so drop them.

### H5 Hospital-reform access atlas
- **Why now:** the KHAG (hospital-reform amendment act) has been in force since 15 Apr 2026.
- **Data:** Bundes-Klinik-Atlas XML export https://bundes-klinik-atlas.de/open-data/ (no explicit licence; small counts suppressed as -1), GISD deprivation index, Zensus grid, OSRM travel times.
- **Methods:** isochrones; closure scenario simulation.

### H6 Doctor supply vs need
- **Data:** KV Berlin needs-plan PDFs; MSS social index. Practice locations would require scraping the KV/116117 doctor search, which is a terms-of-service risk. Weakest data access of all candidates.

## Environment
### E1 Heat vulnerability vs cool rooms
- **Why now:** about 80 official cool rooms, none in Friedrichshain-Kreuzberg or Lichtenberg (BUND); only 7 districts have their own heat plans.
- **Data (Berlin GDI WFS):**
  - `ua_klimaanalyse_2022` (PET/UTCI per block)
  - `ua_umweltgerechtigkeit_2021` (multiple burden)
  - `mss_2023`
  - `lor_2021`
  - `kuehle_raeume` (DL-DE-Zero-2.0)
  - Plus OSM, DWD, optionally Landsat/Copernicus land-surface temperature.
- **Messy parts:** blocks vs LOR vs districts (areal interpolation); EPSG:25833; free-text opening hours.
- **Methods:** robust composite index with Monte Carlo weight sensitivity; LISA clustering; max-coverage facility location; validation against satellite LST.
- **Product:** a map + API with the top-N proposed new cool rooms.

### E2 Street-tree drought
- **Why now:** 5,643 fellings vs 2,736 plantings in 2025.
- **Stakeholders:** SenMVKU (Senate environment department), district green-space offices, CityLAB, BUND.
- **Data:**
  - Gieß den Kiez watering events: https://github.com/technologiestiftung/giessdenkiez-de-opendata (MIT, ~61k rows; git history holds earlier seasons).
  - Tree cadastre WFS https://gdi.berlin.de/services/wfs/baumbestand (434,765 street trees).
  - Felling PDF; DWD soil moisture; klimaanalyse; MSS.
- **Messy parts:** inconsistent tree IDs; tree death inferred only by diffing cadastre snapshots; self-selected volunteers; self-reported litres with outliers.
- **Methods:** robust GLMs; hot-spot analysis; matched survival comparison; forecasting watering demand.
- **Product:** a watering-priority API.

### E3 Tempo-30 rollback
- **Why now:** the Luftreinhalteplan (air-quality plan) 3rd update (Sep 2025) lifts Tempo 30 on 34 routes; DUH threatens to sue.
- **Data:**
  - BLUME API https://luftdaten.berlin.de/api/doc
  - UBA air data API v3
  - Traffic detection https://api.viz.berlin.de/daten/verkehrsdetektion (DL-DE-BY-2.0)
  - DWD weather
- **Methods:** weather normalisation (GBM); difference-in-differences / synthetic control; placebo tests.

### E4 Heat plan vs reality
- **Why now:** Berlin's heat plan (Wärmeplan) was adopted June 2026; the Solarcity target is 25 % PV by 2035.
- **Data:**
  - MaStR daily full export https://download.marktstammdatenregister.de/ via the `open-mastr` package (DL-DE-BY-2.0)
  - WFS `ua_waermeplanung`
  - SMARD
- **Messy parts:** registration lag; geocoding small systems; heat pumps are not in MaStR.
- **Methods:** registration-lag nowcasting; target-gap projection; the end of the balcony-PV subsidy as a natural experiment.

### E5 Heavy rain vs fire-brigade missions
- **Data:** heavy-rain hazard map (WMS, vector access unclear); DWD RADOLAN radar; Feuerwehr technical-rescue missions.
- **Methods:** negative-binomial models; spatial validation of the hazard map.

### E6 Spree low water
- **Data:** Pegel Online REST API (only ~30 days of history, so a collector is needed); DWD evapotranspiration and soil moisture.
- **Methods:** state-space models and quantile GBM; extreme-value analysis.

## Notes
- A shared "Berlin data commons" (DWD, LOR/MSS geometries, RKI data) can be seeded once for all teams.
- Dead ends:
  - The RKI heat-mortality GitHub repo returns 404 (PDF reports only).
  - DWD pollen is forecast-only.
  - daten.berlin.de pages return 403 to bots (use the gdi.berlin.de WFS instead).
  - The noise-map WFS name was not found.
- Licences: RKI, Feuerwehr and DWD are CC BY 4.0; Berlin GDI is DL-DE-BY/Zero-2.0; Gieß den Kiez is MIT; MaStR is DL-DE-BY-2.0; Klinik-Atlas and BfArM have no explicit licence.

## Sources (relevance, 2024–26)
- RKI heat deaths 2025: https://praxis.thieme.de/news/hitzetote-2025-rki-schaetzung-14000-todesfaelle
- Hitzeaktionsplan: https://www.berlin.de/rbmskzl/aktuelles/pressemitteilungen/2025/pressemitteilung.1590922.php · critique https://www.bund-berlin.de/service/presse/detail/news/berliner-hitzekationsplan-ohne-aktion/
- Rescue-service emergency: https://www.berliner-kurier.de/berlin/berlin-streicht-ausnahmezustand-schoensprech-offensive-bei-der-feuerwehr-li.2253230
- AMELAG funding: https://www.aerzteblatt.de/news/abwassermonitoring-soll-trotz-vorlaeufiger-haushaltsfuehrung-2025-weitergehen-8256a1eb-ab35-4f6c-8a30-2bca34c98669
- Drug shortages 2026: https://deutsch.medscape.com/viewarticle/lieferengp%C3%A4sse-hunderte-medikamente-fehlen-und-jahren-2026a10007z8
- Klinik-Atlas open data: https://bundes-klinik-atlas.de/open-data/
- Street trees: https://www.entwicklungsstadt.de/strassenbaeume-in-berlin-erstmals-seit-jahren-verbessert-sich-der-zustand/
- Luftreinhalteplan: https://www.berlin.de/sen/uvk/umwelt/luft/luftreinhaltung/luftreinhalteplan-3-fortschreibung/
- Wärmeplan: https://www.berlin.de/rbmskzl/aktuelles/pressemitteilungen/2026/pressemitteilung.1682253.php
- Spree water: https://www.umweltbundesamt.de/en/press/pressinformation/spree-faces-increased-water-shortage-after-coal
