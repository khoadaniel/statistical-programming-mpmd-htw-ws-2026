# Put the course data here

This folder is empty in the repository. Download `airbnb-v1.zip` and `ebti-v1.zip` from the **course drive** and unzip them here, so that this folder contains:

```
case-study/data/
  airbnb/   listings.parquet, calendar.parquet, reviews_monthly.parquet, weather_daily.parquet, neighbourhoods.geojson, PROVENANCE.md
  ebti/     decisions.parquet, nomenclature/ (Nomenclature EN.xlsx, DE, FR and the Commission's guide), PROVENANCE.md
```

`PROVENANCE.md` says where every file comes from, what was done to it, and the version of the data, the same as in the zip name (`airbnb-v1`, `ebti-v1`). When a zip with a higher version appears on the drive, delete the old folder and unzip the new one.

From the repository folder, if the zips are in your Downloads folder:

```bash
unzip ~/Downloads/airbnb-v1.zip -d case-study/data/
unzip ~/Downloads/ebti-v1.zip -d case-study/data/
```

Git ignores everything here except this README, so the data are never uploaded. What the datasets are: [airbnb/README.md](../airbnb/README.md) and [ebti/README.md](../ebti/README.md). How they were split: [case-study/README.md](../README.md).
