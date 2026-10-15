# Put the course data here

This folder is empty in the repository. Download `airbnb.zip` and `ebti.zip` from the **course drive** and unzip them here, so that this folder contains:

```
case-study/data/
  airbnb/   listings.parquet, calendar.parquet, reviews_monthly.parquet, weather_daily.parquet, neighbourhoods.geojson, snapshot.txt
  ebti/     train.parquet, train_sample.parquet, val.parquet, test.parquet, nomenclature.parquet
```

From the repository folder, if the zips are in your Downloads folder:

```bash
unzip ~/Downloads/airbnb.zip -d case-study/data/
unzip ~/Downloads/ebti.zip -d case-study/data/
```

Git ignores everything here except this README, so the data are never uploaded. What the datasets are: [airbnb/README.md](../airbnb/README.md) and [ebti/README.md](../ebti/README.md). How they were split: [case-study/README.md](../README.md).
