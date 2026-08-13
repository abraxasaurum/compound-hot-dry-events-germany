# Compound Hot-Dry Events in Contrasting German Climate Regions

A reproducible Python workflow for analysing compound hot-dry events from daily Deutscher Wetterdienst (DWD) Climate Data Center station observations.

## Research question

How do frequency, duration and timing of compound hot-dry events differ between:

- a drier, more continental reference region in Brandenburg/Lusatia, and
- a wetter maritime reference region in north-western Germany?

The project is an exploratory, station-composite analysis designed to connect climate extremes, land-surface conditions and forest-relevant hydroclimatic stress.

## Study regions and stations

| Region | DWD stations | Role |
|---|---|---|
| Brandenburg/Lusatia | Cottbus (`00880`), Lindenberg (`03015`) | Drier continental reference region with a forest and sandy-soil context |
| North-western Germany | Norderney (`03631`), Bremen (`00691`) | Maritime reference region influenced by the North Sea |

## Data source

The workflow uses DWD Climate Data Center daily climate observations (`daily/kl`):

- DWD CDC Open Data: <https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/>
- DWD dataset description: <https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/DESCRIPTION_obsgermany-climate-daily-kl_en.pdf>

The ingestion layer merges DWD `historical` data with the current `recent` data. When dates overlap, the quality-controlled historical value has priority.

## Event definition

The primary analysis uses complete calendar years from 1961 to 2025. It identifies compound hot-dry events during April–September using a fixed 1961–1990 baseline:

- Heat condition: regional daily maximum temperature (`TXK`) at or above the calendar-day-specific 90th percentile.
- Dry condition: antecedent 30-day precipitation sum (`RSK`) at or below the calendar-day-specific 10th percentile.
- Calendar-day thresholds: calculated within a plus/minus 15-day window in the fixed baseline.
- Compound event: at least three consecutive days fulfilling both conditions.
- Missing values: never imputed; a regional day is eligible only when both stations in that region have complete core observations (`TMK`, `TXK`, `TNK`, `RSK`).

The 30-day precipitation metric is interpreted as an antecedent meteorological precipitation deficit, not as direct forest soil-moisture drought.

## Repository structure

```text
forest-climate-extremes/
├── data/
│   ├── raw/                    # Local DWD archives; ignored by Git
│   ├── processed/              # Local Parquet outputs; ignored by Git
│   └── outputs/                # Local analysis outputs; ignored by Git
├── notebooks/
│   └── 01_compound_hot_dry_analysis.ipynb
├── src/
│   ├── quality_check.py
│   ├── prepare_analysis_data.py
│   ├── derive_compound_hot_dry_events.py
│   ├── analyze_event_trends.py
│   └── prepare_current_monitoring.py
├── ingest_dwd_daily_kl.py
└── README.md
```

## Installation

Create and activate a dedicated Python environment. The workflow was developed with Python 3.11.

```bash
conda create -n forest-climate python=3.11 -y
conda activate forest-climate
python -m pip install pandas pyarrow requests numpy scipy matplotlib seaborn jupyterlab ipykernel
```

## Reproducing the analysis

Run all commands from the project root.

```bash
# 1. Download and merge historical/current DWD daily observations
python ingest_dwd_daily_kl.py

# 2. Assess data availability and common temporal coverage
python src/quality_check.py

# 3. Build complete regional daily analysis data for 1961–2025
python src/prepare_analysis_data.py

# 4. Derive P10 compound hot-dry events and annual summaries
python src/derive_compound_hot_dry_events.py

# 5. Calculate monotonic trends and create figures
python src/analyze_event_trends.py

# 6. Build current year-to-date monitoring data
python src/prepare_current_monitoring.py

# 7. Open the analysis notebook
jupyter lab
```

## Current monitoring

The fixed historical analysis excludes incomplete calendar years. The monitoring layer applies the same fixed 1961–1990 thresholds to the latest available DWD data and compares the current year only with historical years up to the same calendar date.

For an update, run:

```bash
conda activate forest-climate
python ingest_dwd_daily_kl.py
python src/prepare_current_monitoring.py
```

Use `python ingest_dwd_daily_kl.py --refresh-historical` periodically to refresh already cached historical archives after DWD revisions.

## Outputs

Key generated outputs include:

- `data_quality_summary.csv`: station-level data completeness
- `analysis_region_daily_1961_2025.parquet`: complete regional daily reference data
- `compound_hot_dry_thresholds.csv`: fixed regional event thresholds
- `compound_hot_dry_events_p10.csv`: identified strict compound events
- `compound_hot_dry_annual_p10.csv`: annual event statistics
- `compound_hot_dry_trend_statistics.csv`: Kendall rank and Theil-Sen trend diagnostics
- `monitoring_region_daily_<year>.parquet`: current year-to-date regional monitoring layer
- `current_monitoring_summary_<year>.json`: current monitoring summary

## Limitations and extensions

This is an exploratory two-station-per-region composite analysis. It does not resolve forest-stand microclimate, directly measure soil moisture, correct station inhomogeneities, or establish causality. Sensible next extensions are:

1. P20 dry-threshold and 1951–2025 sensitivity analyses.
2. Comparison with DWD forest soil-moisture products or SPI.
3. Forest-cover, land-cover and soil context around the stations.
4. Autocorrelation-adjusted trend tests.
5. Scheduled automated updates via GitHub Actions.
