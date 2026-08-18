# Compound Hot-Dry Events in Contrasting German Climate Regions

A reproducible Python workflow for analysing compound hot-dry events from daily Deutscher Wetterdienst (DWD) station observations and the E-OBS 0.1° gridded observational dataset.

The project combines a transparent two-station regional DWD reference analysis with an E-OBS validation and Germany-wide spatial context. It is designed as an exploratory forest-climate **hazard-exposure** analysis: it identifies concurrent hot and meteorologically dry conditions, but does not claim to measure direct forest impacts.

## Research questions

1. How do frequency, duration and timing of compound hot-dry events differ between a continental Brandenburg/Lusatia reference region and a maritime north-west German reference region?
2. Does E-OBS reproduce the annual regional compound-event patterns derived from the DWD station composites?
3. How has the spatial frequency of compound hot-dry event days changed across the Germany-plus-margin E-OBS domain?

## Study regions and stations

| Region | DWD stations | Role |
|---|---|---|
| Brandenburg/Lusatia | Cottbus (`00880`), Lindenberg (`03015`) | Continental reference region with a forest and sandy-soil context |
| Northwest | Bremen (`00691`), Norderney (`03631`) | Maritime reference region influenced by the North Sea |

Each regional DWD index is the equal-weighted daily mean of its two stations. A regional day is retained only when both stations have complete core observations.

## Data sources

### DWD station observations

The station workflow uses DWD Climate Data Center daily climate observations (`daily/kl`):

- [DWD CDC Open Data](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/)
- [DWD daily KL dataset description](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/DESCRIPTION_obsgermany-climate-daily-kl_en.pdf)

The ingestion layer merges DWD `historical` data with `recent` data. When dates overlap, the quality-controlled historical value has priority.

### E-OBS gridded observations

The spatial workflow uses E-OBS v31.0e ensemble-mean daily maximum temperature (`tx`) and daily precipitation (`rr`) on a regular 0.1° grid:

- [E-OBS download and documentation](https://www.ecad.eu/download/ensembles/download.php)

The prepared local subset spans 5°E–16°E and 47°N–56°N. It is a Germany-plus-margin bounding box, not an exact political-border or forest mask.

## Compound-event definition

The primary definition is applied during April–September with a fixed 1961–1990 baseline:

- **Heat:** daily maximum temperature (`TXK`) at or above the calendar-day-specific 90th percentile.
- **Dryness:** antecedent 30-day precipitation sum (`RSK`) at or below the calendar-day-specific 10th percentile (P10).
- **Threshold calculation:** a ±15-day calendar window is used around each warm-season day in the fixed baseline.
- **Compound event:** at least three consecutive days fulfilling both conditions.
- **Sensitivity analysis:** P20 dryness is calculated as a less strict, pre-specified alternative.
- **Missing data:** values are never imputed. Incomplete regional days cannot become compound days.

The 30-day precipitation metric is an antecedent meteorological precipitation-deficit indicator. It is not a direct measurement of soil-moisture drought, tree water status, forest vitality or damage.

## Analysis periods

| Analysis component | Period | Purpose |
|---|---|---|
| DWD historical event analysis | 1961–2025 | Station-composite analysis and current data availability |
| DWD–E-OBS validation | 1961–2024 | Shared period for direct comparison |
| E-OBS spatial analysis | 1961–2024 | Germany-plus-margin gridded context |
| Threshold baseline | 1961–1990 | Fixed historical reference climate |
| Recent spatial comparison | 2005–2024 | Recent 20-year period versus baseline |

## Main findings so far

- E-OBS reproduces daily DWD Tmax very closely at the station-matched grid cells. Precipitation agreement is lower, as expected for a gridded product, but remains high.
- For annual compound-event days, DWD–E-OBS Pearson correlations are 0.923–0.942 with P10 dryness and 0.939–0.963 with P20 dryness across the two regions.
- E-OBS agreement supports its use as a spatial context product, not as a substitute for local point observations.
- Norderney is treated as a coastal sensitivity case: its nearest valid E-OBS land cell is about 9.5 km from the DWD station.
- Across the E-OBS Germany-plus-margin domain, mean P10 compound-event days rose from 2.055 per warm season in 1961–1990 to 5.778 in 2005–2024. The corresponding P20 values are 3.394 and 9.122.

## Repository structure

```text
forest-climate-extremes/
├── data/
│   ├── raw/                    # Local source downloads; ignored by Git
│   ├── processed/              # Local Parquet/NetCDF analysis data; ignored by Git
│   └── outputs/                # Local tables, maps and figures; ignored by Git
├── notebooks/
│   ├── 01_compound_hot_dry_analysis.ipynb
│   ├── 02_eobs_spatial_compound_hot_dry.ipynb
│   └── 03_dwd_eobs_event_comparison.ipynb
├── src/
│   ├── quality_check.py
│   ├── prepare_analysis_data.py
│   ├── derive_compound_hot_dry_events.py
│   ├── analyze_event_trends.py
│   ├── prepare_current_monitoring.py
│   ├── download_eobs.py
│   ├── prepare_eobs_germany.py
│   ├── validate_dwd_eobs.py
│   ├── prepare_eobs_region_daily.py
│   ├── compare_dwd_eobs_events.py
│   ├── derive_eobs_spatial_compound_hot_dry.py
│   └── plot_eobs_spatial_compound_hot_dry.py
├── ingest_dwd_daily_kl.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a dedicated environment. The workflow was developed with Python 3.11.

```bash
conda create -n forest-climate python=3.11 -y
conda activate forest-climate
python -m pip install -r requirements.txt
```

## Reproducing the DWD workflow

Run commands from the project root.

```bash
# 1. Download and merge historical/current DWD daily observations
python ingest_dwd_daily_kl.py

# 2. Assess data availability and common temporal coverage
python src/quality_check.py

# 3. Build complete two-station regional daily analysis data
python src/prepare_analysis_data.py

# 4. Derive primary P10 events and P20 sensitivity results
python src/derive_compound_hot_dry_events.py --dataset dwd --run-sensitivity

# 5. Calculate trend diagnostics and DWD figures
python src/analyze_event_trends.py

# 6. Open the DWD analysis notebook
jupyter lab notebooks/01_compound_hot_dry_analysis.ipynb
```

## Reproducing the E-OBS workflow

The E-OBS raw and prepared files are deliberately local because of their size.

```bash
# 1. Download E-OBS source files and prepare the Germany-plus-margin subset
python src/download_eobs.py
python src/prepare_eobs_germany.py

# 2. Validate station-matched E-OBS cells against DWD observations
python src/validate_dwd_eobs.py

# 3. Build matched two-cell E-OBS regional indices
python src/prepare_eobs_region_daily.py

# 4. Derive E-OBS regional P10 and P20 events
python src/derive_compound_hot_dry_events.py --dataset eobs --run-sensitivity

# 5. Compare DWD and E-OBS annual event statistics
python src/compare_dwd_eobs_events.py --dry-percentile 10
python src/compare_dwd_eobs_events.py --dry-percentile 20

# 6. Derive and map compound-event days for every valid E-OBS grid cell
python src/derive_eobs_spatial_compound_hot_dry.py
python src/plot_eobs_spatial_compound_hot_dry.py

# 7. Open the spatial and comparison notebooks
jupyter lab notebooks/02_eobs_spatial_compound_hot_dry.ipynb
jupyter lab notebooks/03_dwd_eobs_event_comparison.ipynb
```

## Key generated outputs

| Output | Description |
|---|---|
| `compound_hot_dry_annual_p10.csv` | DWD annual P10 event statistics |
| `eobs_compound_hot_dry_annual_p10.csv` | Matched E-OBS regional annual P10 event statistics |
| `dwd_eobs_event_comparison_p10_summary.csv` | DWD–E-OBS P10 agreement metrics |
| `dwd_eobs_event_comparison_p20_summary.csv` | DWD–E-OBS P20 sensitivity metrics |
| `eobs_spatial_compound_annual_1961_2024.parquet` | Annual P10/P20 event days at each valid E-OBS cell |
| `eobs_spatial_compound_summary_1961_2024.nc` | Baseline, recent and change fields for spatial mapping |
| `eobs_spatial_compound_event_days_maps.png` | P10/P20 baseline, recent and change maps |
| `eobs_spatial_compound_event_days_timeseries.png` | Domain-mean annual spatial event-day time series |

## Current monitoring

The fixed historical analysis excludes incomplete calendar years. The monitoring layer applies the fixed 1961–1990 thresholds to the latest available DWD data and compares the current year only with historical years up to the same calendar date.

```bash
conda activate forest-climate
python ingest_dwd_daily_kl.py
python src/prepare_current_monitoring.py
```

Use `python ingest_dwd_daily_kl.py --refresh-historical` periodically to refresh cached historical archives after DWD revisions.

## Limitations and next extensions

This is an exploratory two-station-per-region and gridded-climate analysis. It does not resolve forest-stand microclimate, directly measure soil moisture, correct station inhomogeneities, establish causal forest impacts or use an exact forest mask.

Potential next extensions are:

1. Add a forest-cover mask and report exposure specifically for forested E-OBS cells.
2. Compare the meteorological deficit index with DWD forest soil-moisture products, SPI or SPEI.
3. Add tree-species, stand-structure or remote-sensing indicators for impact analysis.
4. Use autocorrelation-aware trend tests and additional baselines as sensitivity analyses.
5. Add lightweight automated tests and a Makefile for routine reproducibility.
