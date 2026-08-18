.PHONY: help check dwd-data dwd-events dwd-trends eobs-prepare eobs-validate eobs-events eobs-compare eobs-spatial notebooks

.DEFAULT_GOAL := help
.RECIPEPREFIX := >

help:
> @echo "Available targets:"
> @echo "  make check         - Syntax-check all Python scripts"
> @echo "  make dwd-data      - Download/check/prepare DWD station data"
> @echo "  make dwd-events    - Derive DWD P10 and P20 compound events"
> @echo "  make dwd-trends    - Create DWD trend outputs"
> @echo "  make eobs-prepare  - Download and prepare E-OBS subsets"
> @echo "  make eobs-validate - Validate E-OBS against DWD stations"
> @echo "  make eobs-events   - Derive matched regional E-OBS events"
> @echo "  make eobs-compare  - Compare DWD and E-OBS events"
> @echo "  make eobs-spatial  - Derive and plot Germany-wide E-OBS events"
> @echo "  make notebooks     - Start JupyterLab"

check:
> python -m py_compile ingest_dwd_daily_kl.py src/*.py
> git diff --check

dwd-data:
> python ingest_dwd_daily_kl.py
> python src/quality_check.py
> python src/prepare_analysis_data.py

dwd-events:
> python src/derive_compound_hot_dry_events.py --dataset dwd --run-sensitivity

dwd-trends:
> python src/analyze_event_trends.py

eobs-prepare:
> python src/download_eobs.py
> python src/prepare_eobs_germany.py

eobs-validate:
> python src/validate_dwd_eobs.py
> python src/prepare_eobs_region_daily.py

eobs-events:
> python src/derive_compound_hot_dry_events.py --dataset eobs --run-sensitivity

eobs-compare:
> python src/compare_dwd_eobs_events.py --dry-percentile 10
> python src/compare_dwd_eobs_events.py --dry-percentile 20

eobs-spatial:
> python src/derive_eobs_spatial_compound_hot_dry.py
> python src/plot_eobs_spatial_compound_hot_dry.py

notebooks:
> jupyter lab
