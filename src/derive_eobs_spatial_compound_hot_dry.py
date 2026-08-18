#!/usr/bin/env python3
"""Derive gridded E-OBS compound hot-dry event-day statistics for Germany.

For every valid 0.1° E-OBS grid cell, the script applies the same core
method used for the regional DWD and E-OBS comparison:
- warm season: April-September;
- heat: calendar-day-specific TXK p90 from the fixed 1961-1990 baseline;
- dry: antecedent 30-day RSK sum <= calendar-day-specific p10 or p20;
- event days: members of a run of at least three compound days.

Processing is done in latitude blocks to keep memory use practical on a
laptop. No missing values are imputed. Annual statistics require complete
warm-season TXK and rolling-precipitation values for that grid cell.

Inputs:
    data/processed/eobs/eobs_germany_*_v31.0e.nc

Outputs:
    data/processed/eobs_spatial_compound_annual_1961_2024.parquet
    data/outputs/eobs_spatial_compound_summary_1961_2024.nc
    data/outputs/eobs_spatial_compound_config.json

Run:
    python src/derive_eobs_spatial_compound_hot_dry.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EOBS_DIR = Path("data/processed/eobs")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
EOBS_VERSION = "31.0e"
START_YEAR = 1961
END_YEAR = 2024
BASELINE_START = 1961
BASELINE_END = 1990
RECENT_START = 2005
RECENT_END = 2024
WARM_MONTHS = (4, 5, 6, 7, 8, 9)
WINDOW_DAYS = 15
EVENT_MIN_DAYS = 3
LATITUDE_BLOCK_SIZE = 10


def eobs_paths() -> list[Path]:
    paths = sorted(EOBS_DIR.glob(f"eobs_germany_*_v{EOBS_VERSION}.nc"))
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected five E-OBS files in {EOBS_DIR}; found {len(paths)}.")
    return paths


def source_metadata(paths: list[Path]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    times = []
    latitudes = None
    longitudes = None
    for path in paths:
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            times.append(pd.DatetimeIndex(dataset.time.values))
            if latitudes is None:
                latitudes = dataset.latitude.values
                longitudes = dataset.longitude.values
    time = pd.DatetimeIndex(np.concatenate(times))
    keep = (time.year >= START_YEAR) & (time.year <= END_YEAR)
    return time[keep], latitudes, longitudes


def read_latitude_block(paths: list[Path], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    tx_chunks = []
    rr_chunks = []
    for path in paths:
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            subset = dataset.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")).isel(
                latitude=slice(start, end)
            )
            tx_chunks.append(subset["tx"].values.astype(np.float32))
            rr_chunks.append(subset["rr"].values.astype(np.float32))
    tx = np.concatenate(tx_chunks, axis=0)
    rr = np.concatenate(rr_chunks, axis=0)
    return tx.reshape(tx.shape[0], -1), rr.reshape(rr.shape[0], -1)


def rolling_sum_30(values: np.ndarray) -> np.ndarray:
    """Strict 30-day rolling sum: NaN if any value in the window is missing."""
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype(np.float64)
    cumulative_sum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(filled, axis=0)])
    cumulative_count = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(valid, axis=0)])

    result = np.full(values.shape, np.nan, dtype=np.float32)
    sums = cumulative_sum[30:] - cumulative_sum[:-30]
    counts = cumulative_count[30:] - cumulative_count[:-30]
    result[29:] = np.where(counts == 30, sums, np.nan)
    return result


def threshold_arrays(
    tx: np.ndarray,
    precip_30d: np.ndarray,
    time: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 183 calendar-day threshold rows for a latitude block."""
    baseline = (time.year >= BASELINE_START) & (time.year <= BASELINE_END)
    clim_dates = pd.to_datetime("2000-" + time.strftime("%m-%d"), format="%Y-%m-%d")
    targets = pd.date_range("2000-04-01", "2000-09-30", freq="D")

    heat = np.full((len(targets), tx.shape[1]), np.nan, dtype=np.float32)
    dry_p10 = np.full_like(heat, np.nan)
    dry_p20 = np.full_like(heat, np.nan)

    for index, target in enumerate(targets):
        within_window = np.abs((clim_dates - target).days) <= WINDOW_DAYS
        selected = baseline & within_window
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            heat[index] = np.nanquantile(tx[selected], 0.90, axis=0)
            dry_p10[index] = np.nanquantile(precip_30d[selected], 0.10, axis=0)
            dry_p20[index] = np.nanquantile(precip_30d[selected], 0.20, axis=0)
    return heat, dry_p10, dry_p20


def event_days(condition: np.ndarray) -> np.ndarray:
    """Keep only compound-condition runs with at least EVENT_MIN_DAYS days."""
    result = np.zeros_like(condition, dtype=bool)
    for column in range(condition.shape[1]):
        values = condition[:, column]
        padded = np.concatenate(([False], values, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(changes[::2], changes[1::2]):
            if end - start >= EVENT_MIN_DAYS:
                result[start:end, column] = True
    return result


def annual_event_days(
    events: np.ndarray,
    complete: np.ndarray,
    time: pd.DatetimeIndex,
    years: np.ndarray,
) -> np.ndarray:
    result = np.full((len(years), events.shape[1]), np.nan, dtype=np.float32)
    for index, year in enumerate(years):
        in_year_warm = (time.year == year) & np.isin(time.month, WARM_MONTHS)
        expected_days = int(in_year_warm.sum())
        complete_days = complete[in_year_warm].sum(axis=0)
        counts = events[in_year_warm].sum(axis=0).astype(np.float32)
        result[index] = np.where(complete_days == expected_days, counts, np.nan)
    return result


def period_mean(annual: np.ndarray, years: np.ndarray, start: int, end: int) -> np.ndarray:
    selected = annual[(years >= start) & (years <= end)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(selected, axis=0).astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = eobs_paths()
    time, latitudes, longitudes = source_metadata(paths)
    years = np.arange(START_YEAR, END_YEAR + 1)
    warm = np.isin(time.month, WARM_MONTHS)
    warm_day_index = np.full(len(time), -1, dtype=int)
    warm_day_index[warm] = pd.to_datetime("2000-" + time[warm].strftime("%m-%d")).dayofyear - 92

    shape = (len(latitudes), len(longitudes))
    fields = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "baseline_mean_event_days_p10", "recent_mean_event_days_p10", "change_event_days_p10",
            "baseline_mean_event_days_p20", "recent_mean_event_days_p20", "change_event_days_p20",
        )
    }
    annual_frames = []

    for start in range(0, len(latitudes), LATITUDE_BLOCK_SIZE):
        end = min(start + LATITUDE_BLOCK_SIZE, len(latitudes))
        print(f"Processing latitude rows {start + 1}-{end} of {len(latitudes)}...")
        tx, rr = read_latitude_block(paths, start, end)
        precip_30d = rolling_sum_30(rr)
        heat, dry_p10, dry_p20 = threshold_arrays(tx, precip_30d, time)

        condition_p10 = np.zeros(tx.shape, dtype=bool)
        condition_p20 = np.zeros(tx.shape, dtype=bool)
        for day in range(len(heat)):
            matching_days = warm_day_index == day
            condition_p10[matching_days] = (
                (tx[matching_days] >= heat[day])
                & (precip_30d[matching_days] <= dry_p10[day])
            )
            condition_p20[matching_days] = (
                (tx[matching_days] >= heat[day])
                & (precip_30d[matching_days] <= dry_p20[day])
            )

        complete = np.isfinite(tx) & np.isfinite(precip_30d)
        annual_p10 = annual_event_days(event_days(condition_p10), complete, time, years)
        annual_p20 = annual_event_days(event_days(condition_p20), complete, time, years)

        block_shape = (end - start, len(longitudes))
        for percentile, annual in ((10, annual_p10), (20, annual_p20)):
            baseline_mean = period_mean(annual, years, BASELINE_START, BASELINE_END).reshape(block_shape)
            recent_mean = period_mean(annual, years, RECENT_START, RECENT_END).reshape(block_shape)
            fields[f"baseline_mean_event_days_p{percentile}"][start:end] = baseline_mean
            fields[f"recent_mean_event_days_p{percentile}"][start:end] = recent_mean
            fields[f"change_event_days_p{percentile}"][start:end] = recent_mean - baseline_mean

        latitude_grid, longitude_grid = np.meshgrid(latitudes[start:end], longitudes, indexing="ij")
        valid_land = np.isfinite(annual_p10).any(axis=0) | np.isfinite(annual_p20).any(axis=0)
        annual_frames.append(
            pd.DataFrame(
                {
                    "year": np.repeat(years, valid_land.sum()),
                    "latitude": np.tile(latitude_grid.ravel()[valid_land], len(years)),
                    "longitude": np.tile(longitude_grid.ravel()[valid_land], len(years)),
                    "event_days_p10": annual_p10[:, valid_land].ravel(),
                    "event_days_p20": annual_p20[:, valid_land].ravel(),
                }
            )
        )

    annual = pd.concat(annual_frames, ignore_index=True)
    annual_path = PROCESSED_DIR / "eobs_spatial_compound_annual_1961_2024.parquet"
    annual.to_parquet(annual_path, index=False)

    summary = xr.Dataset(
        {name: (("latitude", "longitude"), values) for name, values in fields.items()},
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "title": "E-OBS spatial compound hot-dry event-day summary",
            "dataset": f"E-OBS v{EOBS_VERSION} ensemble mean",
            "analysis_period": f"{START_YEAR}-01-01 to {END_YEAR}-12-31",
            "baseline_period": f"{BASELINE_START}-01-01 to {BASELINE_END}-12-31",
            "recent_period": f"{RECENT_START}-01-01 to {RECENT_END}-12-31",
            "heat_definition": "calendar-day-specific TXK p90, +/- 15-day baseline window",
            "dry_definition": "antecedent 30-day RSK sum <= calendar-day-specific p10 or p20",
            "event_definition": f"at least {EVENT_MIN_DAYS} consecutive compound days",
        },
    )
    summary_path = OUTPUT_DIR / "eobs_spatial_compound_summary_1961_2024.nc"
    summary.to_netcdf(summary_path)

    config = {
        "dataset": f"E-OBS v{EOBS_VERSION} ensemble mean",
        "analysis_period": f"{START_YEAR}-01-01 to {END_YEAR}-12-31",
        "baseline_period": f"{BASELINE_START}-01-01 to {BASELINE_END}-12-31",
        "recent_period": f"{RECENT_START}-01-01 to {RECENT_END}-12-31",
        "warm_season": "April to September",
        "calendar_window_days": WINDOW_DAYS,
        "heat_definition": "TXK >= calendar-day-specific p90",
        "dry_definitions": ["antecedent 30-day RSK sum <= calendar-day-specific p10", "antecedent 30-day RSK sum <= calendar-day-specific p20"],
        "event_definition": f"at least {EVENT_MIN_DAYS} consecutive compound days",
        "missing_data_policy": "No imputation; annual values require complete warm-season TXK and rolling precipitation.",
        "processing": f"latitude blocks of {LATITUDE_BLOCK_SIZE} grid rows",
    }
    (OUTPUT_DIR / "eobs_spatial_compound_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote {annual_path}")
    print(f"Wrote {summary_path}")
    print("Wrote spatial analysis configuration to data/outputs/")


if __name__ == "__main__":
    main()
