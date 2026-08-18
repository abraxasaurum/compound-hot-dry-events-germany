#!/usr/bin/env python3
"""Prepare E-OBS regional daily series matching the DWD two-station index.

The DWD analysis represents each region as the equal-weighted mean of two
stations. This script extracts the validated E-OBS cell assigned to each DWD
station and creates the matching equal-weighted two-cell regional series.

Inputs:
    data/processed/eobs/eobs_germany_*_v31.0e.nc
    data/outputs/dwd_eobs_station_mapping.csv

Outputs:
    data/processed/eobs_station_daily_1961_2024.parquet
    data/processed/eobs_region_daily_1961_2024.parquet
    data/outputs/eobs_region_preparation_summary.json

Run:
    python src/prepare_eobs_region_daily.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr


EOBS_DIR = Path("data/processed/eobs")
MAPPING_PATH = Path("data/outputs/dwd_eobs_station_mapping.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
EOBS_VERSION = "31.0e"
START_DATE = pd.Timestamp("1961-01-01")
END_DATE = pd.Timestamp("2024-12-31")
EXPECTED_CELLS_PER_REGION = 2


def eobs_paths() -> list[Path]:
    paths = sorted(EOBS_DIR.glob(f"eobs_germany_*_v{EOBS_VERSION}.nc"))
    if len(paths) != 5:
        raise FileNotFoundError(
            f"Expected five prepared E-OBS files in {EOBS_DIR}; found {len(paths)}."
        )
    return paths


def load_mapping() -> pd.DataFrame:
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAPPING_PATH}. Run src/validate_dwd_eobs.py first."
        )

    required = {
        "station_id", "station_name", "region", "eobs_latitude", "eobs_longitude",
        "grid_distance_km",
    }
    mapping = pd.read_csv(MAPPING_PATH)
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping is missing required columns: {sorted(missing)}")

    mapping["station_id"] = mapping["station_id"].astype(str).str.zfill(5)
    if len(mapping) != 4 or mapping["station_id"].nunique() != 4:
        raise ValueError("Expected one E-OBS mapping row for each of four DWD stations.")
    if not (mapping.groupby("region")["station_id"].nunique() == EXPECTED_CELLS_PER_REGION).all():
        raise ValueError("Expected exactly two mapped E-OBS cells in each region.")
    return mapping.sort_values("station_id").reset_index(drop=True)


def extract_station_series(mapping: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for path in eobs_paths():
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            subset = dataset.sel(time=slice(START_DATE, END_DATE))
            if subset.sizes.get("time", 0) == 0:
                continue

            for station in mapping.itertuples(index=False):
                point = subset.sel(
                    latitude=station.eobs_latitude,
                    longitude=station.eobs_longitude,
                )[["tx", "rr"]].load()

                frame = point.to_dataframe().reset_index().rename(
                    columns={"time": "MESS_DATUM", "tx": "TXK", "rr": "RSK"}
                )
                frame["station_id"] = station.station_id
                frame["station_name"] = station.station_name
                frame["region"] = station.region
                frame["eobs_latitude"] = station.eobs_latitude
                frame["eobs_longitude"] = station.eobs_longitude
                frame["grid_distance_km"] = station.grid_distance_km
                frames.append(frame)

    if not frames:
        raise ValueError("No E-OBS dates overlap the configured analysis period.")

    station = pd.concat(frames, ignore_index=True)
    station["MESS_DATUM"] = pd.to_datetime(station["MESS_DATUM"])
    station = station.drop_duplicates(["station_id", "MESS_DATUM"])
    station = station.sort_values(["region", "station_id", "MESS_DATUM"]).reset_index(drop=True)
    station["core_complete"] = station[["TXK", "RSK"]].notna().all(axis=1)
    return station


def prepare_region_series(station: pd.DataFrame) -> pd.DataFrame:
    region = (
        station.groupby(["region", "MESS_DATUM"], as_index=False)
        .agg(
            n_grid_cells=("station_id", "nunique"),
            n_complete_grid_cells=("core_complete", "sum"),
            TXK=("TXK", "mean"),
            RSK=("RSK", "mean"),
        )
        .sort_values(["region", "MESS_DATUM"])
        .reset_index(drop=True)
    )

    region["region_complete"] = (
        region["n_grid_cells"].eq(EXPECTED_CELLS_PER_REGION)
        & region["n_complete_grid_cells"].eq(EXPECTED_CELLS_PER_REGION)
    )
    region.loc[~region["region_complete"], ["TXK", "RSK"]] = float("nan")
    return region


def make_summary(mapping: pd.DataFrame, station: pd.DataFrame, region: pd.DataFrame) -> dict:
    station_summary = (
        station.groupby(["region", "station_id", "station_name"], as_index=False)
        .agg(days=("MESS_DATUM", "size"), complete_days=("core_complete", "sum"))
        .assign(complete_pct=lambda frame: (100 * frame["complete_days"] / frame["days"]).round(3))
        .merge(
            mapping[["station_id", "eobs_latitude", "eobs_longitude", "grid_distance_km"]],
            on="station_id",
            how="left",
        )
        .to_dict(orient="records")
    )
    region_summary = (
        region.groupby("region", as_index=False)
        .agg(days=("MESS_DATUM", "size"), complete_days=("region_complete", "sum"))
        .assign(complete_pct=lambda frame: (100 * frame["complete_days"] / frame["days"]).round(3))
        .to_dict(orient="records")
    )

    return {
        "dataset": f"E-OBS v{EOBS_VERSION} ensemble mean",
        "analysis_period": {"start": START_DATE.date().isoformat(), "end": END_DATE.date().isoformat()},
        "regional_aggregation": "Equal-weighted mean of the two validated station-matched E-OBS cells.",
        "core_variables": ["TXK", "RSK"],
        "missing_data_policy": "No values are imputed. Incomplete regional days are set to missing.",
        "station_summary": station_summary,
        "region_summary": region_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping()

    print("Extracting mapped E-OBS station-cell series...")
    station = extract_station_series(mapping)
    region = prepare_region_series(station)

    station_path = PROCESSED_DIR / "eobs_station_daily_1961_2024.parquet"
    region_path = PROCESSED_DIR / "eobs_region_daily_1961_2024.parquet"
    summary_path = OUTPUT_DIR / "eobs_region_preparation_summary.json"

    station.to_parquet(station_path, index=False)
    region.to_parquet(region_path, index=False)
    summary = make_summary(mapping, station, region)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {station_path}")
    print(f"Wrote {region_path}")
    print(f"Wrote {summary_path}")
    print("\nRegional completeness (%):")
    print(
        pd.DataFrame(summary["region_summary"])[
            ["region", "days", "complete_days", "complete_pct"]
        ].to_string(index=False)
    )
    print("\nMapped E-OBS cells:")
    print(
        mapping[
            ["station_id", "station_name", "region", "eobs_latitude", "eobs_longitude", "grid_distance_km"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
