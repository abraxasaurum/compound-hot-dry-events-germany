#!/usr/bin/env python3
"""Prepare analysis-ready daily station and regional DWD KL datasets.

The primary analysis period is 1961-01-01 to 2025-12-31. No missing values
are imputed. A regional value is retained only when both stations in that
region have complete core observations on that date.

Inputs:
    data/processed/daily_kl_*.parquet

Outputs:
    data/processed/analysis_station_daily_1961_2025.parquet
    data/processed/analysis_region_daily_1961_2025.parquet
    data/outputs/analysis_preparation_summary.json

Run:
    python src/prepare_analysis_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
START_DATE = pd.Timestamp("1961-01-01")
END_DATE = pd.Timestamp("2025-12-31")
CORE_VARIABLES = ["TMK", "TXK", "TNK", "RSK"]
EXPECTED_STATIONS_PER_REGION = 2


def load_station_files() -> pd.DataFrame:
    paths = sorted(PROCESSED_DIR.glob("daily_kl_*.parquet"))
    if len(paths) != 4:
        raise FileNotFoundError(
            f"Expected four daily_kl_*.parquet files in {PROCESSED_DIR}; found {len(paths)}."
        )

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"])
        missing = set(CORE_VARIABLES) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing core variables: {sorted(missing)}")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_station_data(data: pd.DataFrame) -> pd.DataFrame:
    station = data.loc[
        data["MESS_DATUM"].between(START_DATE, END_DATE)
    ].copy()

    station["core_complete"] = station[CORE_VARIABLES].notna().all(axis=1)
    station = station.sort_values(["region", "station_id", "MESS_DATUM"]).reset_index(drop=True)
    return station


def prepare_region_data(station: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        station.groupby(["region", "MESS_DATUM"], as_index=False)
        .agg(
            n_stations=("station_id", "nunique"),
            n_complete_stations=("core_complete", "sum"),
            TMK=("TMK", "mean"),
            TXK=("TXK", "mean"),
            TNK=("TNK", "mean"),
            RSK=("RSK", "mean"),
        )
        .sort_values(["region", "MESS_DATUM"])
        .reset_index(drop=True)
    )

    grouped["region_complete"] = (
        (grouped["n_stations"] == EXPECTED_STATIONS_PER_REGION)
        & (grouped["n_complete_stations"] == EXPECTED_STATIONS_PER_REGION)
    )

    grouped.loc[~grouped["region_complete"], CORE_VARIABLES] = float("nan")
    return grouped


def make_summary(station: pd.DataFrame, region: pd.DataFrame) -> dict:
    station_summary = (
        station.groupby(["region", "station_id", "station_name"], as_index=False)
        .agg(
            days=("MESS_DATUM", "size"),
            complete_days=("core_complete", "sum"),
        )
        .assign(
            complete_pct=lambda x: (100 * x["complete_days"] / x["days"]).round(3)
        )
        .to_dict(orient="records")
    )

    region_summary = (
        region.groupby("region", as_index=False)
        .agg(
            days=("MESS_DATUM", "size"),
            complete_days=("region_complete", "sum"),
        )
        .assign(
            complete_pct=lambda x: (100 * x["complete_days"] / x["days"]).round(3)
        )
        .to_dict(orient="records")
    )

    return {
        "analysis_period": {"start": START_DATE.date().isoformat(), "end": END_DATE.date().isoformat()},
        "core_variables": CORE_VARIABLES,
        "missing_data_policy": "No values are imputed. Incomplete regional days are set to missing.",
        "station_summary": station_summary,
        "region_summary": region_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_station = load_station_files()
    station = prepare_station_data(raw_station)
    region = prepare_region_data(station)

    station_path = PROCESSED_DIR / "analysis_station_daily_1961_2025.parquet"
    region_path = PROCESSED_DIR / "analysis_region_daily_1961_2025.parquet"
    station.to_parquet(station_path, index=False)
    region.to_parquet(region_path, index=False)

    summary = make_summary(station, region)
    summary_path = OUTPUT_DIR / "analysis_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {station_path}")
    print(f"Wrote {region_path}")
    print(f"Wrote {summary_path}")
    print("\nRegional completeness (%):")
    print(pd.DataFrame(summary["region_summary"])[["region", "days", "complete_days", "complete_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
