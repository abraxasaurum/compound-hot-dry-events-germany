#!/usr/bin/env python3
"""Assess completeness and common availability of processed DWD daily KL data.

Inputs:
    data/processed/daily_kl_*.parquet

Outputs:
    data/outputs/data_quality_summary.csv
    data/outputs/data_quality_annual.csv
    data/outputs/common_complete_coverage.csv
    data/outputs/quality_report.json

No values are imputed. The script only documents availability.

Run:
    python src/quality_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
CORE_VARIABLES = ["TMK", "TXK", "TNK", "RSK"]
OPTIONAL_VARIABLES = ["UPM", "VPM", "SDK", "FM", "FX"]


def station_metadata(frame: pd.DataFrame, path: Path) -> dict:
    fallback_id = path.stem.split("_")[2]
    return {
        "station_id": str(frame["station_id"].iloc[0]).zfill(5)
        if "station_id" in frame.columns
        else fallback_id,
        "station_name": frame["station_name"].iloc[0]
        if "station_name" in frame.columns
        else path.stem,
        "region": frame["region"].iloc[0]
        if "region" in frame.columns
        else "unknown",
    }


def quality_summary(frame: pd.DataFrame, metadata: dict) -> list[dict]:
    rows = []
    for variable in CORE_VARIABLES + OPTIONAL_VARIABLES:
        if variable not in frame.columns:
            rows.append(
                {
                    **metadata,
                    "variable": variable,
                    "available_in_file": False,
                    "n_days": len(frame),
                    "n_valid": 0,
                    "n_missing": len(frame),
                    "completeness_pct": 0.0,
                    "first_valid": None,
                    "last_valid": None,
                }
            )
            continue

        values = frame[["MESS_DATUM", variable]].dropna(subset=[variable])
        rows.append(
            {
                **metadata,
                "variable": variable,
                "available_in_file": True,
                "n_days": len(frame),
                "n_valid": len(values),
                "n_missing": int(frame[variable].isna().sum()),
                "completeness_pct": round(100 * values.shape[0] / len(frame), 3),
                "first_valid": values["MESS_DATUM"].min().date().isoformat()
                if not values.empty
                else None,
                "last_valid": values["MESS_DATUM"].max().date().isoformat()
                if not values.empty
                else None,
            }
        )
    return rows


def annual_summary(frame: pd.DataFrame, metadata: dict) -> list[pd.DataFrame]:
    result = []
    data = frame.copy()
    data["year"] = data["MESS_DATUM"].dt.year
    for variable in CORE_VARIABLES + OPTIONAL_VARIABLES:
        if variable not in data.columns:
            continue
        annual = (
            data.groupby("year", as_index=False)
            .agg(
                calendar_days=("MESS_DATUM", "size"),
                valid_days=(variable, "count"),
            )
            .assign(
                completeness_pct=lambda x: (100 * x["valid_days"] / x["calendar_days"]).round(3),
                variable=variable,
                **metadata,
            )
        )
        result.append(annual)
    return result


def common_coverage(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    availability = []
    for station_id, frame in frames.items():
        missing_columns = set(CORE_VARIABLES) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{station_id} is missing core columns: {sorted(missing_columns)}")
        complete = frame.set_index("MESS_DATUM")[CORE_VARIABLES].notna().all(axis=1)
        availability.append(complete.rename(station_id))

    common = pd.concat(availability, axis=1).sort_index()
    common["all_stations_complete"] = common.all(axis=1)
    common["year"] = common.index.year

    annual = (
        common.groupby("year", as_index=False)
        .agg(
            calendar_days=("all_stations_complete", "size"),
            complete_days=("all_stations_complete", "sum"),
        )
        .assign(
            completeness_pct=lambda x: (100 * x["complete_days"] / x["calendar_days"]).round(3)
        )
    )
    return annual


def main() -> None:
    paths = sorted(PROCESSED_DIR.glob("daily_kl_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No processed files found in {PROCESSED_DIR}. Run ingest_dwd_daily_kl.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    annual_frames = []
    stations = []

    for path in paths:
        frame = pd.read_parquet(path)
        frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"])
        metadata = station_metadata(frame, path)
        frames[metadata["station_id"]] = frame
        stations.append(
            {
                **metadata,
                "n_days": len(frame),
                "first_date": frame["MESS_DATUM"].min().date().isoformat(),
                "last_date": frame["MESS_DATUM"].max().date().isoformat(),
                "file": str(path),
            }
        )
        summary_rows.extend(quality_summary(frame, metadata))
        annual_frames.extend(annual_summary(frame, metadata))

    quality = pd.DataFrame(summary_rows).sort_values(["station_id", "variable"])
    quality.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

    annual_quality = pd.concat(annual_frames, ignore_index=True).sort_values(
        ["station_id", "variable", "year"]
    )
    annual_quality.to_csv(OUTPUT_DIR / "data_quality_annual.csv", index=False)

    common = common_coverage(frames)
    common.to_csv(OUTPUT_DIR / "common_complete_coverage.csv", index=False)

    report = {
        "core_variables": CORE_VARIABLES,
        "optional_variables": OPTIONAL_VARIABLES,
        "stations": stations,
        "common_core_period": {
            "first_date": max(pd.to_datetime(item["first_date"]) for item in stations).date().isoformat(),
            "last_date": min(pd.to_datetime(item["last_date"]) for item in stations).date().isoformat(),
        },
        "rule": "A complete day has non-missing TMK, TXK, TNK and RSK at every station.",
        "note": "No missing values are imputed by this script.",
    }
    (OUTPUT_DIR / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    core_quality = quality.loc[quality["variable"].isin(CORE_VARIABLES)]
    print("Wrote quality-control outputs to data/outputs/")
    print("\nCore-variable completeness by station (%):")
    print(
        core_quality.pivot(index="station_name", columns="variable", values="completeness_pct")
        .round(2)
        .to_string()
    )
    print("\nAnnual common completeness for the last 10 years (%):")
    print(common.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
