#!/usr/bin/env python3
"""Create a year-to-date monitoring layer from current merged DWD station data.

This script uses the fixed 1961-1990 thresholds created by
derive_compound_hot_dry_events.py. It does not modify the fixed 1961-2025
reference analysis. Run this after every DWD ingestion update.

Inputs:
    data/processed/daily_kl_*.parquet
    data/outputs/compound_hot_dry_thresholds.csv
    data/processed/compound_hot_dry_daily_1961_2025.parquet

Outputs (for the latest available year by default):
    data/processed/monitoring_region_daily_<year>.parquet
    data/outputs/current_monitoring_events_<year>.csv
    data/outputs/current_monitoring_ytd_comparison_<year>.csv
    data/outputs/current_monitoring_summary_<year>.json

Run:
    python src/prepare_current_monitoring.py
    python src/prepare_current_monitoring.py --year 2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
THRESHOLD_PATH = OUTPUT_DIR / "compound_hot_dry_thresholds.csv"
REFERENCE_PATH = PROCESSED_DIR / "compound_hot_dry_daily_1961_2025.parquet"
CORE_VARIABLES = ["TMK", "TXK", "TNK", "RSK"]
WARM_MONTHS = {4, 5, 6, 7, 8, 9}
EXPECTED_STATIONS_PER_REGION = 2
MIN_EVENT_DAYS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare current DWD compound-event monitoring data.")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Monitoring year. Defaults to the latest year available in merged DWD data.",
    )
    return parser.parse_args()


def load_all_station_data() -> pd.DataFrame:
    paths = sorted(PROCESSED_DIR.glob("daily_kl_*.parquet"))
    if len(paths) != 4:
        raise FileNotFoundError(f"Expected four daily_kl_*.parquet files; found {len(paths)}.")

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"])
        missing = set(CORE_VARIABLES) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing core variables: {sorted(missing)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_regions(station: pd.DataFrame) -> pd.DataFrame:
    station = station.copy()
    station["core_complete"] = station[CORE_VARIABLES].notna().all(axis=1)

    region = (
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
    region["region_complete"] = (
        (region["n_stations"] == EXPECTED_STATIONS_PER_REGION)
        & (region["n_complete_stations"] == EXPECTED_STATIONS_PER_REGION)
    )
    region.loc[~region["region_complete"], CORE_VARIABLES] = float("nan")
    return region


def label_current_events(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled_frames = []
    event_frames = []

    for region_name, group in data.groupby("region", sort=True):
        group = group.sort_values("MESS_DATUM").copy()
        condition = group["compound_p10"].fillna(False).astype(bool)
        run_id = condition.ne(condition.shift(fill_value=False)).cumsum()
        run_length = condition.groupby(run_id).transform("sum")
        group["is_event_day"] = condition & (run_length >= MIN_EVENT_DAYS)
        group["event_run_id"] = run_id.where(group["is_event_day"], pd.NA)

        lookup = (
            group.loc[group["is_event_day"], ["event_run_id"]]
            .drop_duplicates()
            .sort_values("event_run_id")
            .reset_index(drop=True)
        )
        lookup["event_id"] = lookup.index + 1
        group = group.merge(lookup, on="event_run_id", how="left")
        group["event_id"] = group["event_id"].astype("Int64")
        labelled_frames.append(group)

        events = (
            group.loc[group["is_event_day"]]
            .groupby("event_id", as_index=False)
            .agg(
                start_date=("MESS_DATUM", "min"),
                end_date=("MESS_DATUM", "max"),
                duration_days=("MESS_DATUM", "size"),
                peak_txk=("TXK", "max"),
                minimum_precip_30d=("precip_30d", "min"),
                maximum_heat_excess=("heat_excess", "max"),
                maximum_precip_deficit=("precip_deficit", "max"),
            )
        )
        if not events.empty:
            events.insert(0, "region", region_name)
            events["start_date"] = events["start_date"].dt.date.astype(str)
            events["end_date"] = events["end_date"].dt.date.astype(str)
        event_frames.append(events)

    return pd.concat(labelled_frames, ignore_index=True), pd.concat(event_frames, ignore_index=True)


def historical_ytd_comparison(current: pd.DataFrame, monitoring_year: int) -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Missing {REFERENCE_PATH}. Run derive_compound_hot_dry_events.py first.")

    reference = pd.read_parquet(REFERENCE_PATH)
    reference["MESS_DATUM"] = pd.to_datetime(reference["MESS_DATUM"])
    latest_date = current["MESS_DATUM"].max()
    cutoff = latest_date.strftime("%m-%d")

    reference["month_day"] = reference["MESS_DATUM"].dt.strftime("%m-%d")
    historical = reference.loc[
        (reference["is_warm_season"])
        & (reference["month_day"] <= cutoff)
    ].copy()

    comparison = (
        historical.groupby(["region", historical["MESS_DATUM"].dt.year.rename("year")], as_index=False)
        .agg(
            complete_days=("region_complete", "sum"),
            compound_condition_days=("compound_p10", "sum"),
        )
        .rename(columns={"MESS_DATUM": "year"})
    )

    current_summary = (
        current.loc[current["is_warm_season"]]
        .groupby("region", as_index=False)
        .agg(
            complete_days=("region_complete", "sum"),
            compound_condition_days=("compound_p10", "sum"),
        )
    )
    current_summary["year"] = monitoring_year
    current_summary["record_type"] = "current_ytd"
    comparison["record_type"] = "historical_ytd"

    percentile_rows = []
    for region_name, group in comparison.groupby("region"):
        current_value = current_summary.loc[
            current_summary["region"] == region_name, "compound_condition_days"
        ].iloc[0]
        historical_values = group["compound_condition_days"]
        percentile_rows.append(
            {
                "region": region_name,
                "current_ytd_percentile": round(100 * (historical_values <= current_value).mean(), 2),
            }
        )
    current_summary = current_summary.merge(pd.DataFrame(percentile_rows), on="region", how="left")
    comparison["current_ytd_percentile"] = pd.NA

    return pd.concat([comparison, current_summary], ignore_index=True), latest_date


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    station = load_all_station_data()
    region = aggregate_regions(station)
    region["is_warm_season"] = region["MESS_DATUM"].dt.month.isin(WARM_MONTHS)
    region["calendar_day"] = region["MESS_DATUM"].dt.strftime("%m-%d")
    region["precip_30d"] = region.groupby("region", group_keys=False)["RSK"].apply(
        lambda values: values.rolling(30, min_periods=30).sum()
    )

    thresholds = pd.read_csv(THRESHOLD_PATH)
    region = region.merge(thresholds, on=["region", "calendar_day"], how="left")
    region["compound_p10"] = (
        region["region_complete"]
        & region["is_warm_season"]
        & region["TXK"].ge(region["heat_txk_p90"])
        & region["precip_30d"].le(region["precip_30d_p10"])
    )
    region["heat_excess"] = region["TXK"] - region["heat_txk_p90"]
    region["precip_deficit"] = region["precip_30d_p10"] - region["precip_30d"]

    monitoring_year = args.year or int(region["MESS_DATUM"].dt.year.max())
    current = region.loc[region["MESS_DATUM"].dt.year == monitoring_year].copy()
    if current.empty:
        raise ValueError(f"No data found for monitoring year {monitoring_year}.")

    current, events = label_current_events(current)
    comparison, latest_date = historical_ytd_comparison(current, monitoring_year)

    daily_path = PROCESSED_DIR / f"monitoring_region_daily_{monitoring_year}.parquet"
    events_path = OUTPUT_DIR / f"current_monitoring_events_{monitoring_year}.csv"
    comparison_path = OUTPUT_DIR / f"current_monitoring_ytd_comparison_{monitoring_year}.csv"
    summary_path = OUTPUT_DIR / f"current_monitoring_summary_{monitoring_year}.json"
    current.to_parquet(daily_path, index=False)
    events.to_csv(events_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    current_summary = (
        current.loc[current["is_warm_season"]]
        .groupby("region", as_index=False)
        .agg(
            complete_days=("region_complete", "sum"),
            compound_condition_days=("compound_p10", "sum"),
            event_days=("is_event_day", "sum"),
            event_count=("event_id", "nunique"),
            longest_event_days=("is_event_day", "sum"),
        )
    )
    if not events.empty:
        longest = events.groupby("region", as_index=False)["duration_days"].max().rename(
            columns={"duration_days": "longest_event_days"}
        )
        current_summary = current_summary.drop(columns="longest_event_days").merge(longest, on="region", how="left")
    current_summary["longest_event_days"] = current_summary["longest_event_days"].fillna(0).astype(int)

    percentiles = comparison.loc[comparison["record_type"] == "current_ytd", ["region", "current_ytd_percentile"]]
    current_summary = current_summary.merge(percentiles, on="region", how="left")

    summary = {
        "monitoring_year": monitoring_year,
        "latest_available_date": latest_date.date().isoformat(),
        "definition": "Fixed 1961-1990 calendar-day thresholds; TXK >= p90 and antecedent 30-day RSK <= p10; event >= 3 days.",
        "ytd_comparison": "Current warm-season compound-condition days are compared with every 1961-2025 year through the same month-day.",
        "regional_summary": current_summary.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Latest available DWD date: {latest_date.date().isoformat()}")
    print(f"Wrote {daily_path}")
    print(f"Wrote {events_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {summary_path}")
    print("\nCurrent year-to-date summary:")
    print(current_summary.to_string(index=False))


if __name__ == "__main__":
    main()
