#!/usr/bin/env python3
"""Derive compound hot–dry events from prepared regional DWD daily data.

Primary definition
------------------
Warm season: April–September
Heat: TXK >= calendar-day-specific 90th percentile
Dry condition: antecedent 30-day RSK sum <= calendar-day-specific 10th percentile
Event: >= 3 consecutive compound days
Baseline for thresholds: 1961–1990 (fixed)

The calendar-day thresholds use a +/- 15-day window in the baseline period.
No missing values are imputed. The p20 dry threshold is also calculated and
can be used as an optional sensitivity analysis.

Run primary analysis:
    python src/derive_compound_hot_dry_events.py

Run primary and p20 dry-threshold sensitivity analysis:
    python src/derive_compound_hot_dry_events.py --run-sensitivity

Example with a different minimum duration:
    python src/derive_compound_hot_dry_events.py --event-min-days 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/analysis_region_daily_1961_2025.parquet")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
WARM_MONTHS = {4, 5, 6, 7, 8, 9}
DEFAULT_BASELINE_START = 1961
DEFAULT_BASELINE_END = 1990
DEFAULT_EVENT_MIN_DAYS = 3
WINDOW_DAYS = 15
HEAT_PERCENTILE = 0.90
DRY_PERCENTILES = (0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive compound hot-dry events.")
    parser.add_argument("--baseline-start", type=int, default=DEFAULT_BASELINE_START)
    parser.add_argument("--baseline-end", type=int, default=DEFAULT_BASELINE_END)
    parser.add_argument("--event-min-days", type=int, default=DEFAULT_EVENT_MIN_DAYS)
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Also derive events using the moderate p20 dry-condition threshold.",
    )
    return parser.parse_args()


def climatology_day(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(
    "2000-" + dates.dt.strftime("%m-%d"),
    format="%Y-%m-%d",
    )


def threshold_table(region_data: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    baseline = region_data.loc[
        (region_data["MESS_DATUM"].dt.year.between(baseline_start, baseline_end))
        & region_data["region_complete"]
    ].copy()
    baseline["clim_date"] = climatology_day(baseline["MESS_DATUM"])

    warm_calendar = pd.date_range("2000-04-01", "2000-09-30", freq="D")
    rows = []
    for target in warm_calendar:
        distance = (baseline["clim_date"] - target).abs().dt.days
        window = baseline.loc[distance <= WINDOW_DAYS]
        if window.empty:
            raise ValueError(f"No baseline observations for calendar day {target:%m-%d}.")
        rows.append(
            {
                "calendar_day": target.strftime("%m-%d"),
                "heat_txk_p90": window["TXK"].quantile(HEAT_PERCENTILE),
                "precip_30d_p10": window["precip_30d"].quantile(DRY_PERCENTILES[0]),
                "precip_30d_p20": window["precip_30d"].quantile(DRY_PERCENTILES[1]),
                "baseline_n_days": int(len(window)),
            }
        )
    return pd.DataFrame(rows)


def label_events(data: pd.DataFrame, condition_column: str, min_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.copy()
    condition = frame[condition_column].fillna(False)

    previous_condition = (
        condition.groupby(frame["region"])
        .shift()
        .fillna(False)
    )

    run_id = (
        condition.ne(previous_condition)
        .groupby(frame["region"])
        .cumsum()
    )

    run_length = condition.groupby(
        [frame["region"], run_id]
    ).transform("sum")

    frame["is_event_day"] = condition & (run_length >= min_days)

    frame["event_run_id"] = run_id.where(
        frame["is_event_day"],
        pd.NA,
    )

    event_lookup = (
        frame.loc[
            frame["is_event_day"],
            ["region", "event_run_id"],
        ]
        .drop_duplicates()
        .sort_values(["region", "event_run_id"])
        .reset_index(drop=True)
    )

    event_lookup["event_id"] = (
        event_lookup.groupby("region")
        .cumcount()
        .add(1)
    )

    frame = frame.merge(
        event_lookup,
        on=["region", "event_run_id"],
        how="left",
    )

    frame["event_id"] = frame["event_id"].astype("Int64")

    events = (
        frame.loc[frame["is_event_day"]]
        .groupby(["region", "event_id"], as_index=False)
        .agg(
            start_date=("MESS_DATUM", "min"),
            end_date=("MESS_DATUM", "max"),
            duration_days=("MESS_DATUM", "size"),
            peak_txk=("TXK", "max"),
            mean_txk=("TXK", "mean"),
            minimum_precip_30d=("precip_30d", "min"),
            mean_heat_excess=("heat_excess", "mean"),
            maximum_heat_excess=("heat_excess", "max"),
            mean_precip_deficit=("precip_deficit", "mean"),
            maximum_precip_deficit=("precip_deficit", "max"),
        )
    )
    if not events.empty:
        events["start_date"] = events["start_date"].dt.date.astype(str)
        events["end_date"] = events["end_date"].dt.date.astype(str)
        events["year"] = pd.to_datetime(events["start_date"]).dt.year
    return frame, events


def annual_summary(events: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    years = pd.DataFrame({"year": np.arange(1961, 2026)})
    all_groups = pd.MultiIndex.from_product([regions, years["year"]], names=["region", "year"]).to_frame(index=False)
    if events.empty:
        result = all_groups.copy()
        result[["event_count", "event_days", "longest_event_days", "max_peak_txk"]] = 0
        return result

    summary = (
        events.groupby(["region", "year"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            event_days=("duration_days", "sum"),
            longest_event_days=("duration_days", "max"),
            max_peak_txk=("peak_txk", "max"),
        )
    )
    result = all_groups.merge(summary, on=["region", "year"], how="left")
    result[["event_count", "event_days", "longest_event_days"]] = result[
        ["event_count", "event_days", "longest_event_days"]
    ].fillna(0).astype(int)
    return result


def run_definition(data: pd.DataFrame, dry_percentile: int, min_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_column = f"precip_30d_p{dry_percentile}"
    condition_column = f"compound_p{dry_percentile}"
    frame = data.copy()
    frame[condition_column] = (
        frame["region_complete"]
        & frame["TXK"].ge(frame["heat_txk_p90"])
        & frame["precip_30d"].le(frame[threshold_column])
    )
    frame["heat_excess"] = frame["TXK"] - frame["heat_txk_p90"]
    frame["precip_deficit"] = frame[threshold_column] - frame["precip_30d"]

    labelled, events = label_events(frame, condition_column, min_days)
    annual = annual_summary(events, sorted(frame["region"].unique()))
    return labelled, events, annual


def main() -> None:
    args = parse_args()
    if args.baseline_end < args.baseline_start:
        raise ValueError("baseline-end must be >= baseline-start")
    if args.event_min_days < 1:
        raise ValueError("event-min-days must be at least 1")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run prepare_analysis_data.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(INPUT_PATH)
    data["MESS_DATUM"] = pd.to_datetime(data["MESS_DATUM"])
    data = data.sort_values(["region", "MESS_DATUM"]).reset_index(drop=True)
    data["is_warm_season"] = data["MESS_DATUM"].dt.month.isin(WARM_MONTHS)
    data["calendar_day"] = data["MESS_DATUM"].dt.strftime("%m-%d")
    data["precip_30d"] = data.groupby("region", group_keys=False)["RSK"].apply(
        lambda values: values.rolling(30, min_periods=30).sum()
    )

    threshold_frames = []
    for region, region_data in data.groupby("region", sort=True):
        thresholds = threshold_table(region_data, args.baseline_start, args.baseline_end)
        thresholds.insert(0, "region", region)
        threshold_frames.append(thresholds)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    thresholds.to_csv(OUTPUT_DIR / "compound_hot_dry_thresholds.csv", index=False)

    data = data.merge(thresholds, on=["region", "calendar_day"], how="left")
    data.loc[~data["is_warm_season"], ["heat_txk_p90", "precip_30d_p10", "precip_30d_p20"]] = np.nan

    definitions = [10, 20] if args.run_sensitivity else [10]
    primary_daily = None
    for dry_percentile in definitions:
        daily, events, annual = run_definition(data, dry_percentile, args.event_min_days)
        events.to_csv(OUTPUT_DIR / f"compound_hot_dry_events_p{dry_percentile}.csv", index=False)
        annual.to_csv(OUTPUT_DIR / f"compound_hot_dry_annual_p{dry_percentile}.csv", index=False)
        if dry_percentile == 10:
            primary_daily = daily

    primary_path = PROCESSED_DIR / "compound_hot_dry_daily_1961_2025.parquet"
    primary_daily.to_parquet(primary_path, index=False)

    config = {
        "analysis_period": "1961-01-01 to 2025-12-31",
        "warm_season": "April to September",
        "baseline_period": f"{args.baseline_start}-01-01 to {args.baseline_end}-12-31",
        "calendar_window_days": WINDOW_DAYS,
        "heat_definition": "TXK >= calendar-day-specific p90",
        "dry_definition_primary": "antecedent 30-day RSK sum <= calendar-day-specific p10",
        "event_definition": f"at least {args.event_min_days} consecutive compound days",
        "sensitivity_dry_threshold_p20_created": args.run_sensitivity,
        "missing_data_policy": "No imputation; incomplete regional days cannot be compound days.",
    }
    (OUTPUT_DIR / "compound_hot_dry_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    primary_events = pd.read_csv(OUTPUT_DIR / "compound_hot_dry_events_p10.csv")
    print(f"Wrote {primary_path}")
    print("Wrote thresholds, events, annual summaries and configuration to data/outputs/")
    print("\nPrimary p10 event counts:")
    if primary_events.empty:
        print("No primary events found.")
    else:
        print(primary_events.groupby("region").size().rename("events").to_string())


if __name__ == "__main__":
    main()
