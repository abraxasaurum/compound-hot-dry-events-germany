#!/usr/bin/env python3
"""Compare DWD and E-OBS compound hot-dry event statistics by region.

The comparison is restricted to 1961-2024, the period shared by DWD and
E-OBS. Both inputs must have been created with the same event definition.

Inputs:
    data/outputs/compound_hot_dry_annual_p10.csv
    data/outputs/eobs_compound_hot_dry_annual_p10.csv

Outputs:
    data/outputs/dwd_eobs_event_comparison_p10_annual.csv
    data/outputs/dwd_eobs_event_comparison_p10_summary.csv
    data/outputs/dwd_eobs_event_comparison_p10_timeseries.png
    data/outputs/dwd_eobs_event_comparison_p10_scatter.png

Run:
    python src/compare_dwd_eobs_events.py
    python src/compare_dwd_eobs_events.py --dry-percentile 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, theilslopes


OUTPUT_DIR = Path("data/outputs")
START_YEAR = 1961
END_YEAR = 2024
REGION_LABELS = {
    "brandenburg_lusatia": "Brandenburg/Lusatia",
    "northwest": "Northwest",
}
COLORS = {"DWD": "#1b7837", "E-OBS": "#2166ac"}
METRICS = ("event_count", "event_days")
METRIC_LABELS = {
    "event_count": "Compound events",
    "event_days": "Compound-event days",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare annual DWD and E-OBS event statistics.")
    parser.add_argument(
        "--dry-percentile",
        type=int,
        choices=(10, 20),
        default=10,
        help="Dry-threshold percentile to compare (default: 10).",
    )
    return parser.parse_args()


def load_annual(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run derive_compound_hot_dry_events.py first.")
    frame = pd.read_csv(path)
    required = {"region", "year", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.loc[frame["year"].between(START_YEAR, END_YEAR)].copy()
    frame = frame[["region", "year", *METRICS]].rename(
        columns={metric: f"{source.lower().replace('-', '')}_{metric}" for metric in METRICS}
    )
    return frame


def paired_correlation(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
    values = frame[[left, right]].dropna()
    if len(values) < 3 or values[left].nunique() < 2 or values[right].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(values[left], values[right]).statistic)
    return float(spearmanr(values[left], values[right]).statistic)


def trend_per_decade(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    values = frame[["year", column]].dropna()
    if len(values) < 3 or values[column].nunique() < 2:
        return np.nan, np.nan, np.nan
    slope, _, lower, upper = theilslopes(values[column], values["year"], 0.95)
    return float(slope * 10), float(lower * 10), float(upper * 10)


def make_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in annual.groupby("region", sort=True):
        for metric in METRICS:
            dwd_column = f"dwd_{metric}"
            eobs_column = f"eobs_{metric}"
            difference = group[eobs_column] - group[dwd_column]
            dwd_slope, dwd_low, dwd_high = trend_per_decade(group, dwd_column)
            eobs_slope, eobs_low, eobs_high = trend_per_decade(group, eobs_column)
            dwd_present = group[dwd_column].gt(0)
            eobs_present = group[eobs_column].gt(0)
            union = (dwd_present | eobs_present).sum()

            rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "n_years": len(group),
                    "dwd_mean_annual": group[dwd_column].mean(),
                    "eobs_mean_annual": group[eobs_column].mean(),
                    "mean_bias_eobs_minus_dwd": difference.mean(),
                    "mean_absolute_error": difference.abs().mean(),
                    "pearson_r": paired_correlation(group, dwd_column, eobs_column, "pearson"),
                    "spearman_rho": paired_correlation(group, dwd_column, eobs_column, "spearman"),
                    "both_event_years": int((dwd_present & eobs_present).sum()),
                    "dwd_only_event_years": int((dwd_present & ~eobs_present).sum()),
                    "eobs_only_event_years": int((~dwd_present & eobs_present).sum()),
                    "event_year_jaccard": ((dwd_present & eobs_present).sum() / union) if union else np.nan,
                    "dwd_theil_sen_per_decade": dwd_slope,
                    "dwd_slope_ci_lower_per_decade": dwd_low,
                    "dwd_slope_ci_upper_per_decade": dwd_high,
                    "eobs_theil_sen_per_decade": eobs_slope,
                    "eobs_slope_ci_lower_per_decade": eobs_low,
                    "eobs_slope_ci_upper_per_decade": eobs_high,
                }
            )
    return pd.DataFrame(rows)


def plot_timeseries(annual: pd.DataFrame, output_path: Path, percentile: int) -> None:
    regions = sorted(annual["region"].unique())
    fig, axes = plt.subplots(
        nrows=len(regions), ncols=len(METRICS), figsize=(14, 7), sharex=True, constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for row, region in enumerate(regions):
        group = annual.loc[annual["region"] == region]
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            axis.plot(group["year"], group[f"dwd_{metric}"], color=COLORS["DWD"], linewidth=1.6, label="DWD")
            axis.plot(group["year"], group[f"eobs_{metric}"], color=COLORS["E-OBS"], linewidth=1.6, label="E-OBS")
            axis.set_title(f"{REGION_LABELS.get(region, region)} — {METRIC_LABELS[metric]}")
            axis.set_ylabel("Count")
            axis.grid(alpha=0.25)
            if row == 0 and column == 0:
                axis.legend(frameon=False)

    for axis in axes[-1, :]:
        axis.set_xlabel("Year")
    fig.suptitle(f"DWD versus E-OBS compound hot-dry events (P{percentile}, 1961–2024)", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(annual: pd.DataFrame, output_path: Path, percentile: int) -> None:
    regions = sorted(annual["region"].unique())
    fig, axes = plt.subplots(
        nrows=len(regions), ncols=len(METRICS), figsize=(11, 9), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for row, region in enumerate(regions):
        group = annual.loc[annual["region"] == region]
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            x = group[f"dwd_{metric}"]
            y = group[f"eobs_{metric}"]
            limit = max(x.max(), y.max(), 1)
            axis.scatter(x, y, color="#6a3d9a", alpha=0.75, s=28)
            axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
            axis.set_xlim(-0.1, limit + 0.1)
            axis.set_ylim(-0.1, limit + 0.1)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{REGION_LABELS.get(region, region)} — {METRIC_LABELS[metric]}")
            axis.set_xlabel(f"DWD annual {metric.replace('_', ' ')}")
            axis.set_ylabel(f"E-OBS annual {metric.replace('_', ' ')}")
            axis.grid(alpha=0.25)

    fig.suptitle(f"Annual compound-event agreement (P{percentile}, 1961–2024)", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dwd_path = OUTPUT_DIR / f"compound_hot_dry_annual_p{args.dry_percentile}.csv"
    eobs_path = OUTPUT_DIR / f"eobs_compound_hot_dry_annual_p{args.dry_percentile}.csv"

    dwd = load_annual(dwd_path, "DWD")
    eobs = load_annual(eobs_path, "E-OBS")
    annual = dwd.merge(eobs, on=["region", "year"], how="inner", validate="one_to_one")

    expected_rows = len(annual["region"].unique()) * (END_YEAR - START_YEAR + 1)
    if len(annual) != expected_rows:
        raise ValueError(f"Expected {expected_rows} paired regional years; found {len(annual)}.")

    summary = make_summary(annual)
    prefix = f"dwd_eobs_event_comparison_p{args.dry_percentile}"
    annual.to_csv(OUTPUT_DIR / f"{prefix}_annual.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"{prefix}_summary.csv", index=False)
    plot_timeseries(annual, OUTPUT_DIR / f"{prefix}_timeseries.png", args.dry_percentile)
    plot_scatter(annual, OUTPUT_DIR / f"{prefix}_scatter.png", args.dry_percentile)

    print(f"Compared DWD and E-OBS for {START_YEAR}-{END_YEAR} using P{args.dry_percentile} dryness.")
    print(f"Wrote paired annual data, summary, time-series plot and scatter plot to {OUTPUT_DIR}")
    print("\nComparison summary:")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
