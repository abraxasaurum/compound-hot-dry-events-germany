#!/usr/bin/env python3
"""Analyse threshold sensitivity and robust annual event diagnostics.

Inputs:
    data/outputs/compound_hot_dry_annual_p10.csv
    data/outputs/compound_hot_dry_annual_p20.csv
    data/outputs/compound_hot_dry_events_p10.csv
    data/outputs/compound_hot_dry_events_p20.csv

Outputs:
    data/outputs/robustness_descriptive_statistics.csv
    data/outputs/robustness_trend_statistics.csv
    data/outputs/robustness_region_difference.csv
    data/outputs/robustness_threshold_comparison.csv
    data/outputs/robustness_event_days_trends.png

Methods:
    - Descriptive annual-event statistics, including zero-event years.
    - Kendall tau and Theil-Sen monotonic trend estimates.
    - Lag-1 autocorrelation diagnostic for annual series.
    - Moving-block bootstrap confidence intervals for regional annual
      differences. Blocks preserve short-run dependence better than
      independently resampling individual years.
    - P10 versus P20 dry-threshold sensitivity comparison.

This script is an exploratory robustness layer. It does not claim causal
attribution and does not replace a dedicated autocorrelation-adjusted
Mann-Kendall implementation.

Run:
    python src/analyze_robustness.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, theilslopes


OUTPUT_DIR = Path("data/outputs")
THRESHOLDS = (10, 20)
METRICS = ("event_count", "event_days", "longest_event_days")
N_BOOTSTRAP = 5_000
BLOCK_LENGTH = 5
RANDOM_SEED = 20260813

COLORS = {
    "brandenburg_lusatia": "#b35806",
    "northwest": "#2166ac",
}


def load_annual(threshold: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"compound_hot_dry_annual_p{threshold}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run derive_compound_hot_dry_events.py "
            "with --run-sensitivity first."
        )

    annual = pd.read_csv(path)
    required = {"region", "year", *METRICS}
    missing = required - set(annual.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    annual["threshold"] = f"P{threshold}"
    return annual


def lag1_autocorrelation(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 3 or values.nunique() < 2:
        return np.nan
    return float(values.autocorr(lag=1))


def descriptive_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (threshold, region), group in annual.groupby(["threshold", "region"], sort=True):
        for metric in METRICS:
            values = group[metric].dropna()
            rows.append(
                {
                    "threshold": threshold,
                    "region": region,
                    "metric": metric,
                    "n_years": len(values),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "zero_years": int((values == 0).sum()),
                    "zero_year_share_pct": 100 * (values == 0).mean(),
                    "lag1_autocorrelation": lag1_autocorrelation(values),
                }
            )

    return pd.DataFrame(rows)


def trend_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (threshold, region), group in annual.groupby(["threshold", "region"], sort=True):
        group = group.sort_values("year")

        for metric in METRICS:
            values = group[["year", metric]].dropna()
            tau, p_value = kendalltau(values["year"], values[metric])

            slope, intercept, lower, upper = theilslopes(
                values[metric],
                values["year"],
                0.95,
            )

            rows.append(
                {
                    "threshold": threshold,
                    "region": region,
                    "metric": metric,
                    "n_years": len(values),
                    "kendall_tau": tau,
                    "kendall_p_value": p_value,
                    "theil_sen_slope_per_year": slope,
                    "theil_sen_slope_per_decade": slope * 10,
                    "slope_ci_lower_per_decade": lower * 10,
                    "slope_ci_upper_per_decade": upper * 10,
                    "theil_sen_intercept": intercept,
                    "lag1_autocorrelation": lag1_autocorrelation(values[metric]),
                }
            )

    return pd.DataFrame(rows)


def moving_block_bootstrap_mean(
    values: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return observed mean and percentile bootstrap interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < block_length:
        return float(np.mean(values)), np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(values)
    starts = np.arange(n - block_length + 1)
    sampled_means = np.empty(n_bootstrap)

    for index in range(n_bootstrap):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(values[start : start + block_length])
        sampled_means[index] = np.mean(sample[:n])

    return (
        float(np.mean(values)),
        float(np.quantile(sampled_means, 0.025)),
        float(np.quantile(sampled_means, 0.975)),
    )


def region_difference_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for threshold, threshold_data in annual.groupby("threshold", sort=True):
        regions = sorted(threshold_data["region"].unique())
        if len(regions) != 2:
            raise ValueError(
                f"Expected exactly two regions for {threshold}; found {regions}."
            )

        first_region, second_region = regions

        for metric in METRICS:
            wide = (
                threshold_data.pivot(index="year", columns="region", values=metric)
                .dropna()
                .sort_index()
            )

            difference = wide[first_region] - wide[second_region]
            estimate, lower, upper = moving_block_bootstrap_mean(
                difference.to_numpy(),
                block_length=BLOCK_LENGTH,
                n_bootstrap=N_BOOTSTRAP,
                seed=RANDOM_SEED + int(threshold.replace("P", "")) + len(metric),
            )

            rows.append(
                {
                    "threshold": threshold,
                    "metric": metric,
                    "difference_definition": f"{first_region} minus {second_region}",
                    "n_common_years": len(difference),
                    "mean_annual_difference": estimate,
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "block_length_years": BLOCK_LENGTH,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )

    return pd.DataFrame(rows)


def threshold_comparison(
    annual: pd.DataFrame,
    events_by_threshold: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for region in sorted(annual["region"].unique()):
        for threshold in THRESHOLDS:
            threshold_label = f"P{threshold}"
            annual_subset = annual.loc[
                (annual["region"] == region)
                & (annual["threshold"] == threshold_label)
            ]
            events = events_by_threshold[threshold]
            event_subset = events.loc[events["region"] == region]

            rows.append(
                {
                    "region": region,
                    "threshold": threshold_label,
                    "total_events": int(annual_subset["event_count"].sum()),
                    "years_with_event": int((annual_subset["event_count"] > 0).sum()),
                    "total_event_days": int(annual_subset["event_days"].sum()),
                    "mean_event_days_per_year": annual_subset["event_days"].mean(),
                    "median_event_days_per_year": annual_subset["event_days"].median(),
                    "median_event_duration": event_subset["duration_days"].median(),
                    "mean_event_duration": event_subset["duration_days"].mean(),
                    "maximum_event_duration": event_subset["duration_days"].max(),
                }
            )

    return pd.DataFrame(rows)


def plot_event_day_trends(annual: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )

    for axis, threshold in zip(axes, ("P10", "P20")):
        subset = annual.loc[annual["threshold"] == threshold]

        for region, group in subset.groupby("region", sort=True):
            group = group.sort_values("year")
            color = COLORS.get(region)
            label = region.replace("_", " ")

            axis.plot(
                group["year"],
                group["event_days"],
                color=color,
                alpha=0.28,
                linewidth=0.9,
                marker="o",
                markersize=2.2,
            )

            rolling = group["event_days"].rolling(
                window=5,
                center=True,
                min_periods=3,
            ).mean()
            axis.plot(
                group["year"],
                rolling,
                color=color,
                linewidth=2.2,
                label=label,
            )

            slope, intercept, _, _ = theilslopes(
                group["event_days"],
                group["year"],
                0.95,
            )
            axis.plot(
                group["year"],
                intercept + slope * group["year"],
                color=color,
                linestyle="--",
                linewidth=1.4,
            )

        axis.set_title(
            f"{threshold}: annual compound hot-dry event days "
            "(line = 5-year mean; dashed = Theil-Sen trend)"
        )
        axis.set_ylabel("Event days per year")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=2)

    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(int(annual["year"].min()), int(annual["year"].max()))
    sns.despine(fig=fig)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annual = pd.concat(
        [load_annual(threshold) for threshold in THRESHOLDS],
        ignore_index=True,
    )

    events_by_threshold: dict[int, pd.DataFrame] = {}
    for threshold in THRESHOLDS:
        event_path = OUTPUT_DIR / f"compound_hot_dry_events_p{threshold}.csv"
        if not event_path.exists():
            raise FileNotFoundError(f"Missing {event_path}")
        events_by_threshold[threshold] = pd.read_csv(event_path)

    descriptive = descriptive_statistics(annual)
    trends = trend_statistics(annual)
    region_differences = region_difference_statistics(annual)
    comparison = threshold_comparison(annual, events_by_threshold)

    descriptive.to_csv(
        OUTPUT_DIR / "robustness_descriptive_statistics.csv",
        index=False,
    )
    trends.to_csv(
        OUTPUT_DIR / "robustness_trend_statistics.csv",
        index=False,
    )
    region_differences.to_csv(
        OUTPUT_DIR / "robustness_region_difference.csv",
        index=False,
    )
    comparison.to_csv(
        OUTPUT_DIR / "robustness_threshold_comparison.csv",
        index=False,
    )

    plot_event_day_trends(
        annual,
        OUTPUT_DIR / "robustness_event_days_trends.png",
    )

    print("Wrote robustness-analysis outputs to data/outputs/")
    print("\nThreshold comparison:")
    print(comparison.round(3).to_string(index=False))

    print("\nTrend statistics: event days per decade")
    print(
        trends.loc[
            trends["metric"] == "event_days",
            [
                "threshold",
                "region",
                "kendall_tau",
                "kendall_p_value",
                "theil_sen_slope_per_decade",
                "slope_ci_lower_per_decade",
                "slope_ci_upper_per_decade",
                "lag1_autocorrelation",
            ],
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nAnnual regional differences: Brandenburg/Lusatia minus Northwest")
    print(region_differences.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
