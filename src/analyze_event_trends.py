#!/usr/bin/env python3
"""Analyse annual compound hot-dry event statistics and monotonic trends.

Inputs:
    data/outputs/compound_hot_dry_annual_p10.csv
    data/outputs/compound_hot_dry_events_p10.csv

Outputs:
    data/outputs/compound_hot_dry_trend_statistics.csv
    data/outputs/annual_event_count.png
    data/outputs/annual_event_days.png
    data/outputs/event_duration_distribution.png

Trend method:
    Kendall rank correlation of annual metrics against year, plus Theil-Sen slope
    and its 95% confidence interval. This is an initial monotonic-trend analysis;
    no autocorrelation adjustment is applied in this MVP.

Run:
    python src/analyze_event_trends.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, theilslopes

OUTPUT_DIR = Path("data/outputs")
ANNUAL_PATH = OUTPUT_DIR / "compound_hot_dry_annual_p10.csv"
EVENTS_PATH = OUTPUT_DIR / "compound_hot_dry_events_p10.csv"
TREND_METRICS = ["event_count", "event_days", "longest_event_days"]
COLORS = {
    "brandenburg_lusatia": "#b35806",
    "brandenburg lusatia": "#b35806",
    "northwest": "#2166ac",
}


def trend_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in annual.groupby("region", sort=True):
        for metric in TREND_METRICS:
            values = group[["year", metric]].dropna()
            tau, p_value = kendalltau(values["year"], values[metric])
            slope, intercept, lower_slope, upper_slope = theilslopes(
                values[metric], values["year"], 0.95
            )
            rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "n_years": len(values),
                    "kendall_tau": tau,
                    "p_value": p_value,
                    "theil_sen_slope_per_year": slope,
                    "theil_sen_slope_ci_lower": lower_slope,
                    "theil_sen_slope_ci_upper": upper_slope,
                    "theil_sen_intercept": intercept,
                }
            )
    return pd.DataFrame(rows)


def plot_annual_metric(annual: pd.DataFrame, statistics: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    for region, group in annual.groupby("region", sort=True):
        group = group.sort_values("year")
        color = COLORS.get(region, None)
        label = region.replace("_", " ")
        rolling = group[metric].rolling(5, center=True, min_periods=3).mean()

        ax.plot(
            group["year"], group[metric], color=color, alpha=0.28,
            linewidth=0.9, marker="o", markersize=2.5, label=f"{label}: annual",
        )
        ax.plot(
            group["year"], rolling, color=color, linewidth=2.5,
            label=f"{label}: 5-year mean",
        )

        stat = statistics.loc[
            (statistics["region"] == region)
            & (statistics["metric"] == metric)
        ].iloc[0]
        trend = (
            stat["theil_sen_intercept"]
            + stat["theil_sen_slope_per_year"] * group["year"]
        )
        ax.plot(group["year"], trend, color=color, linestyle="--", linewidth=1.5)

    labels = {
        "event_count": "Compound hot-dry events per year",
        "event_days": "Compound event days per year",
    }
    ax.set_title(labels[metric])
    ax.set_xlabel("Year")
    ax.set_ylabel(labels[metric])
    ax.set_xlim(1961, 2025)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_duration_distribution(events: pd.DataFrame, output_path: Path) -> None:
    plot_data = events.copy()
    plot_data["region_label"] = plot_data["region"].str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    sns.boxplot(
        data=plot_data,
        x="region_label",
        y="duration_days",
        hue="region_label",
        palette=COLORS,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="region_label",
        y="duration_days",
        color="black",
        alpha=0.5,
        size=3,
        ax=ax,
    )
    ax.set_title("Distribution of compound hot-dry event duration")
    ax.set_xlabel("")
    ax.set_ylabel("Event duration (days)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not ANNUAL_PATH.exists() or not EVENTS_PATH.exists():
        raise FileNotFoundError(
            "Missing annual or event CSV. Run derive_compound_hot_dry_events.py first."
        )

    annual = pd.read_csv(ANNUAL_PATH)
    events = pd.read_csv(EVENTS_PATH)
    statistics = trend_statistics(annual)
    statistics.to_csv(OUTPUT_DIR / "compound_hot_dry_trend_statistics.csv", index=False)

    plot_annual_metric(
        annual,
        statistics,
        "event_count",
        OUTPUT_DIR / "annual_event_count.png",
    )
    plot_annual_metric(
        annual,
        statistics,
        "event_days",
        OUTPUT_DIR / "annual_event_days.png",
    )
    plot_duration_distribution(
        events,
        OUTPUT_DIR / "event_duration_distribution.png",
    )

    print("Wrote trend statistics and figures to data/outputs/")
    print("\nTrend statistics:")
    print(
        statistics[
            [
                "region",
                "metric",
                "kendall_tau",
                "p_value",
                "theil_sen_slope_per_year",
                "theil_sen_slope_ci_lower",
                "theil_sen_slope_ci_upper",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
