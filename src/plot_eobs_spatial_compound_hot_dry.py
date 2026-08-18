#!/usr/bin/env python3
"""Create spatial maps and domain-mean time series for E-OBS compound hot-dry events.

Inputs:
    data/outputs/eobs_spatial_compound_summary_1961_2024.nc
    data/processed/eobs_spatial_compound_annual_1961_2024.parquet
    data/outputs/dwd_eobs_station_mapping.csv

Outputs:
    data/outputs/eobs_spatial_compound_event_days_maps.png
    data/outputs/eobs_spatial_compound_event_days_timeseries.png

Run:
    python src/plot_eobs_spatial_compound_hot_dry.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")
SUMMARY_PATH = OUTPUT_DIR / "eobs_spatial_compound_summary_1961_2024.nc"
ANNUAL_PATH = PROCESSED_DIR / "eobs_spatial_compound_annual_1961_2024.parquet"
MAPPING_PATH = OUTPUT_DIR / "dwd_eobs_station_mapping.csv"
BASELINE_LABEL = "1961–1990"
RECENT_LABEL = "2005–2024"
STATION_COLORS = {"brandenburg_lusatia": "#b35806", "northwest": "#2166ac"}


def add_reference_locations(axis: plt.Axes, mapping: pd.DataFrame) -> None:
    for region, group in mapping.groupby("region", sort=True):
        axis.scatter(
            group["longitude"],
            group["latitude"],
            color=STATION_COLORS.get(region, "black"),
            edgecolor="black",
            linewidth=0.6,
            s=42,
            label=region.replace("_", " "),
            zorder=3,
        )


def draw_map(axis: plt.Axes, field: xr.DataArray, title: str, cmap: str, vmin: float, vmax: float, label: str, mapping: pd.DataFrame) -> None:
    image = field.plot(
        ax=axis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": label, "shrink": 0.82},
    )
    add_reference_locations(axis, mapping)
    axis.set_title(title)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")


def make_maps(summary: xr.Dataset, mapping: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), constrained_layout=True)
    for row, percentile in enumerate((10, 20)):
        baseline = summary[f"baseline_mean_event_days_p{percentile}"]
        recent = summary[f"recent_mean_event_days_p{percentile}"]
        change = summary[f"change_event_days_p{percentile}"]
        upper = float(np.nanquantile(np.concatenate([baseline.values.ravel(), recent.values.ravel()]), 0.98))
        change_limit = float(np.nanquantile(np.abs(change.values.ravel()), 0.98))

        draw_map(
            axes[row, 0], baseline, f"P{percentile}: mean event days/year, {BASELINE_LABEL}",
            "YlOrRd", 0, upper, "Event days per warm season", mapping,
        )
        draw_map(
            axes[row, 1], recent, f"P{percentile}: mean event days/year, {RECENT_LABEL}",
            "YlOrRd", 0, upper, "Event days per warm season", mapping,
        )
        draw_map(
            axes[row, 2], change, f"P{percentile}: {RECENT_LABEL} minus {BASELINE_LABEL}",
            "RdBu_r", -change_limit, change_limit, "Change in event days per warm season", mapping,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="DWD references", loc="lower center", ncol=2, frameon=True)
    fig.suptitle("E-OBS spatial compound hot-dry event days", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_timeseries(annual: pd.DataFrame, output_path: Path) -> None:
    domain = annual.groupby("year", as_index=False)[["event_days_p10", "event_days_p20"]].mean()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True, constrained_layout=True)

    for axis, percentile, color in zip(axes, (10, 20), ("#b2182b", "#2166ac")):
        column = f"event_days_p{percentile}"
        rolling = domain[column].rolling(10, center=True, min_periods=5).mean()
        axis.plot(domain["year"], domain[column], color=color, alpha=0.35, linewidth=1, label="Annual grid-cell mean")
        axis.plot(domain["year"], rolling, color=color, linewidth=2.2, label="10-year centred mean")
        axis.axvspan(1961, 1990, color="grey", alpha=0.12, label="Baseline" if percentile == 10 else None)
        axis.axvspan(2005, 2024, color="orange", alpha=0.10, label="Recent period" if percentile == 10 else None)
        axis.set_title(f"P{percentile} dry threshold")
        axis.set_ylabel("Mean event days per grid cell")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=3, fontsize=8)

    axes[-1].set_xlabel("Year")
    fig.suptitle("Germany-box mean E-OBS compound hot-dry event days", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists() or not ANNUAL_PATH.exists() or not MAPPING_PATH.exists():
        raise FileNotFoundError("Missing spatial summary, annual data, or DWD-E-OBS station mapping.")

    summary = xr.open_dataset(SUMMARY_PATH)
    annual = pd.read_parquet(ANNUAL_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    map_path = OUTPUT_DIR / "eobs_spatial_compound_event_days_maps.png"
    timeseries_path = OUTPUT_DIR / "eobs_spatial_compound_event_days_timeseries.png"
    make_maps(summary, mapping, map_path)
    make_timeseries(annual, timeseries_path)

    print(f"Wrote {map_path}")
    print(f"Wrote {timeseries_path}")


if __name__ == "__main__":
    main()
