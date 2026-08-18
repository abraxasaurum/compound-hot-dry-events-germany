#!/usr/bin/env python3
"""Validate E-OBS nearest-grid-cell values against four DWD KL stations.

The script compares daily DWD maximum temperature (TXK) and precipitation
(RSK) against E-OBS v31.0e ensemble-mean daily maximum temperature (tx) and
precipitation (rr) for the nearest 0.1° E-OBS cell to each station.

Inputs:
    data/processed/daily_kl_*.parquet
    data/processed/eobs/eobs_germany_*_v31.0e.nc
    DWD CDC current KL station metadata list

Outputs:
    data/outputs/dwd_eobs_station_mapping.csv
    data/outputs/dwd_eobs_daily_validation.parquet
    data/outputs/dwd_eobs_validation_summary.csv
    data/outputs/dwd_eobs_annual_validation.csv
    data/outputs/dwd_eobs_validation_trends.csv
    data/outputs/dwd_eobs_validation_scatter.png

Notes:
    - Validation is restricted to the shared 1961–2024 period.
    - E-OBS is an interpolated gridded observational product. Exact equality
      with DWD point observations is neither expected nor required.
    - Station coordinates come from DWD's current station metadata. Historic
      station moves or local exposure differences are not corrected here.

Run:
    python src/validate_dwd_eobs.py
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import xarray as xr
from scipy.stats import pearsonr, spearmanr, theilslopes


PROCESSED_DIR = Path("data/processed")
EOBS_DIR = PROCESSED_DIR / "eobs"
OUTPUT_DIR = Path("data/outputs")
EOBS_VERSION = "31.0e"
START_DATE = pd.Timestamp("1961-01-01")
END_DATE = pd.Timestamp("2024-12-31")
WARM_MONTHS = (4, 5, 6, 7, 8, 9)

STATION_METADATA_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/daily/kl/recent/"
    "KL_Tageswerte_Beschreibung_Stationen.txt"
)

VARIABLES = {
    "tx": {"dwd": "TXK", "eobs": "tx", "label": "Daily maximum temperature", "unit": "°C"},
    "rr": {"dwd": "RSK", "eobs": "rr", "label": "Daily precipitation sum", "unit": "mm"},
}

COLORS = {
    "brandenburg_lusatia": "#b35806",
    "northwest": "#2166ac",
}


def load_dwd_station_metadata() -> pd.DataFrame:
    """Read DWD fixed-width KL station metadata and retain needed fields."""
    response = requests.get(STATION_METADATA_URL, timeout=60)
    response.raise_for_status()

    columns = [
        "station_id",
        "valid_from",
        "valid_to",
        "elevation_m",
        "latitude",
        "longitude",
        "station_name_metadata",
        "federal_state",
        "data_release",
    ]
    metadata = pd.read_fwf(
        StringIO(response.text),
        skiprows=2,
        names=columns,
    )
    metadata["station_id"] = metadata["station_id"].astype(str).str.zfill(5)
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")
    return metadata.dropna(subset=["latitude", "longitude"])


def load_dwd_station_data() -> pd.DataFrame:
    paths = sorted(PROCESSED_DIR.glob("daily_kl_*.parquet"))
    if len(paths) != 4:
        raise FileNotFoundError(
            f"Expected four daily_kl_*.parquet files in {PROCESSED_DIR}; found {len(paths)}."
        )

    columns = ["MESS_DATUM", "station_id", "station_name", "region", "TXK", "RSK"]
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"])
        frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
        frames.append(frame)

    dwd = pd.concat(frames, ignore_index=True)
    dwd = dwd.loc[dwd["MESS_DATUM"].between(START_DATE, END_DATE)].copy()
    return dwd.sort_values(["station_id", "MESS_DATUM"]).reset_index(drop=True)


def make_station_mapping(dwd: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    stations = dwd[
        ["station_id", "station_name", "region"]
    ].drop_duplicates().sort_values("station_id")

    mapping = stations.merge(
        metadata[
            ["station_id", "latitude", "longitude", "elevation_m", "station_name_metadata"]
        ],
        on="station_id",
        how="left",
    )

    if mapping[["latitude", "longitude"]].isna().any().any():
        missing = mapping.loc[
            mapping["latitude"].isna() | mapping["longitude"].isna(),
            "station_id",
        ].tolist()
        raise ValueError(f"No DWD metadata coordinates found for station IDs: {missing}")

    return mapping


def eobs_paths() -> list[Path]:
    paths = sorted(EOBS_DIR.glob(f"eobs_germany_*_v{EOBS_VERSION}.nc"))
    if len(paths) != 5:
        raise FileNotFoundError(
            f"Expected five prepared E-OBS files in {EOBS_DIR}; found {len(paths)}."
        )
    return paths


def nearest_valid_eobs_cell(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    """Return the closest grid cell with valid tx and rr data."""
    valid = (
        dataset["tx"].notnull().any(dim="time")
        & dataset["rr"].notnull().any(dim="time")
    ).load()

    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")

    latitude_radians = np.radians(latitude)
    lat_grid_radians = np.radians(lat_grid)
    delta_latitude = lat_grid_radians - latitude_radians
    delta_longitude = np.radians(lon_grid - longitude)

    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_radians)
        * np.cos(lat_grid_radians)
        * np.sin(delta_longitude / 2) ** 2
    )
    distance_km = 6371 * 2 * np.arcsin(np.sqrt(haversine))
    distance_km = np.where(valid.values, distance_km, np.inf)

    latitude_index, longitude_index = np.unravel_index(
        np.argmin(distance_km),
        distance_km.shape,
    )

    return (
        float(latitudes[latitude_index]),
        float(longitudes[longitude_index]),
    )


def extract_eobs_station_series(mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the closest valid E-OBS land-cell time series for each station."""
    chunks = []
    grid_rows = []

    for path in eobs_paths():
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            dataset = dataset.sel(time=slice(START_DATE, END_DATE))

            for station in mapping.itertuples(index=False):
                eobs_latitude, eobs_longitude = nearest_valid_eobs_cell(
                    dataset,
                    station.latitude,
                    station.longitude,
                )

                point = dataset.sel(
                    latitude=eobs_latitude,
                    longitude=eobs_longitude,
                )[["tx", "rr"]].load()

                frame = point.to_dataframe().reset_index()
                frame["station_id"] = station.station_id
                frame["station_name"] = station.station_name
                frame["region"] = station.region
                chunks.append(frame)

                grid_rows.append(
                    {
                        "station_id": station.station_id,
                        "eobs_latitude": eobs_latitude,
                        "eobs_longitude": eobs_longitude,
                    }
                )

    eobs = pd.concat(chunks, ignore_index=True)
    eobs = eobs.rename(
        columns={
            "time": "MESS_DATUM",
            "tx": "eobs_tx",
            "rr": "eobs_rr",
        }
    )
    eobs["MESS_DATUM"] = pd.to_datetime(eobs["MESS_DATUM"])
    eobs = eobs.drop_duplicates(["station_id", "MESS_DATUM"]).sort_values(
        ["station_id", "MESS_DATUM"]
    )

    grid_mapping = pd.DataFrame(grid_rows).drop_duplicates("station_id")
    mapping = mapping.merge(grid_mapping, on="station_id", how="left")

    latitude_difference = np.radians(mapping["eobs_latitude"] - mapping["latitude"])
    longitude_difference = np.radians(mapping["eobs_longitude"] - mapping["longitude"])
    latitude_a = np.radians(mapping["latitude"])
    latitude_b = np.radians(mapping["eobs_latitude"])

    haversine = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_a)
        * np.cos(latitude_b)
        * np.sin(longitude_difference / 2) ** 2
    )
    mapping["grid_distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(haversine))

    return eobs, mapping

def correlation(values: pd.DataFrame, method: str) -> float:
    if len(values) < 3 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(values.iloc[:, 0], values.iloc[:, 1]).statistic)
    return float(spearmanr(values.iloc[:, 0], values.iloc[:, 1]).statistic)


def validation_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for station_id, group in daily.groupby("station_id", sort=True):
        station_info = group.iloc[0]
        for key, config in VARIABLES.items():
            pair = group[[config["dwd"], f"eobs_{key}"]].dropna()
            difference = pair[f"eobs_{key}"] - pair[config["dwd"]]

            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_info["station_name"],
                    "region": station_info["region"],
                    "variable": key,
                    "variable_label": config["label"],
                    "unit": config["unit"],
                    "n_common_days": len(pair),
                    "dwd_mean": pair[config["dwd"]].mean(),
                    "eobs_mean": pair[f"eobs_{key}"].mean(),
                    "mean_bias_eobs_minus_dwd": difference.mean(),
                    "mean_absolute_error": difference.abs().mean(),
                    "root_mean_square_error": np.sqrt(np.mean(difference**2)),
                    "pearson_r": correlation(pair, "pearson"),
                    "spearman_rho": correlation(pair, "spearman"),
                }
            )

    return pd.DataFrame(rows)


def annual_statistics(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = daily.copy()
    work["year"] = work["MESS_DATUM"].dt.year
    work["warm_season"] = work["MESS_DATUM"].dt.month.isin(WARM_MONTHS)
    work = work.loc[work["warm_season"]].copy()

    annual_rows = []
    trend_rows = []

    for station_id, group in work.groupby("station_id", sort=True):
        station_info = group.iloc[0]
        for key, config in VARIABLES.items():
            dwd_column = config["dwd"]
            eobs_column = f"eobs_{key}"
            annual = group.groupby("year", as_index=False)[[dwd_column, eobs_column]].mean()
            annual["station_id"] = station_id
            annual["station_name"] = station_info["station_name"]
            annual["region"] = station_info["region"]
            annual["variable"] = key
            annual_rows.append(annual)

            for source, column in (("DWD", dwd_column), ("E-OBS", eobs_column)):
                values = annual[["year", column]].dropna()
                slope, _, lower, upper = theilslopes(values[column], values["year"], 0.95)
                trend_rows.append(
                    {
                        "station_id": station_id,
                        "station_name": station_info["station_name"],
                        "region": station_info["region"],
                        "variable": key,
                        "source": source,
                        "n_years": len(values),
                        "theil_sen_slope_per_decade": slope * 10,
                        "slope_ci_lower_per_decade": lower * 10,
                        "slope_ci_upper_per_decade": upper * 10,
                    }
                )

    return pd.concat(annual_rows, ignore_index=True), pd.DataFrame(trend_rows)


def plot_scatter(daily: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), constrained_layout=True)

    for axis, key in zip(axes, ("tx", "rr")):
        config = VARIABLES[key]
        dwd_column = config["dwd"]
        eobs_column = f"eobs_{key}"

        for station_id, group in daily.groupby("station_id", sort=True):
            info = group.iloc[0]
            values = group[[dwd_column, eobs_column]].dropna()
            color = COLORS.get(info["region"], "grey")
            axis.scatter(
                values[dwd_column],
                values[eobs_column],
                s=5,
                alpha=0.18,
                color=color,
                label=info["station_name"],
            )

        limits = [
            min(axis.get_xlim()[0], axis.get_ylim()[0]),
            max(axis.get_xlim()[1], axis.get_ylim()[1]),
        ]
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(config["label"])
        axis.set_xlabel(f"DWD {dwd_column} ({config['unit']})")
        axis.set_ylabel(f"E-OBS {key} ({config['unit']})")
        axis.grid(alpha=0.2)
        axis.legend(markerscale=3, frameon=False, fontsize=8)

    fig.suptitle("Daily DWD station observations versus nearest E-OBS grid cell", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DWD station observations...")
    dwd = load_dwd_station_data()

    print("Loading DWD station coordinates...")
    metadata = load_dwd_station_metadata()
    mapping = make_station_mapping(dwd, metadata)

    print("Extracting nearest E-OBS cells...")
    eobs, mapping = extract_eobs_station_series(mapping)
    mapping.to_csv(OUTPUT_DIR / "dwd_eobs_station_mapping.csv", index=False)

    daily = dwd.merge(
        eobs,
        on=["station_id", "station_name", "region", "MESS_DATUM"],
        how="inner",
    )
    daily.to_parquet(OUTPUT_DIR / "dwd_eobs_daily_validation.parquet", index=False)

    summary = validation_summary(daily)
    summary.to_csv(OUTPUT_DIR / "dwd_eobs_validation_summary.csv", index=False)

    annual, trends = annual_statistics(daily)
    annual.to_csv(OUTPUT_DIR / "dwd_eobs_annual_validation.csv", index=False)
    trends.to_csv(OUTPUT_DIR / "dwd_eobs_validation_trends.csv", index=False)

    plot_scatter(daily, OUTPUT_DIR / "dwd_eobs_validation_scatter.png")

    print("\nStation-to-grid mapping:")
    print(
        mapping[
            [
                "station_id", "station_name", "region", "latitude", "longitude",
                "eobs_latitude", "eobs_longitude", "grid_distance_km", "elevation_m",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nDaily validation summary:")
    print(summary.round(3).to_string(index=False))
    print("\nWarm-season Theil-Sen trends per decade:")
    print(trends.round(3).to_string(index=False))
    print("\nWrote validation tables and scatter plot to data/outputs/")


if __name__ == "__main__":
    main()
