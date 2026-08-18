#!/usr/bin/env python3
"""Subset validated E-OBS daily TX and RR data to Germany.

Inputs:
    data/raw/eobs/tx_ens_mean_0.1deg_reg_*_v31.0e.nc
    data/raw/eobs/rr_ens_mean_0.1deg_reg_*_v31.0e.nc

Outputs:
    data/processed/eobs/eobs_germany_1950-1964_v31.0e.nc
    data/processed/eobs/eobs_germany_1965-1979_v31.0e.nc
    data/processed/eobs/eobs_germany_1980-1994_v31.0e.nc
    data/processed/eobs/eobs_germany_1995-2010_v31.0e.nc
    data/processed/eobs/eobs_germany_2011-2024_v31.0e.nc
    data/outputs/eobs_germany_preparation_summary.json

The spatial domain is a deliberately simple geographic bounding box:
    longitude: 5.0°E to 16.0°E
    latitude: 47.0°N to 56.0°N

It includes Germany and a small surrounding margin. It is not yet a
political-border mask; this keeps the first xarray workflow transparent and
avoids making claims at national-boundary precision.

Run:
    python src/prepare_eobs_germany.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr


RAW_DIR = Path("data/raw/eobs")
PROCESSED_DIR = Path("data/processed/eobs")
OUTPUT_DIR = Path("data/outputs")

VERSION = "31.0e"
PERIODS = (
    "1950-1964",
    "1965-1979",
    "1980-1994",
    "1995-2010",
    "2011-2024",
)

LON_MIN = 5.0
LON_MAX = 16.0
LAT_MIN = 47.0
LAT_MAX = 56.0


def raw_path(variable: str, period: str) -> Path:
    return RAW_DIR / f"{variable}_ens_mean_0.1deg_reg_{period}_v{VERSION}.nc"


def coordinate_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(
        f"Could not find coordinate among {candidates}. "
        f"Available coordinates: {list(dataset.coords)}"
    )


def coordinate_slice(values: xr.DataArray, lower: float, upper: float) -> slice:
    first = float(values.values[0])
    last = float(values.values[-1])
    return slice(lower, upper) if first < last else slice(upper, lower)


def select_variable(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    if variable in dataset.data_vars:
        return dataset[variable]

    if len(dataset.data_vars) == 1:
        only_variable = next(iter(dataset.data_vars))
        return dataset[only_variable].rename(variable)

    raise KeyError(
        f"Could not identify E-OBS variable '{variable}'. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def open_and_subset(variable: str, period: str) -> xr.DataArray:
    path = raw_path(variable, period)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/download_eobs.py first."
        )

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        latitude = coordinate_name(dataset, ("latitude", "lat"))
        longitude = coordinate_name(dataset, ("longitude", "lon"))

        data = select_variable(dataset, variable)
        data = data.sel(
            {
                latitude: coordinate_slice(dataset[latitude], LAT_MIN, LAT_MAX),
                longitude: coordinate_slice(dataset[longitude], LON_MIN, LON_MAX),
            }
        )

        rename_map = {}
        if latitude != "latitude":
            rename_map[latitude] = "latitude"
        if longitude != "longitude":
            rename_map[longitude] = "longitude"

        data = data.rename(rename_map).load()

    data.name = variable
    return data


def output_encoding(dataset: xr.Dataset) -> dict[str, dict]:
    return {
        variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for variable in dataset.data_vars
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    print("Preparing E-OBS Germany bounding-box subsets")
    print(f"Longitude: {LON_MIN} to {LON_MAX}")
    print(f"Latitude:  {LAT_MIN} to {LAT_MAX}")
    print()

    for period in PERIODS:
        output_path = PROCESSED_DIR / f"eobs_germany_{period}_v{VERSION}.nc"

        print(f"Processing {period}")
        tx = open_and_subset("tx", period)
        rr = open_and_subset("rr", period)

        if not tx["time"].equals(rr["time"]):
            raise ValueError(f"TX and RR time coordinates differ for {period}.")
        if tx.sizes["latitude"] != rr.sizes["latitude"]:
            raise ValueError(f"TX and RR latitude dimensions differ for {period}.")
        if tx.sizes["longitude"] != rr.sizes["longitude"]:
            raise ValueError(f"TX and RR longitude dimensions differ for {period}.")

        rr = rr.assign_coords(
            latitude=tx["latitude"],
            longitude=tx["longitude"],
        )

        dataset = xr.Dataset({"tx": tx, "rr": rr})
        dataset.attrs.update(
            {
                "title": "E-OBS daily ensemble-mean data subset for Germany",
                "source_dataset": f"E-OBS v{VERSION}, 0.1 degree regular grid",
                "source_url": (
                    "https://surfobs.climate.copernicus.eu/"
                    "dataaccess/access_eobs_chunks.php"
                ),
                "spatial_subset": (
                    f"longitude {LON_MIN} to {LON_MAX}; "
                    f"latitude {LAT_MIN} to {LAT_MAX}"
                ),
                "note": (
                    "Geographic bounding box only; not a political-border mask. "
                    "TX is daily maximum temperature and RR is daily precipitation."
                ),
            }
        )

        dataset.to_netcdf(
            output_path,
            engine="netcdf4",
            encoding=output_encoding(dataset),
        )

        summary = {
            "period": period,
            "file": str(output_path),
            "time_start": str(dataset["time"].min().values)[:10],
            "time_end": str(dataset["time"].max().values)[:10],
            "n_days": int(dataset.sizes["time"]),
            "n_latitude": int(dataset.sizes["latitude"]),
            "n_longitude": int(dataset.sizes["longitude"]),
            "tx_units": dataset["tx"].attrs.get("units", "unknown"),
            "rr_units": dataset["rr"].attrs.get("units", "unknown"),
            "file_size_bytes": output_path.stat().st_size,
        }
        summaries.append(summary)

        print(
            f"  wrote {output_path.name}: "
            f"{summary['n_days']} days, "
            f"{summary['n_latitude']} × {summary['n_longitude']} grid cells"
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"E-OBS ensemble mean, version {VERSION}, 0.1° regular grid",
        "spatial_subset": {
            "longitude_min": LON_MIN,
            "longitude_max": LON_MAX,
            "latitude_min": LAT_MIN,
            "latitude_max": LAT_MAX,
        },
        "variables": {
            "tx": "Daily maximum temperature",
            "rr": "Daily precipitation sum",
        },
        "files": summaries,
    }

    report_path = OUTPUT_DIR / "eobs_germany_preparation_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_size = sum(item["file_size_bytes"] for item in summaries)
    print()
    print(f"Wrote {len(summaries)} Germany subset files to {PROCESSED_DIR}")
    print(f"Total subset size: {total_size / 1024**2:.1f} MiB")
    print(f"Wrote summary: {report_path}")


if __name__ == "__main__":
    main()
