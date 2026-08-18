#!/usr/bin/env python3
"""Download validated E-OBS daily gridded ensemble-mean data.

Downloads E-OBS v31.0e daily maximum temperature (TX) and precipitation
(RR) at 0.1 degree resolution in approximately 15-year NetCDF chunks.

Data source:
https://surfobs.climate.copernicus.eu/dataaccess/access_eobs_chunks.php

Outputs:
    data/raw/eobs/tx_ens_mean_0.1deg_reg_1950-1964_v31.0e.nc
    ...
    data/raw/eobs/rr_ens_mean_0.1deg_reg_2011-2024_v31.0e.nc
    data/raw/eobs/download_manifest.csv

Run:
    python src/download_eobs.py --dry-run
    python src/download_eobs.py
    python src/download_eobs.py --variables tx rr
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import requests


RAW_DIR = Path("data/raw/eobs")
BASE_URL = (
    "https://knmi-ecad-assets-prd.s3.amazonaws.com/"
    "ensembles/data/Grid_0.1deg_reg_ensemble"
)
VERSION = "31.0e"
GRID = "0.1deg_reg"
PERIODS = (
    "1950-1964",
    "1965-1979",
    "1980-1994",
    "1995-2010",
    "2011-2024",
)
VARIABLES = {
    "tx": "daily maximum temperature",
    "rr": "daily precipitation sum",
}
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download validated E-OBS daily NetCDF chunks."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=sorted(VARIABLES),
        default=sorted(VARIABLES),
        help="Variables to download; default: tx rr.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if a non-empty local file exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned downloads without downloading files.",
    )
    return parser.parse_args()


def make_url(variable: str, period: str) -> str:
    filename = f"{variable}_ens_mean_{GRID}_{period}_v{VERSION}.nc"
    return f"{BASE_URL}/{filename}"


def format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "unknown size"

    units = ("B", "KiB", "MiB", "GiB")
    value = float(size_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} GiB"


def download_file(url: str, destination: Path) -> tuple[str, int | None]:
    temporary = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        expected_size = int(content_length) if content_length else None
        downloaded = 0

        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)

        if expected_size is not None and downloaded != expected_size:
            temporary.unlink(missing_ok=True)
            raise IOError(
                f"Incomplete download for {destination.name}: "
                f"received {downloaded} bytes, expected {expected_size} bytes."
            )

    temporary.replace(destination)
    return "downloaded", expected_size


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    print("E-OBS validated ensemble-mean download plan")
    print(f"Version: {VERSION}")
    print(f"Resolution: {GRID}")
    print(f"Variables: {', '.join(args.variables)}")
    print(f"Target directory: {RAW_DIR}")
    print()

    for variable in args.variables:
        for period in PERIODS:
            url = make_url(variable, period)
            filename = url.rsplit("/", maxsplit=1)[-1]
            destination = RAW_DIR / filename

            row = {
                "variable": variable,
                "variable_description": VARIABLES[variable],
                "period": period,
                "version": VERSION,
                "grid": GRID,
                "url": url,
                "local_file": str(destination),
                "retrieved_utc": "",
                "status": "",
                "size_bytes": "",
            }

            if args.dry_run:
                row["status"] = "planned"
                print(f"PLAN  {filename}")
                manifest_rows.append(row)
                continue

            if destination.exists() and destination.stat().st_size > 0 and not args.force:
                row["status"] = "already_present"
                row["size_bytes"] = destination.stat().st_size
                row["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
                print(
                    f"SKIP  {filename} "
                    f"({format_size(destination.stat().st_size)})"
                )
                manifest_rows.append(row)
                continue

            print(f"GET   {filename}")
            try:
                status, expected_size = download_file(url, destination)
                row["status"] = status
                row["size_bytes"] = (
                    expected_size
                    if expected_size is not None
                    else destination.stat().st_size
                )
                row["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
                print(
                    f"DONE  {filename} "
                    f"({format_size(destination.stat().st_size)})"
                )
            except requests.RequestException as error:
                row["status"] = f"failed: {error}"
                print(f"FAIL  {filename}: {error}")
            except OSError as error:
                row["status"] = f"failed: {error}"
                print(f"FAIL  {filename}: {error}")

            manifest_rows.append(row)

    manifest_path = RAW_DIR / "download_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Wrote manifest: {manifest_path}")

    failed = [row for row in manifest_rows if str(row["status"]).startswith("failed")]
    if failed:
        raise SystemExit(f"{len(failed)} E-OBS download(s) failed.")


if __name__ == "__main__":
    main()
