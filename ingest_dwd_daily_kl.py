#!/usr/bin/env python3
"""Download, merge and persist daily DWD CDC KL observations.

Default stations:
- 03631 Norderney, 00691 Bremen
- 00880 Cottbus, 03015 Lindenberg

Run:
    python ingest_dwd_daily_kl.py
    python ingest_dwd_daily_kl.py --station 00880
    python ingest_dwd_daily_kl.py --refresh-historical

Dependencies:
    pip install pandas pyarrow requests
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/daily/kl"
)
HISTORICAL_URL = f"{BASE_URL}/historical/"
RECENT_URL = f"{BASE_URL}/recent/"
REQUEST_TIMEOUT = (60, 300)
MAX_RETRIES = 8

STATIONS = {
    "03631": {"name": "Norderney", "region": "northwest", "role": "coastal_reference"},
    "00691": {"name": "Bremen", "region": "northwest", "role": "inland_reference"},
    "00880": {"name": "Cottbus", "region": "brandenburg_lusatia", "role": "dry_reference"},
    "03015": {"name": "Lindenberg", "region": "brandenburg_lusatia", "role": "regional_reference"},
}

RAW_DIR = Path("data/raw/dwd_daily_kl")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")


##more robust for bad internet
SESSION = requests.Session()

retry_strategy = Retry(
    total=MAX_RETRIES,
    connect=MAX_RETRIES,
    read=MAX_RETRIES,
    status=MAX_RETRIES,
    backoff_factor=2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
)

adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)

SESSION.headers.update(
    {"User-Agent": "forest-climate-extremes/0.1 (research project)"}
)


def get_text(url: str) -> str:
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def download(url: str, destination: Path, overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        return destination
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def historical_archive_url(station_id: str) -> str:
    html = get_text(HISTORICAL_URL)
    pattern = rf'href="(tageswerte_KL_{station_id}_[^"]*_hist\.zip)"'
    matches = re.findall(pattern, html)
    if not matches:
        raise FileNotFoundError(f"No historical KL archive listed for station {station_id}.")
    return HISTORICAL_URL + sorted(matches)[-1]


def recent_archive_url(station_id: str) -> str:
    return RECENT_URL + f"tageswerte_KL_{station_id}_akt.zip"


def observation_member(archive: zipfile.ZipFile) -> str:
    candidates = [
        name for name in archive.namelist()
        if name.lower().endswith(".txt") and "produkt_klima_tag" in name.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one daily observation file, found: {candidates}")
    return candidates[0]


def read_archive(path: Path, source: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = observation_member(archive)
        with archive.open(member) as file:
            frame = pd.read_csv(
                io.BytesIO(file.read()),
                sep=";",
                encoding="latin1",
                na_values=[-999, -999.0, "-999", "-999.0"],
                engine="python",
            )
    frame.columns = frame.columns.str.strip()
    frame = frame.drop(columns="eor", errors="ignore")
    frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"].astype(str), format="%Y%m%d")
    frame["data_source"] = source
    return frame


def merge_observations(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([historical, recent], ignore_index=True, sort=False)
    combined["source_priority"] = combined["data_source"].map({"historical": 0, "recent": 1})
    combined = combined.sort_values(["MESS_DATUM", "source_priority"])
    combined = combined.drop_duplicates(subset=["MESS_DATUM"], keep="first")
    combined = combined.drop(columns="source_priority")
    return combined.sort_values("MESS_DATUM").reset_index(drop=True)


def process_station(station_id: str, refresh_historical: bool) -> dict:
    meta = STATIONS[station_id]
    station_raw_dir = RAW_DIR / station_id

    historical_url = historical_archive_url(station_id)
    historical_path = station_raw_dir / historical_url.rsplit("/", 1)[-1]
    recent_path = station_raw_dir / f"tageswerte_KL_{station_id}_akt.zip"

    download(historical_url, historical_path, overwrite=refresh_historical)
    download(recent_archive_url(station_id), recent_path, overwrite=True)

    historical = read_archive(historical_path, "historical")
    recent = read_archive(recent_path, "recent")
    merged = merge_observations(historical, recent)

    merged["station_id"] = station_id
    merged["station_name"] = meta["name"]
    merged["region"] = meta["region"]
    merged["station_role"] = meta["role"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"daily_kl_{station_id}_{meta['name'].lower().replace(' ', '_')}.parquet"
    merged.to_parquet(output_path, index=False)

    return {
        "station_id": station_id,
        "station_name": meta["name"],
        "region": meta["region"],
        "rows": len(merged),
        "first_date": merged["MESS_DATUM"].min().date().isoformat(),
        "last_date": merged["MESS_DATUM"].max().date().isoformat(),
        "historical_rows": int((merged["data_source"] == "historical").sum()),
        "recent_rows": int((merged["data_source"] == "recent").sum()),
        "parquet": str(output_path),
        "historical_url": historical_url,
        "recent_url": recent_archive_url(station_id),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest DWD CDC daily KL data.")
    parser.add_argument("--station", choices=STATIONS.keys(), action="append", help="Process one station; repeatable.")
    parser.add_argument("--refresh-historical", action="store_true", help="Re-download historical archives even if cached.")
    return parser.parse_args()


def main(station_ids: Iterable[str], refresh_historical: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for station_id in station_ids:
        print(f"Processing {station_id} — {STATIONS[station_id]['name']}")
        results.append(process_station(station_id, refresh_historical))

    run_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DWD CDC daily climate observations (KL)",
        "stations": results,
    }
    summary_path = OUTPUT_DIR / "ingest_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"Done. Wrote {summary_path}")
    print(pd.DataFrame(results)[["station_id", "station_name", "rows", "first_date", "last_date"]].to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(args.station or list(STATIONS), args.refresh_historical)
