"""Download AEMO 5-minute price & demand data (public, no API key).

AEMO publishes one CSV per region per month at a stable URL. Files for
completed months never change, so they are cached in data/raw and only the
current month is re-downloaded.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from gridpulse.config import AEMO_PD_URL, PARQUET_DIR, RAW_DIR, REGIONS

log = logging.getLogger(__name__)


def month_list(months_back: int, today: date | None = None) -> list[str]:
    """Return yyyymm strings for the current month and `months_back` before it."""
    today = today or date.today()
    out = []
    y, m = today.year, today.month
    for _ in range(months_back + 1):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def fetch_month(region: str, yyyymm: str, force: bool = False) -> pd.DataFrame | None:
    """Download one region-month CSV, using the raw-file cache when possible."""
    dest = RAW_DIR / f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
    if dest.exists() and not force:
        return pd.read_csv(dest)
    url = AEMO_PD_URL.format(yyyymm=yyyymm, region=region)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        log.warning("No file for %s %s", region, yyyymm)
        return None
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return pd.read_csv(dest)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw AEMO frame: parse timestamps, keep trading intervals, rename."""
    df = df.rename(
        columns={
            "REGION": "region",
            "SETTLEMENTDATE": "settlement_ts",
            "TOTALDEMAND": "demand_mw",
            "RRP": "rrp",
            "PERIODTYPE": "period_type",
        }
    )
    # Timestamps are NEM time (AEST, UTC+10, no DST); keep naive-local.
    df["settlement_ts"] = pd.to_datetime(df["settlement_ts"], format="%Y/%m/%d %H:%M:%S")
    df = df[df["period_type"] == "TRADE"].drop(columns=["period_type"])
    return df.drop_duplicates(subset=["region", "settlement_ts"])


def ingest_prices(months_back: int = 24, today: date | None = None) -> pd.DataFrame:
    """Download all region-months, normalise, and write partitioned parquet."""
    current = (today or date.today()).strftime("%Y%m")
    frames = []
    for yyyymm in month_list(months_back, today):
        for region in REGIONS:
            df = fetch_month(region, yyyymm, force=(yyyymm == current))
            if df is not None and len(df):
                frames.append(normalise(df))
    prices = pd.concat(frames, ignore_index=True).sort_values(["region", "settlement_ts"])
    out = PARQUET_DIR / "prices.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out, index=False)
    log.info("Wrote %s rows to %s", len(prices), out)
    return prices
