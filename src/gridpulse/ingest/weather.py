"""Daily temperature history + forecasts per NEM region capital (Open-Meteo, free)."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from gridpulse.config import PARQUET_DIR, REGIONS

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_max,temperature_2m_min"


def _get_with_retry(
    url: str, params: dict, attempts: int = 4, timeout: int = 90
) -> requests.Response:
    """Open-Meteo's free tier is occasionally slow; retry with backoff."""
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if i == attempts - 1:
                raise
            wait = 5 * (2**i)
            log.warning("Weather request failed (%s), retrying in %ss", e, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _daily_frame(payload: dict, region: str) -> pd.DataFrame:
    d = payload["daily"]
    return pd.DataFrame(
        {
            "region": region,
            "date": pd.to_datetime(d["time"]).date,
            "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"],
        }
    )


def fetch_history(region: str, start: date, end: date) -> pd.DataFrame:
    meta = REGIONS[region]
    resp = _get_with_retry(
        ARCHIVE_URL,
        params={
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": DAILY_VARS,
            "timezone": "Australia/Brisbane",  # AEST, matches NEM time
        },
    )
    return _daily_frame(resp.json(), region)


def fetch_forecast(region: str, days: int = 3) -> pd.DataFrame:
    meta = REGIONS[region]
    resp = _get_with_retry(
        FORECAST_URL,
        attempts=2,  # callers degrade gracefully; don't stall the pipeline
        timeout=20,
        params={
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "daily": DAILY_VARS,
            "forecast_days": days,
            "past_days": 7,  # cover the archive API's publication lag
            "timezone": "Australia/Brisbane",
        },
    )
    return _daily_frame(resp.json(), region)


def ingest_weather(months_back: int = 24, today: date | None = None) -> pd.DataFrame:
    """History up to ~5 days ago (archive lag) stitched with the live forecast.

    The forecast rows cover the archive's lag window plus the next few days,
    which is also what the spike model needs at prediction time.
    """
    today = today or date.today()
    start = (today - timedelta(days=31 * months_back)).replace(day=1)
    frames = []
    for region in REGIONS:
        hist = fetch_history(region, start, today - timedelta(days=6))
        try:
            # Forecast rows cover the archive's ~5-day lag plus the next days.
            fcst = fetch_forecast(region, days=3)
        except requests.RequestException as e:
            # Forecast API outage must not block the pipeline: recent days
            # will have missing temps, which the model handles natively.
            log.warning("Forecast unavailable for %s (%s); using history only", region, e)
            fcst = hist.iloc[0:0]
        frames.append(pd.concat([hist, fcst], ignore_index=True))
    weather = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["region", "date"], keep="last")
        .sort_values(["region", "date"])
        .dropna(subset=["tmax", "tmin"])
    )
    weather["date"] = pd.to_datetime(weather["date"])
    out = PARQUET_DIR / "weather.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out, index=False)
    log.info("Wrote %s weather rows to %s", len(weather), out)
    return weather
