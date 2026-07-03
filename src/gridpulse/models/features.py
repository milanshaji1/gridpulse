"""Feature engineering for the day-ahead spike-risk model.

One row per (region, date D). The label is whether ANY 5-min price on day D
reached the spike threshold. Every feature is information available the
evening before D:

- price/demand aggregates use dates <= D-1 only
- weather for D uses the day's tmax/tmin (a stand-in for the day-ahead
  temperature forecast, which is what the live system feeds in at
  prediction time)
- calendar features of D itself
"""
from __future__ import annotations

import pandas as pd

from gridpulse import db
from gridpulse.config import PARQUET_DIR, SPIKE_THRESHOLD

LAGS = [1, 2, 3, 7]
ROLLS = [7, 30]


def load_daily() -> pd.DataFrame:
    con = db.connect(read_only=True)
    daily = con.execute("SELECT * FROM daily ORDER BY region, date").fetchdf()
    weather = con.execute("SELECT * FROM weather ORDER BY region, date").fetchdf()
    con.close()
    daily["date"] = pd.to_datetime(daily["date"])
    weather["date"] = pd.to_datetime(weather["date"])
    return daily.merge(weather, on=["region", "date"], how="left")


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_daily()
    df = df.sort_values(["region", "date"]).copy()
    df["spike"] = (df["max_rrp"] >= SPIKE_THRESHOLD).astype(int)

    g = df.groupby("region", group_keys=False)
    feats = df[["region", "date", "spike"]].copy()

    # Lagged daily aggregates (strictly historical).
    for lag in LAGS:
        for col in ["avg_rrp", "max_rrp", "max_demand", "spike"]:
            feats[f"{col}_lag{lag}"] = g[col].shift(lag)

    # Rolling windows over history ending at D-1 (shift first, then roll).
    for win in ROLLS:
        for col, agg in [("avg_rrp", "mean"), ("max_rrp", "max"),
                         ("spike", "mean"), ("max_demand", "mean")]:
            shifted = g[col].shift(1)
            feats[f"{col}_roll{win}_{agg}"] = (
                shifted.groupby(df["region"]).rolling(win, min_periods=win // 2)
                .agg(agg).reset_index(level=0, drop=True)
            )

    # Day-of weather: proxy for the day-ahead forecast available at 6pm D-1.
    feats["tmax"] = df["tmax"]
    feats["tmin"] = df["tmin"]
    feats["tmax_lag1"] = g["tmax"].shift(1)
    # Temperature swing vs yesterday - heat ramps drive demand surprises.
    feats["tmax_delta"] = feats["tmax"] - feats["tmax_lag1"]

    # Calendar.
    feats["dow"] = df["date"].dt.dayofweek
    feats["month"] = df["date"].dt.month
    feats["is_weekend"] = (feats["dow"] >= 5).astype(int)
    feats["region_code"] = feats["region"].astype("category").cat.codes

    # Drop warm-up rows where long rollups are undefined.
    feats = feats.dropna(subset=[f"spike_roll{max(ROLLS)}_mean"]).reset_index(drop=True)
    return feats


FEATURE_COLS = None  # resolved at train time: everything except region/date/spike


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("region", "date", "spike")]


def main() -> None:
    feats = build_features()
    out = PARQUET_DIR / "features.parquet"
    feats.to_parquet(out, index=False)
    rate = feats["spike"].mean()
    print(f"Wrote {len(feats):,} feature rows to {out} (spike base rate {rate:.1%})")


if __name__ == "__main__":
    main()
