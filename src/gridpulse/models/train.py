"""Train the production spike model on all history and forecast tomorrow.

Writes:
- models_artifacts/spike_model.txt      (LightGBM booster)
- `forecasts` table in DuckDB           (one row per region per target date)
"""
from __future__ import annotations

from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd

from gridpulse import db
from gridpulse.config import MODELS_DIR, REGIONS
from gridpulse.ingest.weather import fetch_forecast
from gridpulse.models.backtest import LGB_PARAMS
from gridpulse.models.features import build_features, feature_columns, load_daily


def train_full() -> tuple[lgb.LGBMClassifier, list[str]]:
    feats = build_features()
    cols = feature_columns(feats)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(feats[cols], feats["spike"])
    MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "spike_model.txt"))
    return model, cols

def predict_tomorrow(model: lgb.LGBMClassifier, cols: list[str]) -> pd.DataFrame:
    """Build tomorrow's feature rows from history + live temperature forecast."""
    daily = load_daily()
    tomorrow = pd.Timestamp(date.today() + timedelta(days=1))

    fcst_rows = []
    for region in REGIONS:
        try:
            fc = fetch_forecast(region, days=3)
            fc["date"] = pd.to_datetime(fc["date"])
            row = fc[fc["date"] == tomorrow]
        except Exception:  # forecast API outage: predict with missing temps
            row = pd.DataFrame()
        fcst_rows.append(
            {
                "region": region,
                "date": tomorrow,
                "tmax": float(row["tmax"].iloc[0]) if len(row) else None,
                "tmin": float(row["tmin"].iloc[0]) if len(row) else None,
            }
        )
    future = pd.DataFrame(fcst_rows)

    # Append the future day (no price data yet) and reuse the exact same
    # feature builder so training and serving cannot diverge.
    hist_cols = ["region", "date", "avg_rrp", "max_rrp", "min_rrp", "median_rrp",
                 "avg_demand", "max_demand", "spike_intervals",
                 "negative_intervals", "n_intervals", "tmax", "tmin"]
    combined = pd.concat(
        [daily[hist_cols], future.reindex(columns=hist_cols)], ignore_index=True
    )
    feats = build_features(combined)
    target = feats[feats["date"] == tomorrow].copy()
    target["spike_probability"] = model.predict_proba(target[cols])[:, 1]
    out = target[["region", "date", "spike_probability", "tmax"]].rename(
        columns={"date": "target_date", "tmax": "tmax_forecast"}
    )
    out["generated_at"] = pd.Timestamp.now()
    return out


def save_forecast(forecast: pd.DataFrame) -> None:
    con = db.connect()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            region TEXT, target_date DATE, spike_probability DOUBLE,
            tmax_forecast DOUBLE, generated_at TIMESTAMP
        )
        """
    )
    # Idempotent per target_date: latest run wins.
    con.execute(
        "DELETE FROM forecasts WHERE target_date = ?",
        [forecast["target_date"].iloc[0].date()],
    )
    con.register("forecast_df", forecast)
    con.execute("INSERT INTO forecasts SELECT * FROM forecast_df")
    con.close()


def main() -> None:
    model, cols = train_full()
    forecast = predict_tomorrow(model, cols)
    save_forecast(forecast)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
