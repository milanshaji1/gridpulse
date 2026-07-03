"""Leakage and correctness tests for the feature builder.

Uses a small synthetic frame so these run without the warehouse.
"""
import numpy as np
import pandas as pd
import pytest

from gridpulse.models.features import build_features, feature_columns


def synthetic_daily(days: int = 120, regions=("AAA1", "BBB1")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for region in regions:
        dates = pd.date_range("2025-01-01", periods=days, freq="D")
        for i, d in enumerate(dates):
            max_rrp = 100.0 + (400.0 if rng.random() < 0.1 else 0.0)
            rows.append(
                {
                    "region": region, "date": d,
                    "avg_rrp": 80.0 + rng.normal(0, 5), "max_rrp": max_rrp,
                    "min_rrp": 20.0, "median_rrp": 75.0,
                    "avg_demand": 6000.0, "max_demand": 7000.0 + rng.normal(0, 100),
                    "spike_intervals": 3 if max_rrp > 300 else 0,
                    "negative_intervals": 0, "n_intervals": 288,
                    "tmax": 25.0 + rng.normal(0, 3), "tmin": 15.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def feats():
    return build_features(synthetic_daily())


def test_one_row_per_region_day(feats):
    assert not feats.duplicated(subset=["region", "date"]).any()


def test_no_nans_in_features_after_warmup(feats):
    cols = feature_columns(feats)
    assert feats[cols].isna().sum().sum() == 0


def test_lag_features_are_strictly_historical(feats):
    """Perturbing day D's price data must not change day D's lag features."""
    base = synthetic_daily()
    perturbed = base.copy()
    target_date = pd.Timestamp("2025-03-15")
    mask = (perturbed["date"] == target_date) & (perturbed["region"] == "AAA1")
    perturbed.loc[mask, ["avg_rrp", "max_rrp", "max_demand"]] = 9999.0

    f_base = build_features(base)
    f_pert = build_features(perturbed)
    row_b = f_base[(f_base["date"] == target_date) & (f_base["region"] == "AAA1")]
    row_p = f_pert[(f_pert["date"] == target_date) & (f_pert["region"] == "AAA1")]

    lag_cols = [c for c in feature_columns(f_base) if "lag" in c or "roll" in c]
    pd.testing.assert_frame_equal(
        row_b[lag_cols].reset_index(drop=True),
        row_p[lag_cols].reset_index(drop=True),
    )


def test_perturbation_visible_from_next_day(feats):
    """Sanity check the test above isn't vacuous: D+1 lag1 features DO change."""
    base = synthetic_daily()
    perturbed = base.copy()
    target_date = pd.Timestamp("2025-03-15")
    mask = (perturbed["date"] == target_date) & (perturbed["region"] == "AAA1")
    perturbed.loc[mask, "max_rrp"] = 9999.0

    nxt = target_date + pd.Timedelta(days=1)
    f_base = build_features(base)
    f_pert = build_features(perturbed)
    b = f_base[(f_base["date"] == nxt) & (f_base["region"] == "AAA1")]["max_rrp_lag1"]
    p = f_pert[(f_pert["date"] == nxt) & (f_pert["region"] == "AAA1")]["max_rrp_lag1"]
    assert float(b.iloc[0]) != float(p.iloc[0])


def test_label_matches_threshold(feats):
    raw = synthetic_daily()
    merged = feats.merge(raw[["region", "date", "max_rrp"]], on=["region", "date"])
    assert ((merged["max_rrp"] >= 300) == (merged["spike"] == 1)).all()
