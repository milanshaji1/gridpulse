"""Rolling-origin backtest: the honest way to evaluate a time-series classifier.

For each held-out month M (last N months of data), train on everything
strictly before M and score every region-day in M. Aggregated metrics compare
LightGBM against two baselines any real analyst would demand:

- climatological: each region's historical spike frequency (constant score)
- persistence:    the region's spike frequency over the trailing 7 days

Metrics: PR-AUC (spikes are rare), ROC-AUC, Brier score, and recall at a 20%
alert budget - "if ops will only tolerate alerts on 1 day in 5, how many real
spike days do we catch?"
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from gridpulse.config import PARQUET_DIR, REPORTS_DIR
from gridpulse.models.features import build_features, feature_columns

N_TEST_MONTHS = 6

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    random_state=42,
)


def recall_at_budget(y_true: np.ndarray, score: np.ndarray, budget: float = 0.2) -> float:
    """Recall when alerting only on the top `budget` fraction of days."""
    if y_true.sum() == 0:
        return float("nan")
    k = max(1, int(len(score) * budget))
    top_idx = np.argsort(-score)[:k]
    return float(y_true[top_idx].sum() / y_true.sum())


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "recall_at_20pct": recall_at_budget(y, score, 0.2),
    }


def run_backtest(feats: pd.DataFrame | None = None) -> dict:
    feats = feats if feats is not None else build_features()
    feats = feats.sort_values("date").reset_index(drop=True)
    cols = feature_columns(feats)

    months = sorted(feats["date"].dt.to_period("M").unique())
    test_months = months[-N_TEST_MONTHS:]

    preds = []
    for m in test_months:
        train = feats[feats["date"].dt.to_period("M") < m]
        test = feats[feats["date"].dt.to_period("M") == m]
        if train["spike"].sum() < 10 or len(test) == 0:
            continue

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(train[cols], train["spike"])
        p_model = model.predict_proba(test[cols])[:, 1]

        # Baseline 1: climatological per-region spike rate from training data.
        clim = train.groupby("region")["spike"].mean()
        p_clim = test["region"].map(clim).to_numpy()

        # Baseline 2: persistence - trailing 7-day spike frequency (a feature).
        p_persist = test["spike_roll7_mean"].to_numpy()

        preds.append(
            pd.DataFrame(
                {
                    "region": test["region"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "y": test["spike"].to_numpy(),
                    "p_model": p_model,
                    "p_clim": p_clim,
                    "p_persist": p_persist,
                    "fold": str(m),
                }
            )
        )

    allp = pd.concat(preds, ignore_index=True)
    y = allp["y"].to_numpy()
    results = {
        "n_test_days": int(len(allp)),
        "n_spike_days": int(y.sum()),
        "test_window": f"{allp['date'].min().date()} .. {allp['date'].max().date()}",
        "spike_rate_test": float(y.mean()),
        "lightgbm": metrics(y, allp["p_model"].to_numpy()),
        "climatological": metrics(y, allp["p_clim"].to_numpy()),
        "persistence_7d": metrics(y, allp["p_persist"].fillna(0).to_numpy()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest_results.json").write_text(json.dumps(results, indent=2))
    allp.to_parquet(PARQUET_DIR / "backtest_preds.parquet", index=False)
    return results


def main() -> None:
    results = run_backtest()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
