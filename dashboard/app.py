"""GridPulse dashboard: live prices, spike risk, the daily brief, and eval scores."""
import json
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridpulse.config import BRIEFS_DIR, DB_PATH, EVALS_DIR, REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="GridPulse", page_icon="⚡", layout="wide")
st.title("⚡ GridPulse — AI Market Analyst for Australia's Electricity Grid")


@st.cache_data(ttl=600)
def load_daily() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM daily WHERE date >= current_date - 60").fetchdf()
    con.close()
    return df


@st.cache_data(ttl=600)
def load_forecast() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
    if "forecasts" not in tables:
        con.close()
        return pd.DataFrame()
    df = con.execute(
        "SELECT * FROM forecasts WHERE target_date = (SELECT max(target_date) FROM forecasts)"
    ).fetchdf()
    con.close()
    return df


if not DB_PATH.exists():
    st.warning("Warehouse not built yet — run `make ingest` first.")
    st.stop()

tab_market, tab_brief, tab_evals = st.tabs(["Market & spike risk", "Daily brief", "Eval scores"])

with tab_market:
    daily = load_daily()
    forecast = load_forecast()

    st.subheader("Tomorrow's spike risk (P(any 5-min price ≥ $300/MWh))")
    if len(forecast):
        cols = st.columns(len(forecast))
        for col, (_, row) in zip(cols, forecast.sort_values("region").iterrows()):
            p = row["spike_probability"]
            col.metric(
                row["region"],
                f"{p:.0%}",
                help=f"Forecast tmax {row['tmax_forecast']}°C" if pd.notna(row["tmax_forecast"]) else None,
            )
    else:
        st.info("No forecast yet — run `make train`.")

    st.subheader("Daily average spot price, last 60 days ($/MWh)")
    st.altair_chart(
        alt.Chart(daily).mark_line().encode(
            x="date:T", y=alt.Y("avg_rrp:Q", title="avg RRP $/MWh"),
            color="region:N",
            tooltip=["region", "date:T", alt.Tooltip("avg_rrp:Q", format=".1f")],
        ).properties(height=300),
        use_container_width=True,
    )

    st.subheader("Daily max price (spike days visible), last 60 days")
    st.altair_chart(
        alt.Chart(daily).mark_line().encode(
            x="date:T", y=alt.Y("max_rrp:Q", title="max RRP $/MWh", scale=alt.Scale(type="symlog")),
            color="region:N",
            tooltip=["region", "date:T", alt.Tooltip("max_rrp:Q", format=".0f")],
        ).properties(height=300),
        use_container_width=True,
    )

with tab_brief:
    latest = BRIEFS_DIR / "latest.md"
    if latest.exists():
        st.markdown(latest.read_text())
    else:
        st.info("No brief published yet — run `make brief`.")
    st.caption(
        "Every figure in this brief was programmatically re-verified against the "
        "source database before publication. Briefs that fail verification are blocked."
    )

with tab_evals:
    st.subheader("LLM analyst — golden-set evaluation")
    results_path = EVALS_DIR / "results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text())
        s = data["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{s['accuracy']:.0%}", help=f"{s['n_correct']}/{s['n_questions']} correct")
        c2.metric("Model", s["model"])
        c3.metric("Total cost", f"${s['total_cost_usd']:.3f}")
        c4.metric("Mean latency", f"{s['mean_latency_s']:.1f}s")
        st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
    else:
        st.info("No eval results yet — run `make evals`.")

    st.subheader("Backtest — spike early-warning model")
    bt_path = REPORTS_DIR / "backtest_results.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text())
        st.json(bt)
    else:
        st.info("No backtest yet — run `make backtest`.")
