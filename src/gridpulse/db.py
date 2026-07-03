"""DuckDB warehouse built from the parquet layer."""
from __future__ import annotations

import duckdb

from gridpulse.config import DB_PATH, PARQUET_DIR


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def rebuild() -> None:
    """(Re)build warehouse tables from parquet. Cheap: full rebuild each run."""
    con = connect()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE prices AS
        SELECT * FROM read_parquet('{PARQUET_DIR / "prices.parquet"}');

        CREATE OR REPLACE TABLE weather AS
        SELECT * FROM read_parquet('{PARQUET_DIR / "weather.parquet"}');

        -- One row per region-day: the granularity the model and analyst use.
        CREATE OR REPLACE TABLE daily AS
        SELECT
            region,
            CAST(settlement_ts AS DATE)            AS date,
            avg(rrp)                               AS avg_rrp,
            max(rrp)                               AS max_rrp,
            min(rrp)                               AS min_rrp,
            quantile_cont(rrp, 0.5)                AS median_rrp,
            avg(demand_mw)                         AS avg_demand,
            max(demand_mw)                         AS max_demand,
            sum(CASE WHEN rrp >= 300 THEN 1 ELSE 0 END)  AS spike_intervals,
            sum(CASE WHEN rrp < 0   THEN 1 ELSE 0 END)   AS negative_intervals,
            count(*)                               AS n_intervals
        FROM prices
        GROUP BY region, CAST(settlement_ts AS DATE)
        ORDER BY region, date;
        """
    )
    con.close()
