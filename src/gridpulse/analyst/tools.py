"""Tools the analyst agent can call, plus their schemas.

The SQL tool is the security boundary between the LLM and the warehouse:
read-only connection AND statement allowlist, so a prompt-injected or
hallucinated query cannot mutate data.
"""
from __future__ import annotations

import re

import duckdb

from gridpulse import db

MAX_ROWS = 60

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|install|load|export|pragma|set)\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:  # single statement only
        return False
    if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE):
        return False
    return not _FORBIDDEN.search(stripped)


def run_sql(sql: str) -> str:
    """Execute a read-only query and return a compact text table."""
    if not is_safe_sql(sql):
        return "ERROR: only single read-only SELECT/WITH statements are allowed."
    con = db.connect(read_only=True)
    try:
        rel = con.execute(sql)
        columns = [d[0] for d in rel.description]
        rows = rel.fetchmany(MAX_ROWS + 1)
    except duckdb.Error as e:
        return f"ERROR: {e}"
    finally:
        con.close()
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    lines = ["\t".join(columns)]
    lines += ["\t".join(str(v) for v in row) for row in rows]
    if truncated:
        lines.append(f"... truncated at {MAX_ROWS} rows")
    return "\n".join(lines)


def get_spike_forecast() -> str:
    """Latest model forecast: per-region spike probability for the target day."""
    con = db.connect(read_only=True)
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "forecasts" not in tables:
            return "No forecast available (model has not been run)."
        rows = con.execute(
            """
            SELECT region, target_date, round(spike_probability, 3), tmax_forecast
            FROM forecasts
            WHERE target_date = (SELECT max(target_date) FROM forecasts)
            ORDER BY spike_probability DESC
            """
        ).fetchall()
    finally:
        con.close()
    lines = ["region\ttarget_date\tspike_probability\ttmax_forecast"]
    lines += ["\t".join(str(v) for v in r) for r in rows]
    return "\n".join(lines)


TOOL_SCHEMAS = [
    {
        "name": "run_sql",
        "description": (
            "Run a single read-only SQL query (DuckDB syntax) against the NEM "
            "warehouse and get a small text table back. Tables: "
            "prices(region, settlement_ts, demand_mw, rrp) - 5-minute data; "
            "daily(region, date, avg_rrp, max_rrp, min_rrp, median_rrp, avg_demand, "
            "max_demand, spike_intervals, negative_intervals, n_intervals) - one row "
            "per region-day; weather(region, date, tmax, tmin); "
            "forecasts(region, target_date, spike_probability, tmax_forecast, generated_at). "
            "Regions: NSW1, QLD1, SA1, TAS1, VIC1. Prices are $/MWh in NEM time (AEST). "
            "Always aggregate - never select raw 5-minute rows without a LIMIT."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "The SQL query."}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_spike_forecast",
        "description": (
            "Get the ML model's latest day-ahead price-spike risk forecast: "
            "probability that any 5-minute price exceeds $300/MWh, per region."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "run_sql":
        return run_sql(tool_input["sql"])
    if name == "get_spike_forecast":
        return get_spike_forecast()
    return f"ERROR: unknown tool {name}"
