"""Data-quality gates: run after ingestion, fail loudly on bad data.

These are the checks a production pipeline would run before letting anything
downstream (model, analyst) touch the warehouse.
"""
from datetime import date, timedelta

import duckdb
import pytest

from gridpulse.config import DB_PATH, REGIONS


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("warehouse not built yet - run `make ingest` first")
    c = duckdb.connect(str(DB_PATH), read_only=True)
    yield c
    c.close()


def test_all_regions_present(con):
    regions = {r for (r,) in con.execute("SELECT DISTINCT region FROM prices").fetchall()}
    assert regions == set(REGIONS)


def test_no_duplicate_intervals(con):
    dupes = con.execute(
        "SELECT count(*) FROM (SELECT region, settlement_ts FROM prices "
        "GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()[0]
    assert dupes == 0


def test_timestamps_on_5min_grid(con):
    off_grid = con.execute(
        "SELECT count(*) FROM prices WHERE minute(settlement_ts) % 5 != 0 "
        "OR second(settlement_ts) != 0"
    ).fetchone()[0]
    assert off_grid == 0


def test_interval_completeness(con):
    """Completed days should have 288 5-min intervals; tolerate rare AEMO gaps."""
    short_days = con.execute(
        """
        SELECT count(*) FROM daily
        WHERE n_intervals < 285
          AND date < current_date  -- today is legitimately partial
          AND date > (SELECT min(date) FROM daily)  -- first day partial by ingest window
        """
    ).fetchone()[0]
    total_days = con.execute(
        "SELECT count(*) FROM daily WHERE date < current_date"
    ).fetchone()[0]
    assert short_days / max(total_days, 1) < 0.01, f"{short_days}/{total_days} short days"


def test_prices_within_market_bounds(con):
    """NEM floor is -$1,000/MWh; cap (MPC) is ~$17,500-20,300 depending on year."""
    lo, hi = con.execute("SELECT min(rrp), max(rrp) FROM prices").fetchone()
    assert lo >= -1100
    assert hi <= 25000


def test_demand_plausible(con):
    """SA1 grid demand legitimately goes NEGATIVE when rooftop solar exceeds
    underlying demand - a real feature of the modern NEM, not corruption."""
    lo, hi = con.execute("SELECT min(demand_mw), max(demand_mw) FROM prices").fetchone()
    assert lo > -2000  # deepest observed SA1 minimum is a few hundred MW negative
    assert hi < 20000  # NSW peaks ~14-15 GW; anything above 20 GW is corrupt


def test_negative_demand_only_in_solar_heavy_regions(con):
    regions = {
        r for (r,) in con.execute(
            "SELECT DISTINCT region FROM prices WHERE demand_mw < 0"
        ).fetchall()
    }
    assert regions <= {"SA1", "VIC1"}, f"unexpected negative demand in {regions}"


def test_data_is_fresh(con):
    """Latest settlement should be within the last 3 days (AEMO publishes daily)."""
    latest = con.execute("SELECT max(CAST(settlement_ts AS DATE)) FROM prices").fetchone()[0]
    assert latest >= date.today() - timedelta(days=3)


def test_weather_covers_all_regions_recently(con):
    """Archive data lags ~5-6 days and the forecast API can be down, so the
    gate checks weather exists within the last 10 days rather than daily."""
    rows = con.execute(
        "SELECT region, max(date) FROM weather GROUP BY region"
    ).fetchall()
    stale = {r for r, latest in rows if latest.date() < date.today() - timedelta(days=10)}
    assert not stale, f"stale weather for {stale}"
    assert {r for r, _ in rows} == set(REGIONS)


def test_weather_temps_plausible(con):
    lo, hi = con.execute("SELECT min(tmin), max(tmax) FROM weather").fetchone()
    assert lo > -15 and hi < 55
