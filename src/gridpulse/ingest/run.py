"""Ingestion entrypoint: python -m gridpulse.ingest.run --months 24"""
from __future__ import annotations

import argparse
import logging

from gridpulse import db
from gridpulse.ingest.aemo import ingest_prices
from gridpulse.ingest.weather import ingest_weather

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=24, help="months of history")
    args = parser.parse_args()

    prices = ingest_prices(months_back=args.months)
    weather = ingest_weather(months_back=args.months)
    db.rebuild()
    print(
        f"Ingest complete: {len(prices):,} price rows, "
        f"{len(weather):,} weather rows -> data/gridpulse.duckdb"
    )


if __name__ == "__main__":
    main()
