.PHONY: setup ingest features train backtest brief evals test dashboard daily

PY := .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/pip install -q -e ".[dev]"

ingest:
	$(PY) -m gridpulse.ingest.run --months 24

features:
	$(PY) -m gridpulse.models.features

train:
	$(PY) -m gridpulse.models.train

backtest:
	$(PY) -m gridpulse.models.backtest

brief:
	$(PY) -m gridpulse.analyst.brief

evals:
	$(PY) -m gridpulse.analyst.evals

test:
	$(PY) -m pytest

dashboard:
	.venv/bin/streamlit run dashboard/app.py

daily: ingest brief
