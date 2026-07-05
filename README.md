# ⚡ GridPulse — AI Market Analyst for Australia's Electricity Grid

[![Daily brief](https://github.com/milanshaji1/gridpulse/actions/workflows/daily.yml/badge.svg)](https://github.com/milanshaji1/gridpulse/actions/workflows/daily.yml)

**A production-shaped ML + LLM system on live market data.** GridPulse ingests
5-minute price and demand data from Australia's National Electricity Market
(NEM), forecasts next-day **price-spike risk** per region with a backtested
gradient-boosting model, and every morning an **LLM analyst agent writes a
market brief in which every numeric claim is programmatically re-verified
against the source database before publication**.

> Spot prices in the NEM swing from ~$50/MWh to the market cap (>$17,000/MWh)
> within a single 5-minute interval. Retailers, large industrial users, and
> battery operators pay analysts to watch this and write morning briefs.
> GridPulse automates the watch *and* the brief — and, critically, includes
> the evaluation and verification layer that makes an LLM system trustworthy
> enough to put in front of a client.

## Architecture

```
AEMO price/demand (5-min CSVs) ──┐
Open-Meteo weather ──────────────┼─► Idempotent ingestion ─► DuckDB + Parquet
                                 │        │                   (~1M+ rows)
                                 │        ├─► Spike-risk model (LightGBM)
                                 │        │     rolling-origin backtest vs
                                 │        │     climatology + persistence
                                 │        │
                                 │        └─► LLM analyst (Claude, tool-use)
                                 │              │  every figure cited as {{value|SQL}}
                                 │              ▼
                                 │          VERIFIER re-executes every citation
                                 │          → mismatch = brief BLOCKED
                                 │              │
GitHub Actions (daily cron) ─────┴──────────────┴─► reports/briefs/YYYY-MM-DD.md
Streamlit dashboard ─► prices, spike-risk, latest brief, eval scores
```

## Why this isn't a notebook project

1. **Honest evaluation.** The spike model is evaluated with a rolling-origin
   backtest (train strictly on the past, score the next month, roll forward)
   against two baselines any real analyst would demand: climatological spike
   frequency and 7-day persistence. Metrics that match how the tool is used:
   recall at a fixed 20% alert budget, PR-AUC, calibration.
2. **Hallucination-blocking verification.** The LLM writes every figure as a
   `{{value|SQL}}` citation. Before publication, every query is re-executed
   and compared to the claimed value. One failed citation blocks the brief
   (one repair round is allowed). Trust the database, not the model.
3. **An LLM eval harness.** A 30-question golden set with ground-truth
   answers computed directly from the warehouse, scored deterministically,
   with per-question cost and latency — so swapping models
   (`GRIDPULSE_MODEL`) gives a directly comparable accuracy/cost tradeoff.
4. **Data-quality gates as tests.** Interval completeness, market price
   bounds, timezone sanity (the NEM runs on AEST with no DST), freshness —
   all pytest, all run in CI before anything downstream.
5. **It runs itself.** A GitHub Actions cron ingests, forecasts, writes,
   verifies, and commits a fresh brief every morning.

## Results

<!-- RESULTS:START — filled by measured runs -->
**Data**: 1,054,080 five-minute price/demand observations (25 months × 5
regions) + daily weather, all data-quality gates green (38 tests).

**Spike model — rolling-origin backtest** (6 held-out months, Feb–Jul 2026,
765 region-days, 59 spike days):

| Metric | GridPulse (LightGBM) | Climatology | 7-day persistence |
|---|---|---|---|
| Recall @ 20% alert budget | **71.2%** | 57.6% | 57.6% |
| PR-AUC | **0.349** | 0.174 | 0.189 |
| ROC-AUC | **0.857** | 0.746 | 0.757 |
| Brier score (↓) | **0.062** | 0.107 | 0.072 |

Alerting on just 1 day in 5, the model catches 71% of real spike days —
+13.6 pp over both baselines, with double their PR-AUC.

**LLM analyst — measured on live runs** (claude-sonnet-5):

| Metric | Result |
|---|---|
| Golden-set accuracy (30 questions, deterministic scoring) | **100% (30/30)** |
| Cost per eval question / per daily brief | $0.008 / **$0.07** |
| Citations per brief, all re-verified against the DB | 26 |
| Brief generation latency | ~31s |

The hallucination gate is additionally proven by tests — a seeded fabricated
figure is detected and blocks publication (`tests/test_verify.py`).

A fun by-product of the data-quality gates: they flagged *negative* demand in
South Australia (−311 MW). Investigation showed it's real — SA rooftop solar
now pushes grid demand below zero on sunny days. The gate was corrected to
encode the domain, not a wrong assumption.
<!-- RESULTS:END -->

## Quick start

```bash
make setup                       # venv + deps
make ingest                      # ~25 months x 5 regions of 5-min AEMO data + weather
make test                        # data-quality gates, no-leakage tests, verifier tests
make backtest                    # rolling-origin evaluation of the spike model
make train                       # train on full history + forecast tomorrow
ANTHROPIC_API_KEY=... make brief # generate + verify today's brief
ANTHROPIC_API_KEY=... make evals # run the 30-question golden set
make dashboard                   # Streamlit app
```

## Data sources

| Source | What | Access |
|---|---|---|
| [AEMO price & demand](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data) | 5-min spot price + demand per region | Public CSVs, no key |
| [Open-Meteo](https://open-meteo.com/) | Daily temperatures per capital city | Free API, no key |

Completed months are cached in `data/raw/` and never re-downloaded; only the
current month is refreshed — the pipeline is idempotent and cheap to re-run.
During this build, Open-Meteo had a partial outage: the pipeline degrades
gracefully (recent days get missing temperatures, which LightGBM handles
natively) rather than failing.

## Design decisions & limitations

- **Day-ahead granularity, not dispatch-level.** The model predicts *which
  days* carry spike risk, not which 5-minute interval — that's the decision
  a human analyst actually supports (hedge/alert today or not).
- **Training uses day-of actual temperature as a stand-in for the day-ahead
  forecast** (standard practice; live serving feeds the real forecast). This
  slightly flatters backtest numbers on heat-driven spikes; a next step is
  archiving true day-ahead forecasts and retraining once enough accumulate.
- **The verifier checks figures, not narrative.** A brief could cite correct
  numbers and still mis-narrate causality; the golden-set evals and the
  structural prompt mitigate but don't eliminate this. Case in point from the
  first live run: AEMO's current-month file trailed the market, so
  "yesterday" had one 5-minute interval ingested — every cited figure
  verified, yet the narrative described a full day from a single observation.
  Fix: the generator now anchors the brief to the most recent *complete*
  trading day (`n_intervals >= 285`), determined from the data, not the clock.
- **What I'd do next:** probabilistic price *level* forecasts (pinball loss),
  generator outage data (AEMO MMS) as features, per-region models, and a
  proper LLM-judge for narrative quality tracked over time.
- **The verification pattern is portable.** The same verify-before-publish
  discipline is reused in
  [gesture-canvas](https://github.com/milanshaji1/gesture-canvas) — a
  real-time TouchDesigner graphics project where an LLM "scene director"'s
  JSON output is schema-validated before it touches the live render. Swap
  "SQL re-execution against a warehouse" for "schema validation"; same gate.

## Repo map

```
src/gridpulse/ingest/    AEMO + weather downloaders (idempotent, retrying)
src/gridpulse/models/    features (no-leakage), backtest harness, training
src/gridpulse/analyst/   LLM agent, SQL tools (read-only guard), verifier, evals
tests/                   data-quality gates, leakage tests, verifier tests
dashboard/               Streamlit app
.github/workflows/       daily cron: ingest → forecast → brief → verify → commit
```

---
*Built by [Milan Shaji](https://linkedin.com/in/milan-shaji) — QUT Data
Science / Business (Entrepreneurship & Innovation).*
