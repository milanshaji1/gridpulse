# Resume bullets for GridPulse

Use 2–4 of these under a "Projects" heading. All numbers are measured from
real runs (backtest: reports/backtest_results.json; evals: evals/results.json;
brief cost: reports/run_log.jsonl).

## The bullets

- Built **GridPulse**, an end-to-end AI market-analyst system for Australia's
  National Electricity Market: ingests **1M+ rows** of live 5-minute AEMO
  price and demand data across 5 regions into a DuckDB/Parquet pipeline with
  automated data-quality gates (38 tests, CI-enforced).

- Trained a gradient-boosted **price-spike early-warning model** achieving
  **71% recall of $300+/MWh spike days at a 20% alert budget** in a 6-month
  rolling-origin backtest — **+14 pp over a persistence baseline, with 2×
  the PR-AUC of climatology**.

- Engineered an **LLM analyst agent** (Claude tool-use over SQL) that writes
  daily market briefings in which **every cited figure (26 per brief) is
  programmatically re-verified against the source database before
  publication** — unverified briefs are blocked — at **~$0.07 and ~30 seconds
  per brief**, replacing hours of manual analysis.

- Built an **LLM evaluation harness**: a 30-question golden set with
  ground-truth answers computed from the warehouse, scored deterministically
  with per-question cost and latency — the agent scored **100% (30/30) at
  $0.008 per question**, making model choice a measured accuracy/cost
  tradeoff.

- Deployed as a **fully automated daily pipeline** (GitHub Actions cron:
  ingest → quality gates → forecast → brief → verify → publish) with a public
  Streamlit dashboard — replacing a multi-hour manual analyst workflow.

## The 30-second interview pitch

> "I took a real analyst workflow — watching Australia's electricity spot
> market and writing the morning brief — and automated it end to end. The ML
> side is a spike early-warning model evaluated the honest way: rolling-origin
> backtests against baselines, recall at a fixed alert budget. The GenAI side
> is the part most people skip: every number the LLM writes is a citation
> that gets re-executed against the database, and a brief with one unverified
> figure doesn't publish. That verification-and-evals layer is what makes an
> AI system something you can actually put in front of a client."

## Talking points that differentiate you

- **Why rolling-origin, not random splits**: time series leak future into past
  with random CV; rolling-origin trains strictly on the past for each fold.
- **Why recall at an alert budget**: ops teams tolerate limited alerts;
  "recall@20%" matches how the tool would actually be used.
- **Why the verifier matters**: LLM hallucination is the #1 blocker to
  client-facing GenAI; deterministic re-execution is stronger than asking a
  second LLM to check the first.
- **The negative-demand story**: data-quality gate flagged −311 MW demand in
  SA → investigation → it's real (rooftop solar exceeds grid demand).
  Shows you validate assumptions instead of "fixing" data you don't understand.
- **The outage story**: Open-Meteo went down mid-build; the pipeline was
  reworked to degrade gracefully (missing temps handled natively by the model)
  instead of failing — that's what production-shaped means.
