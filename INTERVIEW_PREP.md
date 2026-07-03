# GridPulse — plain-English explainer & interview prep

Read this the night before any interview. Companion to RESUME_BULLETS.md.

## The world this lives in

Electricity can't be stored cheaply at scale, so it's made the moment it's
used. Eastern Australia runs a live auction — the **NEM** — where every **5
minutes** the operator (AEMO) buys the cheapest power that keeps the lights
on, producing a price per region (NSW1, QLD1, SA1, TAS1, VIC1).

Normally ~$50–100/MWh. But on a 42°C afternoon it can hit the **$17,500 cap**
in five minutes (a **spike**), and on mild sunny days South Australia's
rooftop solar pushes prices **negative**. Retailers, factories, and battery
operators lose or make real money on these swings, so they employ analysts
to watch the market and write a morning brief. **GridPulse automates that
job — with a trust layer.**

## The assembly line (every morning, 7am AEST, unattended)

1. **Collect** — download AEMO's 5-min prices/demand (1M+ rows, 2 years,
   5 regions) + capital-city weather (temperature drives demand).
2. **Inspect** — 38 automatic data-quality checks; anything broken halts
   the line (a chef refusing bad ingredients).
3. **Predict** — an ML model outputs P(spike today) per region, learned from
   two years of history (like "70% chance of rain", for prices).
4. **Write** — an AI language model drafts the brief in plain English.
5. **Fact-check the AI** — every number must carry a *receipt*: the exact
   database query that produced it. A separate program re-runs every receipt;
   one mismatch **blocks publication** (one repair round allowed).
   *Never trust the AI — trust the database.*
6. **Publish** — verified brief committed to the repo + dashboard.

## How I know it works

- **The model sat a fair exam**: rolling-origin backtest (train on the past
  only, predict the next month, roll forward, 6 months) vs two dumb-but-fair
  rivals (historical odds; recent persistence). Result: alerting on the
  riskiest 1 day in 5, it catches **71% of real spike days vs 58%**.
- **The AI sat an exam**: 30 questions with answers computed from the
  database. Score: **30/30**, <1c/question, automated so model swaps become a
  measured accuracy-vs-cost decision.

## The three war stories

1. **Negative demand mystery** — a quality gate flagged "impossible" negative
   demand in SA. Investigated instead of deleting: it's REAL (rooftop solar
   exceeds state demand). Fixed the test's assumption, not the data.
2. **Verified-but-wrong brief** — first live brief passed every fact check
   yet described a whole day from ONE 5-minute reading (AEMO's file lags the
   market). Every figure true; picture false. Fix: anchor briefs to the most
   recent COMPLETE day (288 intervals), decided from the data, not the clock.
   *Verifying facts ≠ verifying context.*
3. **Timezone bug** — first cloud run dated the brief a day behind (CI
   servers run UTC). Fix: use the market's timezone, never the machine's.

## Numbers to memorize

| Number | Meaning |
|---|---|
| 1,054,080 rows | 5-min readings, 2 years × 5 regions |
| 71% @ 20% alert budget | spike days caught alerting 1 day in 5 (baselines 58%) |
| PR-AUC 0.349 vs 0.174 | 2× the climatology baseline |
| 30/30 (100%) | AI golden-set exam score |
| 25–26 citations/brief | all re-verified pre-publication |
| ~$0.07–0.09, ~30s | cost & time per brief |
| 38 tests | quality + correctness gates |

## Q&A

**One sentence?** "A robot market analyst for Australia's electricity grid —
ML predicts spike risk, AI writes the morning brief, and every number is
automatically fact-checked against the source database before it can publish."

**Why build it?** Every company deploying AI hits the same wall: you can't
put it in front of clients if it invents numbers. I built the solution —
verification + evaluation — on a live, valuable problem.

**AI makes things up — how can you trust it?** You can't; that's the design.
Every number carries the query that produced it; deterministic code re-runs
each one and blocks the brief on any mismatch. Tests prove the blocker works
by seeding a fabricated number and confirming it's caught.

**Why not a second AI as the checker?** Then you trust two hallucination-prone
things instead of zero. The verifier is plain code: re-run query, compare.

**How does the model work?** Gradient-boosted decision trees (LightGBM) —
hundreds of learned if-then rules over recent prices, demand, temperature,
calendar. Right-sized for thousands of daily rows; deep learning is overkill.

**Why rolling-origin testing?** Random splits leak the future into training
and inflate scores. I train strictly on the past for each fold, and tests
prove no feature can see the future.

**Why recall@20% instead of accuracy?** Spikes are rare (~8% of test days), so
"never predict spike" is 92% accurate and useless. Ops teams tolerate limited
alerts; the honest question is recall under that budget: 71% vs 58% baseline.

**Who'd pay?** Retailers hedging spot exposure, industrial users curtailing,
battery operators planning, and consultancies — the pattern "automate an
expert workflow with ML+AI, add the client-safe trust layer" sells across
industries; the energy skin is swappable.

**Limitations?** (1) Verifier checks figures, not narrative — hence the
complete-day guard after the partial-day incident. (2) Day-level risk, not
5-minute timing. (3) Training used day-of temps as a forecast stand-in,
slightly flattering the backtest; fix by archiving true forecasts over time.

**Did you build it yourself?** "I built it with an AI coding assistant —
that's the modern professional workflow. I own every design decision and can
defend them all: deterministic verifier over LLM-judge, rolling-origin
evaluation, recall-at-budget. Ask me anything about how it works."

**Hardest part?** The things tutorials skip: a weather API died mid-build
(made the pipeline degrade gracefully), "impossible" data that was real
physics, and a verified-but-misleading brief. Each changed the design —
that's the difference between a notebook and a system.

**Next?** Probabilistic price-level forecasts (pinball loss), generator
outage features (outage + heatwave = classic spike recipe), per-region
models, and a tracked narrative-quality score alongside figure verification.
