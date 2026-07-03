"""Golden-set eval harness for the analyst agent.

30 questions with ground-truth answers computed directly from the warehouse.
The agent answers each using its tools; answers are scored deterministically
(numeric tolerance / exact string). Results include per-question cost and
latency, so model swaps (GRIDPULSE_MODEL) are directly comparable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

from gridpulse import db
from gridpulse.analyst.llm import UsageTracker
from gridpulse.analyst.tools import TOOL_SCHEMAS, execute_tool
from gridpulse.analyst.verify import _values_match
from gridpulse.config import EVALS_DIR, REGIONS

ANSWER_SYSTEM = """\
You are a data analyst answering questions about Australia's National
Electricity Market from a DuckDB warehouse. Use the run_sql tool to compute
the answer. Prices are $/MWh ($ = AUD). Round numeric answers to 2 decimals.
Respond with ONLY the final answer value - a single number or single word,
no units, no explanation, no punctuation.
"""

MAX_TOOL_TURNS = 12


@dataclass
class GoldenQuestion:
    question: str
    truth_sql: str


def build_golden_set(asof: date) -> list[GoldenQuestion]:
    """~30 templated questions over recent, fully-ingested days."""
    d1 = (asof - timedelta(days=2)).isoformat()   # two days ago: definitely complete
    d7 = (asof - timedelta(days=8)).isoformat()
    d30 = (asof - timedelta(days=31)).isoformat()
    qs: list[GoldenQuestion] = []

    for region in REGIONS:  # 5 regions x 4 templates = 20
        qs.append(GoldenQuestion(
            f"What was the average 5-minute spot price in {region} on {d1}?",
            f"SELECT round(avg_rrp, 2) FROM daily WHERE region='{region}' AND date=DATE '{d1}'",
        ))
        qs.append(GoldenQuestion(
            f"What was the maximum 5-minute spot price in {region} on {d1}?",
            f"SELECT round(max_rrp, 2) FROM daily WHERE region='{region}' AND date=DATE '{d1}'",
        ))
        qs.append(GoldenQuestion(
            f"What was the peak demand in MW in {region} on {d1}?",
            f"SELECT round(max_demand, 2) FROM daily WHERE region='{region}' AND date=DATE '{d1}'",
        ))
        qs.append(GoldenQuestion(
            f"How many 5-minute intervals in {region} had a negative price "
            f"between {d7} and {d1} inclusive?",
            f"SELECT sum(negative_intervals) FROM daily WHERE region='{region}' "
            f"AND date BETWEEN DATE '{d7}' AND DATE '{d1}'",
        ))

    # 10 cross-region / time-window questions.
    qs += [
        GoldenQuestion(
            f"Which region had the highest average spot price on {d1}?",
            f"SELECT region FROM daily WHERE date=DATE '{d1}' ORDER BY avg_rrp DESC LIMIT 1",
        ),
        GoldenQuestion(
            f"Which region had the lowest average spot price on {d1}?",
            f"SELECT region FROM daily WHERE date=DATE '{d1}' ORDER BY avg_rrp ASC LIMIT 1",
        ),
        GoldenQuestion(
            f"Which region had the highest peak demand on {d1}?",
            f"SELECT region FROM daily WHERE date=DATE '{d1}' ORDER BY max_demand DESC LIMIT 1",
        ),
        GoldenQuestion(
            f"How many region-days had at least one price spike above $300/MWh "
            f"between {d30} and {d1} inclusive?",
            f"SELECT count(*) FROM daily WHERE spike_intervals > 0 "
            f"AND date BETWEEN DATE '{d30}' AND DATE '{d1}'",
        ),
        GoldenQuestion(
            f"Which region had the most spike intervals (prices >= $300/MWh) "
            f"between {d30} and {d1} inclusive?",
            f"SELECT region FROM daily WHERE date BETWEEN DATE '{d30}' AND DATE '{d1}' "
            f"GROUP BY region ORDER BY sum(spike_intervals) DESC LIMIT 1",
        ),
        GoldenQuestion(
            f"What was the highest single 5-minute price across all regions "
            f"between {d30} and {d1} inclusive?",
            f"SELECT round(max(max_rrp), 2) FROM daily "
            f"WHERE date BETWEEN DATE '{d30}' AND DATE '{d1}'",
        ),
        GoldenQuestion(
            f"What was the average of NSW1's daily average prices between {d7} and {d1}?",
            f"SELECT round(avg(avg_rrp), 2) FROM daily WHERE region='NSW1' "
            f"AND date BETWEEN DATE '{d7}' AND DATE '{d1}'",
        ),
        GoldenQuestion(
            f"On which date did VIC1 record its highest maximum price "
            f"between {d30} and {d1} inclusive?",
            f"SELECT CAST(date AS VARCHAR) FROM daily WHERE region='VIC1' "
            f"AND date BETWEEN DATE '{d30}' AND DATE '{d1}' ORDER BY max_rrp DESC LIMIT 1",
        ),
        GoldenQuestion(
            f"What was the maximum temperature in Brisbane (QLD1) on {d1}?",
            f"SELECT round(tmax, 1) FROM weather WHERE region='QLD1' AND date=DATE '{d1}'",
        ),
        GoldenQuestion(
            f"How many 5-minute price observations are in the warehouse for SA1 on {d1}?",
            f"SELECT n_intervals FROM daily WHERE region='SA1' AND date=DATE '{d1}'",
        ),
    ]
    return qs


def ask(question: str, tracker: UsageTracker) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(MAX_TOOL_TURNS):
        response = tracker.create(
            max_tokens=1000, system=ANSWER_SYSTEM, tools=TOOL_SCHEMAS, messages=messages
        )
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text").strip()
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": b.id,
                 "content": execute_tool(b.name, b.input)}
                for b in response.content if b.type == "tool_use"
            ],
        })
    return "NO_ANSWER"


def main() -> None:
    asof = date.today()
    golden = build_golden_set(asof)

    con = db.connect(read_only=True)
    truths = [con.execute(q.truth_sql).fetchone()[0] for q in golden]
    con.close()

    results = []
    for q, truth in zip(golden, truths):
        tracker = UsageTracker()
        answer = ask(q.question, tracker)
        correct = _values_match(answer, truth)
        results.append({
            "question": q.question,
            "truth": str(truth),
            "answer": answer,
            "correct": correct,
            **{k: tracker.summary()[k] for k in ("cost_usd", "latency_s", "n_calls")},
        })
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] {q.question}\n       truth={truth}  answer={answer}")

    n_correct = sum(r["correct"] for r in results)
    summary = {
        "asof": asof.isoformat(),
        "model": UsageTracker().model,
        "n_questions": len(results),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(results), 3),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
    }
    EVALS_DIR.mkdir(exist_ok=True)
    (EVALS_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
