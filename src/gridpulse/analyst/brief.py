"""Generate, verify, and publish the daily NEM market brief.

Flow: agent drafts brief with cited figures -> verifier re-executes every
citation -> failures go back to the agent for one repair round -> a brief
with any unverified figure is NOT published (non-zero exit).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from gridpulse.analyst import verify
from gridpulse.analyst.llm import UsageTracker, log_run
from gridpulse.analyst.tools import TOOL_SCHEMAS, execute_tool
from gridpulse.config import BRIEFS_DIR, REPORTS_DIR

MAX_TOOL_TURNS = 25

SYSTEM_PROMPT = """\
You are GridPulse, a market analyst covering Australia's National Electricity
Market (NEM). You write a daily morning brief for energy professionals
(retailers, large energy users, battery operators).

Data rules - non-negotiable:
- EVERY numeric figure in the brief must be written as a citation:
  {{value|SQL}} where SQL is a single read-only query whose FIRST returned
  cell equals the value. Example: average price was {{58.06|SELECT round(avg(rrp),2) FROM prices WHERE region='QLD1' AND CAST(settlement_ts AS DATE) = DATE '2026-05-01'}} $/MWh.
- Round values to at most 2 decimals, and round INSIDE the SQL (use round())
  so the query returns exactly what you claim.
- Keep SQL on one line, no braces in SQL. Use the run_sql tool to check every
  figure before citing it - never cite a number you have not queried.
- Model forecast probabilities come from get_spike_forecast; cite them with a
  SQL query against the forecasts table.

Brief structure (markdown):
# NEM Daily Brief - {date}
## Yesterday at a glance  (2-4 sentences: prices, demand, notable regions)
## Notable events         (spikes, negative prices, unusual demand - or say it was quiet)
## Today's spike outlook  (the model's per-region spike probabilities + weather context)
## Bottom line            (1-2 sentences for a busy reader)

Style: precise, plain language, no hype. Explain *why* where the data
supports it (heat, low wind regions, etc.), and say so when it doesn't.
"""


def run_agent(user_prompt: str, tracker: UsageTracker, max_tokens: int = 4000) -> str:
    """Manual tool-use loop; returns the final text."""
    messages = [{"role": "user", "content": user_prompt}]
    for _ in range(MAX_TOOL_TURNS):
        response = tracker.create(
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")
        messages.append({"role": "assistant", "content": response.content})
        results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": execute_tool(block.name, block.input),
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})
    raise RuntimeError("agent exceeded tool-turn limit")


def generate_brief(brief_date: date | None = None) -> tuple[str, dict]:
    brief_date = brief_date or date.today()
    yesterday = brief_date - timedelta(days=1)
    tracker = UsageTracker()

    draft = run_agent(
        f"Write the NEM daily brief for {brief_date.isoformat()}. "
        f"'Yesterday' means {yesterday.isoformat()}. Use the tools to ground "
        f"every figure, then output ONLY the final markdown brief.",
        tracker,
    )

    citations = verify.verify_citations(draft)
    failed = [c for c in citations if not c.ok]

    if failed:
        # One repair round: show the agent exactly what didn't reproduce.
        draft = run_agent(
            "Your draft brief contained figures that FAILED verification "
            "against the database:\n"
            f"{verify.failure_report(citations)}\n\n"
            "Re-check each failed figure with run_sql and output the corrected "
            "full brief. Every figure must be a {{value|SQL}} citation.\n\n"
            f"Original draft:\n{draft}",
            tracker,
        )
        citations = verify.verify_citations(draft)
        failed = [c for c in citations if not c.ok]

    meta = tracker.summary() | {
        "date": brief_date.isoformat(),
        "n_citations": len(citations),
        "n_failed": len(failed),
        "verified": not failed and len(citations) > 0,
    }
    return draft, meta


def main() -> None:
    draft, meta = generate_brief()
    log_run(REPORTS_DIR / "run_log.jsonl", meta)

    if not meta["verified"]:
        print("BLOCKED: brief failed verification, not publishing.", file=sys.stderr)
        print(f"{meta['n_failed']}/{meta['n_citations']} citations failed.", file=sys.stderr)
        failing = [c for c in verify.verify_citations(draft) if not c.ok]
        print(verify.failure_report(failing), file=sys.stderr)
        sys.exit(1)

    published = verify.render_brief(draft)
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    out = BRIEFS_DIR / f"{meta['date']}.md"
    out.write_text(published + f"\n\n---\n*{meta['n_citations']} figures verified against "
                   f"source data. Generated for ${meta['cost_usd']:.4f} in {meta['latency_s']}s.*\n")
    (BRIEFS_DIR / "latest.md").write_text(out.read_text())
    print(f"Published {out} ({meta['n_citations']} citations verified, "
          f"${meta['cost_usd']:.4f}, {meta['latency_s']}s)")


if __name__ == "__main__":
    main()
