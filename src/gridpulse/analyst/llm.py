"""Claude client wrapper with per-call cost and latency tracking.

Model choice is configurable so the eval harness can compare models on
accuracy vs cost - swap via GRIDPULSE_MODEL, re-run `make evals`, compare.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import anthropic

MODEL = os.environ.get("GRIDPULSE_MODEL", "claude-sonnet-5")

# USD per million tokens (input, output) - used for run-cost reporting.
PRICING = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class UsageTracker:
    """Accumulates tokens, cost, and latency across an agent run."""

    model: str = MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0
    latency_s: float = 0.0
    _client: anthropic.Anthropic | None = field(default=None, repr=False)

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit(
                    "ANTHROPIC_API_KEY is not set. The analyst agent needs it for "
                    "`make brief` and `make evals` - get a key at "
                    "https://console.anthropic.com and export it first."
                )
            self._client = anthropic.Anthropic()
        return self._client

    def create(self, **kwargs) -> anthropic.types.Message:
        start = time.monotonic()
        response = self.client.messages.create(model=self.model, **kwargs)
        self.latency_s += time.monotonic() - start
        self.n_calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        return response

    @property
    def cost_usd(self) -> float:
        in_rate, out_rate = PRICING.get(self.model, (5.00, 25.00))
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1e6

    def summary(self) -> dict:
        return {
            "model": self.model,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 5),
            "latency_s": round(self.latency_s, 2),
        }


def log_run(path, record: dict) -> None:
    """Append a JSON line to the run log (cost/latency observability)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
