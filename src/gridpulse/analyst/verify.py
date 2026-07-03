"""Citation verifier: every number the analyst publishes must be reproducible.

The agent writes figures as {{value|SQL}} citations. Before a brief is
published, each citation's query is re-executed against the warehouse and the
first cell of the result must match the claimed value. Any mismatch blocks
publication. This turns "trust the LLM" into "trust the database".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from gridpulse.analyst.tools import is_safe_sql
from gridpulse import db

CITATION_RE = re.compile(r"\{\{\s*([^|{}]+?)\s*\|\s*([^{}]+?)\s*\}\}")

REL_TOLERANCE = 0.005  # numbers may be rounded for prose; allow 0.5%
ABS_TOLERANCE = 0.051  # and rounding to 1-2 decimal places


@dataclass
class Citation:
    claimed: str
    sql: str
    actual: str | None = None
    ok: bool = False
    error: str | None = None


def _to_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", ""))
    except (ValueError, TypeError):
        return None


def _values_match(claimed: str, actual) -> bool:
    c_num, a_num = _to_float(claimed), _to_float(actual)
    if c_num is not None and a_num is not None:
        if c_num == a_num:
            return True
        return abs(c_num - a_num) <= max(ABS_TOLERANCE, abs(a_num) * REL_TOLERANCE)
    return str(claimed).strip().lower() == str(actual).strip().lower()


def extract_citations(text: str) -> list[Citation]:
    return [Citation(claimed=m.group(1), sql=m.group(2)) for m in CITATION_RE.finditer(text)]


def verify_citations(text: str) -> list[Citation]:
    """Re-execute every citation's query and compare against the claimed value."""
    citations = extract_citations(text)
    if not citations:
        return citations
    con = db.connect(read_only=True)
    try:
        for c in citations:
            if not is_safe_sql(c.sql):
                c.error = "unsafe or non-SELECT SQL"
                continue
            try:
                row = con.execute(c.sql).fetchone()
            except Exception as e:  # noqa: BLE001 - report any DB error to the agent
                c.error = f"query failed: {e}"
                continue
            if row is None or row[0] is None:
                c.error = "query returned no rows"
                continue
            c.actual = str(row[0])
            c.ok = _values_match(c.claimed, row[0])
            if not c.ok:
                c.error = f"claimed {c.claimed!r} but query returns {c.actual!r}"
    finally:
        con.close()
    return citations


def render_brief(text: str) -> str:
    """Strip citation SQL for the published brief, keeping the verified value."""
    return CITATION_RE.sub(lambda m: m.group(1), text)


def failure_report(citations: list[Citation]) -> str:
    """Human/LLM-readable list of failed citations, for the repair round."""
    lines = []
    for c in citations:
        if not c.ok:
            lines.append(f"- {{{{{c.claimed}|{c.sql}}}}}: {c.error}")
    return "\n".join(lines)
