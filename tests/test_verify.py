"""Verifier unit tests - including the seeded-wrong-number case that proves
the hallucination gate actually blocks."""
import pytest

from gridpulse.analyst.verify import (
    _values_match,
    extract_citations,
    render_brief,
    verify_citations,
)
from gridpulse.config import DB_PATH


def test_extracts_value_and_sql():
    text = "Price was {{58.06|SELECT round(avg(rrp),2) FROM prices}} today."
    (c,) = extract_citations(text)
    assert c.claimed == "58.06"
    assert c.sql == "SELECT round(avg(rrp),2) FROM prices"


def test_render_strips_sql_keeps_value():
    text = "Price was {{58.06|SELECT 1}} and demand {{6110|SELECT 2}} MW."
    assert render_brief(text) == "Price was 58.06 and demand 6110 MW."


def test_values_match_tolerates_rounding():
    assert _values_match("58.06", 58.0612)
    assert _values_match("58", 58.02)
    assert _values_match("QLD1", "qld1")


def test_values_mismatch_detected():
    assert not _values_match("58.06", 60.0)
    assert not _values_match("QLD1", "NSW1")


@pytest.mark.skipif(not DB_PATH.exists(), reason="warehouse not built")
class TestAgainstWarehouse:
    def test_true_citation_verifies(self):
        text = "There are {{5|SELECT count(DISTINCT region) FROM prices}} regions."
        (c,) = verify_citations(text)
        assert c.ok, c.error

    def test_seeded_wrong_number_is_blocked(self):
        """The load-bearing test: a fabricated figure must fail verification."""
        text = "There are {{99|SELECT count(DISTINCT region) FROM prices}} regions."
        (c,) = verify_citations(text)
        assert not c.ok
        assert "99" in c.error

    def test_unsafe_sql_is_rejected(self):
        text = "Total {{5|DROP TABLE prices}} regions."
        (c,) = verify_citations(text)
        assert not c.ok
        assert "unsafe" in c.error
