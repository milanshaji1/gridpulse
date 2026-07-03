"""SQL guard tests: the LLM's only path to the warehouse must be read-only."""
import pytest

from gridpulse.analyst.tools import is_safe_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM daily LIMIT 5",
        "select avg(rrp) from prices",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT count(*) FROM prices;",
    ],
)
def test_allows_read_only(sql):
    assert is_safe_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE prices",
        "DELETE FROM prices",
        "INSERT INTO prices VALUES (1)",
        "UPDATE daily SET avg_rrp = 0",
        "CREATE TABLE evil AS SELECT 1",
        "SELECT 1; DROP TABLE prices",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "COPY prices TO 'out.csv'",
        "INSTALL httpfs",
        "SET memory_limit='1GB'",
        "SELECT * FROM prices; SELECT 1",
    ],
)
def test_blocks_mutations_and_escapes(sql):
    assert not is_safe_sql(sql)
