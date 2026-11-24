"""Original column names (as in the README / ledger) -> column names in the DB.

The data layer renames original columns to canonical names (bat_dau -> order_time) and records them in _meta_columns.original_name. The prompt includes the
README so the LLM often writes the original name; the DB reports "no such column". Convert back with that mapping (no LLM call, no guessing): only convert
when the original name points to exactly one column (in the table being used, or the whole shop has a single interpretation).
"""
from __future__ import annotations

import re

import duckdb

from ..loader.parsers import fold

_NOT_FOUND = re.compile(r'Referenced column "([^"]+)" not found')


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", fold(str(name))).strip("_")


def alias_map(con) -> dict[str, dict[str, str]]:
    """{table: {original-name key: DB column}} (key = the original name lowercased, accents stripped, joined by _)."""
    out: dict[str, dict[str, str]] = {}
    try:
        rows = con.execute("SELECT table_name, column_name, original_name FROM _meta_columns "
                           "WHERE original_name IS NOT NULL").fetchall()
    except duckdb.Error:
        return out
    for t, c, o in rows:
        if not c or c.endswith("__raw") or _key(o) == _key(c):
            continue
        out.setdefault(t, {})[_key(o)] = c
    return out


def resolve(con, table: str, col, cols) -> str | None:
    """Column `col` is not in `cols` of `table`: return the column it is the original name of, otherwise None."""
    if not isinstance(col, str) or col in cols:
        return None
    hit = alias_map(con).get(table, {}).get(_key(col))
    return hit if hit in cols else None


def fix_sql(con, sql: str, err: str) -> str | None:
    """SQL fails with 'Referenced column "X" not found' where X is the original name of exactly one column in the tables the SQL mentions -> the SQL has been converted."""
    m = _NOT_FOUND.search(err)
    if not m:
        return None
    bad = m.group(1)
    amap = alias_map(con)
    low = sql.lower()
    used = [t for t in amap if re.search(rf'(?<![\w"]){re.escape(t.lower())}(?![\w"])|"{re.escape(t.lower())}"', low)]
    cands = {amap[t][_key(bad)] for t in used if _key(bad) in amap[t]}
    if len(cands) != 1:
        cands = {v[_key(bad)] for v in amap.values() if _key(bad) in v}
    if len(cands) != 1:
        return None
    new = cands.pop()
    out = re.sub(rf'"{re.escape(bad)}"', f'"{new}"', sql)
    out = re.sub(rf'(?<![\w."]){re.escape(bad)}(?![\w"])', f'"{new}"', out)
    return out if out != sql else None
