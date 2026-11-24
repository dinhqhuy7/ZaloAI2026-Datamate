"""Combine the two tiers that solve a question: the LLM composes a JSON plan (query/compose.py) and the LLM writes SQL (query/sql_tier.py).

The two tiers ask independently so their errors rarely coincide: the same result from both is far more trustworthy than either alone; when they differ trust the
stronger tier for that kind of question (familiar question: JSON; unfamiliar question / new rule line: SQL).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from datamate.loader.parsers import fold
from datamate.query.compose import UNANSWERABLE, ExampleBank, RuleBank, ShopContext, _rule_words, near_rules_q, rule_rows, solve
from datamate.query.luat_goc import cham as luat_cham, luat_changed_for_q
from datamate.query.sql_tier import SqlBank, solve_sql, solve_sql_clusters

_NUM_TOL = 0.006

_METRICS = ("so luong", "doanh thu", "lai gop", "so don", "so ngay")


@lru_cache(maxsize=1)
def _train_rule_map() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]      # when run from source: docker/assets; inside Docker: /app/assets
    p = next((x for x in (root / "docker" / "assets" / "rulebook_train.md", root / "assets" / "rulebook_train.md")
              if x.exists()), None)
    if p is None:
        return {}
    out: dict[str, str] = {}
    for line in rule_rows(p.read_text(encoding="utf-8")):
        if line.startswith("|"):
            parts = [x.strip() for x in line.strip("|").split("|")]
            if len(parts) >= 2:
                out[fold(parts[0])] = parts[1]
    return out


# formula changes commonly seen between the training rulebook and the rulebook at run time
_FORMULA_SHIFTS = (
    ("thanh tien", "gia tri ghi tren don"),
    ("thanh tien", "gia tri don"),
    ("don ghi no", "phieu chua tra"),
    ("bo ca don", "chi tru"),
    ("bo ca don", "khong bo ca don"),
    ("ngay dat", "ngay giao"),
    ("ngay dat", "ngay thu"),
)


def _match_train_rule(k: str, tmap: dict[str, str]) -> str | None:
    """A rule-line name that does not exactly equal a training name -> the training line of the same topic (keyword intersection >= 2, ratio >= 0.5)."""
    if k in tmap:
        return tmap[k]
    wk = _rule_words(k)
    if not wk:
        return None
    best_def, best_score = None, 0.0
    for tk, tdef in tmap.items():
        wtk = _rule_words(tk)
        if not wtk:
            continue
        inter = len(wk & wtk)
        if inter >= 2 or k in tk or tk in k:
            score = inter / min(len(wk), len(wtk))
            if score >= 0.5 and score > best_score:
                best_def, best_score = tdef, score
    return best_def


def _rule_drifted(train_def: str, cur_def: str, mo: bool = False) -> bool:
    w1, w2 = _rule_words(train_def), _rule_words(cur_def)
    if not w1 or not w2:
        return False
    if mo:
        g1, g2 = fold(train_def), fold(cur_def)
        for a, b in _FORMULA_SHIFTS:
            if (a in g1 and b in g2 and b not in g1) or (b in g1 and a in g2 and a not in g1):
                return True
    if len(w1 & w2) / min(len(w1), len(w2)) < 0.35:
        return True
    f1, f2 = fold(train_def), fold(cur_def)
    for kw in ("huy", "tra hang", "giam gia", "phi giao", "vat"):
        if kw in f1 and kw in f2:
            neg1 = bool(re.search(rf"\b(?:khong|bo)\b[^.;,|]{{0,20}}\b{kw}\b", f1))
            neg2 = bool(re.search(rf"\b(?:khong|bo)\b[^.;,|]{{0,20}}\b{kw}\b", f2))
            if neg1 != neg2:
                return True
    pos1 = {m for m in _METRICS if m in f1 and not re.search(rf"\b(?:khong|bo)\b[^.;,|]{{0,20}}\b{m}\b", f1)}
    pos2 = {m for m in _METRICS if m in f2 and not re.search(rf"\b(?:khong|bo)\b[^.;,|]{{0,20}}\b{m}\b", f2)}
    if pos1 and pos2 and not (pos1 & pos2):
        return True
    return False


def has_drifted_rule(ctx_rules: str, q: dict, mo: bool = False) -> bool:
    """True if the rule line matches a question that uses the same term as the training rulebook but its definition / metric was changed."""
    tmap = _train_rule_map()
    if not tmap or not ctx_rules:
        return False
    for line in near_rules_q(ctx_rules, q):
        if line.startswith("|"):
            parts = [x.strip() for x in line.strip("|").split("|")]
            if len(parts) >= 2:
                k = fold(parts[0])
                tdef = _match_train_rule(k, tmap) if mo else tmap.get(k)
                if tdef and _rule_drifted(tdef, parts[1], mo):
                    return True
    return False



def _name_keys(x: dict) -> set[str]:
    keys = {re.sub(r"[^\w]+", "", fold(str(n))) for n in x.get("names") or [] if n}
    if x.get("id") is not None:
        keys.add(re.sub(r"[^\w]+", "", fold(str(x["id"]))))
    return keys


def same_value(a, b) -> bool:
    """Whether two execution results are the same answer. A one-element list counts as that element; tables are compared regardless of row order."""
    if isinstance(a, list) and len(a) == 1 and not isinstance(b, list):
        a = a[0]
    if isinstance(b, list) and len(b) == 1 and not isinstance(a, list):
        b = b[0]
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < _NUM_TOL
    if isinstance(a, dict) and isinstance(b, dict) and "bang" in a and "bang" in b:
        ra, rb = a["bang"], b["bang"]
        return len(ra) == len(rb) and all(any(same_value(x, y) for y in rb) for x in ra)
    if isinstance(a, dict) and isinstance(b, dict) and "names" in a and "names" in b:
        return bool(_name_keys(a) & _name_keys(b))
    if isinstance(a, dict) and "names" in a and isinstance(b, str):
        return re.sub(r"[^\w]+", "", fold(b)) in _name_keys(a)
    if isinstance(b, dict) and "names" in b and isinstance(a, str):
        return same_value(b, a)
    if isinstance(a, str) and isinstance(b, str):
        return re.sub(r"[^\w]+", "", fold(a)) == re.sub(r"[^\w]+", "", fold(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return False


POLICIES = ("ca_hai", "dinh_tuyen", "dpc", "json", "sql", "bi_thi_sql", "bi_thi_json", "doi_chieu")
# A question this similar to the example store or more is a familiar form (a JSON plan example exists), below is an unfamiliar question (usually a new rule line, SQL
# reads rules far better). ca_hai: when the two tiers differ, a familiar question trusts JSON, an unfamiliar one trusts SQL; dinh_tuyen: a familiar question asks JSON
# first. Threshold 0.6: leans toward SQL because questions outside the example store are common.
FAMILIAR = 0.6
_NO_DATA = UNANSWERABLE | {"llm_khong_co"}


def _settled(r: dict) -> bool:
    return r["loai"] == "tra_loi" or r["loai"] in UNANSWERABLE


def guard_unanswerable(ctx, r: dict) -> dict:
    """§7 needs evidence that the shop does not record it. A shop for which the data layer did not recognise a sales ledger with dates (README does not describe columns, odd ledger)
    usually has an unread ledger when "missing source" shows up: leave blank instead of submitting cannot-be-answered. A shop whose README describes no column
    is the same (columns guessed by the name dictionary / LLM)."""
    if r["loai"] in UNANSWERABLE and not (ctx.sales_known() and ctx.columns_described()):
        return {**r, "loai": "khong_chac"}
    return r


def solve_both(ctx: ShopContext, q: dict, bank: ExampleBank, sql_bank: SqlBank | None = None, rule_bank: RuleBank | None = None,
               policy: str = "ca_hai", temps=(0, 0.5, 0.9), skip_shop=None, skip_dong=None) -> dict:
    """policy: ca_hai (always ask both, see both) | dinh_tuyen (familiar question: plan first, unfamiliar: SQL first; if the first is stuck
    ask the other) | json / sql (one tier) | bi_thi_sql (plan first) | bi_thi_json (SQL first)
    | doi_chieu (always ask both; blank when they differ). A cannot-be-answered conclusion goes through guard_unanswerable."""
    r = guard_unanswerable(ctx, _solve_both(ctx, q, bank, sql_bank, rule_bank, policy, temps, skip_shop, skip_dong))
    if policy == "dpc":     # a changed rule already pushed the question to SQL at the entry (luat_changed_for_q), not blanked at the end
        return r
    # a default rule hard-coded in the code whose meaning the contest rulebook changed (luat_goc): a question using that source is left blank
    return luat_cham(ctx, r)


def _solve_both(ctx, q, bank, sql_bank, rule_bank, policy, temps, skip_shop, skip_dong) -> dict:
    def j():
        return solve(ctx, q, bank, skip_shop=skip_shop, temps=temps, rule_bank=rule_bank, skip_dong=skip_dong)

    def s():
        return solve_sql(ctx, q, temps=temps, bank=sql_bank, skip_shop=skip_shop, rule_bank=rule_bank, skip_dong=skip_dong)
    if policy in ("ca_hai", "dinh_tuyen", "dpc"):
        familiar = bank.familiarity(q["question"], ctx.names_in(q["question"]), skip_shop=skip_shop) >= FAMILIAR
        if policy == "dpc":
            if familiar:
                rules_text = getattr(ctx, "rules", "")
                if has_drifted_rule(rules_text, q, mo=True) or luat_changed_for_q(ctx, q, near_rules_q(rules_text, q)):
                    familiar = False
            j_res = solve(ctx, q, bank, skip_shop=skip_shop, temps=(0,), rule_bank=rule_bank, skip_dong=skip_dong)
            s_pack = solve_sql_clusters(ctx, q, k=5, bank=sql_bank, skip_shop=skip_shop, rule_bank=rule_bank,
                                        skip_dong=skip_dong)
            return dpc_select(ctx, q, j_res, s_pack, familiar, cautious=not ctx.columns_described())
        if familiar and has_drifted_rule(getattr(ctx, "rules", ""), q):
            familiar = False
        if policy == "ca_hai":
            return both(j(), s(), familiar, cautious=not ctx.columns_described())
        policy = "bi_thi_sql" if familiar else "bi_thi_json"
    if policy == "sql":
        return {**s(), "tang": "sql"}
    if policy == "bi_thi_json":
        first = s()
        return {**first, "tang": "sql"} if _settled(first) else combine(j(), first)
    first = j()
    if policy == "json" or (policy == "bi_thi_sql" and _settled(first)):
        return first
    return combine(first, s(), cross_check=policy == "doi_chieu")


def both(j: dict, s: dict, familiar: bool, cautious: bool = False) -> dict:
    """Result of the two tiers for the same question. Both answer: equal -> submit, different -> a familiar question takes JSON, an unfamiliar one takes SQL
    (cautious: blank). Only one tier answers -> take that tier (even when the other concludes cannot-be-answered). Neither
    answers: keep the cannot-be-answered conclusion (§7) of JSON, then of SQL; none -> blank.
    cautious = a shop whose README describes no column: a disagreement there is mostly both guessing the wrong column,
    while at a shop whose README describes enough columns taking SQL pays off more."""
    sql = {**s, "tang": "sql"}
    if j["loai"] == s["loai"] == "llm_khong_co":
        # both tiers say the shop does not record the kind of information the question needs (gender, cancellation reason, satisfaction level...): §7 cannot be
        return {"loai": "khong_ghi", "notes": (j.get("notes") or []) + (s.get("notes") or [])}
    if j["loai"] == "tra_loi" and s["loai"] == "tra_loi":
        if same_value(j["value"], s["value"]):
            return j
        if cautious:
            return {"loai": "khong_chac", "notes": (j.get("notes") or []) + (s.get("notes") or [])}
        return j if familiar else sql
    # one tier answers, the other says the shop has no such data: the two tiers understood the question differently, left blank
    # (such a question is as likely right as wrong)
    if j["loai"] == "tra_loi" and s["loai"] in _NO_DATA or s["loai"] == "tra_loi" and j["loai"] in _NO_DATA:
        return {"loai": "khong_chac", "notes": (j.get("notes") or []) + (s.get("notes") or [])}
    if j["loai"] == "tra_loi":
        return j
    if s["loai"] == "tra_loi":
        return sql
    if j["loai"] in UNANSWERABLE:
        return j
    return sql if s["loai"] in UNANSWERABLE else j


def combine(j: dict, s: dict, cross_check: bool = False) -> dict:
    """If the plan answers, take the plan (cross-check: when the SQL differs, blank). If the plan concludes cannot-be-answered (§7),
    keep that conclusion. Only a stuck plan (error, no consensus) falls back to SQL."""
    if j["loai"] == "tra_loi":
        if cross_check and s["loai"] == "tra_loi" and not same_value(j["value"], s["value"]):
            return {"loai": "khac_nhau", "notes": (j.get("notes") or []) + (s.get("notes") or [])}
        return j
    if j["loai"] in UNANSWERABLE:
        return j
    if s["loai"] == "tra_loi":
        return {**s, "tang": "sql"}
    return j


# cluster the SQL candidates, cross-check with JSON
def _cluster_to_res(c: dict, j_want, notes: list, tang: str) -> dict:
    return {"loai": "tra_loi", "value": c["value"], "want": c.get("want") or j_want, "plan": c["representative_sql"],
            "tang": tang, "notes": notes}


def _sql_plurality_ok(pool: list[dict]) -> bool:
    """Accept the top SQL cluster when consensus is >= 3/5 votes and it is not empty."""
    return bool(pool and not pool[0]["is_empty_or_null"] and pool[0]["count"] >= 3)


def dpc_select(ctx: ShopContext, q: dict, j_res: dict, s_pack: dict, familiar: bool, cautious: bool = False) -> dict:
    """Cluster the SQL (5 candidates) -> cross-check with JSON (temperature 0) -> a familiar question trusts JSON / an unfamiliar one needs SQL >= 3/5, otherwise
    blank."""
    clusters = s_pack.get("clusters") or []
    no_data = s_pack.get("no_data")
    no_data_count = s_pack.get("no_data_count", 0)
    notes = (j_res.get("notes") or []) + (s_pack.get("notes") or [])
    py_valid = j_res.get("loai") == "tra_loi"
    # missing data (§7): submit cannot-be-answered only when SQL has no cluster and >= 2 SQL samples agree it is missing
    if j_res.get("loai") == "llm_khong_co" and not clusters and no_data and no_data.get("loai") == "llm_khong_co":
        return {"loai": "khong_ghi", "notes": notes}
    if j_res.get("loai") in _NO_DATA:
        if not clusters and no_data and no_data_count >= 2:
            if j_res["loai"] in UNANSWERABLE:
                return j_res
            return no_data if no_data.get("loai") in UNANSWERABLE else j_res
        return {"loai": "khong_chac", "notes": notes}
    if not py_valid and not clusters:
        if no_data and no_data_count >= 2 and no_data.get("loai") in UNANSWERABLE:
            return no_data
        return {"loai": "khong_chac", "notes": notes}
    if py_valid and not clusters and no_data:
        return {"loai": "khong_chac", "notes": notes}
    if len(clusters) == 1:          # one SQL cluster
        c1 = clusters[0]
        if py_valid and same_value(c1["value"], j_res["value"]):
            return {**j_res, "tang": "dpc_agree", "notes": notes}
        if (not py_valid) and _sql_plurality_ok(clusters) and (not cautious):
            return _cluster_to_res(c1, None, notes, "dpc_single_sql")
    if py_valid and any(same_value(c["value"], j_res["value"]) for c in clusters):
        return {**j_res, "tang": "dpc_match", "notes": notes}      # JSON matches any SQL cluster
    non_empty = [c for c in clusters if not c["is_empty_or_null"]]
    pool = non_empty if non_empty else clusters
    if cautious:
        return {"loai": "khong_chac", "notes": notes}
    if familiar and py_valid:
        return {**j_res, "tang": "dpc_fallback_json", "notes": notes}
    if (not familiar) and _sql_plurality_ok(pool):
        return _cluster_to_res(pool[0], None, notes, "dpc_fallback_sql")
    return {"loai": "khong_chac", "notes": notes}
