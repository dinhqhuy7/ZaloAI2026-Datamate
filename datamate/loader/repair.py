"""Cells the rules cannot read -> the LLM rewrites them in standard form, then the rules' OWN reader reads them again (the LLM's own arithmetic is not trusted).

A numeric / date column with a few oddly written cells ("hai trăm nghìn", "khoảng 50k", "x2") turns those cells blank: revenue is
lost silently. A column with too many odd cells falls back to text entirely: it cannot be mapped. Group the failing cells by DISTINCT VALUE,
send them to the LLM with a few readable cells of the same column as examples; the LLM returns one standard string per cell ("45000", "2025-03-05") or null
(the cell is not a value). The code re-reads with parsers and accepts only when it reads and is of the same scale as the other cells of the column.
ENABLED = False (default): does nothing, as if only rules existed.
"""
from __future__ import annotations

import json
import re

from . import llm
from .parsers import fold, looks_like_code, parse_date, parse_number

ENABLED = False
MAX_VALUES = 40
MIN_RATE = 0.6       # a text column where >= 60% of cells read as numbers / dates: try to fix the rest

# extended rule layer (before the LLM, applies to any cell)
# A column with thousands of odd cells, each a different value (money read in words): the LLM can only see a few dozen labels, so
# forms with clear rules are read by rules: numbers read in words, a cell with exactly one number plus text, English-style dates /
# a cell with exactly one date phrase. Runs only for columns already known to be numeric / date, with the same scale check as LLM results.

_VN_DIGIT = {"khong": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "lam": 5, "nham": 5, "sau": 6,
             "bay": 7, "tam": 8, "chin": 9}
_VN_SCALE = {"nghin": 1e3, "ngan": 1e3, "trieu": 1e6, "ty": 1e9, "ti": 1e9}
_VN_SKIP = {"le", "linh", "dong", "chan", "vnd"}


def parse_vn_words(s) -> int | None:
    """'hai trăm bốn mươi lăm nghìn' -> 245000 (also without accents: 'muoi' is mười when no digit precedes it,
    mươi when one does). A word that is not a digit -> None."""
    total = group = 0
    pending, seen = None, False
    for w in fold(s).replace(",", " ").split():
        if w in _VN_DIGIT:
            if pending is not None:          # two digits side by side ("hai ba"): not a way of reading a number
                return None
            pending, seen = _VN_DIGIT[w], True
        elif w == "muoi":
            group += 10 if pending is None else pending * 10
            pending, seen = None, True
        elif w == "tram":
            group += (pending or 0) * 100
            pending = None
        elif w in _VN_SCALE:
            group += pending or 0
            total += group * _VN_SCALE[w]
            group, pending = 0, None
        elif w not in _VN_SKIP:
            return None
    return int(total + group + (pending or 0)) if seen else None


_NUM_TOKEN = re.compile(r"\d[\d.,]*(?:\s*(?:k|nghìn|ngàn|tr|triệu|tỷ|tỉ)(?:\s*\d+)?)?(?:\s*(?:đ|vnđ|vnd|đồng))?", re.I)


def loose_number(v):
    """A number read in words, or a cell with EXACTLY ONE number plus text ('x2', 'TT 388.000', '245.000đ (đã CK)'). With a minus sign
    outside the number (accounting-style negative '30.000-') or several numbers, do not guess."""
    s = str(v).strip()
    if (x := parse_vn_words(s)) is not None:
        return x
    toks = _NUM_TOKEN.findall(s)
    if len(toks) != 1 or "-" in s.replace(toks[0], ""):
        return None
    return parse_number(toks[0].strip())


_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
           "nov": 11, "dec": 12}
_D_EN_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b")       # Mar 05, 2025
_D_EN_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")       # 05 March 2025
_DATE_TOKEN = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def loose_date(v):
    """An English-style date, or a cell with EXACTLY ONE date phrase plus text ('Thứ 4, 05/03/2025')."""
    import datetime as dt
    s = str(v)
    for rx, mi, di in ((_D_EN_MDY, 1, 2), (_D_EN_DMY, 2, 1)):
        if m := rx.search(s):
            mon = _MONTHS.get(m[mi][:3].lower())
            if mon:
                try:
                    return dt.datetime(int(m[3]), mon, int(m[di]))
                except ValueError:
                    return None
    toks = _DATE_TOKEN.findall(s)
    return parse_date(toks[0]) if len(toks) == 1 else None

_SYSTEM = """Bạn chuẩn hoá các ô trong sổ bán hàng của cửa hàng nhỏ ở Việt Nam mà máy không đọc được.
Mỗi ô viết lại thành dạng chuẩn, theo đúng cách hiểu của các ô mẫu cùng cột:
- cột số: số thuần, dấu chấm thập phân, không dấu nghìn, cùng đơn vị với các ô mẫu (vd "245000", "2", "12.5")
- cột ngày: YYYY-MM-DD, có giờ thì YYYY-MM-DD HH:MM
Ô không phải một giá trị (ghi chú, chưa rõ, ký hiệu), hoặc không chắc: null. Không đoán.
Chỉ trả về một JSON object: {"<ô y như đề>": "<chuỗi chuẩn>" hoặc null}."""


def repair_prompt(column: str, kind: str, bad: list[str], good: list[str]) -> tuple[str, str]:
    """(system, user) — shared by the pipeline and (formerly) the data-layer fine-tune data, not included in this repo."""
    user = "\n".join([f"Cột: {column} (kiểu cần: {'số' if kind == 'number' else 'ngày'})",
                      "Ô mẫu đọc được: " + "; ".join(good),
                      "Ô cần chuẩn hoá:", *[f'- "{v}"' for v in bad]])
    return _SYSTEM, user


def _read(kind: str, v, year=None):
    return parse_number(v) if kind == "number" else parse_date(v, year)


def _plausible(kind: str, x, good_vals: list) -> bool:
    """Of the same scale as the readable cells of the column: a number within [min/1000, max*1000] (by absolute value),
    a date within the column's date range widened by 2 years."""
    if not good_vals:
        return True
    if kind == "number":
        mags = [abs(v) for v in good_vals if v]
        return not mags or x == 0 or min(mags) / 1000 <= abs(x) <= max(mags) * 1000
    lo, hi = min(good_vals), max(good_vals)
    return (lo.year - 2) <= x.year <= (hi.year + 2)


def target_kind(res: dict, raw: list) -> str | None:
    """The kind a column should have: numeric / date already accepted (with failing cells left), or a text column most of whose cells read as numbers / dates."""
    if res["kind"] in ("number", "datetime"):
        return res["kind"] if res["n_fail"] else None
    if res["kind"] != "text" or not res["n"]:
        return None
    present = [v for v in raw if v is not None]
    if any(looks_like_code(v) for v in present[:500]):       # phone numbers, barcodes: keep as text
        return None
    if sum(parse_number(v) is not None for v in present) >= MIN_RATE * len(present):
        return "number"
    if sum(parse_date(v) is not None for v in present) >= MIN_RATE * len(present):
        return "datetime"
    return None


def repair_column(column: str, raw: list, years: list, res: dict) -> dict:
    """res from parsers.parse_column -> repaired res (values, kind, n_fail, 'repaired'); unchanged when it cannot be repaired."""
    if not ENABLED or not llm.available():
        return res
    kind = target_kind(res, raw)
    if kind is None:
        return res
    vals = res["values"] if res["kind"] == kind else [_read(kind, v, y) for v, y in zip(raw, years)]
    bad = {}
    for v, p in zip(raw, vals):
        if v is not None and p is None:
            bad[str(v)] = bad.get(str(v), 0) + 1
    if not bad:
        return res
    good_raw = list(dict.fromkeys(str(v) for v, p in zip(raw, vals) if v is not None and p is not None))[:6]
    good_vals = [p for p in vals if p is not None]
    # 1. extended rules for every odd cell
    fixed = {}
    for v in bad:
        x = loose_number(v) if kind == "number" else loose_date(v)
        if x is not None and _plausible(kind, x, good_vals):
            fixed[v] = x
    # 2. LLM for the remaining cells (most rows first)
    top = [v for v, _ in sorted(bad.items(), key=lambda x: -x[1]) if v not in fixed][:MAX_VALUES]
    out = {}
    if top:
        system, user = repair_prompt(column, kind, top, good_raw)
        try:
            text = llm.chat(system, user, max_tokens=40 + 20 * len(top))
            m = re.search(r"\{.*\}", text, re.S)
            out = json.loads(m.group(0)) if m else {}
        except Exception:
            out = {}
    for v in top:
        s = out.get(v) if isinstance(out, dict) else None
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            s = str(s)
        if not isinstance(s, str):
            continue
        x = _read(kind, s)
        if x is not None and _plausible(kind, x, good_vals):
            fixed[v] = x
    if not fixed:
        return res
    new = [p if p is not None or v is None else fixed.get(str(v)) for v, p in zip(raw, vals)]
    n_fail = sum(v is not None and p is None for v, p in zip(raw, new))
    present = sum(v is not None for v in raw)
    if res["kind"] == "text" and n_fail > 0.05 * present:      # still too many odd cells: keep as a text column
        return res
    out_res = dict(res, kind=kind, values=new, n_fail=n_fail, repaired=sum(bad[v] for v in fixed),
                   fail_examples=list(dict.fromkeys(str(v) for v, p in zip(raw, new) if v is not None
                                                    and p is None))[:5])
    out_res["had_text"] = True
    return out_res
