"""Cut a raw cell grid into tables: find the header, drop total / note rows, merge sheets.

Patterns seen in real data:
- Header at row 0..3, above it a title ("SỔ CHI TIẾT BÁN HÀNG"), filter rows
  ("Kỳ báo cáo: toàn bộ", "Trạng thái đơn hàng | Đặt hàng (Đã duyệt)") or blank rows.
- Two-level header: a sparse group row ("Hàng", _, _, "Tiền", ...) right above the real header.
- Total rows in the middle of the data, the label in any column: "Tổng ngày 02/06",
  "Cộng tháng 09/2026", "tổng 06/09", "TỔNG THÁNG", "Tổng cộng".
- Note rows at the end of a sheet: "Ghi chú: số liệu ghi tay".
- One file with several sheets: same header (split by month) -> merge; different header -> separate table.

Cell values are NOT changed here; every dropped row is recorded in `dropped`.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import re
from dataclasses import dataclass, field

from .read_files import RawSheet

SCAN_ROWS = 30
# Ledgers typed without accents ("Tong cong", "Cong thang 9"): an unaccented word alone ("Tong", "Cong") may be a person's name, so
# the unaccented form needs a second word or an accompanying number / date.
SUMMARY_RE = re.compile(
    r"^(?:(tổng|cộng|total|subtotal)"
    r"(\s+(cộng|ngày|tháng|năm|quý|tuần|kỳ|ca))?"
    r"[\s:]*[\d/.\-\s]*"
    r"|(tong|cong)\s+(cong|ngay|thang|nam|quy|tuan|ky|ca)[\s:]*[\d/.\-\s]*"
    r"|(tong|cong)[\s:]*\d[\d/.\-\s]*)$",
    re.IGNORECASE,
)
NOTE_RE = re.compile(r"^(ghi chú|lưu ý|chú thích|note)\s*:", re.IGNORECASE)
# total labels with a number ("Tổng tiền tháng 3", "TC 03/2025"): accepted only when they start with a total-type word (the number pattern must still match)
_TOTAL_WORD = re.compile(r"^(tổng|tong|tông|cộng|cong|t\.?\s?c\.?\b|tc\b|sum|total|subtotal|∑)")


@dataclass
class Table:
    name: str
    file: str
    sheets: list[str | None]
    columns: list[str]
    rows: list[list]
    src: list[tuple]                 # (sheet, row number within the sheet, counted from 1) for each row
    groups: dict[str, str] = field(default_factory=dict)   # column -> upper-level header group
    preamble: list[str] = field(default_factory=list)      # title / filter rows above the header
    footnotes: list[str] = field(default_factory=list)     # "Ghi chú: ..." rows
    dropped: list[tuple] = field(default_factory=list)     # (sheet, row, reason, content)


def _nonempty(row) -> list:
    return [c for c in row if c is not None]


def _header_cell(c) -> bool:
    """Header cell: text; or a year / date used as a column name (tables by year or by month: 'Hàng | 2024 | 2025')."""
    if isinstance(c, str):
        return True
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return float(c).is_integer() and 1900 <= c <= 2100
    return isinstance(c, (dt.date, dt.datetime))


def _is_header_like(row, min_cells: int) -> bool:
    cells = _nonempty(row)
    n_str = sum(isinstance(c, str) for c in cells)
    return len(cells) >= min_cells and all(_header_cell(c) for c in cells) and n_str >= 0.5 * len(cells)


def detect_header(rows: list[list]) -> tuple[int | None, int | None]:
    """Return (header row, group row above it or None)."""
    scan = rows[:SCAN_ROWS]
    width = max((len(_nonempty(r)) for r in scan), default=0)
    if width == 0:
        return None, None
    min_cells = max(1, math.ceil(0.6 * width)) if width > 1 else 1
    for i, row in enumerate(scan):
        if not _is_header_like(row, min_cells):
            continue
        # group row wrongly detected: the row right below is also all text and denser
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            if _is_header_like(nxt, min_cells) and len(_nonempty(nxt)) > len(_nonempty(row)):
                return i + 1, i
        group = None
        if i > 0 and _nonempty(rows[i - 1]) and all(isinstance(c, str) for c in _nonempty(rows[i - 1])):
            prev = rows[i - 1]
            # group row: sparse, its labels sit over positions where the header has a column
            if 1 < len(_nonempty(prev)) < len(_nonempty(row)) and len(prev) > 2:
                group = i - 1
        return i, group
    return None, None


def _row_text(row) -> str:
    return " | ".join(str(c) for c in row if c is not None)


def classify_row(row, header: list) -> str | None:
    """Reason the row is dropped, or None if it is a data row."""
    cells = _nonempty(row)
    if not cells:
        return "empty"
    # check "summary" first: a header may contain a column named "Cộng"
    if [c for c in row[:len(header)]] == header:
        return "repeat_header"
    if any(isinstance(c, str) and SUMMARY_RE.match(c) for c in cells):
        return "summary"
    if len(cells) <= 2 and isinstance(cells[0], str) and NOTE_RE.match(cells[0]):
        return "note"
    return None


def _make_columns(header_row: list, width: int) -> list[str]:
    cols, seen = [], {}
    for j in range(width):
        name = header_row[j] if j < len(header_row) else None
        if isinstance(name, float) and name.is_integer():
            name = int(name)                        # a year used as a column name: "2024", not "2024.0"
        elif isinstance(name, dt.datetime):
            name = name.strftime("%Y-%m-%d")
        name = re.sub(r"\s+", " ", str(name)).strip() if name is not None else f"col_{j + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        cols.append(name)
    return cols


def _drop_unlabeled_totals(data: list, src: list, dropped: list, max_tail: int = 3):
    """A total row WITHOUT a "Tổng/Cộng" word at the end of a sheet: sparse cells, every numeric cell = the column total of the rows above.
    Keeping it would double-count revenue."""
    from .parsers import parse_number
    if len(data) < 3:
        return
    filled = sorted(len(_nonempty(r)) for r in data)
    median = filled[len(filled) // 2]
    for _ in range(max_tail):
        row, body = data[-1], data[:-1]
        filled_cells = [(j, c) for j, c in enumerate(row) if c is not None and str(c).strip() != ""]
        cells = [(j, parse_number(c)) for j, c in filled_cells]
        # text cells (labels "Tong cong", "Sum", "Cộng tiền"... written any way) do not break the check: the evidence is that
        # the NUMERIC cells all equal the column total above. A label may only be short text, not a number / date.
        labels = [c for (j, c), (_, v) in zip(filled_cells, cells) if v is None]
        cells = [(j, v) for j, v in cells if v is not None]
        if not cells or len(filled_cells) > max(1, median // 2) or len(labels) > 1                 or any(not isinstance(c, str) or len(c) > 40 or re.search(r"\d", c) and not _TOTAL_WORD.match(c.strip().lower())
                       for c in labels):
            return
        ok = 0
        for j, v in cells:
            col = [parse_number(r[j]) for r in body if j < len(r) and r[j] is not None]
            col = [x for x in col if x is not None]
            if len(col) < 2 or abs(sum(col) - v) > max(1.0, 0.005 * abs(v)):
                return
            ok += 1
        if not ok:
            return
        sheet, k = src[-1]
        dropped.append((sheet, k, "summary_no_label", _row_text(row)))
        del data[-1], src[-1]


def extract_table(raw: RawSheet) -> Table | None:
    rows = raw.rows
    if not rows:
        return None
    if raw.header_known:
        h, g = 0, None
    else:
        h, g = detect_header(rows)
        if h is None:
            return None

    header_row = rows[h]
    body_start = h + 1
    data, src, dropped, footnotes = [], [], [], []
    for k in range(body_start, len(rows)):
        row = rows[k]
        reason = None if raw.header_known else classify_row(row, header_row)
        if reason is None:
            data.append(row)
            src.append((raw.sheet, k + 1))
        elif reason == "note":
            footnotes.append(_row_text(row))
        elif reason != "empty":
            dropped.append((raw.sheet, k + 1, reason, _row_text(row)))

    if not raw.header_known:
        _drop_unlabeled_totals(data, src, dropped)

    # width = the last position that has a header or data
    last_hdr = max((j for j, c in enumerate(header_row) if c is not None), default=-1)
    last_data = max((max((j for j, c in enumerate(r) if c is not None), default=-1) for r in data), default=-1)
    width = max(last_hdr, last_data) + 1
    columns = _make_columns(header_row, width)
    data = [(r + [None] * width)[:width] for r in data]

    groups = {}
    if g is not None:
        cur = None
        for j, c in enumerate((rows[g] + [None] * width)[:width]):
            cur = c if c is not None else cur
            if cur is not None:
                groups[columns[j]] = cur
    preamble = [_row_text(r) for r in rows[:g if g is not None else h] if _nonempty(r)]

    stem = os.path.splitext(raw.file)[0]
    return Table(name=stem if raw.kind != "sqlite" else raw.sheet, file=raw.file,
                 sheets=[raw.sheet], columns=columns, rows=data, src=src, groups=groups,
                 preamble=preamble, footnotes=footnotes, dropped=dropped)


_LEDGER_KIND = [(re.compile(p), k) for p, k in [
    (r"\b(nhap|mua hang|phieu nhap|pn|purchase|ncc|nha cung cap)\b", "nhap"),
    (r"\b(tra hang|tra lai|hang tra|doi tra|tra|hoan|return|refund)\b", "tra"),
    (r"\b(huy|cancel)\b", "huy"), (r"\b(ton|ton kho|kho|stock|inventory|the kho)\b", "ton"),
    (r"\b(thu no|phieu thu|thu tien|so thu|payment|receipt)\b", "thu"),
    (r"\b(ban|ban hang|don|don hang|hd|hoa don|sale|sales|order|bill)\b", "ban")]]


def ledger_kind(name) -> str | None:
    """Ledger kind by file / sheet / table name (accents stripped, CamelCase split): nhap / tra / huy / ton / thu / ban / None."""
    from .parsers import fold
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name or ""))
    f = re.sub(r"[_\-.]+", " ", fold(s))
    return next((k for rx, k in _LEDGER_KIND if rx.search(f)), None)


def khac_loai(a, b) -> bool:
    """Two ledgers that differ in nature by name (a sales ledger vs a purchase / return / cancel ledger...). Unknown kind, or both sales ledgers: not different."""
    ka, kb = ledger_kind(a), ledger_kind(b)
    ka, kb = (None if ka == "ban" else ka), (None if kb == "ban" else kb)
    return ka != kb and (ka is not None or kb is not None)


def _same_ledger(a: set, b: set, name_a="", name_b="") -> bool:
    """Sheets of the same file: the same column set, or nearly (a later month's sheet adds a few extra columns). A "Bán" sheet and a "Nhập" /
    "Trả" sheet with the same columns are still two ledgers."""
    if khac_loai(name_a, name_b):
        return False
    return a == b or (len(a & b) >= 3 and len(a & b) / len(a | b) >= 0.7)


def merge_sheets(tables: list[Table]) -> list[Table]:
    """Merge the sheets of one file that have the same column set (e.g. one sheet per month); a sheet missing a column leaves that cell empty."""
    out: list[Table] = []
    by_file: dict[str, list[Table]] = {}
    for t in tables:
        by_file.setdefault(t.file, []).append(t)
    for file, ts in by_file.items():
        if ts[0].sheets == [None] or len(ts) == 1 or file.lower().endswith((".sqlite", ".db", ".sqlite3")):
            out.extend(ts)
            continue
        groups: list[list[Table]] = []
        for t in ts:
            for grp in groups:
                if _same_ledger(set(grp[0].columns), set(t.columns), grp[0].sheets[0], t.sheets[0]):
                    grp.append(t)
                    break
            else:
                groups.append([t])
        for grp in groups:
            base = grp[0]
            if len(groups) > 1:
                base.name = f"{base.name}__{base.sheets[0]}" if len(grp) == 1 else base.name
            for t in grp[1:]:
                new = [c for c in t.columns if c not in base.columns]
                if new:
                    base.columns.extend(new)
                    for r in base.rows:
                        r.extend([None] * len(new))
                pos = {c: i for i, c in enumerate(t.columns)}     # reorder by the column order of the first sheet
                base.rows.extend([r[pos[c]] if c in pos else None for c in base.columns] for r in t.rows)
                base.src.extend(t.src)
                base.sheets.extend(t.sheets)
                base.dropped.extend(t.dropped)
                base.footnotes.extend(t.footnotes)
                base.preamble.extend(p for p in t.preamble if p not in base.preamble)
            out.append(base)
    # duplicate names (e.g. 2 sheet groups with the same file name) -> add a suffix
    seen: dict[str, int] = {}
    for t in out:
        if t.name in seen:
            seen[t.name] += 1
            t.name = f"{t.name}_{seen[t.name]}"
        else:
            seen[t.name] = 1
    return out


def load_tables(raw_sheets: list[RawSheet]) -> tuple[list[Table], list[str]]:
    """Return (tables, warnings for sheets where no header was found)."""
    tables, warnings = [], []
    for raw in raw_sheets:
        t = extract_table(raw)
        if t is None:
            if raw.rows:
                warnings.append(f"{raw.file}[{raw.sheet}]: không tìm được header")
            continue
        tables.append(t)
    return merge_sheets(tables), warnings
