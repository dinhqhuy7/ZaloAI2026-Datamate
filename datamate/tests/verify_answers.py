"""Verify a .duckdb file against the training answers, on the simplest question forms.

The SQL here is deliberately plain and shared by every shop (relying only on canonical names + table roles + columns added by build).
A wrong answer means the error is in the data layer (forward-fill, de-duplication, status_norm, qty_base, walk-in customers...),
not in understanding the question. A question that matches no template is skipped.

    python -m datamate.tests.verify_answers <split> <DB folder> [--kinds top_product ...] [-v]
"""
import argparse
import datetime as dt
import json
import math
import os
import re
from collections import Counter

import duckdb

from datamate.loader.build import line_type as line_type_of, split_phone, text_key
from datamate.loader.parsers import fold

# understanding questions (pure templates only)

_WORDS = [(r"\b(bnhiu|bao nhiu|bn|nhieu)\b(?! (tien|nhat))", "bao nhieu"), (r"\b(dc|đc)\b", "duoc"),
          (r"\bmay\b", "bao nhieu"), (r"\bnawm\b", "nam"), (r"\bthangs\b", "thang")]
_LEAD = r"^((cho (minh|toi|em) )?(hoi|biet)|xem giup|check giup|coi giup|coi gium|check gium|cho hoi|cho minh hoi) "
_TAIL = r" (a|nhi|vay|ta|vay a)$"

TEMPLATES = {
    "revenue": [r"^(tong )?doanh thu (trong )?(ngay )?<P>( la)?( bao nhieu)?$",
                r"^<P> (cua hang )?ban duoc (tong cong )?bao nhieu tien$"],
    "orders": [r"^<P> (cua hang )?ban duoc bao nhieu don( hang)?$", r"^so don <P> la bao nhieu$"],
    "cancel_count": [r"^(dem )?so don (da |bi )?huy (trong )?<P>( la bao nhieu)?$",
                     r"^<P> co bao nhieu don( hang)? bi huy$",
                     r"^trong <P> cua hang co bao nhieu don o trang thai huy$"],
    "cancel_rate": [r"^(ti|ty) le huy don <P>( la)?( bao nhieu)? ?%?$",
                    r"^trong <P> (ti|ty) le don huy la( bao nhieu)? ?%?$"],
    "top_product": [r"^(<P> )?(hang|mat hang|mon|san pham) nao ban chay nhat( <P>)?$"],
    "top_customer": [r"^(<P> )?khach (nao )?mua nhieu tien nhat( <P>)?( la ai)?$"],
    "purchase_total": [r"^<P> tong gia tri cac phieu nhap hang( la bao nhieu)?$",
                       r"^<P> cua hang bo ra bao nhieu tien nhap hang$",
                       r"^tong tien nhap hang <P>( la bao nhieu)?$"],
    # rulebook §6 building materials: number of delivery batches in the period (any phrasing with "đợt giao / đợt hàng" + a period)
    "deliveries": [r".*\bdot (giao|hang)\b.*"],
    # debt: recognised in _parse_debt
    "debt_total": [], "debt_customers": [], "debt_aging": [], "debt_buckets": [], "debt_top": [],
    "debt_collected": [], "debt_customer_month": [],
    # end-of-day stock of a product: recognised in _parse_stock
    "stock": [], "stock_split": [],
    # batches, expiry (rulebook §6 pharmacy), out-of-stock products (§5): recognised in _parse_lots
    "lot_nearest": [], "lot_value": [], "lot_count": [], "lot_products": [], "lot_products_count": [],
    "stockout_count": [],
    # cost price, profit, prescription (rulebook §5, §6 pharmacy): recognised in _parse_profit
    "avg_purchase_price": [], "inventory_value": [], "gross_profit": [], "top_profit_product": [],
    "net_profit": [], "rx_revenue": [], "non_rx_revenue": [], "rx_rate": [],
    # group 4: recognised in _parse_sales2
    "avg_order_value": [], "best_day": [], "best_month": [], "group_revenue": [], "group_share": [],
    "qty_sold": [], "price_on_day": [], "price_effect": [],
    # group 5: fallback _parse_g5
    "rev_filter": [], "growth": [], "best_staff": [], "cust_min_orders": [], "group_avg_order": [],
    "group_top_product": [],
}
_TPL = {k: [re.compile(p) for p in v] for k, v in TEMPLATES.items()}


_DEBT_DAY = re.compile(r"het ngay (\d{1,2})/(\d{1,2})/(\d{4})")


_COLLECT = re.compile(r"\btra (bot )?no\b|\bthu hoi\b.*\bcong no\b|\bthu (duoc|dc)\b.*\btra no\b")
_ADD = re.compile(r"\b(ghi no them|no them|mua chiu them)\b")


def _parse_debt(f: str, today: dt.date):
    """Debt (rulebook §4) -> (kind, (day X or period, parameter)) or None. Pure forms only:
    at the end of day X: total debt, number of customers still owing, debt of orders over N days, 3 debt-age groups, top k debtors;
    within a month: debt collected of the whole shop, extra debt + payments of one customer."""
    if not re.search(r"\bno\b|\bmua chiu\b", f):
        return None
    m = _DEBT_DAY.search(f)
    if m and not _ADD.search(f) and not _COLLECT.search(f):
        x = dt.date(int(m[3]), int(m[2]), int(m[1]))
        if re.search(r"\b0 ?[-–] ?7\b.*\b8 ?[-–] ?15\b.*\btren 15\b", f):
            return "debt_buckets", (x, None)
        if a := re.search(r"\b(hon|qua) (\d+) ngay\b", f):
            return "debt_aging", (x, int(a[2]))
        if t := re.search(r"\b(top )?(\d+) khach\b.*\b(nhieu nhat|lon nhat)\b|\b(\d+) khach (hang )?nao\b", f):
            return "debt_top", (x, int(t[2] or t[4]))
        if re.search(r"\b(bao nhieu|so) khach\b", f):
            return "debt_customers", (x, None)
        if re.search(r"\b(tong|tong cong|bao nhieu tien)\b", f) and "khach nao" not in f:
            return "debt_total", (x, None)
        return None
    p = _period(f, today)
    if not p or isinstance(p[0], tuple):
        return None
    lo, hi, pm = p
    if _ADD.search(f):
        rest = re.sub(_LEAD, "", (f[:pm.start()] + " " + f[pm.end():]).strip())
        n = re.search(r"(?:^|\bkhach (?:hang )?)(?:trong )?([a-z][a-z ]*?) +(?:da )?(?:ghi no them|no them|mua chiu them)",
                      re.sub(r"\s+", " ", rest))
        if n:
            name = re.sub(r"^(e |em )?(trong )?(khach (hang )?)?", "", n[1].strip())
            return "debt_customer_month", ((lo, hi), name)
        return None
    if _COLLECT.search(f):
        return "debt_collected", ((lo, hi), None)
    return None


def _month_end(y: int, mo: int) -> dt.date:
    return dt.date(y + (mo == 12), mo % 12 + 1, 1) - dt.timedelta(days=1)


def _no_year(mo: int, today: dt.date) -> int:
    """A month / quarter with no year: this year if not after today, otherwise last year (§2)."""
    return today.year if dt.date(today.year, mo, 1) <= today else today.year - 1


_D = r"(\d{1,2})/(\d{1,2})/(\d{4})"
# (regex, function -> (from, to)). Order: specific before general (a range "từ X đến Y" before a single day X)
_PERIODS = [
    (rf"\btu (?:ngay )?{_D} (?:den|toi) (?:het )?(?:ngay )?{_D}",
     lambda m, t: (dt.date(int(m[3]), int(m[2]), int(m[1])), dt.date(int(m[6]), int(m[5]), int(m[4])))),
    (r"(?:dip |mua )?\btet (?:nam )?(\d{4})", lambda m, t: (("tet", int(m[1])), t)),
    (r"\bthang (\d{1,2})/(\d{4})|\bthang (\d{1,2}) nam (\d{4})",
     lambda m, t: (dt.date(int(m[2] or m[4]), int(m[1] or m[3]), 1), _month_end(int(m[2] or m[4]), int(m[1] or m[3])))),
    (r"\bquy (\d)(?:/| nam )(\d{4})",
     lambda m, t: (dt.date(int(m[2]), 3 * int(m[1]) - 2, 1), _month_end(int(m[2]), 3 * int(m[1])))),
    (r"\bquy (\d)\b",
     lambda m, t: (dt.date(_no_year(3 * int(m[1]) - 2, t), 3 * int(m[1]) - 2, 1),
                   _month_end(_no_year(3 * int(m[1]) - 2, t), 3 * int(m[1])))),
    (rf"\bngay {_D}", lambda m, t: (dt.date(int(m[3]), int(m[2]), int(m[1])),) * 2),
    (r"\bthang nay\b|\btu (dau thang|mung 1|ngay 1)( nay)? (den|toi) (het )?(hom nay|nay)\b",
     lambda m, t: (t.replace(day=1), t)),
    (r"\btuan nay\b", lambda m, t: (t - dt.timedelta(days=t.weekday()), t)),
    (r"\btuan (truoc|roi|vua roi)\b",
     lambda m, t: (t - dt.timedelta(days=t.weekday() + 7), t - dt.timedelta(days=t.weekday() + 1))),
    (r"\b(\d+) ngay (gan nhat|gan day|qua)\b", lambda m, t: (t - dt.timedelta(days=int(m[1]) - 1), t)),
    (r"\btu dau nam\b", lambda m, t: (t.replace(month=1, day=1), t)),
    (r"\bhom nay\b", lambda m, t: (t, t)),
    (r"\bhom qua\b", lambda m, t: (t - dt.timedelta(days=1),) * 2),
    (r"\bthang (\d{1,2})\b(?!/)", lambda m, t: (dt.date(_no_year(int(m[1]), t), int(m[1]), 1),
                                                 _month_end(_no_year(int(m[1]), t), int(m[1])))),
]
_PERIODS = [(re.compile(p), fn) for p, fn in _PERIODS]


def _period(f: str, today: dt.date):
    """The period in a question (rulebook §2) -> (from date, to date, match) or None.
    A date after today counts as not yet happened: the end date is cut at today."""
    for rx, fn in _PERIODS:
        if m := rx.search(f):
            lo, hi = fn(m, today)
            if isinstance(lo, tuple):
                return lo, hi, m
            return lo, min(hi, today), m
    return None


def _parse_sales2(f: str, text: str, today: dt.date):
    """Group 4 -> (kind, (parameter, original question)). The parameter is a period (from, to), except best_month (list of months) and
    price_effect (day D, N days)."""
    if re.search(r"\b(tang|doi|len) gia\b|\blen [\d.,]+ ?(d|dong|k)\b", f):
        d = re.search(rf"\btu ngay {_D}|\bngay {_D}", f)
        n = re.search(r"\b(\d+) ngay\b", f)
        if d and n:
            g = [x for x in d.groups() if x]
            return "price_effect", ((dt.date(int(g[2]), int(g[1]), int(g[0])), int(n[1])), text)
        return None
    if re.search(r"\bthang nao\b", f) and re.search(r"\b(doanh thu|dthu|nhieu tien|cao nhat)\b", f):
        months = [(int(y), int(m)) for m, y in re.findall(r"\b(\d{1,2})/(\d{4})\b", f)]
        if r := re.search(r"\btu thang (\d{1,2})/(\d{4}) (?:den|toi) thang (\d{1,2})/(\d{4})", f):
            a, b = (int(r[2]), int(r[1])), (int(r[4]), int(r[3]))
            months, (y, m) = [], a
            while (y, m) <= b:
                months.append((y, m))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return ("best_month", (months, text)) if months else None
    p = _period(f, today)
    if not p or isinstance(p[0], tuple):
        return None
    period = (p[0], p[1])
    if re.search(r"\bngay nao\b", f) and re.search(r"\b(doanh thu|dthu|nhieu tien|cao nhat)\b", f):
        return "best_day", (period, text)
    avg = re.search(r"\btrung binh\b.*\b(mot|moi|1) don\b|\bgia tri trung binh\b|\b(doanh thu|dthu) trung binh\b", f)
    if avg and "nhap" not in f:
        return ("group_avg_order" if re.search(r"\bnhom\b", f) else "avg_order_value"), (period, text)
    if re.search(r"\bgia ban\b|\bdon gia\b", f) and re.search(rf"\bngay {_D}", f):
        return "price_on_day", (period, text)
    if re.search(r"\bnhom\b", f) and not re.search(r"\b(khach mua|tu \d+ don)\b", f):
        if re.search(r"\b(mat hang|mon|san pham|hang|thuoc) nao\b", f):
            return "group_top_product", (period, text)
        if re.search(r"phan tram|%|\bchiem\b|\b(ti|ty) trong\b|\bdong gop\b", f):
            return "group_share", (period, text)
        if re.search(r"\b(tien|doanh thu|dthu)\b", f):
            return "group_revenue", (period, text)
        return None
    # a product's size / number taken by the catalogue column, not by name: not supported yet -> blank
    if re.search(r"\b(bim|ta|sua bot|sua)\b.*\b(size|so \d)\b|\bsize (s|xs|xl|xxl|xxxl|nb)\b", f) \
            or (re.search(r"\bsize\b", f) and re.search(r"phan tram|%|\bchiem\b", f)):
        return None
    if re.search(r"\bban (duoc|ra)?\s*(tong cong )?bao nhieu (?!tien|don|lan|khach)\w+|\bsize [ml]\b", f):
        return "qty_sold", (period, text)
    return None


def _parse_profit(f: str, text: str, today: dt.date):
    """Average purchase price, stock value at cost, gross / net profit, most profitable product, prescription.
    -> (kind, ((from, to), original question))."""
    if re.search(r"\bgia von\b", f) and re.search(r"\bkho\b", f) and not re.search(r"\blo\b", f):
        m = _DEBT_DAY.search(f)
        x = dt.date(int(m[3]), int(m[2]), int(m[1])) if m else today
        return "inventory_value", ((x, x), text)
    p = _period(f, today)
    if not p or isinstance(p[0], tuple):
        return None
    period = (p[0], p[1])
    if re.search(r"\bgia nhap binh quan\b", f):
        return "avg_purchase_price", (period, text)
    if re.search(r"\blai rong\b", f):
        return "net_profit", (period, text)
    if re.search(r"\b(hang|mat hang|san pham|mon) (nao|naof)\b", f) and re.search(r"\b(lai|loi)\b", f):
        return "top_profit_product", (period, text)
    if re.search(r"\b(lai|loi) gop\b", f):
        return "gross_profit", (period, text)
    if re.search(r"\bke don\b", f):
        if re.search(r"phan tram|%|\b(ti|ty) le\b|\bchiem\b", f):
            return "rx_rate", (period, text)
        if re.search(r"\b(khong|k|hok|hong|ko)\b ke don|\btru\b.*\bke don\b", f):
            return "non_rx_revenue", (period, text)
        return "rx_revenue", (period, text)
    return None


def _parse_lots(f: str, text: str, today: dt.date):
    """Batches / expiry as of today D (rulebook §6 pharmacy) and the number of out-of-stock products at the end of the day (§5).
    -> (kind, ((D, expiry until), original question)). 'expiry until' = the largest date written in the question."""
    days = [dt.date(int(y), int(m), int(d)) for d, m, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", f)]
    if re.search(r"\bmat hang\b.*\b(dang )?het hang\b", f) and not re.search(r"\bhet han\b(?!g)", f):
        return "stockout_count", ((max(days) if days else today, None), text)
    if not re.search(r"\bhet han\b|\bhan (dung|su dung)\b", f):
        return None
    until = max(days) if days else None
    if re.search(r"\blo\b.*\bgan het han nhat\b", f):
        return "lot_nearest", ((today, None), text)
    if until is None:
        return None
    if re.search(r"\btien von\b", f):
        return "lot_value", ((today, until), text)
    if re.search(r"\b(may|bao nhieu) lo\b", f):
        return "lot_count", ((today, until), text)
    if re.search(r"\b(may|bao nhieu) mat hang\b", f):
        return "lot_products_count", ((today, until), text)
    if re.search(r"\b(hang|san pham|sp|thuoc) nao\b|\bnhung (san pham|sp|hang|thuoc)\b", f):
        return "lot_products", ((today, until), text)
    return None


def _parse_stock(f: str, text: str, today: dt.date):
    """A product's stock at the end of a day (today or an explicitly written day). The product is looked up by name in the DB when answering.
    'mấy hộp chẵn và mấy viên lẻ' -> stock_split (rulebook §5: [whole large units, leftover base units])."""
    if not re.search(r"\bton kho\b|\btrong kho\b|\bcon (lai )?(bao nhieu|may)\b", f):
        return None
    if not re.search(r"\bcuoi ngay\b|\bhet (hom nay|ngay)\b", f) or re.search(r"\b(lo|han|tien|gia tri|von)\b", f):
        return None
    m = _DEBT_DAY.search(f)
    x = dt.date(int(m[3]), int(m[2]), int(m[1])) if m else today
    kind = "stock_split" if re.search(r"\b(chan|nguyen)\b.*\ble\b", f) else "stock"
    return kind, (x, text)


def parse_question(text: str, today: dt.date):
    """-> (kind, (from date, to date)) or None. Debt questions: (kind, (day X, parameter))."""
    f = re.sub(r"[?.,!]", " ", fold(text))
    f = re.sub(r"\s+", " ", f).strip()
    for p, r in _WORDS:
        f = re.sub(p, r, f)
    if d := _parse_debt(f, today):
        return d
    if s := _parse_lots(f, text, today):
        return s
    if s := _parse_profit(f, text, today):
        return s
    if s := _parse_sales2(f, text, today):
        return s
    if s := _parse_stock(f, text, today):
        return s
    f = re.sub(_LEAD, "", f)
    f = re.sub(_TAIL, "", f)
    p = _period(f, today)
    if not p:
        return None
    lo, hi, m = p
    rest = (f[:m.start()] + "<P>" + f[m.end():]).strip()
    for kind, rxs in _TPL.items():
        if any(rx.match(rest) for rx in rxs):
            return kind, (lo, hi)
    return _parse_g5(f, text, (lo, hi))


def _parse_g5(f: str, text: str, period):
    """Fallback for revenue / order-count questions with conditions (channel, payment, shift, staff, product) and comparisons
    with the previous period. Specific conditions are looked up when answering, by the values present in the DB."""
    if isinstance(period[0], tuple) and not re.search(r"\btruoc tet\b", f):
        return None
    if (n := re.search(r"\b(?:tu|it nhat) (\d+) don\b", f)) and re.search(r"\bkhach\b", f) \
            and re.search(r"phan tram|%|\b(ti|ty) le\b|\bchiem\b", f) and not isinstance(period[0], tuple):
        return "cust_min_orders", ((period, int(n[1])), text)
    if re.search(r"\b(so voi|so vs|tang hay giam|hon hay kem|cao hay thap|len hay xuong|thay doi|tang truong|"
                 r"cung ky|chenh|cao hon|thap hon)\b|\bca (chieu|sang)\b.*\b(hon|kem)\b", f) \
            and re.search(r"phan tram|%|\bphan\b", f) \
            and re.search(r"\b(doanh thu|dthu|tien|don)\b", f) \
            and not re.search(r"\b(bao hanh|luot|may ban|tu \d+ don|it nhat \d+ don)\b", f):
        return "growth", (period, text)
    if isinstance(period[0], tuple):
        return None
    if re.search(r"\bnhan vien\b.*\b(nao|gioi nhat|ai)\b", f) and re.search(r"\b(doanh thu|dthu|gioi|ban)\b", f):
        return "best_staff", (period, text)
    if re.search(r"\b(doanh thu|dthu|tien|so don|bao nhieu don|may don)\b", f) and not re.search(
            r"\b(nhap|no|ton|lai|loi|von|trung binh)\b", f):
        return "rev_filter", (period, text)
    return None


# SQL built from meta

def _cols(con, t):
    return {r[0] for r in con.execute(f'DESCRIBE "{t}"').fetchall()}


def _tables(con, role):
    rows = con.execute("SELECT table_name, n_rows FROM _meta_tables WHERE role = ? ORDER BY n_rows DESC",
                       [role]).fetchall()
    return [t for t, _ in rows]


LINE_COLS = ["order_id", "product_id", "product_key", "product_name", "qty_base", "line_amount", "line_type",
             "customer_id", "customer_key", "customer_name", "is_walk_in", "status_norm", "order_date", "order_time",
             "channel", "payment_method", "staff_id", "staff_name", "customer_group", "payment_norm", "channel_norm"]


def sales_sql(con) -> str | None:
    """Sales rows joined to orders: order_id, d, st, amt, pid, pkey, pname, qb, cid, ckey, cname, walk, ..., disc (discount on the row)."""
    lines, orders = _tables(con, "sales_lines"), _tables(con, "orders")
    if not lines:
        # a shop with only an order table: order revenue = goods total before the order discount
        for O in orders:
            oc = _cols(con, O)
            # goods amount before the order discount; a service-slip ledger that records only the slip total takes the total
            money = "order_subtotal" if "order_subtotal" in oc else "order_total" if "order_total" in oc else None
            if money and ("status_norm" in oc or money == "order_total") and oc & {"order_date", "order_time"}:
                d = "order_date" if "order_date" in oc else "order_time"
                ck = "customer_key" if "customer_key" in oc else "NULL"
                st = "status_norm" if "status_norm" in oc else "NULL"
                oid = "order_id" if "order_id" in oc else "_src_row"
                return (f'SELECT {oid} AS order_id, CAST("{d}" AS DATE) AS d, {st} AS st, {money} AS amt, '
                        f'NULL AS pid, NULL AS pkey, NULL AS pname, NULL AS qb, NULL AS cid, {ck} AS ckey, '
                        f'{"customer_name" if "customer_name" in oc else "NULL"} AS cname, '
                        f'{"is_walk_in" if "is_walk_in" in oc else "FALSE"} AS walk, NULL AS qty, NULL AS uc, NULL AS up, '
                        f'{"channel_norm" if "channel_norm" in oc else "NULL"} AS ch, '
                        f'{"payment_norm" if "payment_norm" in oc else "NULL"} AS pm, NULL AS chraw, '
                        f'{"payment_method" if "payment_method" in oc else "NULL"} AS pmraw, '
                        f'{"COALESCE(CAST(staff_id AS VARCHAR), staff_name)" if "staff_id" in oc and "staff_name" in oc else "staff_name" if "staff_name" in oc else "CAST(staff_id AS VARCHAR)" if "staff_id" in oc else "NULL"} AS staff, '
                        f'{"order_time" if "order_time" in oc else "NULL"} AS t, NULL AS disc '
                        + (f'FROM (SELECT DISTINCT ON (order_id) * FROM "{O}")' if "order_id" in oc else f'FROM "{O}"'))
        return None
    L = lines[0]
    lc = _cols(con, L)
    amt_col = "line_amount"
    if "line_amount" not in lc:
        # a ledger with one row per session (EV charging...): no line amount, only the session total, the session code does not repeat
        if "order_total" not in lc:
            return None
        if "order_id" in lc and con.execute(f'SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM "{L}"').fetchone()[0] > 0:
            return None
        amt_col = "order_total"
    O = next((o for o in orders if "order_id" in _cols(con, o)), None)
    oc = _cols(con, O) if O else set()
    join = bool(O) and "order_id" in lc and any(c not in lc and c in oc for c in LINE_COLS)

    def pick(c):
        if c in lc:
            return f'l."{c}"'
        if join and c in oc:
            return f'o."{c}"'
        return "NULL"
    # rulebook §1: do not deduct the whole-order discount (voucher, points, discount rows), do not add VAT, do not include delivery fees
    conds = (["l.line_type NOT IN ('voucher', 'giam_gia', 'dung_diem', 'phi_giao', 'thue')"] if "line_type" in lc else [])
    if "is_latest_row" in lc:          # a table recording status history: for each product row only the latest version counts
        conds.append("COALESCE(l.is_latest_row, TRUE)")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    src = f'"{L}" l' + (f' LEFT JOIN (SELECT DISTINCT ON (order_id) * FROM "{O}") o USING (order_id)' if join else "")
    oid = 'l."order_id"' if "order_id" in lc else "NULL"
    lt = 'l."line_type"' if "line_type" in lc else "'hang'"
    amt = f"l.{amt_col}"
    if amt_col == "line_amount" and {"line_discount", "qty", "unit_price"} <= lc:
        # §1 line revenue is after the line discount: a ledger records the amount BEFORE discount (= qty x unit price) on almost every discounted row
        # so the real money = line amount - line discount; a ledger that records after discount stays as is
        n, gross, net = con.execute(f"""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE abs(a - q * p) <= 1),
                   COUNT(*) FILTER (WHERE abs(a - (q * p - x)) <= 1)
            FROM (SELECT TRY_CAST(line_amount AS DOUBLE) a, TRY_CAST(qty AS DOUBLE) q, TRY_CAST(unit_price AS DOUBLE) p,
                         TRY_CAST(line_discount AS DOUBLE) x FROM "{L}") WHERE x > 0 AND a IS NOT NULL""").fetchone()
        # a ROW's discount has a discount / line-amount ratio gathered into a few levels (card 3% / 5%); a whole-bill discount repeated on every row or
        # written on one row ("bớt 2% trên tổng bill") gives a messy ratio per row -> do not deduct (§1)
        top = con.execute(f"""
            SELECT COALESCE(SUM(k), 0) FROM (SELECT COUNT(*) k FROM (
                SELECT round(TRY_CAST(line_discount AS DOUBLE) / (TRY_CAST(qty AS DOUBLE) * TRY_CAST(unit_price AS DOUBLE)), 3) r
                FROM "{L}" WHERE TRY_CAST(line_discount AS DOUBLE) > 0 AND TRY_CAST(qty AS DOUBLE) * TRY_CAST(unit_price AS DOUBLE) > 0)
                GROUP BY r ORDER BY k DESC LIMIT 3)""").fetchone()[0]
        if n >= 20 and gross >= 0.9 * n and net < 0.1 * n and top >= 0.8 * n:
            amt = "(l.line_amount - COALESCE(TRY_CAST(l.line_discount AS DOUBLE), 0))"
    return (f"SELECT {oid} AS order_id, CAST(COALESCE({pick('order_date')}, {pick('order_time')}) AS DATE) AS d, "
            f"{pick('status_norm')} AS st, {amt} AS amt, {lt} AS lt, "
            f"CAST({pick('product_id')} AS VARCHAR) AS pid, CAST({pick('product_key')} AS VARCHAR) AS pkey, "
            f"{pick('product_full') if 'product_full' in lc else pick('product_name')} AS pname, {pick('qty_base')} AS qb, CAST({pick('customer_id')} AS VARCHAR) AS cid, "
            f"CAST({pick('customer_key')} AS VARCHAR) AS ckey, CAST({pick('customer_name')} AS VARCHAR) AS cname, "
            f"{pick('is_walk_in')} AS walk, "
            # cost price recorded on the sales row: for one sold unit, the same unit as the unit price (§5 gross profit)
            f"{'l.qty' if 'qty' in lc else 'NULL'} AS qty, {'l.unit_cost' if 'unit_cost' in lc else 'NULL'} AS uc, "
            f"{'l.unit_price' if 'unit_price' in lc else 'NULL'} AS up, "
            f"CAST({pick('channel_norm')} AS VARCHAR) AS ch, CAST({pick('payment_norm')} AS VARCHAR) AS pm, "
            f"CAST({pick('channel')} AS VARCHAR) AS chraw, CAST({pick('payment_method')} AS VARCHAR) AS pmraw, "
            f"CAST(COALESCE({pick('staff_id')}, {pick('staff_name')}) AS VARCHAR) AS staff, "
            f"CAST(COALESCE({pick('order_time')}, {pick('order_date')}) AS TIMESTAMP) AS t, "
            f"{'l.line_discount' if 'line_discount' in lc else 'NULL'} AS disc "
            f"FROM {src} {where}")


def orders_sql(con) -> str | None:
    """One row per order: order_id, d, st (the order table if present, otherwise aggregated from the sales rows)."""
    for role in ("orders", "sales_lines"):
        for t in _tables(con, role):
            c = _cols(con, t)
            if {"order_id", "status_norm"} <= c and c & {"order_date", "order_time"}:
                d = "order_date" if "order_date" in c else "order_time"
                return (f'SELECT order_id, CAST(MIN("{d}") AS DATE) AS d, MIN(status_norm) AS st '
                        f'FROM "{t}" GROUP BY order_id')
    return None


def purchases_sql(con) -> str | None:
    """Each purchase slip: d, total, cancelled."""
    for P in _tables(con, "purchases"):
        pc = _cols(con, P)
        if {"purchase_total", "purchase_date"} <= pc:
            cx = "purchase_cancelled" if "purchase_cancelled" in pc else "FALSE"
            return (f'SELECT CAST(purchase_date AS DATE) AS d, purchase_total AS total, {cx} AS cancelled '
                    f'FROM (SELECT DISTINCT ON (purchase_id) * FROM "{P}")')
    for L in _tables(con, "purchase_lines"):
        lc = _cols(con, L)
        amt = "line_amount" if "line_amount" in lc else "qty * unit_price" if {"qty", "unit_price"} <= lc else None
        if amt is None:
            continue
        if "purchase_total" in lc and "purchase_id" in lc:
            amt = "purchase_total"
            agg = "ANY_VALUE"
        else:
            agg = "SUM"
        cx = "BOOL_OR(purchase_cancelled)" if "purchase_cancelled" in lc else "FALSE"
        key = "purchase_id" if "purchase_id" in lc else "purchase_date"
        if "purchase_date" in lc:
            return (f'SELECT CAST(MIN(purchase_date) AS DATE) AS d, {agg}({amt}) AS total, {cx} AS cancelled '
                    f'FROM "{L}" GROUP BY {key}')
    return None


OK_ORDER = "(st IS NULL OR st NOT IN ('huy', 'tra_hang'))"


def deliveries_sql(con) -> str | None:
    """One row per delivery batch of each order: d (delivery date). Rulebook §6: including batches of orders that were later returned."""
    for t in _tables(con, "deliveries"):
        c = _cols(con, t)
        if "delivery_date" not in c:
            continue
        key = ", ".join(k for k in ("order_id", "delivery_round") if k in c) or "delivery_date"
        return f'SELECT CAST(MIN(delivery_date) AS DATE) AS d FROM "{t}" GROUP BY {key}'
    return None


# rulebook §4: COD orders and marketplace (e-commerce) orders are not customer debt
_NOT_DEBT = {"payment_method": r"\bcod\b", "channel": r"\bsan\b|shopee|lazada|tiki|tiktok"}


def _debt_base(con):
    """(CTE of orders counted into debt `od`, CTE of payments `pm`) or None.
    od: order_id, tot (order total), d (order date), cid, cname, walk; cancelled orders, COD / marketplace orders removed (§4).
    pm: order_id, pd (payment date), amt. Debt follows the payment ledger; the order's "đã trả" column is not used."""
    P = next((p for p in _tables(con, "payments")
              if {"order_id", "payment_amount", "payment_date"} <= _cols(con, p)), None)
    orders_candidates = [o for o in _tables(con, "orders")
                         if "order_id" in _cols(con, o) and _cols(con, o) & {"order_date", "order_time"}]
    # Prefer a table that has order_total (as before)
    O = next((o for o in orders_candidates if "order_total" in _cols(con, o)), None)
    if O is None and not orders_candidates:
        # invoice and product rows merged into one sheet (order info repeated on every row): a sales-row table with an order total is the order table
        # (group by order_id, ANY_VALUE(order_total) so repeated rows are not summed twice)
        O = next((t for t in _tables(con, "sales_lines") if {"order_id", "order_total"} <= _cols(con, t)
                  and _cols(con, t) & {"order_date", "order_time"}), None)
    upfront = None
    if O is None and not orders_candidates:
        # a product-row sales ledger without an order total but with the money the customer handed over at purchase ("Đưa trước"): order total = line money - discount
        # + delivery fee (as in the branch below), the prepaid money is a payment on the purchase day. Only a "đã trả" column makes it a ledger that tracks debt.
        O = next((t for t in _tables(con, "sales_lines") if {"order_id", "line_amount", "paid"} <= _cols(con, t)
                  and _cols(con, t) & {"order_date", "order_time"}), None)
        upfront = O
        if O is None and P:
            # a product-row sales ledger without an order total, a payment ledger that records every payment (including at purchase): order total = line money - discount
            # on the order (a column in the sales ledger or a separate order-info table) + delivery fee
            O = next((t for t in _tables(con, "sales_lines") if {"order_id", "line_amount"} <= _cols(con, t)
                      and _cols(con, t) & {"order_date", "order_time"}), None)
    nv2 = bool(os.environ.get("DATAMATE_NV2"))   # debt when the order table has no order total
    if O is None and orders_candidates and nv2:
        O = orders_candidates[0]
    if not (P and O):
        return None
    oc = _cols(con, O)

    # Order date d: CAST(MIN(COALESCE(order_date, order_time)) AS DATE)
    if {"order_date", "order_time"} <= oc:
        date_expr = 'COALESCE(order_date, order_time)'
    elif "order_date" in oc:
        date_expr = 'order_date'
    else:
        date_expr = 'order_time'
    d_col = f'CAST(MIN({date_expr}) AS DATE)'

    # Total payable tot
    if "order_total" in oc:
        tot_expr = "ANY_VALUE(order_total)"
        from_orders = f'"{O}"'
    else:
        # The first sales-row table that has order_id and line_amount
        L = next((l for l in _tables(con, "sales_lines")
                  if {"order_id", "line_amount"} <= _cols(con, l)), None)
        if L is None:
            return None
        lc = _cols(con, L)
        # the same conditions as sales_sql: voucher / discount / delivery fee / tax rows are not goods money
        cond_line = ("line_type NOT IN ('voucher', 'giam_gia', 'dung_diem', 'phi_giao', 'thue')"
                     if "line_type" in lc else "TRUE")
        # sum the line money per order BEFORE joining: joining directly adds money twice for an order recorded on two rows of the order table
        ldisc = "ANY_VALUE(order_discount)" if "order_discount" in lc else "NULL"
        lship = "ANY_VALUE(shipping_fee)" if "shipping_fee" in lc else "NULL"
        # a negative row is a customer's return row: rulebook §4 "a returned order still counts if debt remains", "a refund for returned goods is not deducted"
        # -> the order total used for debt is the sale amount, without deducting return rows
        lt = (f'(SELECT order_id, SUM(CASE WHEN {cond_line} AND line_amount > 0 THEN line_amount ELSE 0 END) amt, '
              f'{ldisc} ldisc, {lship} lship FROM "{L}" GROUP BY order_id)')
        disc_expr = ("COALESCE(ANY_VALUE(order_discount), 0)" if "order_discount" in oc
                     else "COALESCE(ANY_VALUE(ldisc), 0)" if "order_discount" in lc else "0")
        xd = None
        if disc_expr == "0":
            # an order discount recorded in a separate table (order info: order code + order discount)
            xd = next((t for (t,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
                       if not t.startswith("_") and t not in (O, L) and {"order_id", "order_discount"} <= _cols(con, t)), None)
            if xd:
                disc_expr = "COALESCE(ANY_VALUE(xd.g), 0)"
        ship_expr = ("COALESCE(ANY_VALUE(shipping_fee), 0)" if "shipping_fee" in oc
                     else "COALESCE(ANY_VALUE(lship), 0)" if "shipping_fee" in lc else "0")
        # delivery fee recorded in a separate order-info table (the same table as the order discount, or a table with only the order code + delivery fee)
        xs = None
        if ship_expr == "0":
            xs = next((t for (t,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
                       if not t.startswith("_") and t not in (O, L) and {"order_id", "shipping_fee"} <= _cols(con, t)), None)
            if xs:
                ship_expr = "COALESCE(ANY_VALUE(xs.s), 0)"
        tot_expr = f"COALESCE(ANY_VALUE(amt), 0) - {disc_expr} + {ship_expr}"
        from_orders = f'"{O}" LEFT JOIN {lt} USING (order_id)'
        if xd:
            from_orders += (f' LEFT JOIN (SELECT CAST(order_id AS VARCHAR) k, ANY_VALUE(TRY_CAST(order_discount AS DOUBLE)) g '
                            f'FROM "{xd}" GROUP BY 1) xd ON xd.k = CAST(order_id AS VARCHAR)')
        if xs:
            from_orders += (f' LEFT JOIN (SELECT CAST(order_id AS VARCHAR) k, ANY_VALUE(TRY_CAST(shipping_fee AS DOUBLE)) s '
                            f'FROM "{xs}" GROUP BY 1) xs ON xs.k = CAST(order_id AS VARCHAR)')

    # per the contest set's rulebook: "Đơn tính vào nợ | đơn ghi nợ"
    has_ghi_no_rule = False
    all_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "_rulebook" in all_tables:
        texts = [r[0] for r in con.execute("SELECT text FROM _rulebook").fetchall() if r[0]]
        combined = fold("\n".join(texts))
        if nv2 and re.search(r"don tinh vao no[^|]*\|[^|]*don \**ghi no", combined):
            has_ghi_no_rule = True

    # COD / marketplace orders do not count as debt (§4): use the common code the data layer normalised (odd labels "Thu hộ", "Ship thu
    # tiền" are mapped to 'cod'); with no common code, scan the original text
    rb = fold("\n".join(r[0] for r in con.execute("SELECT text FROM _rulebook").fetchall() if r[0])) \
        if "_rulebook" in all_tables else ""
    san_no = not rb or bool(re.search(r"qua san|san tmdt|san thuong mai", rb))   # the rulebook says marketplace orders are not debt
    conds = [f"regexp_matches(strip_accents(lower(COALESCE(CAST(\"{c}\" AS VARCHAR), ''))), '{rx}')"
             for c, rx in _NOT_DEBT.items() if c in oc and not (c == "payment_method" and "payment_norm" in oc)
             and (c != "channel" or san_no)]
    if "payment_norm" in oc:
        conds.append("COALESCE(payment_norm = 'cod', FALSE)")
    if "channel_norm" in oc and san_no:
        conds.append("COALESCE(channel_norm IN ('shopee', 'lazada', 'tiktok'), FALSE)")
    ex = " OR ".join(conds) or "FALSE"

    extra_filter = ""
    if has_ghi_no_rule and "payment_norm" in oc:
        extra_filter = " AND payment_norm = 'ghi_no'"

    cid = "CAST(customer_id AS VARCHAR)" if "customer_id" in oc else "customer_key" if "customer_key" in oc else "NULL"
    cname = "customer_name" if "customer_name" in oc else "NULL"
    walk = "is_walk_in" if "is_walk_in" in oc else "FALSE"
    st = "status_norm" if "status_norm" in oc else "NULL"

    order_id_col = "order_id"         # join USING (order_id): the line-money table has only order_id, amt, ldisc, lship

    # extra_filter applies in the outer WHERE (after the subquery), where the columns are aliased:
    # In the subquery: ANY_VALUE(payment_norm) if there is a debt-recording rule
    pay_norm_select = ""
    if has_ghi_no_rule and "payment_norm" in oc:
        pay_norm_select = ", ANY_VALUE(payment_norm) AS payment_norm"
        extra_filter = " AND payment_norm = 'ghi_no'"
    # the payment ledger records only on-credit orders (almost no pay-now order has a payment row): computing by the payment ledger turns a pay-now order into debt
    # for the whole order -> count only on-credit orders (rulebook §4: extra debt / debt payments are only for on-credit orders)
    if "payment_norm" in oc and not extra_filter:
        r = con.execute(f"""SELECT COUNT(DISTINCT o.order_id) FILTER (WHERE COALESCE(o.payment_norm, '') <> 'ghi_no'),
                COUNT(DISTINCT o.order_id) FILTER (WHERE COALESCE(o.payment_norm, '') <> 'ghi_no' AND p.k IS NOT NULL),
                COUNT(DISTINCT o.order_id) FILTER (WHERE o.payment_norm = 'ghi_no')
                FROM "{O}" o LEFT JOIN (SELECT DISTINCT CAST(order_id AS VARCHAR) k FROM "{P}") p
                ON CAST(o.order_id AS VARCHAR) = p.k""").fetchone()
        if r[2] and r[0] >= 20 and r[1] < 0.05 * r[0]:
            pay_norm_select = ", ANY_VALUE(payment_norm) AS payment_norm"
            extra_filter = " AND payment_norm = 'ghi_no'"

    od = (f'od AS (SELECT * FROM (SELECT {order_id_col} AS order_id, {tot_expr} AS tot, {d_col} AS d, '
          f'MIN({st}) st, BOOL_OR({ex}) ex, ANY_VALUE({cid}) cid, ANY_VALUE({cname}) cname, '
          f'COALESCE(BOOL_OR({walk}), FALSE) walk{pay_norm_select} FROM {from_orders} GROUP BY {order_id_col}) '
          f"WHERE COALESCE(st, '') <> 'huy' AND NOT COALESCE(ex, FALSE){extra_filter})")
    pm = f'pm AS (SELECT order_id, CAST(payment_date AS DATE) pd, payment_amount amt FROM "{P}"'
    if upfront:
        dcol = "order_date" if "order_date" in oc else "order_time"
        pm += (f' UNION ALL SELECT CAST(order_id AS VARCHAR), CAST(MIN({dcol}) AS DATE), '
               f'ANY_VALUE(TRY_CAST(paid AS DOUBLE)) FROM "{upfront}" WHERE order_id IS NOT NULL GROUP BY order_id')
    pm += ")"
    return od, pm


def debt_sql(con) -> str | None:
    """One row per order counted into debt: order_id, d, owe (still owed at the end of day X), cid, cname, walk.
    Debt = order total minus payments up to the end of day X. Parameters: X (payments), X (order date)."""
    b = _debt_base(con)
    if b is None:
        return None
    od, pm = b
    return (f"WITH {od}, {pm}, pay AS (SELECT order_id, SUM(amt) p FROM pm WHERE pd <= ? GROUP BY order_id) "
            f"SELECT order_id, d, tot - COALESCE(p, 0) AS owe, cid, cname, walk "
            f"FROM od LEFT JOIN pay USING (order_id) WHERE d <= ?")


def flow_sql(con) -> str | None:
    """Extra debt / debt payments in a period (§4), one row per order: order_id, cid, cname, add, paid_later.
    add: the part not paid at once on the purchase day, of orders placed in the period. paid_later: payments AFTER the purchase day with
    a payment date in the period (every on-credit order, including orders of earlier periods, returned orders). Parameters: lo, hi, lo, hi."""
    b = _debt_base(con)
    if b is None:
        return None
    od, pm = b
    return (f"WITH {od}, {pm}, same AS (SELECT order_id, SUM(amt) p FROM pm JOIN od USING (order_id) "
            f"WHERE pd <= d GROUP BY order_id), "
            f"later AS (SELECT order_id, SUM(amt) p FROM pm JOIN od USING (order_id) "
            f"WHERE pd > d AND pd BETWEEN ? AND ? GROUP BY order_id) "
            f"SELECT order_id, cid, cname, walk, "
            f"CASE WHEN d BETWEEN ? AND ? THEN GREATEST(tot - COALESCE(same.p, 0), 0) ELSE 0 END AS add, "
            f"COALESCE(later.p, 0) AS paid_later "
            f"FROM od LEFT JOIN same USING (order_id) LEFT JOIN later USING (order_id)")


def find_product(con, text: str):
    """The product whose name (key) is the longest one contained in the question -> (code, name key, name) or None."""
    q = f" {text_key(text)} "
    best, score = None, None
    for role in ("products", "stock", "batches", "purchase_lines", "sales_lines"):
        for t in _tables(con, role):
            c = _cols(con, t)
            if "product_name" not in c:
                continue
            idc = "CAST(product_id AS VARCHAR)" if "product_id" in c else "NULL"
            for pid, nm in con.execute(f'SELECT DISTINCT {idc}, product_name FROM "{t}" '
                                       f'WHERE product_name IS NOT NULL').fetchall():
                k = text_key(nm)
                if k and f" {k} " in q and (score is None or (len(k), pid is not None) > score):
                    best, score = (pid, k, nm), (len(k), pid is not None)
    return best


def stock_at(con, pid, pkey, x: dt.date):
    """Stock (base unit) at the end of day x. Sources in order: the stock table (a snapshot of the last data day, used only
    when x >= that day), the stock card (stock after the last entry <= x), batches (purchased per batch - sold per batch + returned by customers)."""
    data_end = con.execute("SELECT CAST(data_end AS DATE) FROM _meta_shop").fetchone()[0]

    def where(c, alias=""):
        if pid is not None and "product_id" in c:
            return f"CAST({alias}product_id AS VARCHAR) = ?", pid
        return f"{alias}product_key = ?", pkey
    for t in _tables(con, "stock"):
        c = _cols(con, t)
        if "stock_qty" in c and data_end and x >= data_end:
            w, v = where(c)
            r = con.execute(f'SELECT SUM(stock_qty) FROM "{t}" WHERE {w}', [v]).fetchone()[0]
            if r is not None:
                return r, f"bảng tồn {t}"
    for t in _tables(con, "stock_moves"):
        c = _cols(con, t)
        tcol = next((k for k in ("order_time", "order_date", "created_date") if k in c), None)
        if "stock_after" in c and tcol:
            w, v = where(c)
            r = con.execute(f'SELECT stock_after FROM "{t}" WHERE {w} AND CAST("{tcol}" AS DATE) <= ? '
                            f'ORDER BY "{tcol}" DESC, _src_row DESC LIMIT 1', [v, x]).fetchone()
            if r:
                return r[0], f"thẻ kho {t}"
    B = next((b for b in _tables(con, "batches") if {"batch_id", "qty_base"} <= _cols(con, b)), None)
    SB = next((s for s in _tables(con, "sales_batches") if {"batch_id", "qty_base"} <= _cols(con, s)), None)
    if B and SB:
        bc, sc = _cols(con, B), _cols(con, SB)
        w, v = where(bc)
        ret = "COALESCE(s.qty_returned_base, 0)" if "qty_returned_base" in sc else "0"
        r = con.execute(f'WITH b AS (SELECT batch_id, SUM(qty_base) q FROM "{B}" WHERE {w} GROUP BY batch_id) '
                        f'SELECT (SELECT SUM(q) FROM b) - COALESCE(SUM(s.qty_base), 0) + COALESCE(SUM({ret}), 0) '
                        f'FROM "{SB}" s JOIN b USING (batch_id)', [v]).fetchone()[0]
        if r is not None:
            return r, f"lô {B} - {SB}"
    # stock by purchase / sale / return rows: turned on with DATAMATE_TON_DONG=1 (run.sh turns it on by default)
    r = _flow_stock(con, pid, pkey, x) if os.environ.get("DATAMATE_TON_DONG") else None
    if r is not None:
        return r, "nhập - xuất + khách trả"
    return None, "không có nguồn tồn"


def _flow_stock(con, pid, pkey, x: dt.date):
    """No stock table / stock card / batches: stock = purchases (excluding cancelled slips) − shipments by non-cancelled orders + customer returns (by return date),
    base unit, from the start of the data to the end of day x. Needs both purchase rows and sales rows with qty_base."""
    def key(c):
        if pid is not None and "product_id" in c:
            return "CAST(product_id AS VARCHAR) = ?", pid
        if "product_key" in c:
            return "product_key = ?", pkey
        return None, None
    PL = next((p for p in _tables(con, "purchase_lines") if {"qty_base"} <= _cols(con, p)), None)
    SL = next((s for s in _tables(con, "sales_lines") if {"qty_base"} <= _cols(con, s)), None)
    if not (PL and SL):
        return None
    pc, sc = _cols(con, PL), _cols(con, SL)
    wp, vp = key(pc)
    ws, vs = key(sc)
    if wp is None or ws is None:
        return None
    # purchases: the date on the row, or the slip's date
    if "purchase_date" in pc:
        pdate, pjoin = "CAST(purchase_date AS DATE)", ""
    else:
        P = next((p for p in _tables(con, "purchases") if {"purchase_id", "purchase_date"} <= _cols(con, p)), None)
        if not (P and "purchase_id" in pc):
            return None
        pdate = "CAST(pp.d AS DATE)"
        pjoin = f'LEFT JOIN (SELECT purchase_id, MIN(purchase_date) d FROM "{P}" GROUP BY 1) pp USING (purchase_id)'
    pcanc = "AND NOT COALESCE(purchase_cancelled, FALSE)" if "purchase_cancelled" in pc else ""
    got_in = con.execute(f'SELECT SUM(qty_base) FROM "{PL}" {pjoin} WHERE {wp} AND {pdate} <= ? {pcanc}',
                         [vp, x]).fetchone()[0]
    # shipments: the date / status on the row, or the order's if absent
    dcol = next((c for c in ("order_date", "order_time") if c in sc), None)
    if dcol and "status_norm" in sc:
        sdate, sst, sjoin = f'CAST("{dcol}" AS DATE)', "status_norm", ""
    else:
        O = next((o for o in _tables(con, "orders") if "order_id" in _cols(con, o)
                  and _cols(con, o) & {"order_date", "order_time"}), None)
        if not (O and "order_id" in sc):
            return None
        oc = _cols(con, O)
        od = next(c for c in ("order_date", "order_time") if c in oc)
        ost = "ANY_VALUE(status_norm)" if "status_norm" in oc else "NULL"
        sjoin = f'JOIN (SELECT order_id, MIN("{od}") d, {ost} st FROM "{O}" GROUP BY 1) oo USING (order_id)'
        sdate, sst = "CAST(oo.d AS DATE)", "oo.st"
    got_out = con.execute(f'SELECT SUM(qty_base) FROM "{SL}" {sjoin} WHERE {ws} AND {sdate} <= ? '
                          f"AND COALESCE({sst}, '') <> 'huy'", [vs, x]).fetchone()[0]
    back = 0
    for R in _tables(con, "returns"):
        rc = _cols(con, R)
        wr, vr = key(rc)
        q = "qty_returned_base" if "qty_returned_base" in rc else None
        if wr and q and "return_date" in rc:
            back += con.execute(f'SELECT COALESCE(SUM({q}), 0) FROM "{R}" WHERE {wr} AND CAST(return_date AS DATE) <= ?',
                                [vr, x]).fetchone()[0] or 0
    if got_in is None and got_out is None:
        return None
    return (got_in or 0) - (got_out or 0) + back


def lots_sql(con) -> str | None:
    """One row per batch: pid, batch_id, exp (expiry), remain (stock left, base unit, at the end of day ?), cost_base
    (the batch's own purchase price per base unit). Stock left = purchased - sold from the batch + returned by customers to the batch (§6: returned goods go
    back to the batch they were sold from); batches purchased / orders sold after day ? count as not yet happened. Parameters: day X (purchases), X (sales)."""
    B = next((b for b in _tables(con, "batches") if {"batch_id", "qty_base", "expiry_date"} <= _cols(con, b)), None)
    SB = next((s for s in _tables(con, "sales_batches") if {"batch_id", "qty_base"} <= _cols(con, s)), None)
    if not (B and SB):
        return None
    bc, sc = _cols(con, B), _cols(con, SB)
    # the batch's purchase date: a column on the batch table, or the purchase slip's date
    P = next((p for p in _tables(con, "purchases") if {"purchase_id", "purchase_date"} <= _cols(con, p)), None)
    if "purchase_date" in bc:
        bdate, bjoin = "CAST(b.purchase_date AS DATE)", ""
    elif P and "purchase_id" in bc:
        bdate = "CAST(p.purchase_date AS DATE)"
        bjoin = f'LEFT JOIN (SELECT purchase_id, MIN(purchase_date) purchase_date FROM "{P}" GROUP BY 1) p USING (purchase_id)'
    else:
        bdate, bjoin = "NULL", ""
    # sales date of a batch row: the order's order date
    O = next((o for role in ("orders", "sales_lines") for o in _tables(con, role)
              if "order_id" in _cols(con, o) and _cols(con, o) & {"order_date", "order_time"}), None)
    if O and "order_id" in sc:
        oc = _cols(con, O)
        od = "order_date" if "order_date" in oc else "order_time"
        sdate = "CAST(o.d AS DATE)"
        sjoin = f'LEFT JOIN (SELECT order_id, MIN("{od}") d FROM "{O}" GROUP BY 1) o USING (order_id)'
    else:
        sdate, sjoin = "NULL", ""
    ret = "COALESCE(s.qty_returned_base, 0)" if "qty_returned_base" in sc else "0"
    # the batch's purchase price: line amounts of the batch's purchase rows / purchased quantity converted to base; if absent, the unit price written on the batch
    PL = next((p for p in _tables(con, "purchase_lines") if "batch_id" in _cols(con, p)
               and (_cols(con, p) & {"line_amount"} or {"qty", "unit_price"} <= _cols(con, p))), None)
    if PL:
        amt = "line_amount" if "line_amount" in _cols(con, PL) else "qty * unit_price"
        cost_cte = f', pc AS (SELECT batch_id, SUM({amt}) amt FROM "{PL}" GROUP BY 1)'
        cost, cjoin = "pc.amt / NULLIF(b.q, 0)", "LEFT JOIN pc USING (batch_id)"
    elif "unit_price" in bc:
        cost_cte, cost, cjoin = "", "b.up", ""
    else:
        cost_cte, cost, cjoin = "", "NULL", ""
    up = "ANY_VALUE(b.unit_price)" if "unit_price" in bc else "NULL"
    return (f'WITH b AS (SELECT CAST(b.product_id AS VARCHAR) pid, b.batch_id, MIN(CAST(b.expiry_date AS DATE)) exp, '
            f'SUM(b.qty_base) q, {up} up FROM "{B}" b {bjoin} WHERE COALESCE({bdate} <= ?, TRUE) GROUP BY 2, 1), '
            f's AS (SELECT s.batch_id, SUM(s.qty_base) sold, SUM({ret}) ret FROM "{SB}" s {sjoin} '
            f'WHERE COALESCE({sdate} <= ?, TRUE) GROUP BY 1){cost_cte} '
            f'SELECT b.pid, b.batch_id, b.exp, b.q - COALESCE(s.sold, 0) + COALESCE(s.ret, 0) AS remain, '
            f'{cost} AS cost_base FROM b LEFT JOIN s USING (batch_id) {cjoin}')


def product_names(con) -> dict:
    """product code -> name in the catalogue."""
    out = {}
    for t in _tables(con, "products") + _tables(con, "stock"):
        if {"product_id", "product_name"} <= _cols(con, t):
            for pid, nm in con.execute(f'SELECT DISTINCT CAST(product_id AS VARCHAR), product_name FROM "{t}"').fetchall():
                out.setdefault(pid, nm)
    return out


def _money(v):
    """Money rounded to the dong, half a dong rounds up (§9)."""
    return None if v is None else int(math.floor(v + 0.5))


def rx_ids(con) -> set | None:
    """The codes of products marked prescription in the drug catalogue (§6); None if the shop has no drug catalogue."""
    for t in _tables(con, "drug_info") + _tables(con, "products"):
        c = _cols(con, t)
        if {"product_id", "is_prescription"} <= c:
            ids = set()
            for pid, v in con.execute(f'SELECT CAST(product_id AS VARCHAR), is_prescription FROM "{t}"').fetchall():
                if v is True or fold(str(v)) in ("co", "x", "yes", "true", "1", "ke don", "thuoc ke don"):
                    ids.add(pid)
            return ids
    return None


def purchase_lines_sql(con) -> str | None:
    """Purchase rows: pid, pkey, qty, amt, d (slip date), cancelled (cancelled slip)."""
    for L in _tables(con, "purchase_lines"):
        lc = _cols(con, L)
        if "qty" not in lc or not (lc & {"line_amount"} or "unit_price" in lc):
            continue
        amt = "l.line_amount" if "line_amount" in lc else "l.qty * l.unit_price"
        P = next((p for p in _tables(con, "purchases") if {"purchase_id", "purchase_date"} <= _cols(con, p)), None)
        if "purchase_date" in lc:
            d, j = "l.purchase_date", ""
            cx = "l.purchase_cancelled" if "purchase_cancelled" in lc else "FALSE"
        elif P and "purchase_id" in lc:
            pc = _cols(con, P)
            d, cx = "p.purchase_date", "p.cx"
            j = (f'LEFT JOIN (SELECT purchase_id, MIN(purchase_date) purchase_date, '
                 f'{"BOOL_OR(purchase_cancelled)" if "purchase_cancelled" in pc else "FALSE"} cx '
                 f'FROM "{P}" GROUP BY 1) p USING (purchase_id)')
        else:
            continue
        pid = "CAST(l.product_id AS VARCHAR)" if "product_id" in lc else "NULL"
        pk = "CAST(l.product_key AS VARCHAR)" if "product_key" in lc else "NULL"
        return (f'SELECT {pid} pid, {pk} pkey, l.qty, {amt} amt, CAST({d} AS DATE) d, '
                f'COALESCE({cx}, FALSE) cancelled FROM "{L}" l {j}')
    return None


def group_maps(con, customer: bool) -> dict:
    """Group value -> set of keys (code or name key) in the group. Product group: the category column of the catalogue / sales rows;
    customer group: customer_group of the customer list / orders."""
    col, role_ids = ("customer_group", ("customer_id", "customer_key")) if customer else \
        ("category", ("product_id", "product_key"))
    out: dict[str, set] = {}
    roles = ("customers", "orders", "sales_lines") if customer else ("products", "sales_lines", "stock")
    for role in roles:
        for t in _tables(con, role):
            c = _cols(con, t)
            if col not in c:
                continue
            for idc in role_ids:
                if idc in c:
                    for v, k in con.execute(f'SELECT DISTINCT "{col}", CAST("{idc}" AS VARCHAR) FROM "{t}" '
                                            f'WHERE "{col}" IS NOT NULL AND "{idc}" IS NOT NULL').fetchall():
                        out.setdefault(str(v), set()).add(k)
    return out


def pick_group(groups: dict, text: str):
    """The group whose name (key) is the longest one contained in the question, exact name (§3: do not merge groups with different names)."""
    q = f" {text_key(text)} "
    hits = [g for g in groups if text_key(g) and f" {text_key(g)} " in q]
    return max(hits, key=lambda g: len(text_key(g))) if hits else None


def base_unit_of(con) -> dict:
    """product code / name key -> base unit (catalogue, stock, sales rows)."""
    out = {}
    for role in ("products", "stock", "sales_lines"):
        for t in _tables(con, role):
            c = _cols(con, t)
            if "base_unit" not in c:
                continue
            for idc in ("product_id", "product_key"):
                if idc in c:
                    for k, u in con.execute(f'SELECT DISTINCT CAST("{idc}" AS VARCHAR), base_unit FROM "{t}" '
                                            f'WHERE base_unit IS NOT NULL').fetchall():
                        out.setdefault(k, u)
    return out


_PAY_WORDS = [(r"\b(tien mat|tm)\b", "tien mat"), (r"\b(chuyen khoan|ck)\b", "chuyen khoan"), (r"\bcod\b", "cod"),
              (r"\bvi( dien tu)?\b|\bmomo\b|\bzalopay\b", "vi"), (r"\b(the|quet the)\b", "the"),
              (r"\bghi no\b", "ghi no")]
_PAY_Q = [("tien_mat", r"\btien mat\b|\btm\b"), ("chuyen_khoan", r"\bchuyen khoan\b|\bck\b"),
          ("vi", r"\bvi( dien tu)?\b|\bmomo\b|\bzalo ?pay\b"), ("cod", r"\bcod\b|\bthu ho\b"),
          ("the", r"\b(quet )?the\b"), ("ghi_no", r"\bghi no\b|\bmua chiu\b")]
_CHANNEL_Q = [("zalo", r"\bzalo\b"), ("dien_thoai", r"\bdien thoai\b|\bhotline\b"),
              ("tai_quay", r"\btai (quay|cua hang|quan)\b|\btruc tiep\b"), ("facebook", r"\bfacebook\b|\bfb\b"),
              ("shopee", r"\bshopee\b"), ("lazada", r"\blazada\b"), ("tiktok", r"\btiktok\b"),
              ("grab", r"\bgrab"), ("website", r"\bwebsite\b|\bonline\b")]
_SHIFT = {"sang": (6, 13), "chieu": (14, 21)}      # §2: by order creation time, 06:00–13:59 / 14:00–21:59
_PLAIN_WORDS = set("""cho hoi toi minh em biet xem giup gium check coi tinh tong cong doanh thu dthu cua hang
    trong ngay la bao nhieu bnhiu bn nhiu may ban duoc dc ra tien so don co bi huy da o trang thai phan tram %
    ti ty le the vay a nhi ha z nhe oi e ee ê do ta rieng total revenue nao hok k khong thu ve mang ve dat
    duoc tat ca het ky""".split())


def g5_filters(con, s: str, text: str) -> list[tuple[int, str, str]]:
    """Conditions found in the question -> [(position in the question, label, SQL condition)], in order of appearance."""
    f = fold(text)
    out = []
    # channel / payment method: words in the question -> common code, compared with the DB's channel_norm / payment_norm
    for col, words in (("ch", _CHANNEL_Q), ("pm", _PAY_Q)):
        have = {v for v, in con.execute(f"SELECT DISTINCT {col} FROM ({s}) WHERE {col} IS NOT NULL").fetchall()}
        for code, rx in words:
            if code in have and (m := re.search(rx, f)):
                out.append((m.start(), code, f"{col} = '{code}'"))
    for name, (h1, h2) in _SHIFT.items():
        if (m := re.search(rf"\bca {name}\b", f)):
            out.append((m.start(), f"ca {name}", f"hour(t) BETWEEN {h1} AND {h2}"))
    if (m := re.search(r"\bnhan vien (?:ma )?([a-z0-9 ]+?)(?= thang| tuan| quy| ngay| nam| ban| co| doanh| dthu| trong"
                       r"| hom| tu| den|$)", f)):
        who = m[1].strip()
        # an order may record only the staff code: a name in the question -> code via the staff table
        ids = {who}
        for t in _tables(con, "staff"):
            c = _cols(con, t)
            if {"staff_id", "staff_name"} <= c:
                ids |= {str(i).lower() for i, nm in con.execute(
                    f'SELECT staff_id, staff_name FROM "{t}"').fetchall() if text_key(nm) == text_key(who)}
        lst = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
        out.append((m.start(), f"nv {who}", f"(lower(staff) IN ({lst}) OR text_key_sql IN ({lst}))"
                    .replace("text_key_sql", "lower(strip_accents(staff))")))
    if not out and (p := find_product(con, text)):
        pid, key = p[0], p[1]
        # "phí ship", "voucher"... equal names of rows that are not products: not a question about the revenue of one product
        if line_type_of(p[2]) != "hang":
            return [(0, "dòng không phải mặt hàng", "FALSE")]
        cond = f"(pid = '{pid}' OR pkey = '{key}')" if pid else f"pkey = '{key}'"
        out.append((f.find(fold(p[2])) if fold(p[2]) in f else 0, p[2], cond))
    return sorted(out)


def _prev_period(f: str, a: dt.date, b: dt.date):
    """Comparison period (§1, §2): the same period last year; this month -> the same number of days of the previous month; a whole month -> the whole previous month;
    another period -> the same length right before."""
    if re.search(r"\bcung (ky|thang|quy) nam\b|\bnam (truoc|ngoai|roi|qua)\b", f):
        return a.replace(year=a.year - 1), b.replace(year=b.year - 1)
    if a.day == 1 and (b + dt.timedelta(days=1)).day == 1:          # a whole month
        pe = a - dt.timedelta(days=1)
        return pe.replace(day=1), pe
    if a.day == 1:                                                   # this month up to today
        pe = a - dt.timedelta(days=1)
        return pe.replace(day=1), min(pe, pe.replace(day=min(b.day, pe.day)))
    n = (b - a).days + 1
    return a - dt.timedelta(days=n), a - dt.timedelta(days=1)


def answer(con, kind, lo, hi):
    if kind == "cust_min_orders":     # §1/§3: a customer who bought = has >= 1 valid order in the period; a walk-in customer is not a customer
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        ((a, b), n), _ = lo, hi
        r = con.execute(f"SELECT COUNT(*), COUNT(*) FILTER (k >= ?) FROM (SELECT COALESCE(cid, ckey) c, "
                        f"COUNT(DISTINCT order_id) k FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER} "
                        f"AND NOT COALESCE(walk, FALSE) AND COALESCE(cid, ckey) IS NOT NULL GROUP BY 1)",
                        [n, a, b]).fetchone()
        return (round(100 * r[1] / r[0], 2) if r[0] else None), f"{r[1]}/{r[0]} khách"
    if kind == "best_staff":            # §3: by the staff member's §1 revenue in the period, not by order count
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        (a, b), _ = lo, hi
        r = con.execute(f"SELECT staff, SUM(amt) v FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER} "
                        f"AND staff IS NOT NULL GROUP BY staff ORDER BY v DESC, staff LIMIT 1", [a, b]).fetchone()
        if not r:
            return None, "đơn không ghi nhân viên"
        names = [r[0]]
        for t in _tables(con, "staff"):
            if {"staff_id", "staff_name"} <= _cols(con, t):
                names += [n for n, in con.execute(f'SELECT staff_name FROM "{t}" WHERE CAST(staff_id AS VARCHAR) = ?',
                                                   [r[0]]).fetchall()]
        return {"names": names, "id": r[0], "a": r[1]}, ""
    if kind in ("rev_filter", "growth"):
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        (a, b), text = lo, hi
        f = fold(text)
        orders = bool(re.search(r"\b(so don|(bao nhieu|bao nhiu|bnhiu|bn|may|nhieu) don|don hang)\b", f)) and not re.search(
            r"\b(doanh thu|dthu|tien)\b", f)
        meas = "COUNT(DISTINCT order_id)" if orders else "SUM(amt)"

        def val(x, y, cond="TRUE"):
            return con.execute(f"SELECT {meas} FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER} AND {cond}",
                               [x, y]).fetchone()[0] or 0
        if isinstance(a, tuple):                    # Tet of year Y vs the same number of days right before Tet (§2)
            r = con.execute("SELECT CAST(start_date AS DATE), CAST(end_date AS DATE) FROM _tet_periods "
                            "WHERE tet_year = ?", [a[1]]).fetchone()
            if not r:
                return None, "không có kỳ Tết"
            a, b = r[0], min(r[1], b)
        filters = g5_filters(con, s, text)
        if kind == "growth":
            if len(filters) >= 2 and all(x[1].startswith("ca ") for x in filters[:2]):
                base_c = next(x for x in filters if x[1] == "ca sang")[2]
                other = next(x for x in filters if x[1] != "ca sang")[2]
                cur, prev = val(a, b, other), val(a, b, base_c)
            else:
                cond = filters[0][2] if filters else "TRUE"
                if "truoc tet" in f:
                    n = (b - a).days + 1
                    pa, pb = a - dt.timedelta(days=n), a - dt.timedelta(days=1)
                elif (sv := re.search(r"\bso (?:voi|vs) (.*)", f)) and (p2 := _period(sv[1], b + dt.timedelta(days=366))) \
                        and not isinstance(p2[0], tuple) and (p2[0], p2[1]) != (a, b) \
                        and re.search(r"\d{4}", p2[2][0]):     # only a period with an explicit year; "cùng kỳ tháng trước" per §2
                    pa, pb = p2[0], p2[1]            # an explicit comparison period: "tháng 8/2026 so với tháng 8/2025"
                else:
                    pa, pb = _prev_period(f, a, b)
                cur, prev = val(a, b, cond), val(pa, pb, cond)
            return (round(100 * (cur - prev) / prev, 2) if prev else None), f"{prev} -> {cur}"
        if any(c == "FALSE" for _, _, c in filters):
            return None, "không phải câu doanh thu mặt hàng"
        if not filters:
            # a plain question (all remaining words are words asking for revenue / order count / cancellations) -> computed without conditions;
            # a question with a condition the script does not understand is dropped, not guessed
            pm_ = _period(f, b)
            rest = set((f[:pm_[2].start()] + " " + f[pm_[2].end():]).split()) if pm_ else set(f.split())
            if not rest <= _PLAIN_WORDS:
                return None, "không thấy điều kiện"
            if re.search(r"\bhuy\b", f):
                n, h = con.execute(f"SELECT COUNT(DISTINCT order_id), COUNT(DISTINCT order_id) FILTER (st = 'huy') "
                                   f"FROM ({s}) WHERE d BETWEEN ? AND ?", [a, b]).fetchone()
                if re.search(r"phan tram|%|\b(ti|ty) le\b", f):
                    return (round(100 * h / n, 2) if n else None), "tỉ lệ huỷ"
                return h, "số đơn huỷ"
            return (val(a, b) if orders else _money(val(a, b))), "không điều kiện"
        share = re.search(r"phan tram|%|\bchiem\b|\b(ti|ty) trong\b", f)
        # two values of the same kind (two channels, two payment methods): they cannot hold at once -> ask each value, giving a list
        same_col = len({c.split(" ")[0] for _, _, c in filters}) < len(filters)
        if len(filters) >= 2 and (same_col or re.search(r"\b(lan luot|moi loai|moi ben)\b", f)) and not share:
            return [(_money(val(a, b, c)) if not orders else val(a, b, c)) for _, _, c in filters], \
                ", ".join(x[1] for x in filters)
        cond = " AND ".join(c for _, _, c in filters)
        v = val(a, b, cond)
        if share:
            tot = val(a, b)
            return (round(100 * v / tot, 2) if tot else None), ", ".join(x[1] for x in filters)
        return (v if orders else _money(v)), ", ".join(x[1] for x in filters)
    if kind in ("avg_order_value", "best_day", "best_month", "group_revenue", "group_share", "qty_sold",
                "price_on_day", "price_effect", "group_avg_order", "group_top_product"):
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        par, text = lo, hi
        if kind == "best_month":
            rows = [(f"{y:04d}-{m:02d}", con.execute(
                f"SELECT SUM(amt) FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER}",
                [dt.date(y, m, 1), _month_end(y, m)]).fetchone()[0] or 0) for y, m in par]
            return max(rows, key=lambda r: r[1])[0], ""
        if kind == "price_effect":      # §5: N days from D (including D) vs the N days right before D, excluding cancelled / returned orders
            d, n = par
            p = find_product(con, text)
            if p is None:
                return None, "không thấy tên hàng"
            q = (f"SELECT SUM(qb) FROM ({s}) WHERE (pid = ? OR pkey = ?) AND d BETWEEN ? AND ? AND {OK_ORDER}")
            after = con.execute(q, [p[0], p[1], d, d + dt.timedelta(days=n - 1)]).fetchone()[0] or 0
            before = con.execute(q, [p[0], p[1], d - dt.timedelta(days=n), d - dt.timedelta(days=1)]).fetchone()[0] or 0
            return (round(100 * (after - before) / before, 2) if before else None), f"{before} -> {after}"
        a, b = par
        base = f"FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER}"
        if kind == "avg_order_value":
            r = con.execute(f"SELECT SUM(amt), COUNT(DISTINCT order_id) {base}", [a, b]).fetchone()
            return (_money(r[0] / r[1]) if r[1] else None), ""
        if kind == "best_day":           # on a tie the earlier date
            r = con.execute(f"SELECT d, SUM(amt) v {base} GROUP BY d ORDER BY v DESC, d LIMIT 1", [a, b]).fetchone()
            return (r[0].isoformat() if r else None), ""
        if kind in ("group_revenue", "group_share", "group_avg_order", "group_top_product"):
            customer = bool(re.search(r"\bkhach\b[^.]*\bnhom\b|\bnhom khach\b", fold(text)))
            if kind == "group_top_product" and customer:
                return None, "mặt hàng đầu của nhóm khách: chưa hỗ trợ"
            groups = group_maps(con, customer)
            g = pick_group(groups, text)
            if g is None:
                return None, f"không thấy tên nhóm {'khách' if customer else 'hàng'}"
            keys = groups[g]
            rows = con.execute(f"SELECT pid, pkey, cid, ckey, walk, lt, order_id, pname, SUM(amt), SUM(qb) {base} "
                               f"GROUP BY ALL", [a, b]).fetchall()
            tot = sum(r[8] or 0 for r in rows)
            if customer:
                mine = [r for r in rows if not r[4] and (r[2] in keys or r[3] in keys)]
            else:
                mine = [r for r in rows if r[5] == "hang" and (r[0] in keys or r[1] in keys)]
            grp = sum(r[8] or 0 for r in mine)
            if kind == "group_share":
                return (round(100 * grp / tot, 2) if tot else None), g
            if kind == "group_avg_order":    # group revenue / group order count
                n = len({r[6] for r in mine if r[6] is not None})
                return (_money(grp / n) if n else None), g
            if kind == "group_top_product":  # by money if the question asks money, otherwise by base quantity (§3 best-selling)
                by_money = bool(re.search(r"\b(tien|doanh thu|dthu)\b", fold(text)))
                agg: dict = {}
                for r in mine:
                    k = r[0] or r[1]
                    v = agg.setdefault(k, [0, r[7], r[0]])
                    v[0] += (r[8] if by_money else r[9]) or 0
                if not agg:
                    return None, ""
                best = sorted(agg.items(), key=lambda kv: (-kv[1][0], str(kv[0])))[0][1]
                names = [best[1]] if best[1] else []
                return {"names": names + (names_by_id(con, "products", "product_id", "product_name", best[2])
                                          if best[2] else []), "id": best[2], "q": best[0]}, g
            return _money(grp), g
        f = fold(text)
        if kind == "price_on_day":       # §5: per 1 base unit, before line discount, including cancelled / returned orders
            d = re.search(rf"\bngay {_D}", f)
            day = dt.date(int(d[3]), int(d[2]), int(d[1]))
            p = find_product(con, text)
            if p is None:
                return None, "không thấy tên hàng"
            # unit price converted to 1 base unit: unit price x qty / base qty (a large-unit sales row); the date comes via the order table
            vals = sorted({round(v, 2) for v, in con.execute(
                f"SELECT up * COALESCE(qty / NULLIF(qb, 0), 1) FROM ({s}) WHERE (pid = ? OR pkey = ?) AND d = ?",
                [p[0], p[1], day]).fetchall() if v is not None})
            if not vals:
                return None, "không có dòng bán / đơn giá ngày đó"
            return (vals[0] if len(vals) == 1 else None), (f"nhiều mức giá {vals}" if len(vals) > 1 else "")
        # qty_sold: one product; size M/L; or every product whose base unit is <unit> (§3)
        p = find_product(con, text)
        if p is not None:
            r = con.execute(f"SELECT SUM(qb) {base} AND (pid = ? OR pkey = ?)", [a, b, p[0], p[1]]).fetchone()[0]
            return (_money(r) if r is not None else 0), p[2]
        if sz := re.search(r"\bsize ([ml])\b", f):     # §6 coffee: the name ends with (M) / (L), case-insensitive
            names = product_names(con)
            rows = con.execute(f"SELECT pid, pname, SUM(qb) {base} AND lt = 'hang' GROUP BY ALL", [a, b]).fetchall()
            tot = sum(q or 0 for pid, nm, q in rows
                      if str(nm or names.get(pid, "")).strip().lower().endswith(f"({sz[1]})"))
            return _money(tot), f"size {sz[1]}"
        u = re.search(r"\b(?:bao nhieu|bao nhiu|bnhiu|bn|may) (\w+)", f)
        bu = base_unit_of(con)
        if not u or not bu:
            return None, "không rõ đơn vị"
        keys = {k for k, v in bu.items() if fold(v) == u[1]}
        if not keys:
            return None, f"không hàng nào có đơn vị cơ bản '{u[1]}'"
        rows = con.execute(f"SELECT pid, pkey, SUM(qb) {base} AND lt = 'hang' GROUP BY ALL", [a, b]).fetchall()
        return _money(sum(r[2] or 0 for r in rows if r[0] in keys or r[1] in keys)), f"đơn vị {u[1]}"
    if kind == "avg_purchase_price":    # the average by purchased quantity, by purchase unit, excluding cancelled slips (§5)
        (a, b), text = lo, hi
        s, p = purchase_lines_sql(con), find_product(con, text)
        if s is None or p is None:
            return None, "không có dòng nhập / không thấy tên hàng"
        # a purchase row may record only the name (a notebook): match by code or by name key
        r = con.execute(f"SELECT SUM(amt) / NULLIF(SUM(qty), 0) FROM ({s}) WHERE (pid = ? OR pkey = ?) "
                        f"AND d BETWEEN ? AND ? AND NOT cancelled", [p[0], p[1], a, b]).fetchone()[0]
        return _money(r), ""
    if kind == "inventory_value":       # end-of-day-x stock of each product x the cost price per 1 base unit in the catalogue (§5)
        (x, _), _ = lo, hi
        P = next((t for t in _tables(con, "products") if {"product_id", "unit_cost"} <= _cols(con, t)), None)
        if P is None:
            return None, "danh mục không có giá vốn"
        tot, src = 0, None
        for pid, cost in con.execute(f'SELECT DISTINCT CAST(product_id AS VARCHAR), unit_cost FROM "{P}"').fetchall():
            q, src_ = stock_at(con, pid, None, x)
            src = src or (src_ if q is not None else None)
            if q and q > 0 and cost:
                tot += q * cost
        return (_money(tot) if src else None), src or "không có nguồn tồn"
    if kind in ("gross_profit", "top_profit_product", "net_profit", "rx_revenue", "non_rx_revenue", "rx_rate"):
        (a, b), _ = lo, hi
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        base = f"FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER}"
        if kind in ("rx_revenue", "non_rx_revenue", "rx_rate"):
            ids = rx_ids(con)
            if ids is None:
                return None, "không có danh mục thuốc"
            # a sales row that records only the name: join prescription code -> name in the catalogue -> name key
            names = product_names(con)
            keys = {text_key(names[i]) for i in ids if i in names}
            rows = con.execute(f"SELECT pid, pkey, SUM(amt) {base} GROUP BY pid, pkey", [a, b]).fetchall()
            tot = sum(v or 0 for _, _, v in rows)
            rx = sum(v or 0 for p, k, v in rows if p in ids or (p is None and k in keys))
            if kind == "rx_rate":
                return (round(100 * rx / tot, 2) if tot else None), ""
            return _money(rx if kind == "rx_revenue" else tot - rx), ""
        # gross profit = revenue (§1) - qty x the cost price recorded on the sales row; the cost price is not recomputed
        has_uc = con.execute(f"SELECT COUNT(uc) {base}", [a, b]).fetchone()[0]
        if not has_uc:
            return None, "dòng bán không ghi giá vốn"
        # a cost price present but quantity missing: do not treat the cost price as 0 (gross profit would become revenue) -> cannot be computed
        if con.execute(f"SELECT COUNT(*) {base} AND uc IS NOT NULL AND qty IS NULL AND lt = 'hang'",
                       [a, b]).fetchone()[0]:
            return None, "dòng bán có giá vốn nhưng thiếu số lượng"
        if kind == "top_profit_product":
            r = con.execute(f"SELECT COALESCE(pid, pkey) k, SUM(amt - COALESCE(qty * uc, 0)) g, LIST(DISTINCT pname), "
                            f"ANY_VALUE(pid) {base} AND lt = 'hang' GROUP BY k ORDER BY g DESC, k LIMIT 1",
                            [a, b]).fetchone()
            if not r:
                return None, ""
            return {"names": [n for n in r[2] if n] + (names_by_id(con, "products", "product_id", "product_name", r[3])
                                                       if r[3] else []), "id": r[3], "q": r[1]}, ""
        g = con.execute(f"SELECT SUM(amt - COALESCE(qty * uc, 0)) {base}", [a, b]).fetchone()[0] or 0
        if kind == "gross_profit":
            return _money(g), ""
        E = next((e for e in _tables(con, "expenses") if {"expense_amount", "expense_date"} <= _cols(con, e)), None)
        if E is None:
            return None, "không có sổ chi phí"
        ex = con.execute(f'SELECT SUM(expense_amount) FROM "{E}" WHERE CAST(expense_date AS DATE) BETWEEN ? AND ?',
                         [a, b]).fetchone()[0] or 0
        return _money(g - ex), ""
    if kind.startswith("lot_"):
        s = lots_sql(con)
        if s is None:
            return None, "không có dữ liệu lô / hạn dùng"
        (d, until), text = lo, hi
        if kind == "lot_nearest":        # batches with stock left and not yet expired as of today, earliest expiry
            p = find_product(con, text)
            if p is None or p[0] is None:
                return None, "không thấy mã hàng"
            r = con.execute(f"SELECT MIN(exp) FROM ({s}) WHERE pid = ? AND remain > 0 AND exp >= ?",
                            [d, d, p[0], d]).fetchone()[0]
            return (r.isoformat() if r else None), ""
        base = f"FROM ({s}) WHERE remain > 0 AND exp BETWEEN ? AND ?"
        args = [d, d, d, until]
        if kind == "lot_value":
            v = con.execute(f"SELECT SUM(remain * cost_base) {base}", args).fetchone()[0]
            return (int(round(v)) if v is not None else 0), ""
        if kind == "lot_count":
            return con.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0], ""
        if kind == "lot_products_count":
            return con.execute(f"SELECT COUNT(DISTINCT pid) {base}", args).fetchone()[0], ""
        # list by the product's earliest expiry, the same expiry by code (§9)
        names = product_names(con)
        rows = con.execute(f"SELECT pid, MIN(exp) e {base} GROUP BY pid ORDER BY e, pid", args).fetchall()
        return [names.get(p, p) for p, _ in rows], ""
    if kind == "stockout_count":
        (d, _), _ = lo, hi
        for t in _tables(con, "stock"):
            if "stock_qty" in _cols(con, t):
                return con.execute(f'SELECT COUNT(*) FROM "{t}" WHERE stock_qty = 0').fetchone()[0], f"bảng tồn {t}"
        return None, "không có bảng tồn"
    if kind in ("stock", "stock_split"):
        p = find_product(con, hi)
        if p is None:
            return None, "không thấy tên hàng trong câu"
        q, src = stock_at(con, p[0], p[1], lo)
        if q is None or kind == "stock":
            return q, src
        # large unit: the conversion unit of the product whose name is in the question, otherwise the largest unit
        conv = con.execute("SELECT unit, factor FROM _unit_conversions WHERE CAST(product_id AS VARCHAR) = ? "
                           "OR CAST(product_key AS VARCHAR) = ? ORDER BY factor DESC", [p[0], p[1]]).fetchall()
        # find the unit the question names OUTSIDE the product name ("hộp 10 vỉ x 10 viên ... mấy vỉ chẵn": vỉ, not hộp)
        fq = f" {text_key(hi)} ".replace(f" {p[1]} ", " ")
        f = next((fa for u, fa in conv if f" {text_key(u)} " in fq), conv[0][1] if conv else None)
        if not f:
            return None, f"không có quy đổi ({src})"
        return [int(q // f), int(q % f)], src
    if kind in ("debt_collected", "debt_customer_month"):
        s = flow_sql(con)
        if s is None:
            return None, "không dựng được SQL công nợ"
        (a, b), name = lo, hi
        if kind == "debt_collected":
            v = con.execute(f"SELECT SUM(paid_later) FROM ({s})", [a, b, a, b]).fetchone()[0]
            return (int(round(v)) if v is not None else 0), ""
        ids = customer_ids_by_name(con, name)
        if len(ids) != 1:
            return None, f"tên khách '{name}' khớp {len(ids)} mã"
        r = con.execute(f"SELECT SUM(add), SUM(paid_later) FROM ({s}) WHERE cid = ?", [a, b, a, b, ids[0]]).fetchone()
        return [int(round(v or 0)) for v in r], ""
    if kind == "debt_buckets":
        s = debt_sql(con)
        if s is None:
            return None, "không dựng được SQL công nợ"
        r = con.execute(f"SELECT SUM(owe) FILTER (? - d <= 7), SUM(owe) FILTER (? - d BETWEEN 8 AND 15), "
                        f"SUM(owe) FILTER (? - d > 15) FROM ({s}) WHERE owe > 0", [lo, lo, lo, lo, lo]).fetchone()
        return [int(round(v or 0)) for v in r], ""
    if kind.startswith("debt_"):
        s = debt_sql(con)
        if s is None:
            return None, "không dựng được SQL công nợ"
        x, n = lo, hi
        base = f"FROM ({s}) WHERE owe > 0"
        if kind == "debt_total":
            v = con.execute(f"SELECT SUM(owe) {base}", [x, x]).fetchone()[0]
            return (int(round(v)) if v is not None else 0), ""
        if kind == "debt_aging":        # debt age counted from the order date; "hơn N ngày" is > N
            v = con.execute(f"SELECT SUM(owe) {base} AND ? - d > ?", [x, x, x, n]).fetchone()[0]
            return (int(round(v)) if v is not None else 0), ""
        # debt by customer: a walk-in customer has no row (rulebook §4)
        people = f"{base} AND NOT walk AND cid IS NOT NULL GROUP BY cid"
        if kind == "debt_customers":
            return con.execute(f"SELECT COUNT(*) FROM (SELECT cid {people})", [x, x]).fetchone()[0], ""
        rows = con.execute(f"SELECT cid, SUM(owe) a, LIST(DISTINCT cname) {people} ORDER BY a DESC, cid LIMIT ?",
                           [x, x, n]).fetchall()
        return [{"id": c, "a": a, "names": [v for v in nm if v] + names_by_id(con, "customers", "customer_id",
                                                                              "customer_name", c)}
                for c, a, nm in rows], ""
    if kind in ("revenue", "top_product", "top_customer", "orders"):
        s = sales_sql(con)
        if s is None:
            return None, "không dựng được SQL dòng bán"
        base = f"FROM ({s}) WHERE d BETWEEN ? AND ? AND {OK_ORDER}"
        if kind == "revenue":
            v = con.execute(f"SELECT SUM(amt) {base}", [lo, hi]).fetchone()[0]
            return (int(round(v)) if v is not None else 0), ""
        if kind == "orders":
            return con.execute(f"SELECT COUNT(DISTINCT COALESCE(order_id, CAST(random() AS VARCHAR))) {base}",
                               [lo, hi]).fetchone()[0], ""
        if kind == "top_product":
            r = con.execute(f"SELECT COALESCE(pid, pkey) k, SUM(qb) q, LIST(DISTINCT pname) AS nm, ANY_VALUE(pid) "
                            f"{base} AND lt = 'hang' GROUP BY k ORDER BY q DESC, k LIMIT 1", [lo, hi]).fetchone()
            if r and r[3] is not None:
                r = (*r[:2], [n for n in r[2] if n] + names_by_id(con, "products", "product_id", "product_name", r[3]), r[3])
            return (r and {"names": r[2], "id": r[3], "q": r[1]}), ""
        r = con.execute(f"SELECT COALESCE(cid, ckey) k, SUM(amt) a, LIST(DISTINCT cname) AS nm, ANY_VALUE(cid) "
                        f"{base} AND NOT COALESCE(walk, FALSE) AND COALESCE(cid, ckey) IS NOT NULL "
                        f"GROUP BY k ORDER BY a DESC, k LIMIT 1", [lo, hi]).fetchone()
        if r and r[3] is not None:
            r = (*r[:2], [n for n in r[2] if n] + names_by_id(con, "customers", "customer_id", "customer_name", r[3]), r[3])
        return (r and {"names": r[2], "id": r[3], "a": r[1]}), ""
    if kind in ("cancel_count", "cancel_rate"):
        s = orders_sql(con)
        if s is None:
            return None, "không dựng được SQL đơn"
        n, h = con.execute(f"SELECT COUNT(*), COUNT(*) FILTER (st = 'huy') FROM ({s}) WHERE d BETWEEN ? AND ?",
                           [lo, hi]).fetchone()
        return (h if kind == "cancel_count" else (round(100 * h / n, 2) if n else None)), ""
    if kind == "deliveries":
        s = deliveries_sql(con)
        if s is None:
            return None, "không dựng được SQL giao hàng"
        return con.execute(f"SELECT COUNT(*) FROM ({s}) WHERE d BETWEEN ? AND ?", [lo, hi]).fetchone()[0], ""
    if kind == "purchase_total":
        s = purchases_sql(con)
        if s is None:
            return None, "không dựng được SQL nhập"
        v = con.execute(f"SELECT SUM(total) FROM ({s}) WHERE d BETWEEN ? AND ? AND NOT COALESCE(cancelled, FALSE)",
                        [lo, hi]).fetchone()[0]
        return (int(round(v)) if v is not None else 0), ""
    return None, "?"


def names_by_id(con, role: str, id_col: str, name_col: str, key) -> list:
    """A sales row that records only the code: the name is looked up in the product catalogue / customer list (rulebook §8: a file with a code goes by the code)."""
    out = []
    for t in _tables(con, role):
        c = _cols(con, t)
        if {id_col, name_col} <= c:
            out += [r[0] for r in con.execute(f'SELECT DISTINCT "{name_col}" FROM "{t}" '
                                              f'WHERE CAST("{id_col}" AS VARCHAR) = ?', [key]).fetchall()]
    return out


def customer_ids_by_name(con, name: str) -> list[str]:
    """The customer code whose name matches the name in the question (key with accents / case / punctuation removed, rulebook §3).
    A shop with no customer code: the name key is the customer."""
    key = text_key(name)
    ids = set()
    for role in ("customers", "orders"):
        for t in _tables(con, role):
            c = _cols(con, t)
            if "customer_name" not in c:
                continue
            idc = "CAST(customer_id AS VARCHAR)" if "customer_id" in c else "customer_key"
            nmc = "customer_name_clean" if "customer_name_clean" in c else "customer_name"
            for cid, nm, nmc_ in con.execute(f'SELECT DISTINCT {idc}, customer_name, {nmc} FROM "{t}"').fetchall():
                if cid is not None and key in (text_key(nm), text_key(nmc_)):
                    ids.add(cid)
    return sorted(ids)


_SALES_KINDS = {"revenue", "orders", "cancel_count", "cancel_rate", "top_product", "top_customer", "avg_order_value",
                "best_day", "group_revenue", "group_share", "qty_sold", "rev_filter", "best_staff", "gross_profit",
                "top_profit_product", "rx_revenue", "non_rx_revenue", "rx_rate"}


def out_of_sales_range(con, kind, lo, hi) -> bool:
    """§7: the period lies COMPLETELY outside the date range that has sales orders (_meta_shop.sales_start / sales_end) -> cannot be answered."""
    if kind not in _SALES_KINDS:
        return False
    a, b = lo if isinstance(lo, tuple) and len(lo) == 2 and isinstance(lo[0], dt.date) else (lo, hi)
    if not (isinstance(a, dt.date) and isinstance(b, dt.date)):
        return False
    try:
        s, e = con.execute("SELECT CAST(sales_start AS DATE), CAST(sales_end AS DATE) FROM _meta_shop").fetchone()
    except duckdb.Error:
        return False
    return s is not None and e is not None and (b < s or a > e)


def same(got, exp) -> bool:
    if got is None:
        return False
    if isinstance(got, list) and got and not isinstance(got[0], dict):    # a list in the order of the question
        return isinstance(exp, list) and len(exp) == len(got) and all(same(g, e) for g, e in zip(got, exp))
    if isinstance(got, str):           # a date "YYYY-MM-DD", a name: case, spaces and punctuation are ignored
        return isinstance(exp, str) and re.sub(r"[^\w]+", "", got.lower()) == re.sub(r"[^\w]+", "", exp.lower())
    if isinstance(got, list):          # a table [name, number]: all rows present, each answer row matches one data row (§9)
        rows = exp.get("rows") if isinstance(exp, dict) else None
        if not rows or len(rows) != len(got):
            return False
        return all(any(same(g, r[0]) and same(g["a"], r[1]) for g in got) for r in rows)
    if isinstance(got, dict):          # a name: any spelling in the file, or the code, is accepted; a name with a phone number is also accepted (§9)
        if not isinstance(exp, str):
            return False
        key = re.sub(r"[^\w]+", "", fold(split_phone(exp)[0]))
        return any(re.sub(r"[^\w]+", "", fold(split_phone(n or "")[0])) == key for n in got["names"]) or \
            (got["id"] is not None and fold(got["id"]) == fold(exp))
    return isinstance(exp, (int, float)) and abs(got - exp) < 0.006


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_root", help="folder with questions.jsonl, answers.jsonl")
    ap.add_argument("split_out", help="output folder holding <shop>/<shop>.duckdb")
    ap.add_argument("--kinds", nargs="*", help="check only these kinds")
    ap.add_argument("-v", action="store_true", help="also print the correct questions")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(a.split_root, "questions.jsonl"), encoding="utf-8")]
    ans = {r["id"]: r for r in map(json.loads, open(os.path.join(a.split_root, "answers.jsonl"), encoding="utf-8"))}
    stats, cons = Counter(), {}
    for q in qs:
        today = dt.date.fromisoformat(q["as_of_date"])
        p = parse_question(q["question"], today)
        if not p or (a.kinds and p[0] not in a.kinds):
            continue
        kind, (lo, hi) = p
        shop, gold = q["business_id"], ans[q["id"]]
        db = os.path.join(a.split_out, shop, f"{shop}.duckdb")
        if shop not in cons:
            cons[shop] = duckdb.connect(db, read_only=True) if os.path.exists(db) else None
        if cons[shop] is None:
            got, note = None, "không có db"
        else:
            if isinstance(lo, tuple) and lo[0] == "tet":    # ("tet", Y): rulebook §2, read from _tet_periods
                r = cons[shop].execute("SELECT CAST(start_date AS DATE), CAST(end_date AS DATE) FROM _tet_periods "
                                       "WHERE tet_year = ?", [lo[1]]).fetchone()
                lo, hi = (r[0], min(r[1], hi)) if r else (None, None)
            got, note = answer(cons[shop], kind, lo, hi) if lo else (None, "không có kỳ Tết")
            if got is not None and out_of_sales_range(cons[shop], kind, lo, hi):
                got, note = None, "kỳ nằm hoàn toàn ngoài khoảng ngày có đơn bán (§7)"
        if gold["type"] == "unanswerable":
            # §7: the shop does not record the kind of information the question needs -> the DB must NOT be able to build the source (no 0 comes out)
            stats[(kind, "u_ok" if got is None else "u_miss")] += 1
            if got is not None or a.v:
                print(f"{'OK ' if got is None else 'UNA'} {kind:<14} {shop} {q['as_of_date']} | "
                      f"{q['question'][:58]:<58} | đáp án không trả lời được | data {str(got)[:70]} {note}")
            continue
        ok = same(got, gold["answer"])
        stats[(kind, ok)] += 1
        if not ok or a.v:
            print(f"{'OK ' if ok else 'SAI'} {kind:<14} {shop} {q['as_of_date']} | {q['question'][:58]:<58} "
                  f"| đáp án {json.dumps(gold['answer'], ensure_ascii=False)[:40]} | data {str(got)[:70]} {note}")
    print()
    for kind in TEMPLATES:
        t = stats[(kind, True)] + stats[(kind, False)]
        u = stats[(kind, "u_ok")] + stats[(kind, "u_miss")]
        print(f"{kind:<19} đúng {stats[(kind, True)]:>3}/{t:<3}" +
              (f"  | không trả lời được: DB báo thiếu nguồn {stats[(kind, 'u_ok')]}/{u}" if u else ""))


if __name__ == "__main__":
    main()
