"""Map original columns -> canonical names (see schema.md).

Order of evidence sources (decreasing trust):
  1. readme_col   : the description in this shop's README section "Cột trong `file`"
  2. readme_hint  : a sentence "**Column** is <meaning>" in the conventions section
  3. name_dict    : column name (accents stripped) that another shop's README once described clearly
  4. value_pattern: values that look like codes (ORD..., SP..., KH..., PN..., PAY..., NCC...)
  5. value_rule   : a column still empty whose values reveal its role: all order statuses, contains "Khách lẻ",
                    all units of measure, = qty x unit price on each row, equal to product names in another table
  6. llm          : columns the rules left empty (map_shop(use_llm=True), function `llm_map`)
Every result is re-checked by value type (a money column must be numeric, a date column must be a date).
When unsure KEEP THE ORIGINAL NAME: a wrong name is worse than an odd one.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .parsers import fold, parse_number, split_name

# (regex on the accent-stripped description, canonical name). Specific first, general last.
DESC_RULES: list[tuple[str, str | None]] = [
    (r"cot trong, khong co du lieu", None),
    # a specific phrase changes the meaning of the general word after it ("mã đơn NHẬP", "số lượng KHÁCH", "ngày nhập DỮ LIỆU"):
    # it must come before the general rules, the same kind of error as "total amount of the whole slip" != purchase-slip amount
    (r"^ma (don|phieu) nhap", "purchase_id"),
    (r"^ma (don|phieu) tra", "return_id"),
    (r"tong tien (cua )?(ca )?(don|phieu) nhap", "purchase_total"),
    (r"trang thai (cua )?(don|phieu) nhap", "purchase_status"),
    (r"^so luong khach\b(?! (tra|mua|dat))", "guests"),      # not "quantity of customers returned / bought"
    (r"^so luong (ton|con lai)", "stock_qty"),
    (r"^so luong (khach )?tra lai\b(?!.*co ban)", "qty_returned"),
    (r"thanh tien (cua )?ca (don|hoa don|phieu)", "money"),
    (r"^ngay nhap (du lieu|vao (phan mem|he thong|may))", "created_date"),
    (r"^ngay (khach )?tra hang|^ngay tra\b", "return_date"),
    # services (repair, beauty...): each ledger row is a ticket; worker, commission, vehicle / licence plate
    (r"^ngay (sua|lam|thuc hien|lam dich vu|nhan xe)\b", "order_date"),
    (r"^ho ten (tho|nhan vien)|^(tho|nhan vien) (lam|ban|phu trach|thuc hien)|^tho\b", "staff_name"),
    (r"hoa hong", "commission_pct"),
    (r"^ngay vao lam", "start_date"),
    (r"^dong xe va bien so|^xe va bien so", "vehicle"),
    (r"^(cac )?viec (da )?lam|^noi dung (sua|cong viec|dich vu)|^cong viec\b", "service_desc"),
    # codes
    (r"^ma don|^so hoa don|so thu tu don\b", "order_id"),   # \b: does not match "số thứ tự dòng" (row serial number)
    (r"so thu tu dong", "line_no"),
    (r"^ma phieu nhap|^ma don nhap", "purchase_id"),
    (r"^ma phieu thu", "payment_id"),
    (r"^ma hang|^ma san pham|^ma sku", "product_id"),
    (r"^ma khach", "customer_id"),
    (r"^ma nha cung cap|^ma ncc", "supplier_id"),
    (r"^ma nhan vien", "staff_id"),
    (r"^ma ca", "shift_id"),
    (r"^ma khuyen mai", "promo_id"),
    (r"^so lo|^ma lo", "batch_id"),
    (r"^imei", "imei"),
    # names
    (r"^ten hang|^hang ban|^ten san pham", "product_name"),
    (r"^ten khach", "customer_name"),
    (r"^nha cung cap$|^ten nha cung cap", "supplier_name"),
    (r"^ten$", "name"),
    (r"^nhom hang", "category"),
    (r"^nhom khach", "customer_group"),
    # units, conversion
    (r"^don vi tinh", "unit"),
    (r"^don vi co ban", "base_unit"),
    (r"^don vi lon", "big_unit"),
    (r"^don vi nhap", "unit"),
    (r"^he so|quy doi don vi lon", "conv_factor"),
    # quantity
    (r"so luong khach tra lai", "qty_returned_base"),
    (r"so luong ban lay tu lo", "qty_base"),
    (r"so luong (da doi ra |nhap \()don vi co ban|so luong .*\(don vi co ban\)", "qty_base"),
    (r"^ton kho|^ton cuoi", "stock_qty"),
    (r"^ton sau lan", "stock_after"),
    (r"^so luong", "qty"),
    # order amount comes before line amount: "order total goods amount (sum of the lines' line amounts)" contains the words "thành tiền"
    (r"tong tien hang|tien hang cua don", "order_subtotal"),
    # line amount
    (r"^don gia|gia nhap mot don vi", "unit_price"),
    (r"thanh tien|tien nhap cua ca dong|^tien ban", "line_amount"),
    (r"tien giam (cua )?(ca )?dong|^giam gia (tren |cua )?(tung |moi )?dong", "line_discount"),
    (r"^gia von", "unit_cost"),
    (r"gia ban hien tai", "price"),
    (r"gia ban ap dung tu ngay", "new_price"),
    (r"gia truoc lan doi", "old_price"),
    # order amount
    (r"giam gia cua ca don|tien bot cho ca don", "order_discount"),
    (r"^giam gia$", "discount"),
    (r"tong tien (cua ca )?phieu nhap", "purchase_total"),
    # "phiếu" not specified (sales slip, repair slip, receipt...): generic money, the table's context decides
    (r"tong tien (cua )?(ca )?phieu", "money"),
    (r"tong tien don|phai tra cho ca don|can tra cho ca don"
     r"|tong tien (cua )?(ca )?(hoa don|don|bill)\b|tong (so )?tien (khach )?(phai tra|thanh toan) (cua |cho )?(ca )?(don|hoa don)",
     "order_total"),
    (r"phi giao", "shipping_fee"),
    (r"^da thanh toan|khach da tra cho ca don|so tien khach tra cho ca don", "paid"),
    (r"con no", "debt"),
    (r"so tien tra lan do|^so tien$", "payment_amount"),
    # date and time
    (r"^ngay ban", "order_date"),
    # rulebook §2: morning / afternoon shift by the order creation time
    (r"thoi diem (dat|tao) don|gio (dat|tao) don", "order_time"),
    (r"^ngay tao", "created_date"),
    (r"^ngay nhap", "purchase_date"),
    (r"thoi diem thu", "payment_date"),
    (r"han su dung", "expiry_date"),
    (r"^ngay giao", "delivery_date"),
    (r"han thanh toan", "due_date"),
    (r"ngay bat dau ap dung gia", "price_from"),
    # status, classification
    (r"^trang thai don", "status"),
    (r"trang thai phieu nhap", "purchase_status"),
    (r"hinh thuc thanh toan", "payment_method"),
    (r"kenh ban", "channel"),
    (r"hoat chat", "active_ingredient"),
    (r"thuoc ke don", "is_prescription"),
    (r"con kinh doanh", "active"),
    # delivery, debt
    (r"dot giao", "delivery_round"),
    (r"tai xe", "driver"),
    (r"bien so xe", "plate"),
    (r"so ngay duoc no", "credit_days"),
    # restaurant
    (r"^ban$", "table_no"),
    (r"khu vuc cua ban", "area"),
    (r"so khach cua don", "guests"),
    # contact, notes
    (r"so dien thoai", "phone"),
    (r"^dia chi", "address"),
    (r"^ghi chu cua don", "order_note"),
    (r"^ghi chu", "note"),
]
_DESC_RULES = [(re.compile(p), c) for p, c in DESC_RULES]

NUMBER = {"qty", "qty_base", "qty_returned_base", "stock_qty", "unit_price", "line_amount", "line_discount",
          "unit_cost", "price", "new_price", "old_price", "order_subtotal", "order_discount", "discount",
          "purchase_total", "order_total", "shipping_fee", "paid", "debt", "payment_amount",
          "conv_factor", "delivery_round", "credit_days", "guests", "stock_after", "money", "qty_returned",
          "commission_pct", "expense_amount"}
DATE = {"order_date", "order_time", "created_date", "purchase_date", "payment_date", "expiry_date", "delivery_date",
        "due_date", "price_from", "return_date", "start_date", "expense_date"}
# codes are usually strings; but sqlite/Excel may store a numeric code -> only a date type is forbidden
CODE_PREFIX = {"ORD": "order_id", "SP": "product_id", "KH": "customer_id", "PN": "purchase_id",
               "PAY": "payment_id", "NCC": "supplier_id", "NV": "staff_id"}

# Common column names of sales software / notebooks (key name_key), used when the dictionary from READMEs
# has no such name. Only names whose meaning is clear in every table; ambiguous names are left to context rules / the LLM.
BUILTIN_NAMES = {
    "sl co ban": "qty_base", "sl (dv co ban)": "qty_base", "sl (dvt co ban)": "qty_base",
    "sl quy ra le": "qty_base", "sl nhap (dvt co ban)": "qty_base", "slnhap quy doi": "qty_base",
    "sl tra (dvt co ban)": "qty_returned_base", "sltra quy doi": "qty_returned_base",
    "sl tra (dv co ban)": "qty_returned_base", "dvt nhap": "unit", "dv nhap": "unit",
    "he so quy doi": "conv_factor", "bang": "conv_factor",
    "hsd": "expiry_date", "han dung": "expiry_date",
    "con no": "debt", "cong no": "debt",
    "tien hang truoc giam": "order_subtotal",
    "gio tao don": "order_time", "ngay hd": "order_date",
    "ngay tt": "payment_date", "ngay gio thu": "payment_date", "ngay nhap kho": "purchase_date",
    "ke don": "is_prescription", "thuoc ke don": "is_prescription", "ban theo don": "is_prescription",
    "thanh phan chinh": "active_ingredient", "hoat chat": "active_ingredient", "hoat chat chinh": "active_ingredient",
    "ngay chi": "expense_date", "khoan chi": "expense_item", "muc chi": "expense_item", "tien chi": "expense_amount",
    "so tien chi": "expense_amount",
    "dong": "line_no",
    "so lo": "batch_id", "ma lo": "batch_id", "lo": "batch_id", "lo hang": "batch_id",
    # generic name: a product / customer / supplier / staff name depending on the table's code (_resolve_context)
    "ten": "name", "ten goi": "name", "ho ten": "name", "ho va ten": "name",
    "gia von": "unit_cost", "gia von hang": "unit_cost", "gia von hien tai": "unit_cost",
    "so tien": "money",          # generic money: _resolve_context sorts it into receipt amount / payment amount / line amount by table
    "con kinh doanh": "active",
    # common column names of sales ledgers (invoice code, document date, time, state, discount)
    "ma hd": "order_id", "so hd": "order_id", "ma hoa don": "order_id", "so hoa don": "order_id",
    "ngay chung tu": "order_date", "ngay gio": "order_time", "luc": "order_time",
    "tinh trang": "status", "ma sp": "product_id",
    "ck dong": "line_discount", "chiet khau dong": "line_discount", "giam gia dong": "line_discount",
    "chiet khau": "discount", "giam": "discount", "giam gia": "discount",
    # the README only gives the table names (sqlite, invoices only): frequent abbreviated column names
    "slg": "qty", "so lg": "qty", "t tien": "line_amount", "t.tien": "line_amount",
    "ma nv": "staff_id", "nguon don": "channel", "tong cong": "money",
    # English column names of POS software / machines (SQLite exported as is, ids flattened in read_files)
    "order id": "order_id", "order code": "order_id", "order no": "order_id", "invoice no": "order_id",
    "invoice id": "order_id", "product id": "product_id", "product code": "product_id", "sku": "product_id",
    "product name": "product_name", "customer id": "customer_id", "customer code": "customer_id",
    "customer name": "customer_name", "customer group": "customer_group", "group name": "customer_group",
    "supplier id": "supplier_id", "supplier name": "supplier_name", "staff code": "staff_id", "staff id": "staff_id",
    "staff name": "staff_name", "payment id": "payment_id", "purchase id": "purchase_id", "return id": "return_id",
    "quantity": "qty", "unit price": "unit_price", "line total": "line_amount", "line amount": "line_amount",
    "shipping fee": "shipping_fee", "payment method": "payment_method", "created at": "order_time",
    "order date": "order_date", "order time": "order_time", "paid at": "payment_date", "payment date": "payment_date",
    "base unit": "base_unit", "expiry date": "expiry_date", "due date": "due_date",
    "status": "status", "channel": "channel", "category": "category", "discount": "discount", "note": "note",
    "phone": "phone", "address": "address", "name": "name", "unit": "unit", "amount": "money", "total": "money",
    # base unit (taken separately from NV4; the NV4 group as a whole stays off)
    "dv co ban": "base_unit", "dvt co ban": "base_unit", "don vi co ban": "base_unit",
    # Shopee order export file (the marketplace's original column names)
    "gia goc": "unit_price", "nguoi ban tro gia": "line_discount", "tong gia ban (san pham)": "line_amount",
    "ma giam gia cua shop": "order_discount", "phi van chuyen (du kien)": "shipping_fee",
}
# common column names, off by default (turn on with DATAMATE_NV4=1)
_NV4_NAMES = {
    "dv co ban": "base_unit", "dvt co ban": "base_unit", "don vi co ban": "base_unit",
    "tong don": "order_total", "tong phai tra": "order_total",
    "ma hv": "customer_id", "ma hoi vien": "customer_id",
    "thanh toan": "payment_method", "hinh thuc thanh toan": "payment_method", "httt": "payment_method", "hinh thuc tt": "payment_method",
}
if os.environ.get("DATAMATE_NV4"):   # off by default (changes the LLM's column-matching prompt)
    BUILTIN_NAMES.update(_NV4_NAMES)

# words naming the subject in a column name "Tên ..." (customer checked first: "khách hàng" contains the word "hàng")
_ENTITY_WORDS = [(re.compile(r"\bkhach\b|\bkh\b|customer"), "customer_name"),
                 (re.compile(r"nha cung cap|\bncc\b|supplier"), "supplier_name"),
                 (re.compile(r"\bhang\b|san pham|\bsp\b|\bmon\b|\bthuoc\b|product|item"), "product_name")]
_LINE_QTY = {"qty", "unit_price", "line_amount"}
_EXPENSE_TABLE = re.compile(r"\bchi ?phi\b|\bphieu chi\b|\bso chi\b|\bkhoan chi\b|\bexpense")


def desc_to_canonical(desc: str) -> tuple[bool, str | None]:
    """(matching rule, canonical name). A match with canonical name None means the column is empty per the README."""
    f = fold(desc)
    for rx, canon in _DESC_RULES:
        if rx.search(f):
            return True, canon
    return False, None


@dataclass
class ColumnMap:
    table: str
    column: str
    canonical: str | None
    source: str | None          # readme_col | readme_hint | name_dict | value_pattern | llm
    description: str | None     # README description used as evidence
    note: str = ""


def build_name_dict(readmes) -> dict[str, str]:
    """Dictionary of column names (accents stripped) -> canonical name, collected from many shops' READMEs.

    Keep only names where >= 80% of occurrences give the same canonical name (e.g. 'don gia' in a sales table and a purchase table
    is unit_price in both -> keep; 'giam gia' is sometimes line-level, sometimes order-level -> still 'discount').
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    forms: dict[str, set] = defaultdict(set)
    for info in readmes:
        for d in info.columns:
            ok, canon = desc_to_canonical(d.description)
            if ok and canon:
                # the same family merges slips, so the table-context rule decides (line/order discount,
                # "Thời gian" is sales date / purchase date / receipt time, "Trạng thái" of a sales order / purchase slip)
                canon = _FAMILY.get(canon, canon)
                votes[name_key(d.column)][canon] += 1
                forms[name_key(d.column)].add(_syllables(d.column))
    out = NameDict(forms=forms)
    for name, c in votes.items():
        # the generic "name" supports the specific name that holds the majority ('Tên hàng' in a catalogue README just writes "tên")
        specific = [k for k, _ in c.most_common() if k in _NAME_FAMILY]
        if "name" in c and specific:
            c = c.copy()
            c[specific[0]] += c.pop("name")
        canon, n = c.most_common(1)[0]
        if n >= 0.8 * sum(c.values()):
            out[name] = canon
    return out


class NameDict(dict):
    """Dictionary of column names (accent-stripped key) -> canonical name, with the accented spellings seen for each key."""

    def __init__(self, *a, forms=None, **k):
        super().__init__(*a, **k)
        self.forms = forms or {}


def with_base(names: NameDict, base: NameDict) -> NameDict:
    """Add to the dictionary being built the names that its own README never described, taken from the base dictionary (training READMEs).
    A set whose README does not describe columns has an almost empty dictionary: even "Mã hàng", "Trạng thái" cannot be mapped.
    Names that the README being built has described (including names dropped because shops described them differently) are not overridden by the base."""
    forms = getattr(names, "forms", {})
    out = NameDict(names, forms=dict(forms))
    for k, v in base.items():
        if k not in names and k not in forms:
            out[k] = v
            out.forms[k] = getattr(base, "forms", {}).get(k, set())
    return out


_TONES = {"̀", "́", "̃", "̉", "̣"}     # huyền, sắc, ngã, hỏi, nặng (the five Vietnamese tone marks)


def _syllables(col: str) -> tuple:
    """Each syllable -> (base letters keeping ă â ê ô ơ ư đ, tone). Where the tone mark is placed does not count: 'hoá' = 'hóa'.
    A syllable without any tone mark -> None (not used for comparison)."""
    import unicodedata
    out = []
    for w in name_key_raw(col).split():
        nfd = unicodedata.normalize("NFD", w)
        tone = next((c for c in nfd if c in _TONES), "")
        base = unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _TONES))
        out.append((base, tone) if (tone or base != fold(base)) else None)
    return tuple(out)


def accent_clash(col: str, key: str, name_dict) -> bool:
    """A column written with accents whose tone differs from every spelling of the key in the dictionary: 'Hạng' (customer rank) vs 'Hàng',
    'Thuê' vs 'Thuế'. Equal once accents are stripped but different words. A syllable of an unaccented column is not compared."""
    seen = getattr(name_dict, "forms", {}).get(key)
    mine = _syllables(col)
    if not seen or all(s is None for s in mine):
        return False

    def fits(form):
        return len(form) == len(mine) and all(a is None or b is None or a == b for a, b in zip(mine, form))
    accented = [f for f in seen if any(s is not None for s in f)]
    return bool(accented) and not any(fits(f) for f in accented)


_NAME_FAMILY = ("product_name", "customer_name", "supplier_name")
# The column-name dictionary only decides the FAMILY of the concept; the specific name is decided by the table's context (_resolve_context).
# The same words "ĐVT", "Giá", "Tổng tiền" mean different things in a catalogue / sales row / order table / cash book.
_FAMILY = {"line_discount": "discount", "order_discount": "discount",
           "purchase_date": "order_date", "payment_date": "order_date",
           "purchase_status": "status",
           "base_unit": "unit", "price": "unit_price",
           "line_amount": "money", "order_total": "money", "payment_amount": "money", "purchase_total": "money"}


def name_key_raw(col: str) -> str:
    """Like name_key but keeps accents: 'TênSảnPhẩm' -> 'tên sản phẩm'."""
    return split_name(col)


def name_key(col: str) -> str:
    """Dictionary lookup key: 'TenSanPham' / 'ten_san_pham' / 'Tên sản phẩm' -> 'ten san pham'."""
    return fold(name_key_raw(col))


def _value_prefix(samples: list) -> str | None:
    """'ORD000123' x all values -> 'ORD'."""
    vals = [v for v in samples if isinstance(v, str)]
    if len(vals) < 3:
        return None
    pref = Counter(m[1] for v in vals if (m := re.fullmatch(r"([A-Z]{2,4})\d{3,}", v)))
    if pref and pref.most_common(1)[0][1] >= 0.95 * len(vals):
        return pref.most_common(1)[0][0]
    return None


TIME_ONLY = re.compile(r"^\s*([01]?\d|2[0-3])\s*[:hg]\s*([0-5]\d)(\s*:\s*[0-5]\d)?\s*$", re.I)


def time_only(vals) -> bool:
    """A column recording only the time of day ("18:51", "9h30"): the date is taken from the order-date column."""
    vals = [str(v) for v in vals if v is not None and str(v).strip()]
    return len(vals) >= 5 and sum(bool(TIME_ONLY.match(v)) for v in vals) >= 0.95 * len(vals)


def _kind_ok(canon: str, kind: str, vals=()) -> bool:
    if canon == "payment_method" and kind == "number":
        return False
    if canon == "order_time" and kind == "text":
        return time_only(vals)
    if canon in NUMBER:
        return kind in ("number", "empty")
    if canon in DATE:
        return kind in ("datetime", "empty")
    return kind != "datetime"


def map_table(table: str, columns: list[str], kinds: dict[str, str], samples: dict[str, list],
              docs, name_dict: dict[str, str], parsed: dict[str, list] | None = None,
              unit_vocab: set[str] | None = None) -> list[ColumnMap]:
    """Map the columns of one table.

    docs: the README ColumnDoc entries that apply to this table. samples / parsed: original / parsed values of the
    first N rows, columns in the same row order (to check arithmetic between columns). unit_vocab: the known units.
    """
    by_name = {fold(d.column): d for d in docs}
    result: list[ColumnMap] = []
    for col in columns:
        f = fold(col)
        cand: list[tuple[str | None, str, str | None]] = []   # (canonical, source, description)
        d = by_name.get(f)
        # a description in brackets right after the column name (README style "**Name** (note)") is only a note, it may refer
        # to another column ("= sum of line amounts − ..."): if the column name is already in the dictionary, trust the name first
        if d is not None and d.source == "inline" and (
                (name_key(col) in name_dict and not accent_clash(col, name_key(col), name_dict))
                or name_key(col) in BUILTIN_NAMES):
            d = None
        if d is not None:
            ok, canon = desc_to_canonical(d.description)
            if ok:
                cand.append((canon, "readme_col" if d.source == "columns" else "readme_hint", d.description))
            elif not ((k := name_key(col)) in name_dict and not accent_clash(col, k, name_dict)) and k not in BUILTIN_NAMES:
                result.append(ColumnMap(table, col, None, None, d.description, "mô tả README chưa có luật"))
                continue
            # a description with no rule yet but the column name is in the dictionary: use the name (the same ledger, different files map the same)
        if not cand and (k := name_key(col)) in name_dict and not accent_clash(col, k, name_dict):
            cand.append((name_dict[k], "name_dict", None))
        if not cand and (k := name_key(col)) in BUILTIN_NAMES:
            cand.append((BUILTIN_NAMES[k], "builtin", None))
        if not cand and (p := _value_prefix(samples.get(col, []))) and p in CODE_PREFIX:
            cand.append((CODE_PREFIX[p], "value_pattern", None))
        if not cand:
            result.append(ColumnMap(table, col, None, None, None))
            continue
        canon, source, desc = cand[0]
        # code column: values 'PAY...' while the name suggests order_id -> trust the values (unless the README describes it directly)
        pref = _value_prefix(samples.get(col, []))
        by_value = CODE_PREFIX.get(pref) if pref else None
        if by_value and canon in CODE_PREFIX.values() and canon != by_value and source != "readme_col":
            canon, source = by_value, "value_pattern"
        if canon is None:
            result.append(ColumnMap(table, col, None, source, desc, "README: cột trống"))
        elif not _kind_ok(canon, kinds.get(col, "text"), samples.get(col, [])):
            result.append(ColumnMap(table, col, None, source, desc,
                                    f"loại: {canon} cần kiểu khác, cột là {kinds.get(col)}"))
        else:
            result.append(ColumnMap(table, col, canon, source, desc))

    _resolve_context(result)
    _resolve_duplicates(result)
    _value_rules(result, kinds, samples, parsed or {}, unit_vocab or DEFAULT_UNITS)
    _resolve_context(result)
    _resolve_duplicates(result)
    return result


# Common units; each shop adds the units in its README and in columns mapped as units
DEFAULT_UNITS = {
    "cái", "chiếc", "ly", "cốc", "chai", "lon", "hộp", "gói", "kg", "g", "thùng", "lốc", "bao", "cây", "m",
    "m2", "m3", "bình", "cuộn", "viên", "vỉ", "ống", "lọ", "tuýp", "đĩa", "phần", "tô", "nồi", "con", "lần",
    "bộ", "quả", "bó", "tấm", "tấn", "xe", "bịch", "túi", "đôi", "lít", "ml", "suất", "bát", "chén", "set",
    "máy", "miếng", "hũ", "can", "khay", "vé", "giờ", "kệ", "cặp",
}


# a table with these columns is a sales table (order or order line), including hand-written ledgers without order ids
SALES_SIGNALS = {"order_id", "qty", "qty_base", "line_amount", "unit_price", "product_name", "order_date",
                 "order_total", "order_subtotal"}

_PAY_METHOD = re.compile(r"^(tien mat|chuyen khoan|ck|the|quet the|vi( dien tu)?|momo|zalopay|vnpay|cod|"
                         r"ghi no|no|cash|transfer|ewallet|card|debt|bank)\b")


# customer tiers / groups common in sales software and notebooks
_TIER = re.compile(r"^(khach )?(le|si|ban si|ban le|than thiet|vip|thuong|moi|tiem nang|bac|vang|dong|kim cuong|"
                   r"bach kim|dai ly|ctv|cong tac vien|member|silver|gold|platinum|diamond|regular|wholesale|retail)"
                   r"( \d)?$")


def _distinct(vals) -> list:
    return list(dict.fromkeys(v for v in vals if v is not None))


def _value_rules(maps: list[ColumnMap], kinds, raw: dict[str, list], parsed: dict[str, list], units: set[str]):
    """VALUE rules for a column still empty (no README description, not in the name dictionary)."""
    from .parsers import is_walk_in, norm_status

    present = {m.canonical for m in maps if m.canonical}
    has_lines = bool(present & {"qty", "unit_price", "line_amount", "qty_base"})
    units_f = {fold(u) for u in units}
    for m in maps:
        if m.canonical or m.description:      # already mapped, or the README has said something about this column
            continue
        kind, vals = kinds.get(m.column), [v for v in raw.get(m.column, []) if v is not None]
        if len(vals) < 5:
            continue
        uniq = _distinct(vals)
        if kind == "text":
            st = [norm_status(v) for v in vals]
            if present & SALES_SIGNALS and "purchase_id" not in present and len(uniq) <= 30 \
                    and sum(s is not None for s in st) >= 0.95 * len(vals) and len(set(st)) >= 2:
                m.canonical, m.source, m.note = "status", "value_rule", "giá trị là trạng thái đơn"
            elif (present & {"customer_id", "customer_name"} and "customer_group" not in present
                  and not present & {"product_id", "product_name", "qty", "qty_base"} and 2 <= len(uniq) <= 12
                  and sum(bool(_TIER.search(fold(v))) for v in uniq) >= 0.5 * len(uniq)):
                # a customer tier / type column with values "lẻ / sỉ / thân thiết / VIP": customer group (rulebook §1),
                # whatever the column name ("Hạng", "LoaiKhachHang"); "Loại hình khách" = "Cá nhân" is not
                m.canonical, m.source, m.note = "customer_group", "value_rule", "giá trị là hạng / nhóm khách"
                present.add("customer_group")
            elif len(uniq) <= 15 and sum(_PAY_METHOD.search(fold(v)) is not None for v in vals) >= 0.95 * len(vals):
                m.canonical, m.source, m.note = "payment_method", "value_rule", "giá trị là hình thức thanh toán"
            # customer name: contains "Khách lẻ" AND many different names (a customer-group column "lẻ / thân thiết" has only a few values)
            elif sum(is_walk_in(v) for v in vals) >= 0.03 * len(vals) and len(uniq) >= 10:
                m.canonical, m.source, m.note = "customer_name", "value_rule", "có giá trị khách lẻ"
            elif len(uniq) <= 80 and sum(fold(v) in units_f for v in vals) >= 0.95 * len(vals):
                is_line = has_lines or present & {"order_id", "purchase_id"}
                m.canonical = "unit" if is_line else "base_unit"
                m.source, m.note = "value_rule", "giá trị là đơn vị tính"
        elif (kind == "datetime" and present & {"order_id", "order_date"} and "order_time" not in present
              and not present & {"purchase_id", "payment_id", "delivery_round", "expense_date", "return_id",
                                 "return_date", "qty_returned", "qty_returned_base"}
              and present & {"order_total", "order_subtotal", "line_amount", "qty", "paid", "status",
                             "customer_id", "customer_name", "channel"}):
            # order table / sales rows: an unmapped datetime column where almost every cell has a time of day -> order creation time
            # (morning / afternoon shift by order creation time, rulebook §2). The order date usually records only the date.
            ts = [v for v in parsed.get(m.column, []) if hasattr(v, "hour")]
            if len(ts) >= 5 and sum(v.hour or v.minute for v in ts) >= 0.9 * len(ts):
                m.canonical, m.source, m.note = "order_time", "value_rule", "ngày giờ có giờ trong ngày ở bảng đơn"
                present.add("order_time")
        elif kind == "number" and {"qty", "unit_price"} <= present and "line_amount" not in present:
            col_of = {mm.canonical: mm.column for mm in maps if mm.canonical}
            q, p, a = parsed.get(col_of["qty"], []), parsed.get(col_of["unit_price"], []), parsed.get(m.column, [])
            trip = [(x, y, z) for x, y, z in zip(q, p, a)
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))]
            if len(trip) >= 20:
                exact = sum(abs(abs(z) - abs(x * y)) < 1 for x, y, z in trip)
                within = sum(0.5 * abs(x * y) - 1 <= abs(z) <= abs(x * y) + 1 for x, y, z in trip)
                if exact >= 0.7 * len(trip) and within >= 0.95 * len(trip):
                    m.canonical, m.source = "line_amount", "value_rule"
                    m.note = f"= SL x đơn giá ở {exact}/{len(trip)} dòng mẫu"


@dataclass
class TableInput:
    """What mapping needs to know about a table (not all the data)."""
    name: str
    file: str
    sheets: list
    columns: list[str]
    kinds: dict[str, str]
    raw: dict[str, list]       # first N rows, original values
    parsed: dict[str, list]    # the same N rows, parsed values

    @property
    def is_sqlite(self) -> bool:
        return self.file.lower().endswith((".sqlite", ".db", ".sqlite3"))


def shop_unit_vocab(readme) -> set[str]:
    """The shop's known units: the common list + units in the README's conversion table."""
    units = set(DEFAULT_UNITS)
    if readme is not None:
        for c in readme.unit_conversions:
            units.update({c.big_unit, c.base_unit})
    return units


def map_shop(tables: list[TableInput], readme, name_dict: dict[str, str], use_readme: bool = True,
             use_llm: bool = False) -> dict[str, list[ColumnMap]]:
    """Map every table of the shop, then run the cross-table pass. use_readme=False to measure the fallback.
    use_llm: columns the rules left empty are sent to the LLM (loader/llm.py), applied last so it does not override rules."""
    units = shop_unit_vocab(readme if use_readme else None)
    results = {}
    for t in tables:
        docs = readme.docs_for(t.file, t.sheets, table=t.name if t.is_sqlite else None) if use_readme else []
        results[t.name] = map_table(t.name, t.columns, t.kinds, t.raw, docs, name_dict, t.parsed, units)
    cross_table_names(results, {t.name: t.kinds for t in tables}, {t.name: t.raw for t in tables})
    for maps in results.values():
        _resolve_duplicates(maps)
    if use_llm:
        _llm_pass(tables, results, units)
        link_line_ids(tables, results)
    link_payment_books(tables, results)
    order_discount_names(tables, results)
    return_book_names(tables, results)
    sale_ids_not_purchase(results)
    payment_method_values(tables, results)
    return results


def line_discount_values(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """An unmapped numeric column in a sales-row table whose ratio to the line amount takes only a few levels (card discount 3% / 5%...) -> line discount.
    By value, not by column name. sales_sql subtracts it itself when the ledger records the line amount BEFORE discount."""
    for t in tables:
        maps = results[t.name]
        present = {m.canonical for m in maps}
        if "line_discount" in present or "line_amount" not in present:
            continue
        amt_col = next(m.column for m in maps if m.canonical == "line_amount")
        amt = t.parsed.get(amt_col) or []
        for m in maps:
            if m.canonical is not None or t.kinds.get(m.column) != "number":
                continue
            vals = t.parsed.get(m.column) or []
            rs = []
            for x, a_ in zip(vals, amt):
                try:
                    x, a_ = float(x), float(a_)
                except (TypeError, ValueError):
                    continue
                if x == x and a_ == a_ and x > 0 and a_ > 0:
                    rs.append(round(x / a_, 3))
            if len(rs) < 8:
                continue
            top = sorted(((rs.count(r), r) for r in set(rs)), reverse=True)[:3]
            if sum(n for n, _ in top) >= 0.8 * len(rs) and all(0 < r < 0.5 for _, r in top):
                m.canonical, m.source = "line_discount", m.source or "value_rule"
                m.note = (m.note + " | " if m.note else "") + \
                    "tỉ lệ / tiền dòng gom vào " + ", ".join(f"{r:.0%}" for _, r in top) + ": giảm giá dòng"
                break


_PAY_COL = re.compile(r"thanh toan|hinh thuc|pttt|\btt\b|tra kieu|kieu tra|cach tra")


def payment_method_values(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """An unmapped text column "Thanh toán" (the dictionary assigns "paid" but the values are text): >= 80% of values normalise to a
    payment method (cash, transfer, wallet, credit...) -> payment_method."""
    from .build import _PAYMENT_NORM
    for t in tables:
        maps = results[t.name]
        present = {m.canonical for m in maps}
        if "payment_method" in present or not present & {"order_id", "order_date", "line_amount", "order_total"}:
            continue
        for m in maps:
            if m.canonical is not None or not _PAY_COL.search(name_key(m.column)):
                continue
            vs = [fold(str(v)).replace("_", " ") for v in t.raw.get(m.column, []) if v is not None and str(v).strip()]
            if len(vs) >= 20 and sum(any(rx.search(v) for rx, _ in _PAYMENT_NORM) for v in vs) >= 0.8 * len(vs):
                m.canonical, m.source = "payment_method", m.source or "value_rule"
                m.note = (m.note + " | " if m.note else "") + "giá trị là hình thức thanh toán"
                break


def sale_ids_not_purchase(results: dict[str, list[ColumnMap]]) -> None:
    """The LLM assigned a transaction code as a purchase-slip code (purchase_id) in a table with no sign of purchasing (supplier, purchase date,
    purchase-slip status) but with a sales date / time or a payment method: it is a sales order code (e.g. an EV charging session, a service ticket)."""
    for maps in results.values():
        present = {m.canonical for m in maps}
        if "order_id" in present or present & {"supplier_name", "supplier_id", "purchase_date", "purchase_status"}:
            continue
        if not present & {"order_date", "order_time", "payment_method"} or not present & {"order_total", "line_amount"}:
            continue
        for m in maps:
            if m.canonical == "purchase_id" and m.source in ("llm", "name_dict", "builtin"):
                m.canonical = "order_id"
                m.note = (m.note + " | " if m.note else "") + "bảng không có dấu hiệu nhập hàng, có ngày bán / thanh toán: mã đơn"


_ORDER_DISC_NAME = re.compile(r"^(tien |giam )?voucher$|^(giam|bot|ck|chiet khau) (gia )?(ca )?(don|bill|hoa don|hd)$")


def order_discount_names(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """A numeric column "Voucher" / "Giảm giá đơn" unmapped in a sales table -> whole-order discount (rulebook: revenue does not deduct it, debt does).
    A column holding the voucher code (text) is left alone. Runs after the LLM so the DB-build prompt is unchanged."""
    for t in tables:
        maps = results[t.name]
        present = {m.canonical for m in maps}
        if "order_discount" in present or not present & {"order_id", "line_amount", "order_total"}:
            continue
        for m in maps:
            if m.canonical is None and _ORDER_DISC_NAME.search(name_key(m.column)) and t.kinds.get(m.column) == "number":
                m.canonical, m.source = "order_discount", m.source or "builtin"
                m.note = (m.note + " | " if m.note else "") + "tiền voucher / giảm cả đơn"
                break


def return_book_names(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """A customer-return ledger (with a return quantity, order id / product): an unmapped date column named "Ngày trả..." is the return date.
    In other ledgers "Ngày trả" may be a payment / debt-payment date, so it is not in the common dictionary."""
    for t in tables:
        maps = results[t.name]
        present = {m.canonical for m in maps}
        if not present & {"qty_returned", "qty_returned_base"} or "return_date" in present \
                or not present & {"order_id", "product_id", "product_name"} or present & {"payment_amount", "payment_id"}:
            continue
        for m in maps:
            if m.canonical is None and t.kinds.get(m.column) == "datetime" and re.match(r"ngay (khach )?tra\b", name_key(m.column)):
                m.canonical, m.source = "return_date", m.source or "builtin"
                m.note = (m.note + " | " if m.note else "") + "ngày ở sổ trả hàng"
                break


def _shape(v) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", str(v).strip()))


def link_payment_books(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """A debt-collection ledger by instalment ("Ngày trả", "Khách", "Hóa đơn", "Số tiền"): has a payment date, no order id, no product rows.
    An unmapped text column whose values are order ids of another table (>= 50% equal or >= 90% of the same shape) -> order_id; generic money /
    order money of that table -> the amount paid that time (payment_amount). Without this step the table becomes a customer list and debt is not computed."""
    raw = {t.name: t.raw for t in tables}
    ids, shapes = set(), set()
    for n, maps in results.items():
        for m in maps:
            if m.canonical == "order_id":
                vs = [v for v in raw[n].get(m.column, []) if v is not None and str(v).strip()]
                ids.update(fold(str(v)).strip() for v in vs)
                shapes.update(_shape(v) for v in vs)
    if len(ids) < 10:
        return
    for n, maps in results.items():
        canons = {m.canonical for m in maps}
        # variant: "Mã HĐ" assigned a receipt code by the LLM, "Ngày thu" assigned an order date: a date named thu / trả / nộp / góp
        # in a table with no product rows and a column holding order ids of another table -> payment date
        if "payment_date" not in canons and not canons & {"order_id", "product_name", "product_id", "qty", "qty_base",
                                                          "line_amount", "purchase_id"}:
            pay_day = [m for m in maps if m.canonical in ("order_date", "order_time") and m.source != "readme_col"
                       and re.search(r"\b(thu|tra|nop|gop)\b", name_key(m.column))]
            carrier = next((m for m in maps if m.canonical in (None, "payment_id")
                            and (m.canonical is None or m.source == "llm")
                            and (lambda vs: len(vs) >= 10 and (sum(fold(str(v)).strip() in ids for v in vs) >= 0.5 * len(vs) or sum(_shape(v) in shapes for v in vs) >= 0.9 * len(vs) and any(c.isalpha() for c in str(vs[0]))))(
                                [v for v in raw[n].get(m.column, []) if v is not None and str(v).strip()])), None)
            if pay_day and carrier:
                for m in pay_day:
                    m.canonical = "payment_date"
                    m.note = (m.note + " | " if m.note else "") + "ngày thu / trả của sổ thu nợ"
                canons = {m.canonical for m in maps}
        if "payment_date" not in canons or canons & {"order_id", "product_name", "product_id", "qty", "qty_base",
                                                      "line_amount", "order_date", "purchase_id"}:
            continue
        for m in maps:
            if m.canonical is not None and not (m.canonical == "payment_id" and m.source == "llm"):
                continue
            vs = [v for v in raw[n].get(m.column, []) if v is not None and str(v).strip()]
            if len(vs) < 10 or not all(isinstance(v, str) for v in vs):
                continue
            hit = sum(fold(v).strip() in ids for v in vs) / len(vs)
            same = sum(_shape(v) in shapes for v in vs) / len(vs)
            if hit >= 0.5 or same >= 0.9 and re.search(r"[A-Za-z]", vs[0]):
                m.canonical, m.source = "order_id", m.source or "value_pattern"
                m.note = (m.note + " | " if m.note else "") + f"giá trị là mã đơn bảng khác (trùng {hit:.0%}, cùng dạng {same:.0%})"
                if "payment_amount" not in canons:
                    for x in maps:
                        if x.canonical in ("order_total", "money") and x.source != "readme_col":
                            x.canonical = "payment_amount"
                            x.note = (x.note + " | " if x.note else "") + "tiền của sổ thu nợ: tiền trả lần đó"
                            break
                break


def cross_table_names(results: dict[str, list[ColumnMap]], kinds: dict[str, dict], raw: dict[str, dict]):
    """Cross-table pass: an unmapped text column whose values lie in the set of the shop's known product / customer names."""
    known: dict[str, set] = {"product_name": set(), "customer_name": set()}
    for t, maps in results.items():
        for m in maps:
            if m.canonical in known:
                known[m.canonical].update(fold(v) for v in raw[t].get(m.column, []) if isinstance(v, str))
    for canon, names in known.items():
        if len(names) < 5:
            continue
        for t, maps in results.items():
            if canon in {m.canonical for m in maps}:
                continue
            for m in maps:
                if m.canonical or m.description or kinds[t].get(m.column) != "text":
                    continue
                uniq = {fold(v) for v in raw[t].get(m.column, []) if isinstance(v, str)}
                if len(uniq) >= 5 and len(uniq & names) >= 0.9 * len(uniq):
                    m.canonical, m.source = canon, "value_rule"
                    m.note = f"{len(uniq & names)}/{len(uniq)} giá trị trùng {canon} ở bảng khác"
                    break


def _resolve_context(maps: list[ColumnMap]):
    """Generic name ('name', 'discount') -> specific name by the other columns in the table."""
    # pass 1: words in the COLUMN NAME state it clearly ("Tên khách hàng", "Đơn vị cơ bản") -> trust the column name
    for m in maps:
        if m.source == "readme_col":
            continue
        f = fold(m.column)
        if m.canonical in ("name",) + _NAME_FAMILY:
            hit = next((c for rx, c in _ENTITY_WORDS if rx.search(f)), None)
            if hit:
                m.canonical = hit
        elif m.canonical in ("unit", "base_unit") and re.search(r"co ban|\bgoc\b", f):
            m.canonical = "base_unit"
        elif m.canonical in ("unit", "base_unit") and re.search(r"quy doi|\blon\b", f):
            m.canonical = "big_unit"            # "Đơn vị quy đổi", "ĐV lớn"
        elif m.canonical == "status" and m.source != "value_rule" and re.search(
                r"thanh toan|hddt|hoa don dien tu|xuat hoa don|\b(khach|kh|ncc|tai khoan)\b", f):
            # payment / e-invoice / customer status: not the order status.
            # Do not use "TT" (payment / status / amount are all written TT); if the values are order statuses, trust the values
            m.canonical, m.note = None, "trạng thái không phải của đơn (theo tên cột)"
    # a receipt writes both "Mã phiếu" and "Mã đơn" (with a receipt date): "Mã phiếu" is the receipt code. The training dictionary has "Mã phiếu" = order code
    # (a repair-shop service ticket), so both columns become order_id and the real "Mã đơn" column is dropped during de-duplication
    orders = [m for m in maps if m.canonical == "order_id"]
    if len(orders) >= 2 and any(name_key(m.column) == "ma don" for m in orders) \
            and any(m.canonical == "payment_date" for m in maps):
        for m in orders:
            if m.source != "readme_col" and re.fullmatch(r"ma phieu|so phieu", name_key(m.column)):
                m.canonical, m.note = "payment_id", "mã phiếu thu (bảng có cả mã đơn và ngày thu)"

    # A receipt table without a slip-code column. Off by default (turn on with DATAMATE_NV3=1)
    if maps and os.environ.get("DATAMATE_NV3") and re.search(r"\bphieu ?thu\b|\bthu ?tien\b|\bpayments?\b|\breceipts?\b", name_key(maps[0].table or "")):
        canons = {m.canonical for m in maps}
        if "order_id" in canons \
                and (canons & {"money", "order_total", "payment_amount"}) \
                and (canons & {"order_date", "order_time", "payment_date"}) \
                and not (canons & {"product_id", "product_name", "qty", "qty_base", "unit_price", "line_amount"}):
            for m in maps:
                if m.source != "readme_col":
                    if m.canonical in ("order_date", "order_time"):
                        m.canonical = "payment_date"
                        m.note = "phiếu thu (theo tên bảng)"
                    elif m.canonical in ("money", "order_total"):
                        m.canonical = "payment_amount"
                        m.note = "phiếu thu (theo tên bảng)"

    # a sales table (has customer + product rows, no order code / purchase slip / supplier): an unmapped column "Số phiếu" / "Mã phiếu" / "Số chứng
    # từ" is the order code (a cash register prints a sales slip). The training dictionary has no such name for a sales table
    present = {m.canonical for m in maps}
    if ("order_id" not in present and "purchase_id" not in present and present & {"customer_name", "customer_id"}
            and present & {"qty", "line_amount", "unit_price"} and not present & {"supplier_name", "supplier_id"}):
        for m in maps:
            if m.canonical is None and re.fullmatch(r"(so|ma) (phieu|ct|chung tu)", name_key(m.column)):
                m.canonical, m.source, m.note = "order_id", "builtin", "số phiếu ở bảng bán: mã đơn"
                break
    # pass 2: table context. A product-row table = has quantity / unit price / line amount, or has product + order / date
    present = {m.canonical for m in maps}
    items = present & {"product_id", "product_name", "qty", "qty_base", "unit_price"}
    is_line = bool(present & _LINE_QTY) or bool(present & {"product_id", "product_name"}
                                                 and present & {"order_id", "purchase_id", "order_date"})
    # expense ledger (the table / file name says "chi phí", "phiếu chi", "sổ chi", or has an expense date), no product rows:
    # money is expense money, date is expense date (rulebook §5 net profit). Otherwise generic money becomes order money -> revenue
    if maps and not items and "order_id" not in present and (
            "expense_date" in present or _EXPENSE_TABLE.search(name_key(maps[0].table))):
        for m in maps:
            if m.canonical in ("money", "order_total", "payment_amount") and m.source != "readme_col":
                m.canonical = "expense_amount"
            elif m.canonical in ("order_date", "order_time", "payment_date") and m.source != "readme_col":
                m.canonical = "expense_date"
        present = {m.canonical for m in maps}
    for m in maps:
        if m.canonical == "money":     # even when the README describes it: "total of the whole slip" does not say which slip
            # generic money ("Tiền", "Tổng tiền"): cash book -> receipt money; purchase slip -> slip amount;
            # a table with product rows -> line amount; a table with orders only -> whole-order amount
            # the description says "of the whole slip / whole order" -> order-level money, even in a table with product rows
            whole = bool(m.description and re.search(r"\bca (phieu|don|hoa don)\b", fold(m.description)))
            line = bool(items) and not whole
            m.canonical = ("payment_amount" if "payment_id" in present else
                           ("line_amount" if line else "purchase_total")
                           if "purchase_id" in present and "order_id" not in present else
                           "line_amount" if line else "order_total")
        elif m.source == "readme_col":
            pass                       # keep when the README clearly describes the unit / price
        elif (m.canonical == "unit" and "conv_factor" not in present and not present & _LINE_QTY
              and (not is_line or "qty_base" in present
                   and present & {"delivery_round", "delivery_date", "driver", "plate"})
              and not re.search(r"\bnhap\b|\bban\b", fold(m.column))):
            # a unit goes with the quantity number of the table: a catalogue / stock table has no row quantity,
            # a delivery slip has only the base quantity -> "ĐVT" is the product's base unit
            m.canonical = "base_unit"
        elif m.canonical == "unit_price" and not present & ({"qty", "line_amount", "qty_base"} |
                                                            {"order_id", "purchase_id"}):
            # "Giá" in a product catalogue: the current selling price, not the unit price on a row
            m.canonical = "price"
        elif m.canonical == "price" and present & {"qty", "line_amount"} and present & {"order_id", "order_date"}:
            m.canonical = "unit_price"
        elif (m.canonical == "customer_group" and present & {"product_id", "product_name"}
              and not present & {"customer_id", "customer_name", "order_id"} and not re.search(r"\bkhach\b|\bkh\b", fold(m.column))):
            # "Nhóm" in a product catalogue (no customer, no order): product group. The training READMEs only describe "Nhóm" in a customer table
            m.canonical = "category"
    for m in maps:
        if m.canonical == "name":
            m.canonical = ("product_name" if present & {"product_id", "qty", "unit_price", "line_amount"} else
                           "customer_name" if "customer_id" in present else
                           "supplier_name" if "supplier_id" in present else
                           "staff_name" if "staff_id" in present else "name")
        elif (m.canonical == "product_name" and m.source in ("name_dict", "builtin")
              and not _ENTITY_WORDS[2][0].search(fold(m.column))
              and present & {"order_total", "purchase_total", "order_subtotal", "paid", "money"}
              and not present & {"qty", "qty_base", "unit_price", "line_amount", "product_id"}):
            # a generic text column ("Diễn giải", "Nội dung") in a slip / order header table (has a total, no product
            # rows): it is the slip's note, not a product name (e.g. phieu_nhap.DienGiai = "nhập đầu kỳ")
            m.canonical, m.note = "note", "cột chữ chung ở bảng đầu phiếu: ghi chú, không phải tên hàng"
        elif m.canonical == "discount":
            # an explicit order-level name ("Giảm giá HĐ"), or a table with no product rows -> order discount
            of_order = re.search(r"\b(hd|hoa don|don)\b", fold(m.column))
            m.canonical = ("order_discount" if of_order or not present & {"product_id", "product_name"}
                           else "line_discount")
        elif m.canonical in ("order_date", "order_time") and m.source != "readme_col":
            # "Thời gian" in a cash book is the receipt time, in a purchase slip it is the purchase date
            if "payment_id" in present:
                m.canonical = "payment_date"
            elif "purchase_id" in present and "order_id" not in present:
                m.canonical = "purchase_date"
        elif m.canonical == "status" and "purchase_id" in present and "order_id" not in present:
            m.canonical = "purchase_status"
        elif m.canonical == "status" and not present & SALES_SIGNALS:
            # e.g. customer status "Đang hoạt động": not a sales order status.
            # A hand-written sales ledger may have no order id, so consider every sign of a sales table, not only order_id
            m.canonical, m.note = None, "trạng thái không thuộc bảng bán"
        elif (m.canonical == "qty" and m.source != "readme_col" and "batch_id" in present
              and "order_id" in present and not present & {"purchase_id", "unit", "unit_price", "line_amount"}):
            # an "order - batch - quantity" table (order ledger by batch): the quantity comes from the batch in base units.
            # Does not apply to a purchase table: there the quantity is in the purchase unit.
            m.canonical = "qty_base"
        elif m.canonical == "base_unit" and m.source != "readme_col" and present & {"qty", "line_amount"} \
                and present & {"order_id", "order_date", "purchase_id"}:
            # 'DonVi' in a catalogue is the base unit, but on sales / purchase rows it is the row's unit
            m.canonical = "unit"
        elif (m.canonical in ("unit", "base_unit") and m.source != "readme_col" and "conv_factor" in present
              and "big_unit" not in present and not present & {"qty", "order_id", "purchase_id"}
              and not re.search(r"co ban|\bgoc\b", fold(m.column))
              and sum(x.canonical in ("unit", "base_unit") for x in maps) == 1):
            # a unit-conversion table (Unit | Equals | Product code): the unit that comes with the factor is the large unit
            m.canonical = "big_unit"
            present.add("big_unit")


_SOURCE_RANK = {"readme_col": 0, "readme_hint": 1, "name_dict": 2, "builtin": 2, "value_pattern": 3,
                "value_rule": 4, "llm": 5}


def _resolve_duplicates(maps: list[ColumnMap]):
    """Two columns with the same canonical name: keep the one with stronger evidence (or the earlier), the other keeps its original name."""
    groups = defaultdict(list)
    for m in maps:
        if m.canonical:
            groups[m.canonical].append(m)
    for canon, ms in groups.items():
        if len(ms) < 2:
            continue
        ms.sort(key=lambda m: _SOURCE_RANK.get(m.source, 9))
        for m in ms[1:]:
            m.note = f"trùng {canon} với cột '{ms[0].column}'"
            m.canonical = None


_SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.md")


def canonical_vocab(path: str = _SCHEMA) -> dict[str, str]:
    """Canonical name -> meaning, read from the 'Tên chuẩn | Ý nghĩa' tables of schema.md (one source for people and the LLM)."""
    out, inside = {}, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("| Tên chuẩn |"):
                inside = True
            elif inside and line.startswith("|") and not line.startswith("|---"):
                names, meaning = line.strip().strip("|").split("|", 1)
                for n in re.findall(r"`([a-z_]+)`", names):
                    out[n] = meaning.strip()
            elif not line.startswith("|"):
                inside = False
    return out


_LLM_SYSTEM = """Bạn chuẩn hoá tên cột dữ liệu bán hàng của cửa hàng nhỏ ở Việt Nam.
Với mỗi cột cần map, dựa vào tên cột, giá trị mẫu, mô tả (nếu có) và các cột khác của bảng:
1. "nghia": cột này ghi gì (tối đa 12 chữ).
2. "ten": tên chuẩn trong danh sách dưới đây CHỈ KHI nghĩa của cột trùng đúng với mô tả của tên đó,
   tức là ai hỏi về tên chuẩn đó thì cột này là câu trả lời. Cột ghi thông tin khác, gần nghĩa,
   ngưỡng / giới hạn, thông tin phụ không có trong danh sách, hoặc không chắc -> null.
   Tên sai tệ hơn null. Không dùng lại tên chuẩn mà bảng đã có.
Chỉ trả về một JSON object: {"<tên cột>": {"nghia": "...", "ten": "<tên chuẩn>" hoặc null}}.

Tên chuẩn:
"""


def llm_map_prompt(table: str, unmapped: list[dict], known: list[tuple[str, str]], vocab: dict[str, str]
                   ) -> tuple[str, str]:
    """(system, user) — shared by the pipeline and (formerly) the data-layer fine-tune data, not included in this repo."""
    system = _LLM_SYSTEM + "\n".join(f"- {k}: {v}" for k, v in vocab.items())
    lines = [f"Bảng: {table}",
             "Cột đã biết: " + (", ".join(f'"{c}" = {k}' for c, k in known) or "(chưa có)"),
             "Cột cần map:"]
    for u in unmapped:
        smp = "; ".join(str(v)[:40] for v in u["samples"]) or "(trống)"
        desc = f", mô tả: {u['description']}" if u.get("description") else ""
        lines.append(f'- "{u["column"]}" (kiểu {u["kind"]}{desc}): {smp}')
    return system, "\n".join(lines)


def llm_map(table: str, unmapped: list[dict], known: list[tuple[str, str]], vocab: dict[str, str]
            ) -> dict[str, dict]:
    """Ask the LLM for canonical names of the unmapped columns of ONE table. unmapped: column, kind, samples, description.
    Return {column: {"nghia": ..., "ten": canonical name or None}} as the model returns it; the caller does the validation."""
    from . import llm

    system, user = llm_map_prompt(table, unmapped, known, vocab)
    text = llm.chat(system, user, max_tokens=80 + 60 * len(unmapped))
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in out.items() if isinstance(v, dict)} if isinstance(out, dict) else {}


# The LLM may only change the role of a table the rules have NOT yet recognised (other). A table the rules recognised (sales, purchase, customer...) stays
# as is: "Slg" -> conv_factor once turned an invoice-detail table into a conversion table (losing the whole sales table), "Tổng điểm"
# -> stock turned a customer list into a stock table. An 'other' table becomes a sales / purchase table (added to revenue, purchase money)
# only with enough order signs (>= 3 core columns) and the table name is not an expense ledger ("Ngày chi" -> order_date once turned an expense
# ledger into an order table).
_ROLE_GROUP = {"sales_lines": "ban", "orders": "ban", "order_info": "ban", "sales_batches": "ban",
               "deliveries": "ban", "payments": "ban",
               "purchases": "nhap", "purchase_lines": "nhap", "batches": "nhap"}
_PEOPLE = {"customers", "staff", "suppliers"}
_UNIT_NAMES = {"unit", "base_unit", "big_unit"}
_TX_CORE = {"order_id", "purchase_id", "product_name", "product_id", "qty", "qty_base", "unit_price", "line_amount",
            "order_total", "purchase_total", "order_date", "purchase_date"}


def _role_change_ok(old: str, new: str, cols: set[str] | None = None, table: str = "") -> bool:
    if old == new:
        return True
    g0, g1 = _ROLE_GROUP.get(old), _ROLE_GROUP.get(new)
    if g0 in ("ban", "nhap"):
        return g0 == g1          # refinement within the group (order info + quantity -> product rows); the sales table is not lost
    if old in _PEOPLE:
        return False
    if g1 in ("ban", "nhap"):    # becoming a sales / purchase table (order code + product name that looks like a catalogue): enough order signs
        return len((cols or set()) & _TX_CORE) >= 3 and not _EXPENSE_TABLE.search(fold(table))
    return old == "other"


_ORDER_SIGNALS = {"order_id", "qty", "qty_base", "line_amount", "order_total", "order_subtotal"}


def _llm_family(props: list[tuple[ColumnMap, str]], used: set[str], sales: bool) -> None:
    """The LLM is only trustworthy at the FAMILY of a concept; the specific name inside the family is decided by the table's context (like the name dictionary).
    - quantity: a table with no row unit -> quantity in the base unit (rulebook §10)
    - moment: a sales-group table (sales: not a catalogue or stock card) with no order date yet -> the first moment
      column is the order date; the creation moment of the order itself is the order time"""
    def present():
        return used | {c for _, c in props}
    for i, (m, c) in enumerate(props):
        if c in ("qty", "qty_returned") and "unit" not in present():
            props[i] = (m, c + "_base")
    for i, (m, c) in enumerate(props):
        cur = present() - {c}
        if not sales or not cur & _ORDER_SIGNALS:
            continue
        if c in ("order_time", "created_date") and "order_date" not in cur:
            props[i] = (m, "order_date")
        elif c == "created_date" and "order_id" in cur and "order_time" not in cur:
            props[i] = (m, "order_time")


_CLASS_NAMES = {"category", "customer_group", "channel", "payment_method"}


def _llm_values_ok(c: str, vals: list, units: set[str]) -> bool:
    """Values must fit the name the LLM chose: a unit column is all units of measure; a code column does not carry another code's prefix;
    a classification column (product group, channel...) must have at least 2 values (one value for every row classifies nothing)."""
    if c in _CLASS_NAMES and len(vals) >= 10 and len({fold(str(v)) for v in vals}) < 2:
        return False
    if c in _UNIT_NAMES:
        uniq = {fold(v) for v in vals if isinstance(v, str)}
        return bool(uniq) and len(uniq & {fold(u) for u in units}) >= 0.8 * len(uniq)
    if c.endswith("_id") and (p := _value_prefix(vals)) and CODE_PREFIX.get(p, c) != c:
        return False
    return True


def _resolve_qty_pair(props: list[tuple[ColumnMap, str]], t: TableInput) -> None:
    """Two columns that are both quantities (written on the row / converted to the base unit, the LLM easily collapses them to one name): the column that is always >= the other
    is qty_base (conversion factor >= 1). When they cannot be told apart, leave as is."""
    idx = [i for i, (_, c) in enumerate(props) if c in ("qty", "qty_base")]
    if len(idx) != 2:
        return
    a, b = (props[i][0] for i in idx)
    pairs = [(parse_number(x), parse_number(y)) for x, y in zip(t.parsed.get(a.column, []), t.parsed.get(b.column, []))]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 5:
        return
    if all(x <= y for x, y in pairs):
        props[idx[0]], props[idx[1]] = (a, "qty"), (b, "qty_base")
    elif all(x >= y for x, y in pairs):
        props[idx[0]], props[idx[1]] = (a, "qty_base"), (b, "qty")


def _resolve_llm_duplicates(props: list[tuple[ColumnMap, str]], t: TableInput) -> list[tuple[ColumnMap, str]]:
    """Several columns assigned the same canonical name by the LLM: do not take the earlier column (e.g. a product-group column with 9 values
    took the product-name column's name away from a dish-name column with 50 values). Name / code (*_name, *_id): the column with the most distinct values
    (a product name is finer than a group name); other canonical names: drop all, when unsure do not map."""
    by: dict[str, list] = defaultdict(list)
    for m, c in props:
        by[c].append(m)
    out = []
    for m, c in props:
        group = by[c]
        if len(group) == 1:
            out.append((m, c))
        elif c.endswith(("_name", "_id")):
            card = {g.column: len({fold(str(v)) for v in t.raw.get(g.column, []) if v is not None}) for g in group}
            best = max(group, key=lambda g: card[g.column])
            if m is best and sorted(card.values())[-1] > sorted(card.values())[-2]:
                out.append((m, c))
    return out


HDR = os.environ.get("DATAMATE_HDR", "1") != "0"


def _header_ok(t: TableInput, maps: list[ColumnMap], cand: list[tuple[ColumnMap, str]]) -> bool:
    """An order-header ledger (one order per row: order id + order date, product rows in another table): the order id almost never repeats."""
    col = next((m.column for m, c in cand if c == "order_id"), None) or next(
        (m.column for m in maps if m.canonical == "order_id"), None)
    # the order date may already have been assigned earlier by a name rule (in maps, not in cand)
    if not col or not (any(c == "order_date" for _, c in cand) or any(m.canonical == "order_date" for m in maps)):
        return False
    vals = [fold(str(v)) for v in t.raw.get(col, []) if v is not None and str(v).strip()]
    return len(vals) >= 20 and len(set(vals)) >= 0.9 * len(vals)


def link_line_ids(tables: list[TableInput], results: dict[str, list[ColumnMap]]) -> None:
    """A slip code the LLM assigned as purchase_id whose values lie in the order ids of another table (details of an order-header ledger,
    joined by slip code) and in no purchase slip code -> order_id."""
    if not HDR:
        return
    raw = {t.name: t.raw for t in tables}

    def vals(name, canon):
        return {fold(str(v)) for m in results[name] if m.canonical == canon
                for v in raw[name].get(m.column, []) if v is not None and str(v).strip()}
    orders = {n: vals(n, "order_id") for n in results}
    buys = {n: vals(n, "purchase_id") for n in results}
    for n, maps in results.items():
        if "order_id" in {m.canonical for m in maps}:
            continue
        for m in maps:
            if m.canonical != "purchase_id" or m.source != "llm":
                continue
            mine = buys[n]
            if len(mine) < 20 or any(len(mine & b) for k, b in buys.items() if k != n):
                continue
            if any(k != n and len(mine & o) >= 0.9 * len(mine) for k, o in orders.items()):
                m.canonical, m.note = "order_id", m.note + " | mã nằm trong mã đơn bảng khác (HDR)"


def _llm_pass(tables: list[TableInput], results: dict[str, list[ColumnMap]], units: set[str]):
    """Final pass: columns the rules left empty (no evidence, or the README describes it but no rule yet) -> LLM.
    Accept only names that exist in the schema, with the right type and values, not already in the table, and that do not move the table to another entity
    (_role_change_ok). The meaning the LLM states is written to
    note even when not mapped, so the query side can read an odd column."""
    from .build import table_role

    vocab = canonical_vocab()
    for t in tables:
        maps = results[t.name]
        todo = [m for m in maps if m.canonical is None and m.source is None
                and m.note in ("", "mô tả README chưa có luật") and t.kinds.get(m.column) != "empty"]
        if not todo:
            continue
        used = {m.canonical for m in maps if m.canonical}
        cols = [{"column": m.column, "kind": t.kinds.get(m.column), "description": m.description,
                 "samples": _distinct(t.raw.get(m.column, []))[:8]} for m in todo]
        try:
            res = llm_map(t.name, cols, [(m.column, m.canonical) for m in maps if m.canonical], vocab)
        except Exception as e:          # network / server error: keep the rule result
            for m in todo:
                m.note = f"LLM lỗi: {type(e).__name__}"
            continue
        props = []
        for m in todo:
            r = res.get(m.column, {})
            c, meaning = r.get("ten"), str(r.get("nghia") or "")[:80]
            if meaning:
                m.note = f"LLM: {meaning}"
            if c in vocab and _llm_values_ok(c, [v for v in t.raw.get(m.column, []) if v is not None], units):
                props.append((m, c))
        # the LLM shows this is a product-row table: a unit column that the value rules assigned base_unit is the row's unit
        if {c for _, c in props} & {"qty", "qty_base", "line_amount"} and "base_unit" in used and "unit" not in used:
            for m in maps:
                if m.canonical == "base_unit" and m.source == "value_rule":
                    m.canonical = "unit"
                    used = (used - {"base_unit"}) | {"unit"}
        r0 = table_role(used)
        # a table with order ids but guessed by the rules as customer / unknown (an order-header ledger without a date yet) is still a sales-group table
        hdr_sales = HDR and r0 in ("customers", "other") and "order_id" in used | {c for _, c in props}
        _llm_family(props, used, _ROLE_GROUP.get(r0) == "ban" or hdr_sales)
        _resolve_qty_pair(props, t)
        props = _resolve_llm_duplicates(props, t)
        cand, seen = [], set(used)
        for m, c in props:
            if c not in seen and _kind_ok(c, t.kinds.get(m.column, "text")):
                cand.append((m, c))
                seen.add(c)
        # judge the role over the WHOLE proposed column set (a sales table whose columns are all renamed only shows enough signs when combined);
        # if that fails, accept column by column those that do not change the table's role
        whole = _role_change_ok(r0, table_role(seen), seen, t.name) or (
            hdr_sales and table_role(seen) == "orders" and _header_ok(t, maps, cand))
        for m, c in cand:
            if whole or _role_change_ok(r0, table_role(used | {c}), used | {c}, t.name):
                m.canonical, m.source = c, "llm"
                used.add(c)
