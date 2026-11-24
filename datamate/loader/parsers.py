"""Parse cell values: numbers / money, dates, order status, walk-in customers, names with a phone number.

Every function takes a raw cell (str / int / float / datetime / None) and returns None when it
cannot parse, without guessing. The forms below all come from real data.

Numbers, money (rule: after a , or . exactly 3 digits always => thousands separator;
a decimal separator is seen only with 'tr' or a '.0000' tail):
    "560,000"  "25.000đ"  "1.039.500đ"  "25,000.0000"  "129 000"  "-49.000đ"
    "25k"  "1168k"  "1tr440"  "15tr738"  "16,56tr"  "3,3225tr"  "1tr"  "3 chai"
Dates:
    Excel datetime/date, Excel serial number 45689.0, "01/02/2025",
    "2025-01-01 06:03:59", "1 thg 6 2025", "1/1" (no year -> take the sheet's year)
"""
from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata

# text utilities


def fold(s: str) -> str:
    """Lowercase, strip accents, đ -> d, collapse whitespace: 'Đã Hủy  đơn' -> 'da huy don'."""
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


# numbers, money

_ERROR_CELLS = {"#value!", "#n/a", "#ref!", "#div/0!", "#name?", "#num!", "#null!", "-", "--", "—", "n/a"}
_SCALE = {"k": 1e3, "nghìn": 1e3, "ngàn": 1e3, "tr": 1e6, "triệu": 1e6, "tỷ": 1e9, "tỉ": 1e9}
_NUM_RE = re.compile(
    r"(?P<sign>[-+])?\s*"
    r"(?P<num>\d[\d.,\s]*?)\s*"
    r"(?:(?P<scale>k|nghìn|ngàn|tr|triệu|tỷ|tỉ)\s*(?P<tail>\d+)?)?\s*"
    r"(?P<cur>đ|vnđ|vnd|đồng)?"
    # unit: the first word starts with a letter; later words may have numbers ("bao 25kg", "hộp 30", "xe 4m3")
    r"(?:\s+(?P<unit>[^\W\d_][\w/]*(?:\s[\w/]+){0,2}))?",
    re.IGNORECASE,
)


def _to_number(x: float):
    """a whole float -> int (VND money, quantities); otherwise keep the float."""
    x = round(x, 6)
    return int(x) if float(x).is_integer() else x


def _plain_number(s: str, decimal_if_single_sep: bool) -> float | None:
    s = re.sub(r"\s+", "", s)
    seps = re.findall(r"[.,]", s)
    if not seps:
        return float(s)
    if "," in s and "." in s:
        # form "25,000.0000": the separator that appears last is the decimal separator
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
        return float(s.replace(thou, "").replace(dec, "."))
    sep = seps[0]
    parts = s.split(sep)
    if len(parts) == 2 and decimal_if_single_sep:
        return float(parts[0] + "." + parts[1])
    if all(len(p) == 3 for p in parts[1:]) and parts[0] and parts[0].lstrip("+-") != "0":
        return float("".join(parts))          # thousands separator ("0.500" is 0.5: a thousands group does not start with 0)
    if len(parts) == 2:
        return float(parts[0] + "." + parts[1])  # "1,5", "0.0000"
    return None                                 # messy "1.2.3": do not guess


def parse_number_unit(v) -> tuple[float | int | None, str | None]:
    """Return (value, accompanying unit text if any). E.g. "3 chai" -> (3, "chai")."""
    if v is None or isinstance(v, bool):
        return None, None
    if isinstance(v, (int, float)):
        return (None, None) if isinstance(v, float) and math.isnan(v) else (_to_number(float(v)), None)
    if isinstance(v, (dt.date, dt.datetime)):
        return None, None
    s = str(v).strip()
    if s.lower() in _ERROR_CELLS or not s:
        return None, None
    # currency symbols: "45,000₫", "VND 45,000"
    s = re.sub(r"^(?:vnd|vnđ)\s*", "", s.replace("₫", "đ"), flags=re.I)
    neg = False
    if s.startswith("(") and s.endswith(")"):    # accounting: (1,000) is a negative number
        neg, s = True, s[1:-1].strip()
    if s.endswith("%"):
        # "10%", "-5 %": keep the percentage number, unit "%" (build marks which cells are percentages)
        val, unit = parse_number_unit(s[:-1].strip())
        if val is None or unit:
            return None, None
        return (-val if neg else val), "%"
    m = _NUM_RE.fullmatch(s)
    # a unit is always written in lowercase ("chai", "bao 25kg"); "452 Tân Kỳ Tân Quý" is an address, not a number
    if not m or (m["unit"] and not m["unit"][0].islower() and not _unit_word(m["unit"])):
        return None, None
    scale = (m["scale"] or "").lower()
    if scale and m["tail"]:
        # "1tr440" = 1.440 million; "15tr738" = 15.738 million
        base = float(re.sub(r"[.,\s]", "", m["num"]) + "." + m["tail"])
    else:
        base = _plain_number(m["num"], decimal_if_single_sep=scale in ("tr", "triệu", "tỷ", "tỉ"))
        if base is None:
            return None, None
    val = base * _SCALE.get(scale, 1)
    if m["sign"] == "-" or neg:
        val = -val
    return _to_number(val), m["unit"]


def parse_number(v):
    return parse_number_unit(v)[0]


def looks_like_code(v) -> bool:
    """An all-digit string that is a code: phone / tax id with a leading 0, barcode, IMEI, citizen id (>= 12 digits)."""
    if not isinstance(v, str):
        return False
    s = v.replace(" ", "")
    return s.isdigit() and ((len(s) > 1 and s[0] == "0") or len(s) >= 12)


# dates

_EXCEL_EPOCH = dt.datetime(1899, 12, 30)
_TIME = r"(?:[ T,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?"
_D_ISO = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})" + _TIME)    # year first: 2025-03-05, 2025/03/05
_D_DMY = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})" + _TIME)
_D_THG = re.compile(r"(\d{1,2})\s*(?:thg|tháng)\s*(\d{1,2})\s*,?\s*(?:năm\s*)?(\d{4})" + _TIME, re.I)
_D_NGAY = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*(?:năm\s*)?(\d{4})" + _TIME, re.I)
_D_NOYEAR = re.compile(r"(\d{1,2})/(\d{1,2})")
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _mk(y, mo, d, hh=None, mi=None, ss=None) -> dt.datetime | None:
    try:
        y = int(y)
        y = y + 2000 if y < 100 else y
        return dt.datetime(y, int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0))
    except (ValueError, TypeError):
        return None


def excel_serial(x: float) -> dt.datetime | None:
    """Excel serial number (45689 = 2025-02-01). Only 1954..2119 is accepted, for safety."""
    if 20000 <= x <= 80000:
        # round to the second: a fractional part stored with 6 digits comes out as 19:25:59.98 instead of 19:26 (a question by time slot would put it in the earlier hour)
        return _EXCEL_EPOCH + dt.timedelta(seconds=round(float(x) * 86400))
    return None


def parse_date(v, default_year: int | None = None, allow_serial: bool = False,
               month_first: bool = False) -> dt.datetime | None:
    """Cell -> datetime (hour = 0 when there is no time). Default day before month (dd/mm)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, (int, float)):
        return excel_serial(v) if allow_serial and not math.isnan(v) else None
    s = str(v).strip()
    if m := _D_ISO.fullmatch(s):
        return _mk(m[1], m[2], m[3], m[4], m[5], m[6])
    if m := _D_DMY.fullmatch(s):
        d, mo = (m[2], m[1]) if month_first else (m[1], m[2])
        # a column mixing two formats: if reading by the column's format gives an invalid date ("12/31/2025"), read it swapped
        return _mk(m[3], mo, d, m[4], m[5], m[6]) or _mk(m[3], d, mo, m[4], m[5], m[6])
    if m := (_D_THG.fullmatch(s) or _D_NGAY.fullmatch(s)):
        return _mk(m[3], m[2], m[1], m[4], m[5], m[6])
    if default_year and (m := _D_NOYEAR.fullmatch(s)):
        d, mo = (m[2], m[1]) if month_first else (m[1], m[2])
        return _mk(default_year, mo, d) or _mk(default_year, d, mo)
    if allow_serial and re.fullmatch(r"\d{5}(\.\d+)?", s):     # the fractional part = time of day ("45916.809722")
        return excel_serial(float(s))
    return None


def year_from_text(*texts) -> int | None:
    """Year in a sheet / file name: 'Tháng 01-2025' -> 2025."""
    for t in texts:
        if t and (m := YEAR_RE.search(str(t))):
            return int(m[1])
    return None


# order status

# ASCII codes so SQL is not troubled by "huỷ" / "hủy" (two ways of placing the accent).
STATUS_DONE, STATUS_CANCEL, STATUS_RETURN, STATUS_SHIPPING = "hoan_thanh", "huy", "tra_hang", "dang_giao"


EXTRA_STATUS: dict[str, str] = {}


_UNIT_WORDS = {"cai", "chiec", "ly", "coc", "chai", "lon", "hop", "goi", "kg", "g", "thung", "loc", "bao", "cay", "m",
               "m2", "m3", "binh", "cuon", "vien", "vi", "ong", "lo", "tuyp", "dia", "phan", "to", "noi", "con", "lan",
               "bo", "qua", "tam", "tan", "xe", "bich", "tui", "doi", "lit", "ml", "suat", "bat", "chen", "set", "may",
               "mieng", "hu", "can", "khay", "ve", "gio", "ke", "cap", "kien", "sot", "lang", "hu", "vi", "tep", "cuc"}


def _unit_word(unit: str) -> bool:
    """'Bó', 'Chai', 'Lít', 'Thùng 24' are units; '452 Tân Kỳ Tân Quý' (an address) is not: a unit word is capitalised at the start,
    the rest (if any) is lowercase or a number."""
    w = unit.split()
    return 1 <= len(w) <= 2 and fold(w[0]) in _UNIT_WORDS and all(x.islower() or x.isdigit() for x in w[1:])


def norm_status(v) -> str | None:
    """Order status -> one of the 4 codes of rulebook §8, None if not recognised.

    The in-delivery group and 'Không giao được' are inferred from the distribution in the training data
    (per shop: the big group = completed, ~1-3% = cancelled, small = returned, a few dozen = in delivery).
    """
    if v is None:
        return None
    f = fold(v)
    if EXTRA_STATUS and f.strip() in EXTRA_STATUS:      # the self-repair loop (loader/tu_sua.py) defines unusual values
        return EXTRA_STATUS[f.strip()]
    # cancelled: including failed delivery / fake orders ("không giao được" is cancelled despite the word "không")
    # "lỗi" / "bị lỗi" alone is a failed transaction, no money collected (a failed charging session...); "trả hàng - lỗi" is still a return
    # the customer no longer wants it ("ko lấy nữa", "không lấy") is cancelled
    if re.search(r"\bhuy\b|khong giao duoc|giao (that bai|khong thanh cong)|^that bai$|\bbom\b|cancel|fail|^(bi )?loi$"
                 r"|\b(ko|k|khong) lay( nua)?\b|\bbo don\b|\bkhach bo\b", f) \
            and not re.search(r"\b(khong|chua) huy\b", f):
        return STATUS_CANCEL
    # returned GOODS. The word "trả" alone is not enough: paying, instalments, prepaying, paying later are about payment
    # "kiểm tra lại" (agreed to check again) is not a return
    if re.search(r"tra (lai )?hang|hang tra (lai|ve)|(?<!kiem )\btra lai\b|\btra ve\b|mang (ve )?tra|(khach|\bkh) tra\s*$|doi tra|"
                 r"hoan (hang|tra|tien)|return|refund", f) \
            and not re.search(r"tra (tien|gop|truoc|sau|no|du|het|coc)", f):
        return STATUS_RETURN
    neg = re.search(r"\b(khong|chua)\b", f)
    # the customer stands / sits waiting to pick up on the spot: not an order in delivery (a marketplace's "chờ lấy hàng" is still in delivery)
    if re.search(r"\bdang\b|cho (giao|lay|xac nhan|xu ly)|chua giao|shipping|ship|van chuyen|delivering|pending|"
                 r"processing", f) and not re.search(r"\b(khach|kh) (dung |ngoi )?cho\b", f):
        return STATUS_SHIPPING
    # negation ("chưa hoàn thành", "không thành công"): do not conclude completed, leave it to the later layer (LLM)
    if not neg and re.search(r"hoan thanh|hoan tat|\bxong\b|da lay|thanh cong|^ok$|complete|da giao|giao roi|lay roi|done|success|"
                             r"delivered", f):
        return STATUS_DONE
    return None


# customers

_WALK_IN = {"khach le", "kl", "le", "khach vang lai", "vang lai", "khach", "walk-in", "walk in", "guest", "k/l"}


_WALK_IN_PREFIX = re.compile(r"^(khach le|khach vang lai|vang lai|kl|k/l|walk[- ]?in|guest)(?=$|\s*[-–:(\[#.,/]|\s*\d)")


def is_walk_in(v) -> bool:
    """A customer-name cell clearly stating a walk-in customer, even with a suffix: 'Khách lẻ - 0901...', 'Khách lẻ 1', 'Khách vãng lai (A Tú)'.
    An empty cell does NOT count here (it may be a merged row-group cell)."""
    if v is None:
        return False
    f = fold(v).strip(" .")
    if f == "le":       # with accents stripped, "Lê" (a surname) is also "le": compare with accents
        return str(v).strip(" .").lower() == "lẻ"
    return f in _WALK_IN or bool(_WALK_IN_PREFIX.match(f))


_PHONE_RE = re.compile(r"[\s\-–(]*\(?((?:\+?84|0)(?:[\s.]?\d){8,10})\)?\s*$")


def split_name_phone(v) -> tuple[str | None, str | None]:
    """'Đào Dương Giang - 0912 365 003' -> ('Đào Dương Giang', '0912365003')."""
    if v is None:
        return None, None
    s = str(v).strip()
    m = _PHONE_RE.search(s)
    if not m or m.start() == 0:
        return s, None
    phone = re.sub(r"\D", "", m[1])
    if phone.startswith("84"):
        phone = "0" + phone[2:]
    return s[:m.start()].strip(" -–,"), phone


# bool

_TRUE = {"true", "1", "co", "x", "yes", "dung", "v"}
_FALSE = {"false", "0", "khong", "no", "sai"}


def parse_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    f = fold(v)
    if f in ("1.0",):
        return True
    if f in ("0.0",):
        return False
    return True if f in _TRUE else False if f in _FALSE else None


# applied to a whole column

def split_name(col) -> str:
    """Column name into words, keeping accents: 'TênSảnPhẩm' / 'ten_san_pham' -> 'tên sản phẩm' / 'ten san pham'."""
    s = str(col).replace("_", " ")
    # split CamelCase: insert a space before an UPPERCASE letter that follows a lowercase letter / digit (isupper is correct also for accented letters)
    out = [c if not (i and c.isupper() and (s[i - 1].islower() or s[i - 1].isdigit())) else " " + c
           for i, c in enumerate(s)]
    return re.sub(r"\s+", " ", "".join(out).lower()).strip()


_DATE_HINT = re.compile(r"ngay|thoi gian|thoi diem|date|time|\bhan\b|hsd|han dung|han su dung|sinh nhat|\bgio\b|ngay lap")
_DMY_PARTS = re.compile(r"(\d{1,2})[/.\-](\d{1,2})(?:[/.\-]\d{2,4})?")


def _month_first(values) -> bool:
    """Is a date column written month first (mm/dd)? True only with evidence: the first part <= 12 and the second part > 12."""
    first_big = second_big = False
    for v in values:
        if isinstance(v, str) and (m := _DMY_PARTS.match(v)) and not _D_ISO.match(v):
            first_big |= int(m[1]) > 12
            second_big |= int(m[2]) > 12
    return second_big and not first_big


def _as_text(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, dt.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v)


def parse_column(name: str, values: list, years: list[int | None] | None = None) -> dict:
    """Infer the type and parse a column.

    Return a dict: kind ('number' | 'datetime' | 'bool' | 'text' | 'empty'), values (parsed),
    n (number of cells with a value), n_fail, fail_examples, units (unit text seen in number cells),
    month_first, had_text (the original column has string-form cells -> keep a _raw copy).
    """
    present = [v for v in values if v is not None]
    out = {"kind": "empty", "values": values, "n": len(present), "n_fail": 0,
           "fail_examples": [], "units": {}, "month_first": False,
           "had_text": any(isinstance(v, str) for v in present)}
    if not present:
        return out
    years = years or [None] * len(values)

    def finish(kind, parsed):
        fails = [v for v, p in zip(values, parsed) if v is not None and p is None]
        out.update(kind=kind, values=parsed, n_fail=len(fails),
                   fail_examples=list(dict.fromkeys(map(str, fails)))[:5])
        return out

    # bool: all True/False
    if all(isinstance(v, bool) or (isinstance(v, str) and fold(v) in ("true", "false")) for v in present):
        return finish("bool", [parse_bool(v) for v in values])

    # date: date-like strings / datetime; a serial number is accepted only when the column name suggests a date
    # column names that denote money ("Giá ngày lễ", "Đơn giá giờ", "Tiền ngày công", "Hạn mức"): a number 20000-80000 is money, not
    # an Excel date serial number
    _ten = fold(split_name(name))
    hint = bool(_DATE_HINT.search(_ten)) and not re.search(r"\bgia\b|\btien\b|\bmuc\b|\bphi\b|\bluong\b|\bcong\b", _ten)
    mf = _month_first(present)
    dates = [parse_date(v, y, allow_serial=hint, month_first=mf) for v, y in zip(values, years)]
    ok = sum(d is not None for d in dates)
    non_numeric = any(not isinstance(v, (int, float)) for v in present)
    if ok >= 0.9 * len(present) and (hint or non_numeric):
        out["month_first"] = mf
        return finish("datetime", dates)

    # number: do not accept a column whose codes are numeric (phone 0..., barcode, IMEI)
    if not any(looks_like_code(v) for v in present):
        pairs = [parse_number_unit(v) for v in values]
        ok = sum(p[0] is not None for p, v in zip(pairs, values) if v is not None)
        units = {}
        for _, unit in pairs:
            if unit:
                units[unit] = units.get(unit, 0) + 1
        # too many different "units" => the text after the number is content (address, name), not a unit
        if ok >= 0.95 * len(present) and len(units) <= 30:
            out["units"] = units
            return finish("number", [p[0] for p in pairs])

    out.update(kind="text", values=[_as_text(v) for v in values])
    return out
