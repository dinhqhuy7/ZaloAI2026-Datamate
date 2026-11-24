"""Unrecognised rule labels -> common code, via LLM (rulebook §8: 4 order statuses; §1, §3: payment, walk-in customer).

Rules run first (parsers.norm_status, build._PAYMENT_NORM, parsers.is_walk_in...). The rest is grouped by
DISTINCT VALUE with its row count (a 50k-row column usually has only a few labels) and sent to the LLM along with the code definitions.
The LLM returns a code or 'khong_ro'; the code only accepts valid codes. When unsure leave it empty: later layers see it as unclear
and do not silently treat it as completed / cash. ENABLED = False (default): every function returns empty, as if only rules existed.
"""
from __future__ import annotations

import json
import re

from . import llm
from .parsers import fold

ENABLED = False
UNKNOWN = "khong_ro"
MAX_VALUES = 60
RULE_LINES: list[str] = []      # the rulebook's rule lines (set by build_shop); a label mentioned in a rule is sent together with that line


def set_rules(texts) -> None:
    """Split the rulebook into lines / table cells to find the line that mentions a label."""
    out = []
    for t in texts or []:
        for line in str(t).splitlines():
            for cell in (line.split("|") if line.count("|") >= 2 else [line]):
                cell = cell.strip(" -*")
                if len(cell) >= 8:
                    out.append(cell)
    RULE_LINES[:] = out


def _label_key(v: str) -> str:
    """Main part of a label to look up in the rules: brackets, numbers and punctuation removed; at least 2 letters."""
    v = re.sub(r"\(.*?\)|\[.*?\]", " ", str(v))
    v = re.sub(r"[^\w]+|\d+", " ", fold(v)).strip()
    return v if len(v.split()) >= 2 else ""


def rules_for(values) -> list[str]:
    """Rule lines (at most 6, each at most 300 characters) that contain the main part of a label."""
    keys = {k for k in (_label_key(v) for v in values) if k}
    if not keys or not RULE_LINES:
        return []
    hits = []
    for line in RULE_LINES:
        f = " " + re.sub(r"[^\w]+", " ", fold(line)) + " "
        if any(f" {k} " in f for k in keys) and line[:300] not in hits:
            hits.append(line[:300])
        if len(hits) >= 6:
            break
    return hits

_STATUS = {
    "hoan_thanh": "đơn đã xong: đã giao, khách đã nhận, đã thanh toán, đã chốt xong",
    "huy": "đơn không thành: huỷ, khách bom hàng, giao không được, khách không lấy",
    "tra_hang": "khách trả lại hàng sau khi mua (cả đơn hoặc một phần), hoàn hàng, đổi trả",
    "dang_giao": "đơn chưa xong: đang giao, đang vận chuyển, chờ giao, đã gửi đi mà khách chưa nhận",
}
CODESETS = {
    "status": _STATUS,
    # free-text note: classify only when the note clearly states the order's final status, otherwise it is not mentioned
    "note_status": {**{k: v for k, v in _STATUS.items() if k != "hoan_thanh"},
                    "khong_nhac": "ghi chú không nói đơn bị huỷ, trả hàng hay đang giao (ghi chú việc khác)"},
    "payment": {
        "tien_mat": "tiền mặt",
        "chuyen_khoan": "chuyển khoản ngân hàng, quét mã QR ngân hàng",
        "the": "quẹt thẻ (ATM, Visa, Master, máy POS)",
        "vi": "ví điện tử (MoMo, ZaloPay, VNPay, ShopeePay...)",
        "cod": "giao hàng thu tiền (COD, thu hộ, ship thu tiền)",
        "ghi_no": "ghi nợ, mua chịu, trả sau",
    },
    "channel": {
        "tai_quay": "khách mua trực tiếp tại cửa hàng",
        "dien_thoai": "đặt qua gọi điện thoại, hotline",
        "zalo": "đặt qua Zalo",
        "facebook": "đặt qua Facebook, Messenger, fanpage",
        "shopee": "sàn Shopee",
        "lazada": "sàn Lazada",
        "tiktok": "TikTok Shop",
        "grab": "app giao đồ ăn (GrabFood, ShopeeFood, Baemin, Be...)",
        "website": "website, đặt online của shop",
    },
}

_SYSTEM = """Bạn xếp các nhãn ghi trong sổ bán hàng của cửa hàng nhỏ ở Việt Nam vào bộ mã cho trước.
Mỗi nhãn chọn đúng một mã theo NGHĨA của nhãn (nhãn có thể viết tắt, không dấu, tiếng Anh, văn nói).
Nhãn không thuộc mã nào, hoặc không chắc, ghi "khong_ro".
Chỉ trả về một JSON object: {"<nhãn y như đề>": "<mã>"}."""


def _ask(system: str, user: str, n: int) -> dict:
    text = llm.chat(system, user, max_tokens=60 + 25 * n)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    return out if isinstance(out, dict) else {}


def _match_keys(out: dict, values: list[str]) -> dict:
    """The key the LLM returns may differ from the original label in accents / case: match it back by fold."""
    by_fold = {fold(v): v for v in values}
    res = {}
    for k, v in out.items():
        orig = k if k in values else by_fold.get(fold(str(k)))
        if orig is not None:
            res[orig] = v
    return res


def label_prompt(kind: str, counts: dict[str, int], column: str = "") -> tuple[str, str, list]:
    """(system, user, labels sent) — shared by the pipeline and (formerly) the data-layer fine-tune data, not included in this repo."""
    codes = CODESETS[kind]
    top = sorted(counts.items(), key=lambda x: -x[1])[:MAX_VALUES]
    user = "\n".join([f"Cột: {column}" if column else "Cột nhãn",
                      "Mã:", *[f"- {c}: {d}" for c, d in codes.items()],
                      "Nhãn (số dòng):", *[f'- "{v}" ({n})' for v, n in top]]
                     + (["LUẬT CỦA CỬA HÀNG có nhắc tới nhãn (luật nói nghĩa của nhãn thì theo luật; luật nói nhãn đó "
                         "không phải bán / trả / huỷ thì không xếp vào mã ấy):", *[f"- {r}" for r in rl]]
                        if (rl := rules_for([v for v, _ in top])) else []))
    return _SYSTEM, user, top


def classify(kind: str, counts: dict[str, int], column: str = "", total: int | None = None) -> dict[str, str]:
    """{label: row count} (labels the rules do not recognise) -> {label: code}. Keep only valid codes; drop when unsure."""
    if not ENABLED or not counts or not llm.available():
        return {}
    codes = CODESETS[kind]
    system, user, top = label_prompt(kind, counts, column)
    try:
        out = _match_keys(_ask(system, user, len(top)), [v for v, _ in top])
    except Exception:                 # network / server error: keep the rule result
        return {}
    res = {v: c for v, c in out.items() if c in codes}
    if kind == "status" and total:
        # a label covering over half of the rows that the LLM classifies as cancelled / returned / in delivery: not trusted (the shop mainly sells)
        res = {v: c for v, c in res.items() if c == "hoan_thanh" or counts.get(v, 0) <= 0.5 * total}
    return res


_WALK_SYSTEM = """Các giá trị dưới đây xuất hiện nhiều nhất ở cột tên khách của sổ bán hàng một cửa hàng ở Việt Nam.
Giá trị nào là NHÃN chỉ khách lẻ / khách vãng lai / khách không xưng tên (không phải tên một người, một cơ quan,
một khách quen cụ thể)? Không chắc thì false.
Chỉ trả về một JSON object: {"<giá trị y như đề>": true hoặc false}."""


def walk_in_prompt(counts: dict[str, int], total: int, top_n: int = 12) -> tuple[str, str, list]:
    """(system, user, names sent): the most frequent names (from 2 rows)."""
    top = [(v, n) for v, n in sorted(counts.items(), key=lambda x: -x[1])[:top_n] if n >= 2]
    user = "\n".join([f"Tổng số dòng: {total}", "Giá trị (số dòng):", *[f'- "{v}" ({n})' for v, n in top]])
    return _WALK_SYSTEM, user, top


def walk_in_markers(counts: dict[str, int], total: int, top_n: int = 12) -> set[str]:
    """The shop's own walk-in customer labels among the most frequent names (the rule parsers.is_walk_in does not know them yet)."""
    if not ENABLED or not counts or not llm.available():
        return set()
    system, user, top = walk_in_prompt(counts, total, top_n)
    if not top:
        return set()
    try:
        out = _match_keys(_ask(system, user, len(top)), [v for v, _ in top])
    except Exception:
        return set()
    return {v for v, flag in out.items() if flag is True}
