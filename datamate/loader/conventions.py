"""Shop conventions written in the README (opening paragraph, "Quy ước" section) -> structured configuration, via LLM.

A README states fixed rules that cannot all be anticipated: money written in thousands of dong, order status kept in a
note column, a shop-specific walk-in customer label, kinds of information the ledger does not record... The LLM reads the
README and fills the fixed fields (CONVENTION_KEYS); fields the README does not mention stay null / empty. The code checks
each field (valid value, column really exists) before build applies it, and checks again against the data
(build.apply_money_scale). Conventions outside the fixed fields go into 'khac' for the query side to read.
"""
from __future__ import annotations

import json
import re

from . import llm
from .parsers import fold

MONEY_SCALES = (1, 1000, 1000000)
NOT_RECORDED = ("ma_don", "ten_khach", "ma_khach", "ma_hang", "gio_ban", "trang_thai", "gia_von", "nha_cung_cap",
                "nhan_vien", "don_vi")

_SYSTEM = f"""Bạn đọc README mô tả dữ liệu bán hàng của một cửa hàng nhỏ ở Việt Nam và rút ra các quy ước GHI SỐ LIỆU
ảnh hưởng tới cách tính. Chỉ dựa vào chữ trong README, không đoán. Trả về đúng một JSON object với các khoá:
- "don_vi_tien": các cột tiền ghi theo đơn vị nào: 1 (đồng), 1000 (nghìn đồng), 1000000 (triệu đồng). README không nói rõ thì null.
- "cot_trang_thai": nếu README nói trạng thái đơn (huỷ, trả hàng, đang giao...) ghi trong một cột KHÔNG phải cột trạng thái
  riêng (ví dụ cột ghi chú): tên cột đó y như README; không nói thì null.
- "trong_la_hoan_thanh": true nếu README nói đơn không ghi trạng thái là đơn hoàn thành; không nói thì null.
- "nhan_khach_le": các chữ README nói dùng để ghi khách lẻ / khách vãng lai; không có thì [].
- "khong_ghi": thông tin README VIẾT RÕ là sổ không ghi (vd "không ghi mã đơn", "không ghi tên khách"), chọn trong
  {list(NOT_RECORDED)}; README không viết rõ thì [].
- "khac": quy ước khác về cách ghi số liệu mà người tính cần biết, mỗi ý một câu ngắn; không có thì [].
Không giải thích, không markdown."""


def readme_prompt(readme, columns: dict[str, list[str]]) -> str:
    """Opening paragraph + the "Quy ước" section (without the unit-conversion list, which is read separately) + the real column names of each table."""
    parts = ["README:", (readme.intro or "").strip(), "", "Quy ước:"]
    for topic, text in readme.conventions:
        if fold(topic).startswith("quy doi don vi"):
            parts.append(f"- {topic}: (bảng quy đổi đơn vị, đã đọc riêng)")
        else:
            parts.append(f"- {topic}: {text}")
    parts += ["", "Các cột có trong file:"]
    parts += [f"- {t}: {', '.join(cols)}" for t, cols in columns.items()]
    return "\n".join(parts)


def _validate(out: dict, columns: dict[str, list[str]]) -> dict:
    res = {}
    scale = out.get("don_vi_tien")
    if isinstance(scale, (int, float)) and scale in MONEY_SCALES and scale != 1:
        res["money_scale"] = int(scale)
    col = out.get("cot_trang_thai")
    if isinstance(col, str):
        # column names must really exist in the file (compared accent-stripped)
        hit = next((c for cols in columns.values() for c in cols if fold(c) == fold(col)), None)
        if hit:
            res["status_column"] = hit
    if out.get("trong_la_hoan_thanh") is True:
        res["blank_is_done"] = True
    walk = [str(x)[:40] for x in (out.get("nhan_khach_le") or []) if isinstance(x, str) and x.strip()]
    if walk:
        res["walk_in_labels"] = walk[:10]
    nr = [x for x in (out.get("khong_ghi") or []) if x in NOT_RECORDED]
    if nr:
        res["not_recorded"] = nr
    other = [str(x)[:200] for x in (out.get("khac") or []) if isinstance(x, str) and x.strip()]
    if other:
        res["other"] = other[:8]
    return res


# "not recorded" information -> canonical name of the column that records it: if the shop has that column, the LLM misread it
_NOT_RECORDED_COLS = {"ma_don": {"order_id"}, "ten_khach": {"customer_name"}, "ma_khach": {"customer_id"},
                      "ma_hang": {"product_id"}, "gio_ban": {"order_time"}, "trang_thai": {"status"},
                      "gia_von": {"unit_cost"}, "nha_cung_cap": {"supplier_name", "supplier_id"},
                      "nhan_vien": {"staff_name", "staff_id"}, "don_vi": {"unit", "base_unit"}}


def drop_contradicted(conv: dict, present: set[str]) -> dict:
    """Drop a 'not recorded' entry when the data has a column that records exactly that information (canonical name in `present`)."""
    if not conv.get("not_recorded"):
        return conv
    keep = [x for x in conv["not_recorded"] if not (_NOT_RECORDED_COLS.get(x, set()) & present)]
    out = {k: v for k, v in conv.items() if k != "not_recorded"}
    if keep:
        out["not_recorded"] = keep
    if len(keep) < len(conv["not_recorded"]):
        out["not_recorded_dropped"] = [x for x in conv["not_recorded"] if x not in keep]
    return out


def _salvage(text: str) -> dict:
    """Truncated JSON (the LLM wrote too long a 'khac', hit max_tokens): recover each fixed field that comes before 'khac'."""
    out = {}
    if m := re.search(r'"don_vi_tien"\s*:\s*(null|\d+(?:\.\d+)?)', text):
        out["don_vi_tien"] = None if m[1] == "null" else float(m[1])
    if m := re.search(r'"cot_trang_thai"\s*:\s*(null|"((?:[^"\\]|\\.)*)")', text):
        out["cot_trang_thai"] = m[2]
    if m := re.search(r'"trong_la_hoan_thanh"\s*:\s*(true|false|null)', text):
        out["trong_la_hoan_thanh"] = {"true": True, "false": False}.get(m[1])
    for key in ("nhan_khach_le", "khong_ghi"):
        if m := re.search(rf'"{key}"\s*:\s*(\[[^\]]*\])', text):
            try:
                out[key] = json.loads(m[1])
            except json.JSONDecodeError:
                pass
    return out


def read_conventions(readme, columns: dict[str, list[str]]) -> dict:
    """README + {table: [original column names]} -> {money_scale, status_column, blank_is_done, walk_in_labels,
    not_recorded, other} (only the fields the README states). LLM not connected / error -> {}."""
    if readme is None or not llm.available():
        return {}
    try:
        text = llm.chat(_SYSTEM, readme_prompt(readme, columns), max_tokens=500)
    except Exception:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else _salvage(text)
    except json.JSONDecodeError:
        out = _salvage(text)
    return _validate(out, columns) if isinstance(out, dict) else {}
