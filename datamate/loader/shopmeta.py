"""Information shared across a split (calendar, rulebook) and the shop's industry, written into each DuckDB.

- _calendar      : the split's calendar.csv (solar date, weekday, lunar day/month/year, leap, holidays)
- _tet_periods   : the Tet period per rulebook §2: 23rd of the 12th lunar month (lunar year Y-1) -> end of the 10th of the 1st lunar month (lunar year Y)
- _rulebook      : the split's rulebook.md cut by section, so the query side can fetch the right section (e.g. §6 for the shop's industry)
- industry       : inferred from the README, columns, product names; with evidence, and the matching §6 section name if the rulebook has one
Read from the split folder (not hard-coded): each contest set's rulebook may differ.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd

from .parsers import fold


def split_root(split_dir: str) -> str:
    """'.../train/train/data' -> '.../train/train' (where calendar.csv and rulebook.md are)."""
    d = os.path.abspath(split_dir)
    for cand in (d, os.path.dirname(d)):
        if os.path.exists(os.path.join(cand, "calendar.csv")) or os.path.exists(os.path.join(cand, "rulebook.md")):
            return cand
    return os.path.dirname(d)


def load_calendar(root: str) -> pd.DataFrame | None:
    p = os.path.join(root, "calendar.csv")
    if not os.path.exists(p):
        return None
    # an odd calendar (different column names, unreadable) must not kill the whole set: with no calendar only the parts that use it are lost
    try:
        c = pd.read_csv(p)
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if "date" not in c.columns:
        # a date column with another name ("ngay"...): the first column in which almost all cells read as dates
        dates = [col for col in c.columns
                 if c[col].astype(str).str.fullmatch(r"\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*").mean() >= 0.95]
        if not dates:
            return None
        c = c.rename(columns={dates[0]: "date"})
    c["date"] = pd.to_datetime(c["date"].astype(str), errors="coerce", format="mixed")
    c = c[c["date"].notna()]
    if "lunar_leap" in c.columns:
        c["lunar_leap"] = c["lunar_leap"].astype(str).str.lower().isin(["true", "1"])
    return c


def tet_periods(cal: pd.DataFrame | None) -> pd.DataFrame:
    """For each lunar year Y: from the solar date of the lunar 23/12 (lunar year Y-1, non-leap month) to the end of the solar date of the lunar 10/1 (year Y)."""
    cols = ["tet_year", "start_date", "end_date", "days"]
    if cal is None or not {"lunar_day", "lunar_month", "lunar_year"} <= set(cal.columns):
        return pd.DataFrame(columns=cols)
    c = cal[~cal.get("lunar_leap", False)] if "lunar_leap" in cal.columns else cal
    start = c[(c.lunar_month == 12) & (c.lunar_day == 23)].set_index("lunar_year")["date"]
    end = c[(c.lunar_month == 1) & (c.lunar_day == 10)].set_index("lunar_year")["date"]
    rows = [(y, start[y - 1], end[y], (end[y] - start[y - 1]).days + 1) for y in end.index if y - 1 in start.index]
    return pd.DataFrame(rows, columns=cols)


def parse_rulebook(root: str) -> pd.DataFrame:
    """Cut the rulebook by heading: (order, level, heading, heading path, content)."""
    p = os.path.join(root, "rulebook.md")
    cols = ["seq", "level", "heading", "path", "text"]
    if not os.path.exists(p):
        return pd.DataFrame(columns=cols)
    rows, stack, cur, buf = [], [], None, []

    def flush():
        if cur is not None:
            rows.append((len(rows), cur[0], cur[1], " > ".join(h for _, h in stack), "\n".join(buf).strip()))
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            m = re.match(r"^(#{1,6})\s+(.*)", line.rstrip("\n"))
            if m:
                flush()
                level, head = len(m[1]), m[2].strip()
                stack = [(lv, h) for lv, h in stack if lv < level] + [(level, head)]
                cur, buf = (level, head), []
            else:
                buf.append(line.rstrip("\n"))
    flush()
    return pd.DataFrame(rows, columns=cols)


# industry

# (industry, README keywords, product-name keywords). Industry names match how rulebook §6 names its sections.
INDUSTRIES = [
    ("Cà phê", r"ca phe|cafe|coffee|tra sua|quan nuoc", r"ca phe|tra sua|bac xiu|tra dao|matcha|\((m|l)\)$"),
    ("Nhà thuốc", r"nha thuoc|hieu thuoc|quay thuoc|duoc pham", r"\bvien\b|\bvi\b|siro|\bmg\b|thuoc|khau trang y te"),
    ("Vật liệu xây dựng", r"vat lieu xay dung|\bvlxd\b|xi mang|sat thep", r"xi mang|thep|gach|\bton\b|da 1x2|cat (xay|vang)"),
    ("Đồ trẻ em", r"me va be|me & be|\bme be\b|tre em|tre so sinh|do so sinh|\bem be\b", r"\bsua\b.*\b(so|g)\b|\bbim\b|\bta (dan|quan)\b|binh sua|khan uot"),
    ("Sửa xe", r"sua xe|tiem xe|garage|gara", r"thay nhot|bugi|vo xe|ruot xe|ma phanh"),
    ("Cửa hàng điện thoại", r"dien thoai|di dong|smartphone", r"iphone|samsung|\bgb\b|op lung|cuong luc|sac nhanh"),
    ("Nhà hàng", r"nha hang|quan an|quan nhau|quan lau", r"\blau\b|\bcom\b|\bxao\b|nuong|chien|goi\b"),
    ("Tạp hoá", r"tap hoa|bach hoa|sieu thi mini|minimart", r"\bmi\b|nuoc mam|dau an|bot giat|snack|gao\b"),
]
# characteristic column (canonical name) -> industry
COLUMN_SIGNALS = {"active_ingredient": "Nhà thuốc", "is_prescription": "Nhà thuốc", "batch_id": "Nhà thuốc",
                  "delivery_round": "Vật liệu xây dựng", "driver": "Vật liệu xây dựng",
                  "imei": "Cửa hàng điện thoại", "table_no": "Nhà hàng", "guests": "Nhà hàng"}


def infer_industry(readme, built, rulebook: pd.DataFrame) -> dict:
    """Score of each industry = README (3) + characteristic columns (2 per column) + share of sales rows with an industry keyword (at most 3).
    Return {industry, industry_score, industry_evidence, rulebook_section}."""
    score, why = Counter(), {}
    text = fold(" ".join([readme.intro if readme else ""] + [b.label for b in built]))
    for name, kw, _ in INDUSTRIES:
        if re.search(kw, text):
            score[name] += 3
            why.setdefault(name, []).append("README")
    cols = {c for b in built for c in b.df.columns}
    for col, name in COLUMN_SIGNALS.items():
        if col in cols:
            score[name] += 2
            why.setdefault(name, []).append(f"cột {col}")
    # product name in the sales rows; if the sales rows only record the product code, take it from the catalogue
    names = []
    for roles in (("sales_lines",), ("products", "stock", "purchase_lines", "price_history")):
        names = [n for b in built if b.role in roles and "product_name" in b.df.columns
                 for n in b.df["product_name"].dropna().astype(str).unique()]
        if names:
            break
    names = pd.Series(names)
    if len(names):
        f = names.map(fold)
        for name, _, kw in INDUSTRIES:
            rx = re.compile(kw)
            share = f.map(lambda s: bool(rx.search(s))).mean()
            if share >= 0.05:
                score[name] += round(3 * min(share / 0.4, 1), 2)
                why.setdefault(name, []).append(f"{share:.0%} tên hàng")
    if not score:
        return {"industry": None, "industry_score": 0, "industry_evidence": None, "rulebook_section": None}
    best, s = score.most_common(1)[0]
    section = None
    if rulebook is not None and len(rulebook):
        heads = rulebook[rulebook.path.str.contains("Theo ngành", na=False) & (rulebook.level >= 3)]
        hit = heads[heads.heading.map(fold) == fold(best)]
        if not len(hit):            # the section name differs from the industry name: match the industry's README keywords ("Tiệm cà phê", "Mẹ & bé")
            kw = next((k for n, k, _ in INDUSTRIES if n == best), None)
            if kw:
                # drop a section specific to one shop ("... (<shop id>)"): the answering side picks that section by shop id
                hit = heads[heads.heading.map(lambda h: bool(re.search(kw, fold(h)))
                                              and not re.search(r"\(\s*[a-z]+_\d+\s*\)", fold(h)))]
        section = hit.heading.iloc[0] if len(hit) else None
        if section:
            best = section         # industry label exactly as the rulebook's section name at run time (the answering side picks §6 by this name)
    ev = "; ".join(f"{k}: {', '.join(v)} ({score[k]:g})" for k, v in sorted(why.items(), key=lambda x: -score[x[0]]))
    return {"industry": best, "industry_score": float(s), "industry_evidence": ev, "rulebook_section": section}
